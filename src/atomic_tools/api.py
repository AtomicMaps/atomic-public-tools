"""Library entry points for web services and ``validate --api-mode``.

The validate functions never prompt, print or raise; failures come back as
``status == "error"``. They block on exiftool/pdal, so call them from a worker
thread in async code (a plain ``def`` FastAPI endpoint does this).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from atomic_tools.client_sidecar import SidecarMergeError
from atomic_tools.commands.sidecar import (
    NoSupportedFilesError,
    PointCloudCRSError,
    _build_sidecar,
    _crs_failures,
)
from atomic_tools.errors import InvalidInputError
from atomic_tools.models import (
    SCHEMA_VERSION,
    DependencyReport,
    ErrorInfo,
    ToolStatus,
    ValidationResponse,
)
from atomic_tools.utils.aws_errors import find_auth_error
from atomic_tools.utils.extractors import PdalNotFoundError, find_pdal_bin
from atomic_tools.utils.utils import DataTypeEnum, ExiftoolNotFoundError, find_exiftool_bin
from atomic_tools.validators.constants import DEFAULT_ROW_NAME
from atomic_tools.validators.report import LintReport
from atomic_tools.validators.sidecar import _as_csv_strings, lint_sidecar_df

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SECTIONS",
    "SCHEMA_VERSION",
    "SECTIONS",
    "UploadedFile",
    "check_dependencies",
    "validate_path",
    "validate_upload",
]

SECTIONS: tuple[str, ...] = ("findings", "missing_data", "metadata", "details")
# metadata is opt-in: one dict per file gets large.
DEFAULT_SECTIONS: frozenset[str] = frozenset({"findings", "missing_data", "details"})

# Some findings embed click colours (e.g. the geography histogram).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str | None) -> str | None:
    return _ANSI_RE.sub("", text) if text else text


@dataclass
class ValidationResult:
    df: Any  # pandas.DataFrame with all-string cells
    report: LintReport
    detected_types: set[DataTypeEnum]
    skipped_vector: list[str]
    crs_failures: list[str]
    unclassified: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.report.has_errors() or bool(self.crs_failures)

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        if self.report.warnings():
            return "passed_with_warnings"
        return "passed"


def parse_include(value: str | Iterable[str] | None) -> frozenset[str]:
    """``None``/empty → :data:`DEFAULT_SECTIONS`; ``"all"`` → every section."""
    if value is None:
        return DEFAULT_SECTIONS
    if isinstance(value, str):
        parts = value.split(",")
    else:
        try:
            parts = list(value)
        except TypeError:
            raise InvalidInputError(
                f"include must be a comma-separated string or a list of section "
                f"names, not {type(value).__name__}."
            ) from None
        bad = [p for p in parts if not isinstance(p, str)]
        if bad:
            raise InvalidInputError(f"include entries must be strings, got {bad!r}.")
    names = {p.strip().lower() for p in parts if p and p.strip()}
    if not names:
        return DEFAULT_SECTIONS
    if "all" in names:
        return frozenset(SECTIONS)
    unknown = sorted(names - set(SECTIONS))
    if unknown:
        raise InvalidInputError(
            f"Unknown include section(s): {unknown}. "
            f"Choose from {list(SECTIONS)} or 'all'."
        )
    return frozenset(names)


def run_validation(
    path: str,
    *,
    data_type: DataTypeEnum | None = None,
    client_sidecar: str | None = None,
    client_schema: str | None = None,
    full: bool = False,
    spatial_reference: str | None = None,
    ignore_missing_orientation: bool = False,
    coco: str | None = None,
    quiet: bool = False,
) -> ValidationResult:
    """Build the sidecar in memory and lint it. Unlike :func:`validate_path`, raises."""
    logger.info(
        f"Starting validation — path={path!r} data_type={data_type} "
        f"client_sidecar={client_sidecar!r} client_schema={client_schema!r} "
        f"full={full} spatial_reference={spatial_reference!r} coco={coco!r}"
    )
    unclassified: list[str] = []
    df, backend, detected_types, skipped_vector = _build_sidecar(
        directory=path,
        data_type=data_type,
        client_sidecar=client_sidecar,
        client_schema=client_schema,
        full=full,
        spatial_reference=spatial_reference,
        quiet=quiet,
        unclassified_out=unclassified,
    )
    df = _as_csv_strings(df)
    crs_failures = _crs_failures(df)

    logger.info("Linting in-memory sidecar (not saved)…")
    report = lint_sidecar_df(
        df,
        normalize=False,
        final=True,
        data_type=data_type,
        input_files_path=path,
        input_backend=backend,
        ignore_missing_orientation=ignore_missing_orientation,
        coco_path=coco,
        # Informational run: only sidecar generation blocks on this.
        coco_not_on_disk_is_error=False,
    )
    return ValidationResult(
        df=df,
        report=report,
        detected_types=set(detected_types),
        skipped_vector=list(skipped_vector),
        crs_failures=list(crs_failures),
        unclassified=unclassified,
    )


def build_payload(result: ValidationResult, include: Iterable[str]) -> ValidationResponse:
    """Serialise a :class:`ValidationResult` to the response model."""
    include = set(include)
    report = result.report
    df = result.df
    filenames = df["Filename"].astype(str) if "Filename" in df.columns else None
    files_scanned = (
        int((filenames != DEFAULT_ROW_NAME).sum()) if filenames is not None else len(df)
    )

    # Unrequested sections stay unset so exclude_unset drops them.
    fields: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": result.status,
        "summary": {
            "errors": len(report.errors()),
            "warnings": len(report.warnings()),
            "infos": len(report.infos()),
            "files_scanned": files_scanned,
            "detected_types": sorted(t.value for t in result.detected_types),
            "crs_failures": len(result.crs_failures),
            "skipped_vector": len(result.skipped_vector),
            "unclassified": len(result.unclassified),
        },
        "error": None,
    }
    if "findings" in include:
        fields["findings"] = [
            {**f, "message": _plain(f["message"]), "fix_hint": _plain(f["fix_hint"])}
            for f in report.findings_as_dicts()
        ]
    if "missing_data" in include:
        fields["missing_data"] = (
            report.missing_data.to_dict() if report.missing_data is not None else None
        )
    if "metadata" in include:
        fields["metadata"] = df.to_dict(orient="records")
    if "details" in include:
        fields["details"] = {
            "crs_failures": result.crs_failures,
            "skipped_vector": result.skipped_vector,
            "unclassified": result.unclassified,
            "coco": report.coco_impact.to_dict() if report.coco_impact else None,
        }
    return ValidationResponse.model_validate(fields)


def error_payload(error_type: str, message: str) -> ValidationResponse:
    return ValidationResponse(
        schema_version=SCHEMA_VERSION,
        status="error",
        summary=None,
        error=ErrorInfo(type=error_type, message=_plain(message) or ""),
    )


def _classify_exception(exc: BaseException) -> str:
    if find_auth_error(exc) is not None:
        return "auth_error"
    if isinstance(exc, (PdalNotFoundError, ExiftoolNotFoundError)):
        return "dependency_missing"
    if isinstance(exc, NoSupportedFilesError):
        return "no_data"
    if isinstance(
        exc,
        (
            # Not bare ValueError: one from pandas, pyproj, … is a bug.
            InvalidInputError,
            NotImplementedError,
            FileNotFoundError,
            SidecarMergeError,
            PointCloudCRSError,
        ),
    ):
        return "invalid_input"
    return "internal_error"


def validate_path(
    path: str,
    *,
    data_type: DataTypeEnum | str | None = None,
    client_sidecar: str | None = None,
    client_schema: str | None = None,
    full: bool = False,
    spatial_reference: str | None = None,
    ignore_missing_orientation: bool = False,
    coco: str | None = None,
    include: str | Iterable[str] | None = None,
) -> ValidationResponse:
    """Validate a directory/prefix or a single file/object. Never raises.

    A folder is scanned recursively, so never point this at a shared one.
    """
    try:
        sections = parse_include(include)
        if not path:
            raise InvalidInputError("path is required.")
        dt = DataTypeEnum(data_type) if isinstance(data_type, str) else data_type
    except ValueError as e:
        return error_payload("invalid_input", str(e))

    try:
        result = run_validation(
            path,
            data_type=dt,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
            spatial_reference=spatial_reference,
            ignore_missing_orientation=ignore_missing_orientation,
            coco=coco,
            quiet=True,
        )
    except Exception as e:  # noqa: BLE001 — API contract: never raise
        logger.error(f"Validation failed: {e}", exc_info=logger.isEnabledFor(logging.INFO))
        return error_payload(_classify_exception(e), str(e))

    try:
        return build_payload(result, sections)
    except Exception as e:  # noqa: BLE001 — API contract: never raise
        logger.exception("Could not build the validation response")
        return error_payload("internal_error", f"Could not build the response: {e}")


# ---- uploads -------------------------------------------------------------------


@dataclass
class UploadedFile:
    """A file received over the wire: its bytes (or a binary stream) and the
    name the user's file had. The name matters — type detection, filename
    dates, and client-sidecar/COCO matching all use it."""

    content: bytes | IO[bytes]
    filename: str


# Accepted for every upload argument: an UploadedFile, or anything shaped like
# FastAPI/Starlette's UploadFile (``.file`` stream + ``.filename``).
UploadLike = UploadedFile | Any

_COPY_CHUNK = 1024 * 1024
_EXTRA_ROLES = ("client_sidecar", "client_schema", "coco")
_MAX_NAME_BYTES = 255


class UploadError(InvalidInputError):
    """An upload that can't be validated as given (bad name, too large…)."""


def _safe_filename(name: str | None, role: str) -> str:
    """Basename of a client-supplied filename, safe to create in our temp dir.

    Keeps the original name (detection depends on it) but drops any directory
    part, so ``../../etc/x.jpg`` becomes ``x.jpg``.
    """
    if not name or not name.strip():
        raise UploadError(f"The {role} upload has no filename; send its original name.")
    base = re.split(r"[\\/]", name)[-1]
    base = "".join(ch for ch in base if ch.isprintable()).strip()
    if base in ("", ".", ".."):
        raise UploadError(f"The {role} upload has an invalid filename: {name!r}.")
    if len(base.encode("utf-8")) > _MAX_NAME_BYTES:
        stem, ext = os.path.splitext(base)
        budget = _MAX_NAME_BYTES - len(ext.encode("utf-8"))
        if budget < 1:
            raise UploadError(f"The {role} upload's filename is too long: {name[:40]!r}…")
        stem = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
        base = stem + ext
    return base


def _unpack_upload(obj: UploadLike, filename: str | None, role: str) -> tuple[Any, str]:
    if isinstance(obj, UploadedFile):
        return obj.content, filename or obj.filename
    if hasattr(obj, "file") and hasattr(obj, "filename"):  # Starlette UploadFile
        return obj.file, filename or obj.filename
    if isinstance(obj, str):
        raise UploadError(
            f"The {role} upload is a str; pass bytes or a binary stream, not text."
        )
    if isinstance(obj, (bytes, bytearray, memoryview)) or hasattr(obj, "read"):
        if not filename:
            raise UploadError(
                f"The {role} upload is raw bytes/stream; pass its original filename "
                "(or wrap it in UploadedFile)."
            )
        return obj, filename
    raise UploadError(f"Unsupported {role} upload type: {type(obj).__name__}.")


def _is_empty_form_field(obj: UploadLike) -> bool:
    """An optional file input the user left empty.

    HTML forms still send an empty file part, which FastAPI/Starlette turn into
    an ``UploadFile`` with ``filename=""`` and no bytes rather than ``None``.
    """
    if isinstance(obj, UploadedFile) or not (hasattr(obj, "file") and hasattr(obj, "filename")):
        return False
    if obj.filename:
        return False
    stream = obj.file
    try:
        pos = stream.tell()
        empty = stream.read(1) in (b"", "")
        stream.seek(pos)
    except (AttributeError, OSError, ValueError):
        return False  # can't tell; let _save_upload report it
    return empty


def _save_upload(
    obj: UploadLike,
    dest_dir: Path,
    role: str,
    *,
    filename: str | None = None,
    max_bytes: int | None = None,
    require_extension: bool = False,
) -> Path:
    content, raw_name = _unpack_upload(obj, filename, role)
    name = _safe_filename(raw_name, role)
    if require_extension and not Path(name).suffix:
        raise UploadError(
            f"The {role} upload {name!r} has no file extension; the file type is "
            "detected from it, so send the file under its original name."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    if isinstance(content, str):
        raise UploadError(f"The {role} upload {name!r} is text; pass bytes or a binary stream.")
    if not isinstance(content, (bytes, bytearray, memoryview)) and not hasattr(content, "read"):
        raise UploadError(
            f"The {role} upload {name!r} has unsupported content: {type(content).__name__}."
        )

    written = 0
    with open(dest, "wb") as out:
        if isinstance(content, (bytes, bytearray, memoryview)):
            written = memoryview(content).nbytes
            if max_bytes is None or written <= max_bytes:
                out.write(content)  # no intermediate copy of a large buffer
        else:
            # Rewind if possible; an unseekable stream is read from where it is.
            with contextlib.suppress(AttributeError, OSError, ValueError):
                content.seek(0)
            while chunk := content.read(_COPY_CHUNK):
                if isinstance(chunk, str):
                    raise UploadError(
                        f"The {role} upload {name!r} was opened in text mode; "
                        "open it in binary mode ('rb')."
                    )
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    break
                out.write(chunk)
    if max_bytes is not None and written > max_bytes:
        raise UploadError(f"The {role} upload {name!r} exceeds the {max_bytes}-byte limit.")
    return dest


def _scrub(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_scrub(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, replacements) for k, v in value.items()}
    return value


_UPLOAD_SUBDIRS = ("data", *(f"extras{os.sep}{role}" for role in _EXTRA_ROLES))


def _path_replacements(tmp_root: Path, saved: list[Path]) -> list[tuple[str, str]]:
    """Map server temp paths (as given and as resolved — /var vs /private/var
    on macOS) to the bare uploaded filename, longest first. Covers files that
    were never saved too (e.g. an OSError naming the destination path)."""
    pairs: dict[str, str] = {}
    for p in saved:
        for variant in {str(p), str(p.resolve())}:
            pairs[variant] = p.name
    for root in {str(tmp_root), str(tmp_root.resolve())}:
        for sub in _UPLOAD_SUBDIRS:
            pairs[f"{root}{os.sep}{sub}{os.sep}"] = ""
        pairs[root + os.sep] = ""
        pairs[root] = "<upload>"
    return sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True)


def _scrubbed(
    response: ValidationResponse, tmp_root: Path, saved: list[Path]
) -> ValidationResponse:
    try:
        cleaned = _scrub(
            response.model_dump(mode="json", exclude_unset=True),
            _path_replacements(tmp_root, saved),
        )
        return ValidationResponse.model_validate(cleaned)
    except Exception:  # noqa: BLE001 — API contract: never raise (or leak paths)
        logger.exception("Could not scrub the validation response")
        return error_payload("internal_error", "Could not build the response.")


def validate_upload(
    file: UploadLike,
    *,
    filename: str | None = None,
    client_sidecar: UploadLike | None = None,
    client_schema: UploadLike | None = None,
    coco: UploadLike | None = None,
    data_type: DataTypeEnum | str | None = None,
    full: bool = False,
    spatial_reference: str | None = None,
    ignore_missing_orientation: bool = False,
    include: str | Iterable[str] | None = None,
    work_dir: str | os.PathLike[str] | None = None,
    max_bytes: int | None = None,
) -> ValidationResponse:
    """Validate one uploaded data file (plus optional companion uploads). Never raises.

    ``file`` (and each of ``client_sidecar``/``client_schema``/``coco``) may be a
    FastAPI/Starlette ``UploadFile``, an :class:`UploadedFile`, or — for ``file``
    only — raw bytes / a binary stream together with ``filename``. ``filename``
    overrides the upload's own name for ``file``.

    Each upload is written under its original basename to a fresh private temp
    dir (inside ``work_dir`` if given), validated with :func:`validate_path`,
    and deleted afterwards. Temp paths in messages are replaced by the original
    filename. ``max_bytes`` caps each upload's size.
    """
    try:
        tmp_root = Path(tempfile.mkdtemp(prefix="amtools-", dir=work_dir))
    except OSError as e:
        # Log the path for operators; don't send it to the HTTP client.
        logger.error(f"Could not create a temp dir in {work_dir!r}: {e}")
        reason = e.strerror or type(e).__name__
        return error_payload("internal_error", f"Could not create a temp dir ({reason}).")

    saved: list[Path] = []
    try:
        try:
            response = _save_and_validate(
                tmp_root,
                saved,
                file,
                filename=filename,
                extras={"client_sidecar": client_sidecar, "client_schema": client_schema,
                        "coco": coco},
                max_bytes=max_bytes,
                data_type=data_type,
                full=full,
                spatial_reference=spatial_reference,
                ignore_missing_orientation=ignore_missing_orientation,
                include=include,
            )
        except UploadError as e:
            response = error_payload("invalid_input", str(e))
        except OSError as e:
            logger.error(f"Could not save upload: {e}")
            response = error_payload("internal_error", f"Could not save upload: {e}")
        except Exception as e:  # noqa: BLE001 — API contract: never raise
            logger.exception("Unexpected error while handling an upload")
            response = error_payload("internal_error", f"Unexpected error: {e}")
        # Scrub before the temp dir goes, so paths still resolve the same way.
        return _scrubbed(response, tmp_root, saved)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _save_and_validate(
    tmp_root: Path,
    saved: list[Path],
    file: UploadLike,
    *,
    filename: str | None,
    extras: dict[str, UploadLike | None],
    max_bytes: int | None,
    **validate_kwargs: Any,
) -> ValidationResponse:
    """Write the uploads into ``tmp_root`` (recording each in ``saved``) and
    validate the data file. Raises UploadError/OSError on save problems."""
    data_path = _save_upload(
        file,
        tmp_root / "data",
        "data file",
        filename=filename,
        max_bytes=max_bytes,
        require_extension=True,
    )
    saved.append(data_path)
    extra_paths: dict[str, str | None] = {}
    for role in _EXTRA_ROLES:
        obj = extras.get(role)
        if obj is None or _is_empty_form_field(obj):
            extra_paths[role] = None
            continue
        p = _save_upload(
            obj, tmp_root / "extras" / role, role.replace("_", " "), max_bytes=max_bytes
        )
        saved.append(p)
        extra_paths[role] = str(p)

    return validate_path(
        str(data_path),
        client_sidecar=extra_paths["client_sidecar"],
        client_schema=extra_paths["client_schema"],
        coco=extra_paths["coco"],
        **validate_kwargs,
    )


# ---- health ---------------------------------------------------------------------


def _probe_tool(path: str, args: list[str], pattern: str | None = None) -> ToolStatus:
    """Run ``path args`` and report whether it works, with its version."""
    try:
        proc = subprocess.run(
            [path, *args],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return ToolStatus(found=True, usable=False, path=path, detail=f"Could not run: {e}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()
        return ToolStatus(
            found=True,
            usable=False,
            path=path,
            detail=f"Exited {proc.returncode}: {err[0] if err else 'no output'}",
        )
    out = proc.stdout.strip()
    version = None
    if pattern and (m := re.search(pattern, out)):
        version = m.group(1)
    elif out:
        version = out.splitlines()[0]
    return ToolStatus(found=True, usable=True, path=path, version=version)


def check_dependencies() -> DependencyReport:
    """Report whether exiftool, pdal and pye57 are usable. Never raises.

    Each binary is located *and run*, so a container where a tool is present
    but broken (e.g. missing shared libraries) is caught at deploy time by a
    ``/health`` endpoint rather than on the first upload.
    """
    try:
        exiftool = _probe_tool(find_exiftool_bin(), ["-ver"])
    except ExiftoolNotFoundError as e:
        exiftool = ToolStatus(found=False, usable=False, detail=str(e))

    try:
        pdal = _probe_tool(find_pdal_bin(), ["--version"], r"pdal\s+(\S+)")
    except PdalNotFoundError as e:
        pdal = ToolStatus(found=False, usable=False, detail=str(e).strip().splitlines()[0])

    try:
        from importlib.metadata import version

        import pye57  # noqa: F401

        pye57_status = ToolStatus(found=True, usable=True, version=version("pye57"))
    except Exception as e:  # noqa: BLE001 — ImportError, broken native lib, …
        pye57_status = ToolStatus(found=False, usable=False, detail=f"{type(e).__name__}: {e}")

    return DependencyReport(
        ok=exiftool.usable and pdal.usable and pye57_status.usable,
        exiftool=exiftool,
        pdal=pdal,
        pye57=pye57_status,
    )
