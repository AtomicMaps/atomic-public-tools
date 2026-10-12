"""Programmatic entry points — for web services and the ``validate --api-mode`` CLI.

* :func:`validate_upload` — validate a file received over HTTP (e.g. a FastAPI
  ``UploadFile``), plus optional client sidecar / schema / COCO uploads. It saves
  them under their original names in a private temp dir, validates, scrubs the
  temp paths out of the messages, and cleans up.
* :func:`validate_path` — validate a directory/prefix or a single file that is
  already on disk or in an object store.
* :func:`check_dependencies` — report whether exiftool/pdal/pye57 are usable
  (for a ``/health`` endpoint).

The validate functions never prompt, never print, and never raise: failures come
back as ``status == "error"`` with an ``error`` object. They return a
:class:`~atomic_tools.models.ValidationResponse`; sections not requested via
``include`` are left unset (serialise with ``exclude_unset=True``).

All of these are synchronous and block on subprocesses (exiftool/pdal). In an
async framework call them from a worker thread (in FastAPI, a plain ``def``
endpoint does this automatically).
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

from atomic_tools.commands.sidecar import _build_sidecar, _crs_failures
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
# metadata is opt-in: one dict per file can be large for big directories.
DEFAULT_SECTIONS: frozenset[str] = frozenset({"findings", "missing_data", "details"})

# Some findings embed click-coloured text (e.g. the geography histogram); a JSON
# consumer wants plain text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str | None) -> str | None:
    return _ANSI_RE.sub("", text) if text else text


@dataclass
class ValidationResult:
    """Everything a validation run produces, before any rendering."""

    df: Any  # pandas.DataFrame — the assembled (unsaved) sidecar
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
    """Normalise an ``include`` spec (comma string or iterable) to section names.

    ``None``/empty → :data:`DEFAULT_SECTIONS`; ``"all"`` → every section.
    Raises ``ValueError`` naming any unknown section.
    """
    if value is None:
        return DEFAULT_SECTIONS
    parts = value.split(",") if isinstance(value, str) else list(value)
    names = {p.strip().lower() for p in parts if p and p.strip()}
    if not names:
        return DEFAULT_SECTIONS
    if "all" in names:
        return frozenset(SECTIONS)
    unknown = sorted(names - set(SECTIONS))
    if unknown:
        raise ValueError(
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
    """Build the sidecar in memory and lint it without persisting anything.

    Shared by the interactive CLI and :func:`validate_path`. Raises on
    extraction/build failures; callers decide how to surface them.
    """
    logger.info(
        f"Starting validation — path={path!r} data_type={data_type} "
        f"client_sidecar={client_sidecar!r} client_schema={client_schema!r} "
        f"full={full} spatial_reference={spatial_reference!r} coco={coco!r}"
    )
    unclassified: list[str] = []
    # validate never persists, so the backend is unused.
    df, _, detected_types, skipped_vector = _build_sidecar(
        directory=path,
        data_type=data_type,
        client_sidecar=client_sidecar,
        client_schema=client_schema,
        full=full,
        spatial_reference=spatial_reference,
        quiet=quiet,
        unclassified_out=unclassified,
    )
    crs_failures = _crs_failures(df)

    logger.info("Linting in-memory sidecar (not saved)…")
    report = lint_sidecar_df(
        df,
        final=True,
        data_type=data_type,
        input_files_path=path,
        ignore_missing_orientation=ignore_missing_orientation,
        coco_path=coco,
        # validate is informational — a COCO referencing images absent from
        # the scanned directory is reported as a warning here, not a hard
        # failure (it blocks only when generating a sidecar that ships).
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

    # Only requested sections are passed, so the rest stay *unset* and are
    # dropped by exclude_unset serialisation.
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
        fields["metadata"] = _as_csv_strings(df).to_dict(orient="records")
    if "details" in include:
        fields["details"] = {
            "crs_failures": result.crs_failures,
            "skipped_vector": result.skipped_vector,
            "unclassified": result.unclassified,
            "coco": report.coco_impact.to_dict() if report.coco_impact else None,
        }
    return ValidationResponse.model_validate(fields)


def error_payload(error_type: str, message: str) -> ValidationResponse:
    """The response returned when validation could not run to completion."""
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
    if isinstance(exc, (ValueError, NotImplementedError, FileNotFoundError)):
        # FileNotFoundError: from_path() was given a local path that doesn't exist.
        return "invalid_input"
    if isinstance(exc, RuntimeError):
        # _build_sidecar raises RuntimeError when no supported files are found.
        return "no_data"
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
    """Validate ``path`` (a directory/prefix or a single file/object). Never raises.

    ``data_type`` accepts a :class:`DataTypeEnum` or its string value; ``None``
    auto-detects per file. ``include`` selects response sections (see
    :func:`parse_include`).

    Point ``path`` at the file itself or a folder holding only this request's
    data — a folder is scanned recursively.
    """
    try:
        sections = parse_include(include)
        if not path:
            raise ValueError("path is required.")
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

    return build_payload(result, sections)


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
_MAX_NAME_BYTES = 255


class UploadError(ValueError):
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
        stem = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
        base = stem + ext
    return base


def _unpack_upload(obj: UploadLike, filename: str | None, role: str) -> tuple[Any, str]:
    if isinstance(obj, UploadedFile):
        return obj.content, filename or obj.filename
    if hasattr(obj, "file") and hasattr(obj, "filename"):  # Starlette UploadFile
        return obj.file, filename or obj.filename
    if isinstance(obj, (bytes, bytearray, memoryview)) or hasattr(obj, "read"):
        if not filename:
            raise UploadError(
                f"The {role} upload is raw bytes/stream; pass its original filename "
                "(or wrap it in UploadedFile)."
            )
        return obj, filename
    raise UploadError(f"Unsupported {role} upload type: {type(obj).__name__}.")


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

    written = 0
    with open(dest, "wb") as out:
        if isinstance(content, (bytes, bytearray, memoryview)):
            data = bytes(content)
            written = len(data)
            if max_bytes is None or written <= max_bytes:
                out.write(data)
        else:
            # Rewind if possible; an unseekable stream is read from where it is.
            with contextlib.suppress(AttributeError, OSError, ValueError):
                content.seek(0)
            while chunk := content.read(_COPY_CHUNK):
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


def _path_replacements(tmp_root: Path, saved: list[Path]) -> list[tuple[str, str]]:
    """Map server temp paths (as given and as resolved — /var vs /private/var
    on macOS) to the bare uploaded filename, longest first."""
    pairs: dict[str, str] = {}
    for p in saved:
        for variant in {str(p), str(p.resolve())}:
            pairs[variant] = p.name
    for variant in {str(tmp_root), str(tmp_root.resolve())}:
        pairs[variant + os.sep] = ""
        pairs[variant] = "<upload>"
    return sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True)


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
        return error_payload("internal_error", f"Could not create a temp dir: {e}")

    saved: list[Path] = []
    try:
        try:
            data_path = _save_upload(
                file,
                tmp_root / "data",
                "data file",
                filename=filename,
                max_bytes=max_bytes,
                require_extension=True,
            )
            saved.append(data_path)
            extras: dict[str, str | None] = {}
            for role, obj in (
                ("client_sidecar", client_sidecar),
                ("client_schema", client_schema),
                ("coco", coco),
            ):
                if obj is None:
                    extras[role] = None
                    continue
                p = _save_upload(
                    obj, tmp_root / "extras" / role, role.replace("_", " "),
                    max_bytes=max_bytes,
                )
                saved.append(p)
                extras[role] = str(p)
        except UploadError as e:
            return error_payload("invalid_input", str(e))
        except OSError as e:
            logger.error(f"Could not save upload: {e}")
            return error_payload("internal_error", f"Could not save upload: {e}")

        response = validate_path(
            str(data_path),
            data_type=data_type,
            client_sidecar=extras["client_sidecar"],
            client_schema=extras["client_schema"],
            full=full,
            spatial_reference=spatial_reference,
            ignore_missing_orientation=ignore_missing_orientation,
            coco=extras["coco"],
            include=include,
        )
        scrubbed = _scrub(
            response.model_dump(mode="json", exclude_unset=True),
            _path_replacements(tmp_root, saved),
        )
        return ValidationResponse.model_validate(scrubbed)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


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
