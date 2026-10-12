"""Programmatic entry point for validation — the engine behind ``validate --api-mode``.

:func:`validate_path` validates a directory/prefix or a single file that is
already on disk or in an object store. It never prompts, never prints, and
never raises: failures come back as ``status == "error"`` with an ``error``
object. It returns a :class:`~atomic_tools.models.ValidationResponse`; sections
not requested via ``include`` are left unset (serialise with
``exclude_unset=True``).

It is synchronous and blocks on subprocesses (exiftool/pdal). In an async
framework call it from a worker thread.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

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
    ErrorInfo,
    ValidationResponse,
)
from atomic_tools.utils.aws_errors import find_auth_error
from atomic_tools.utils.extractors import PdalNotFoundError
from atomic_tools.utils.utils import DataTypeEnum, ExiftoolNotFoundError
from atomic_tools.validators.constants import DEFAULT_ROW_NAME
from atomic_tools.validators.report import LintReport
from atomic_tools.validators.sidecar import _as_csv_strings, lint_sidecar_df

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SECTIONS",
    "SCHEMA_VERSION",
    "SECTIONS",
    "validate_path",
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

    df: Any  # pandas.DataFrame — the assembled (unsaved) sidecar, all-string cells
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
    Raises :class:`InvalidInputError` naming any unknown section, or when
    ``value`` isn't a string or an iterable of strings.
    """
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
    # Normalise once to the CSV string form; lint and the metadata section both
    # use this frame.
    df = _as_csv_strings(df)
    crs_failures = _crs_failures(df)

    logger.info("Linting in-memory sidecar (not saved)…")
    report = lint_sidecar_df(
        df,
        normalize=False,
        final=True,
        data_type=data_type,
        input_files_path=path,
        # Reuse the scan's backend: no second session/listing for the
        # file-inventory check.
        input_backend=backend,
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
    if isinstance(exc, NoSupportedFilesError):
        return "no_data"
    if isinstance(
        exc,
        (
            # Raised deliberately for bad paths, flag values and uploads. A bare
            # ValueError (pandas, JSON, …) is a bug, not the caller's fault.
            InvalidInputError,
            NotImplementedError,
            # from_path() was given a local path (or a client file) that doesn't exist.
            FileNotFoundError,
            # Bad client sidecar/schema, or point clouds whose CRS can't be used.
            SidecarMergeError,
            PointCloudCRSError,
        ),
    ):
        return "invalid_input"
    # Anything else: storage/network failures wrapped as RuntimeError, and bugs
    # (including stray ValueErrors from pandas, pyproj, …).
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
