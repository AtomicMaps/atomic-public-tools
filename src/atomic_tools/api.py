"""Programmatic entry point for validation — the engine behind ``validate --api-mode``.

:func:`validate_directory` scans a directory, builds the sidecar in memory, lints
it, and returns a JSON-serialisable dict. It never prompts, never prints to
stdout, never writes a file, and never raises: failures come back as
``status == "error"`` with an ``error`` object, so an API handler can return the
dict as-is.

Payload shape (``schema_version`` 1)::

    {
      "schema_version": 1,
      "status": "passed" | "passed_with_warnings" | "failed" | "error",
      "summary": {...},          # always present
      "findings": [...],         # --include findings
      "missing_data": {...},     # --include missing_data (null if no rules ran)
      "metadata": [...],         # --include metadata (one dict per sidecar row)
      "details": {...},          # --include details (CRS / skipped / COCO)
      "error": null | {"type": ..., "message": ...},
    }

Sections not requested via ``include`` are omitted entirely.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from atomic_tools.commands.sidecar import _build_sidecar, _crs_failures
from atomic_tools.utils.aws_errors import find_auth_error
from atomic_tools.utils.extractors import PdalNotFoundError
from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.constants import DEFAULT_ROW_NAME
from atomic_tools.validators.report import LintReport
from atomic_tools.validators.sidecar import _as_csv_strings, lint_sidecar_df

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SECTIONS: tuple[str, ...] = ("findings", "missing_data", "metadata", "details")
# metadata is opt-in: one dict per file can be large for big directories.
DEFAULT_SECTIONS: frozenset[str] = frozenset({"findings", "missing_data", "details"})


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
    directory: str,
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

    Shared by the interactive CLI and :func:`validate_directory`. Raises on
    extraction/build failures; callers decide how to surface them.
    """
    logger.info(
        f"Starting validation — directory={directory!r} data_type={data_type} "
        f"client_sidecar={client_sidecar!r} client_schema={client_schema!r} "
        f"full={full} spatial_reference={spatial_reference!r} coco={coco!r}"
    )
    unclassified: list[str] = []
    # validate never persists, so the backend is unused.
    df, _, detected_types, skipped_vector = _build_sidecar(
        directory=directory,
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
        input_files_path=directory,
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


def build_payload(result: ValidationResult, include: Iterable[str]) -> dict:
    """Serialise a :class:`ValidationResult` to the JSON payload."""
    include = set(include)
    report = result.report
    df = result.df
    filenames = df["Filename"].astype(str) if "Filename" in df.columns else None
    files_scanned = (
        int((filenames != DEFAULT_ROW_NAME).sum()) if filenames is not None else len(df)
    )

    payload: dict[str, Any] = {
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
    }
    if "findings" in include:
        payload["findings"] = report.findings_as_dicts()
    if "missing_data" in include:
        payload["missing_data"] = (
            report.missing_data.to_dict() if report.missing_data is not None else None
        )
    if "metadata" in include:
        payload["metadata"] = _as_csv_strings(df).to_dict(orient="records")
    if "details" in include:
        payload["details"] = {
            "crs_failures": result.crs_failures,
            "skipped_vector": result.skipped_vector,
            "unclassified": result.unclassified,
            "coco": report.coco_impact.to_dict() if report.coco_impact else None,
        }
    payload["error"] = None
    return payload


def error_payload(error_type: str, message: str) -> dict:
    """The payload returned when validation could not run to completion."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "summary": None,
        "error": {"type": error_type, "message": message},
    }


def _classify_exception(exc: BaseException) -> str:
    if find_auth_error(exc) is not None:
        return "auth_error"
    if isinstance(exc, PdalNotFoundError):
        return "dependency_missing"
    if isinstance(exc, (ValueError, NotImplementedError)):
        return "invalid_input"
    if isinstance(exc, (FileNotFoundError, RuntimeError)):
        # _build_sidecar raises RuntimeError when no supported files are found.
        return "no_data"
    return "internal_error"


def validate_directory(
    directory: str,
    *,
    data_type: DataTypeEnum | str | None = None,
    client_sidecar: str | None = None,
    client_schema: str | None = None,
    full: bool = False,
    spatial_reference: str | None = None,
    ignore_missing_orientation: bool = False,
    coco: str | None = None,
    include: str | Iterable[str] | None = None,
) -> dict:
    """Validate ``directory`` and return the JSON payload (never raises).

    ``data_type`` accepts a :class:`DataTypeEnum` or its string value; ``None``
    auto-detects per file. ``include`` selects payload sections (see
    :func:`parse_include`).
    """
    try:
        sections = parse_include(include)
        if not directory:
            raise ValueError("directory is required.")
        dt = DataTypeEnum(data_type) if isinstance(data_type, str) else data_type
    except ValueError as e:
        return error_payload("invalid_input", str(e))

    try:
        result = run_validation(
            directory,
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
