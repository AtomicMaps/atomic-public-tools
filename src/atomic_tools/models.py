"""Pydantic response models for embedding ``atomic_tools`` in a web service.

Unrequested sections are left unset: serialise with ``exclude_unset=True``
(FastAPI: ``response_model_exclude_unset=True``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

Status = Literal["passed", "passed_with_warnings", "failed", "error"]
ErrorType = Literal[
    "invalid_input", "no_data", "auth_error", "dependency_missing", "internal_error"
]


class Finding(BaseModel):
    level: Literal["error", "warning", "info"]
    message: str
    fix_hint: str | None = None
    location: str | None = None


class Summary(BaseModel):
    errors: int
    warnings: int
    infos: int
    files_scanned: int
    detected_types: list[str]
    crs_failures: int
    skipped_vector: int
    unclassified: int


class MissingData(BaseModel):
    """Per-file table of missing required fields (``"MISSING"`` or ``""``)."""

    filename_column: str
    field_columns: list[str]
    columns: list[str]
    rows: list[dict[str, str]]


class CocoTier(BaseModel):
    tier: str
    images: int
    labels: int
    severity: Literal["ok", "warn", "error"]


class CocoVerdict(BaseModel):
    report_name: str
    tier: str
    labels: int
    reasons: list[str]
    row_filename: str | None = None


class CocoDetails(BaseModel):
    coco_path: str
    images_in_coco: int
    total_labels: int
    affected_labels: int
    unusable_labels: int
    not_on_disk_is_error: bool
    out_of_scope: int = Field(
        default=0, description="Single-file runs: COCO images outside the run, not checked."
    )
    tiers: list[CocoTier]
    verdicts: list[CocoVerdict]


class Details(BaseModel):
    crs_failures: list[str]
    skipped_vector: list[str]
    unclassified: list[str]
    coco: CocoDetails | None = None


class ErrorInfo(BaseModel):
    type: ErrorType
    message: str


class ValidationResponse(BaseModel):
    """``summary`` is None only when ``status == "error"``."""

    schema_version: int = SCHEMA_VERSION
    status: Status
    summary: Summary | None = None
    findings: list[Finding] | None = None
    missing_data: MissingData | None = None
    metadata: list[dict[str, str]] | None = Field(
        default=None, description="One dict per sidecar row (all values strings)."
    )
    details: Details | None = None
    error: ErrorInfo | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)


class ToolStatus(BaseModel):
    found: bool = Field(description="The executable/module was located.")
    usable: bool = Field(
        description="It actually ran (catches e.g. a binary with missing shared libraries)."
    )
    path: str | None = None
    version: str | None = None
    detail: str | None = Field(
        default=None, description="Why the tool is unavailable, when it isn't."
    )


class DependencyReport(BaseModel):
    """External tools validation needs; see :func:`atomic_tools.check_dependencies`.

    ``ok`` is true when every tool is usable: exiftool (images/video), pdal
    (point clouds) and pye57 (.e57 capture dates).
    """

    ok: bool
    exiftool: ToolStatus
    pdal: ToolStatus
    pye57: ToolStatus
