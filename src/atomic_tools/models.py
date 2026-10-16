"""Pydantic response models for embedding ``atomic_tools`` in a web service.

Unrequested sections are left unset: serialise with ``exclude_unset=True``
(FastAPI: ``response_model_exclude_unset=True``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 2

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
    skipped_unsupported: int = 0
    # Set only when --limit/--sample cut the file list: how many files there
    # were before the cut (files_scanned is how many were checked).
    files_total: int | None = None


class MissingFile(BaseModel):
    """One file's missing required fields (``missing_data`` groups these by data type)."""

    file: str
    missing: list[str]
    # Set only when a COCO file was supplied.
    coco_status: str | None = None
    coco_labels: int | None = None


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


class SkippedFile(BaseModel):
    file: str
    reason: str


class FileResults(BaseModel):
    """Every checked file, by outcome. Paths are as in ``details``, which lists
    the files that weren't checked.

    A file is ``failed`` when an error names it, ``passed_with_warnings`` when
    only warnings do; a file a finding names is listed even if it has no
    sidecar row. Batch-level findings (e.g. an unreadable COCO file, or COCO
    images missing from the input) name no file, so they set ``status``
    without moving any file out of ``passed``.
    Telemetry paired with a video is part of that video and isn't listed.
    """

    passed: list[str]
    passed_with_warnings: list[str]
    failed: list[str]


class Details(BaseModel):
    crs_failures: list[str]
    skipped_vector: list[str]
    unclassified: list[str] = Field(description="Files that matched no data type.")
    skipped_unsupported: list[SkippedFile] = Field(
        default_factory=list,
        description="Recognized files Flow can't process yet (e.g. Gaussian splats, .zlas).",
    )
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
    missing_data: dict[str, list[MissingFile]] | None = Field(
        default=None,
        description=(
            "Files missing required fields, keyed by data type ('unknown' for COCO "
            "images with no sidecar row)."
        ),
    )
    optional_missing: dict[str, list[MissingFile]] | None = Field(
        default=None,
        description=(
            "Files missing optional fields (orientation when it's ignored), keyed "
            "by data type. These warn rather than fail."
        ),
    )
    metadata: list[dict[str, str]] | None = Field(
        default=None, description="One dict per sidecar row (all values strings)."
    )
    details: Details | None = None
    files: FileResults | None = None
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
    optional: bool = Field(
        default=False, description="Validation still runs without it, with less detail."
    )


class DependencyReport(BaseModel):
    """``ok`` is true when every required (non-optional) tool is usable."""

    ok: bool
    exiftool: ToolStatus
    pdal: ToolStatus
    pye57: ToolStatus
    ecw: ToolStatus = Field(
        description="GDAL with its ECW driver, used to read .ecw headers."
    )
