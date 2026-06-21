"""COCO label-impact analysis for sidecar linting.

Given a COCO label file alongside a sidecar, this maps each labelled image to
its sidecar row, classifies the image's metadata quality into a tier, and rolls
the result up into label (annotation) impact — i.e. how many labels sit on
images the Flow pipeline can't fully use.

This folds the reporting from the standalone ``extract_exif.py`` into amtools,
but reuses amtools' own field model (``required_fields.py``) instead of a
hand-copied alias list, so the two never drift. The tiers are mapped onto
amtools' required/optional split:

  * ``complete``    — every required and optional field group is satisfied.
  * ``degraded``    — all required groups satisfied, but an optional group
                      (e.g. orientation for oriented images) is missing.
  * ``unusable``    — a required group is missing (a hard lint error), or the
                      COCO entry reports a zero width/height.
  * ``not_on_disk`` — the COCO references an image with no matching sidecar row
                      (the file was never extracted), the most unusable case.

Affected labels = annotations on degraded/unusable/not_on_disk images.
Unusable labels = annotations on unusable/not_on_disk images.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomic_tools.utils.utils import (
    DataTypeEnum,
    _is_path_tail_suffix,
    _split_path_components,
    has_value,
    read_text_uri,
)
from atomic_tools.validators.constants import DEFAULT_ROW_NAME, MAX_LISTED_FILES

if TYPE_CHECKING:
    import pandas as pd

    from atomic_tools.validators.report import MissingDataReport

# COCO label impact only makes sense for imagery — point clouds and video have
# no COCO label set in this pipeline.
IMAGE_DATA_TYPES: frozenset[DataTypeEnum] = frozenset(
    {
        DataTypeEnum.ortho_image,
        DataTypeEnum.oriented_image,
        DataTypeEnum.spherical_image,
    }
)

# Tier names (also used as the ``coco_status`` value in the failed-rows CSV).
TIER_COMPLETE = "complete"
TIER_DEGRADED = "degraded"
TIER_UNUSABLE = "unusable"
TIER_NOT_ON_DISK = "not_on_disk"


class CocoError(ValueError):
    """Raised when a COCO file can't be located or parsed."""


@dataclass(frozen=True)
class CocoImage:
    """One COCO ``images[]`` entry, reduced to what impact analysis needs."""

    report_name: str  # file_name (or s3_uri) used when reporting this image
    candidate_names: frozenset[str]  # raw file_name + s3_uri (kept whole for tail-suffix matching)
    width: float | None
    height: float | None
    label_count: int  # number of annotations referencing this image


@dataclass(frozen=True)
class CocoVerdict:
    """The metadata verdict for one COCO image after matching to the sidecar."""

    report_name: str
    tier: str
    labels: int
    reasons: list[str]
    row_filename: str | None  # matched sidecar Filename, if any


@dataclass
class CocoImpact:
    coco_path: str
    images_in_coco: int = 0
    complete: int = 0
    degraded: int = 0
    unusable: int = 0
    not_on_disk: int = 0
    total_labels: int = 0
    affected_labels: int = 0
    unusable_labels: int = 0
    verdicts: list[CocoVerdict] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        return [
            f"Label impact ({os.path.basename(self.coco_path)}): "
            f"{self.images_in_coco} image(s) in COCO — "
            f"complete={self.complete} degraded={self.degraded} "
            f"unusable={self.unusable} not_on_disk={self.not_on_disk}.",
            f"Labels: {self.total_labels} total, {self.affected_labels} affected "
            f"(degraded+unusable+not_on_disk), {self.unusable_labels} unusable "
            f"(unusable+not_on_disk).",
        ]


def _looks_like_coco(obj: object) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("images"), list)


def _find_coco_in_dir(directory: str) -> str | None:
    """Find a COCO json inside a local directory (non-recursive).

    Order: ``input.coco.json`` → first ``*.coco.json`` → first ``*.json`` that
    structurally looks like a COCO. Mirrors ``extract_exif.py``.
    """
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return None
    jsons = [n for n in names if n.lower().endswith(".json")]
    if "input.coco.json" in jsons:
        return os.path.join(directory, "input.coco.json")
    for n in jsons:
        if n.lower().endswith(".coco.json"):
            return os.path.join(directory, n)
    for n in jsons:
        p = os.path.join(directory, n)
        try:
            if _looks_like_coco(json.loads(read_text_uri(p))):
                return p
        except (ValueError, OSError):
            continue
    return None


def _candidate_names(img: dict) -> set[str]:
    """COCO ``file_name`` is often a flattened S3 path that doesn't match the
    on-disk name, while the real path lives in ``s3_uri`` — so register both
    candidate names. They're kept whole (not basenamed) so the matcher can use
    path tail-suffixes to disambiguate images that share a basename across
    folders (mirrors ``extract_exif.py``'s dual registration, but path-aware).
    """
    names: set[str] = set()
    for key in ("file_name", "s3_uri"):
        v = img.get(key)
        if v:
            names.add(str(v))
    return names


def _as_float(value: object) -> float | None:
    # float() raises TypeError for non str/number-like inputs, which we catch.
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_coco_path(path: str) -> str:
    """Resolve ``path`` to a concrete COCO file: a directory is searched, a file
    is used directly. Raises :class:`CocoError` if none is found.
    """
    if os.path.isdir(path):
        found = _find_coco_in_dir(path)
        if not found:
            raise CocoError(
                f"No COCO .json found in {path} (looked for input.coco.json, "
                "*.coco.json, then any *.json with an images[] array)."
            )
        return found
    return path


def load_coco(path: str) -> tuple[str, list[CocoImage]]:
    """Load a COCO file (local path, ``s3://…`` URI, or a directory containing
    one) into ``(resolved_path, images)``.
    """
    resolved = resolve_coco_path(path)
    try:
        coco = json.loads(read_text_uri(resolved))
    except (ValueError, OSError) as e:
        raise CocoError(f"Could not read COCO file {resolved}: {e}") from e
    if not _looks_like_coco(coco):
        raise CocoError(f"{resolved} is not a COCO file (no images[] array).")

    # image id -> annotation (label) count
    labels_by_id: dict[object, int] = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        labels_by_id[img_id] = labels_by_id.get(img_id, 0) + 1

    images: list[CocoImage] = []
    for img in coco.get("images", []):
        names = _candidate_names(img)
        if not names:
            continue
        img_id = img.get("id")
        label_count = labels_by_id.get(img_id, 0)
        images.append(
            CocoImage(
                report_name=str(img.get("file_name") or img.get("s3_uri") or ""),
                candidate_names=frozenset(names),
                width=_as_float(img.get("width")),
                height=_as_float(img.get("height")),
                label_count=label_count,
            )
        )
    return resolved, images


def _group_coverage(
    df: pd.DataFrame,
    groups: list[list[str]],
    columns_set: set[str],
    default_row_idx: int | None,
) -> tuple[dict[str, list[str]], dict[str, bool]]:
    """Return (present_fields_by_canonical, default_covers_by_canonical) — the
    same per-group resolution ``_build_missing_data_report`` uses.
    """
    default_row = df.iloc[default_row_idx] if default_row_idx is not None else None
    present_by_field = {
        group[0]: [f for f in group if f in columns_set] for group in groups
    }
    default_covers = {
        canonical: default_row is not None and any(has_value(default_row[f]) for f in present)
        for canonical, present in present_by_field.items()
    }
    return present_by_field, default_covers


def analyze_coco_impact(
    df: pd.DataFrame,
    coco_path: str,
    coco_images: list[CocoImage],
    required_groups: list[list[str]],
    optional_groups: list[list[str]],
    columns_set: set[str],
    default_row_idx: int | None,
) -> CocoImpact:
    """Classify each COCO image against the sidecar and tally label impact."""
    impact = CocoImpact(coco_path=coco_path)
    if df.shape[1] == 0:
        return impact

    file_col = df.columns[0]
    sidecar_filenames = df[file_col].astype(str).str.strip()
    non_default_mask = sidecar_filenames != DEFAULT_ROW_NAME

    # Bucket sidecar rows by basename for tail-suffix matching (same scheme as
    # the file-inventory check), so a COCO path with parent dirs resolves the
    # right row even when basenames collide across folders.
    rows_by_basename: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
    for idx, name in sidecar_filenames[non_default_mask].items():
        parts = _split_path_components(name)
        if parts:
            rows_by_basename.setdefault(parts[-1], []).append((int(idx), parts))

    req_present, req_default = _group_coverage(df, required_groups, columns_set, default_row_idx)
    opt_present, opt_default = _group_coverage(df, optional_groups, columns_set, default_row_idx)

    for image in coco_images:
        impact.images_in_coco += 1
        match_idx = _match_row(image, rows_by_basename)
        reasons: list[str] = []

        if match_idx is None:
            tier = TIER_NOT_ON_DISK
            reasons.append("no_sidecar_row")
            row_filename = None
        else:
            row = df.iloc[match_idx]
            row_filename = str(row[file_col]).strip()

            missing_required = [
                canonical
                for canonical, present in req_present.items()
                if not req_default[canonical] and not any(has_value(row[f]) for f in present)
            ]
            missing_optional = [
                canonical
                for canonical, present in opt_present.items()
                if not opt_default[canonical] and not any(has_value(row[f]) for f in present)
            ]
            zero_size = (image.width is not None and image.width <= 0) or (
                image.height is not None and image.height <= 0
            )

            if missing_required:
                reasons += [f"missing_required:{c}" for c in missing_required]
            if zero_size:
                reasons.append("zero_size")
            if missing_optional:
                reasons += [f"missing_optional:{c}" for c in missing_optional]

            if missing_required or zero_size:
                tier = TIER_UNUSABLE
            elif missing_optional:
                tier = TIER_DEGRADED
            else:
                tier = TIER_COMPLETE

        _tally(impact, tier, image.label_count)
        if tier != TIER_COMPLETE:
            impact.verdicts.append(
                CocoVerdict(
                    report_name=image.report_name,
                    tier=tier,
                    labels=image.label_count,
                    reasons=reasons,
                    row_filename=row_filename,
                )
            )

    return impact


def _match_row(
    image: CocoImage,
    rows_by_basename: dict[str, list[tuple[int, tuple[str, ...]]]],
) -> int | None:
    """Return the sidecar row index matching this COCO image, or None.

    A candidate name matches a row when their path components share a tail
    suffix (``flightA/1.jpg`` matches a row named ``1.jpg`` or ``flightA/1.jpg``,
    but not ``flightB/1.jpg``). The first matching row wins.
    """
    matched: list[int] = []
    for name in image.candidate_names:
        parts = _split_path_components(name)
        if not parts:
            continue
        for idx, row_parts in rows_by_basename.get(parts[-1], []):
            if _is_path_tail_suffix(parts, row_parts) and idx not in matched:
                matched.append(idx)
    return min(matched) if matched else None


def _tally(impact: CocoImpact, tier: str, labels: int) -> None:
    impact.total_labels += labels
    if tier == TIER_COMPLETE:
        impact.complete += 1
        return
    if tier == TIER_DEGRADED:
        impact.degraded += 1
        impact.affected_labels += labels
    elif tier == TIER_UNUSABLE:
        impact.unusable += 1
        impact.affected_labels += labels
        impact.unusable_labels += labels
    elif tier == TIER_NOT_ON_DISK:
        impact.not_on_disk += 1
        impact.affected_labels += labels
        impact.unusable_labels += labels


def augment_missing_data(missing_data: MissingDataReport, impact: CocoImpact) -> None:
    """Fold COCO tier + label counts into the failed-rows CSV.

    Existing rows (missing a required field) gain ``coco_status``/``coco_labels``
    cells; degraded and not-on-disk images that aren't already listed are
    appended so the CSV captures every label-affecting image, tagged by tier.
    """
    from atomic_tools.validators.report import COCO_LABELS_COLUMN, COCO_STATUS_COLUMN

    missing_data.include_coco = True
    fname_col = missing_data.filename_column
    verdict_by_file = {v.row_filename: v for v in impact.verdicts if v.row_filename is not None}

    seen: set[str] = set()
    for row in missing_data.rows:
        fname = row.get(fname_col, "")
        seen.add(fname)
        verdict = verdict_by_file.get(fname)
        row[COCO_STATUS_COLUMN] = verdict.tier if verdict else TIER_UNUSABLE
        row[COCO_LABELS_COLUMN] = str(verdict.labels if verdict else 0)

    for verdict in impact.verdicts:
        key = verdict.row_filename if verdict.row_filename is not None else verdict.report_name
        if key in seen:
            continue
        seen.add(key)
        new_row: dict[str, str] = {fname_col: key}
        for col in missing_data.field_columns:
            new_row[col] = ""
        new_row[COCO_STATUS_COLUMN] = verdict.tier
        new_row[COCO_LABELS_COLUMN] = str(verdict.labels)
        missing_data.rows.append(new_row)


def flagged_sample(
    impact: CocoImpact, tiers: set[str] | None = None
) -> tuple[list[CocoVerdict], int]:
    """Return up to ``MAX_LISTED_FILES`` verdicts (worst tier first, most labels
    first) and the number truncated, for a compact console listing. ``tiers``
    optionally restricts which verdict tiers are listed.
    """
    verdicts = (
        impact.verdicts if tiers is None else [v for v in impact.verdicts if v.tier in tiers]
    )
    order = {TIER_NOT_ON_DISK: 0, TIER_UNUSABLE: 1, TIER_DEGRADED: 2}
    ranked = sorted(verdicts, key=lambda v: (order.get(v.tier, 9), -v.labels))
    shown = ranked[:MAX_LISTED_FILES]
    return shown, max(0, len(ranked) - len(shown))
