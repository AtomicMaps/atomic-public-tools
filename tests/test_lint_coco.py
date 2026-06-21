"""Tests for COCO label-impact analysis and its wiring into the sidecar linter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators import coco as coco_mod
from atomic_tools.validators.report import MissingDataReport
from atomic_tools.validators.sidecar import lint_sidecar_file

# ---- helpers -----------------------------------------------------------

_HEADER = ["Filename", "GPSLatitude", "GPSLongitude", "GPSAltitude", "CreateDate", "Heading"]


def _write_sidecar(tmp_path: Path, rows: list[list[str]]) -> Path:
    p = tmp_path / "sidecar.csv"
    lines = [",".join(_HEADER)] + [",".join(r) for r in rows]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_coco(
    tmp_path: Path, images: list[dict], annotations: list[dict], name="labels.coco.json"
) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")
    return p


def _df(rows: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame([dict(zip(_HEADER, r, strict=True)) for r in rows])


_REQUIRED = [["GPSLatitude"], ["GPSLongitude"], ["GPSAltitude"], ["CreateDate", "DateTimeOriginal"]]
_OPTIONAL = [["Heading", "Yaw"]]


def _analyze(df: pd.DataFrame, coco_path: str):
    resolved, images = coco_mod.load_coco(coco_path)
    return coco_mod.analyze_coco_impact(
        df=df,
        coco_path=resolved,
        coco_images=images,
        required_groups=_REQUIRED,
        optional_groups=_OPTIONAL,
        columns_set=set(df.columns),
        default_row_idx=0,
    )


# ---- load_coco ---------------------------------------------------------


def test_load_coco_counts_labels_and_basenames(tmp_path):
    coco = _write_coco(
        tmp_path,
        images=[
            {"id": 1, "file_name": "flightA/1.jpg"},
            {"id": 2, "s3_uri": "s3://bucket/path/2.jpg", "file_name": "flat_2.jpg"},
        ],
        annotations=[{"image_id": 1}, {"image_id": 1}, {"image_id": 2}],
    )
    _resolved, images = coco_mod.load_coco(str(coco))
    by_name = {img.report_name: img for img in images}
    assert by_name["flightA/1.jpg"].label_count == 2
    # both file_name and s3_uri are registered as (whole) candidate names
    assert {"flat_2.jpg", "s3://bucket/path/2.jpg"} <= set(by_name["flat_2.jpg"].candidate_names)


def test_load_coco_finds_file_in_directory(tmp_path):
    _write_coco(tmp_path, images=[{"id": 1, "file_name": "1.jpg"}], annotations=[])
    resolved, images = coco_mod.load_coco(str(tmp_path))  # pass the directory
    assert resolved.endswith("labels.coco.json")
    assert len(images) == 1


def test_load_coco_rejects_non_coco(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "coco"}), encoding="utf-8")
    with pytest.raises(coco_mod.CocoError):
        coco_mod.load_coco(str(bad))


# ---- tier classification ----------------------------------------------


def test_tiers_and_label_tallies(tmp_path):
    df = _df(
        [
            ["DEFAULT", "", "", "", "", ""],
            ["a.jpg", "51", "-114", "100", "2024:01:01", "90"],  # complete
            ["b.jpg", "51", "-114", "100", "2024:01:01", ""],  # degraded (no heading)
            ["c.jpg", "", "", "", "", ""],  # unusable (missing required)
        ]
    )
    coco = _write_coco(
        tmp_path,
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 4000, "height": 3000},
            {"id": 2, "file_name": "b.jpg", "width": 4000, "height": 3000},
            {"id": 3, "file_name": "c.jpg", "width": 4000, "height": 3000},
            {"id": 4, "file_name": "d.jpg", "width": 4000, "height": 3000},  # not_on_disk
        ],
        annotations=[
            {"image_id": 1},
            {"image_id": 2}, {"image_id": 2},
            {"image_id": 3}, {"image_id": 3}, {"image_id": 3},
            {"image_id": 4},
        ],
    )
    impact = _analyze(df, str(coco))
    assert (impact.complete, impact.degraded, impact.unusable, impact.not_on_disk) == (1, 1, 1, 1)
    assert impact.total_labels == 7
    assert impact.affected_labels == 2 + 3 + 1  # degraded(2) + unusable(3) + not_on_disk(1)
    assert impact.unusable_labels == 3 + 1  # unusable(3) + not_on_disk(1)
    tiers = {v.report_name: v.tier for v in impact.verdicts}
    assert tiers == {"b.jpg": "degraded", "c.jpg": "unusable", "d.jpg": "not_on_disk"}


def test_zero_size_is_unusable(tmp_path):
    df = _df([["DEFAULT", "", "", "", "", ""], ["a.jpg", "51", "-114", "100", "2024:01:01", "90"]])
    coco = _write_coco(
        tmp_path,
        images=[{"id": 1, "file_name": "a.jpg", "width": 0, "height": 3000}],
        annotations=[{"image_id": 1}],
    )
    impact = _analyze(df, str(coco))
    assert impact.unusable == 1
    verdict = impact.verdicts[0]
    assert "zero_size" in verdict.reasons


def test_missing_width_height_not_flagged(tmp_path):
    # absent width/height (common in COCOs) must NOT be treated as zero-size.
    df = _df([["DEFAULT", "", "", "", "", ""], ["a.jpg", "51", "-114", "100", "2024:01:01", "90"]])
    coco = _write_coco(
        tmp_path, images=[{"id": 1, "file_name": "a.jpg"}], annotations=[{"image_id": 1}]
    )
    impact = _analyze(df, str(coco))
    assert impact.complete == 1
    assert impact.verdicts == []


def test_tail_suffix_disambiguates_colliding_basenames(tmp_path):
    df = _df(
        [
            ["DEFAULT", "", "", "", "", ""],
            ["flightA/1.jpg", "", "", "", "", ""],  # unusable
            ["flightB/1.jpg", "51", "-114", "100", "2024:01:01", "90"],  # complete
        ]
    )
    coco = _write_coco(
        tmp_path,
        images=[{"id": 1, "file_name": "flightB/1.jpg", "width": 10, "height": 10}],
        annotations=[{"image_id": 1}],
    )
    impact = _analyze(df, str(coco))
    # Must match the flightB row (complete), not the colliding flightA row.
    assert impact.complete == 1
    assert impact.unusable == 0


# ---- CSV augmentation --------------------------------------------------


def test_augment_missing_data_adds_columns_and_rows(tmp_path):
    df = _df(
        [
            ["DEFAULT", "", "", "", "", ""],
            ["b.jpg", "51", "-114", "100", "2024:01:01", ""],  # degraded
            ["c.jpg", "", "", "", "", ""],  # unusable
        ]
    )
    coco = _write_coco(
        tmp_path,
        images=[
            {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10},
            {"id": 3, "file_name": "c.jpg", "width": 10, "height": 10},
            {"id": 4, "file_name": "d.jpg", "width": 10, "height": 10},  # not_on_disk
        ],
        annotations=[{"image_id": 2}, {"image_id": 3}, {"image_id": 4}],
    )
    impact = _analyze(df, str(coco))
    md = MissingDataReport(
        filename_column="Filename",
        field_columns=["GPSLatitude", "GPSLongitude", "GPSAltitude", "CreateDate"],
        rows=[{"Filename": "c.jpg", "GPSLatitude": "MISSING"}],
    )
    coco_mod.augment_missing_data(md, impact)
    assert md.include_coco is True
    by_file = {r["Filename"]: r for r in md.rows}
    assert by_file["c.jpg"]["coco_status"] == "unusable"
    assert by_file["b.jpg"]["coco_status"] == "degraded"  # appended
    assert by_file["d.jpg"]["coco_status"] == "not_on_disk"  # appended

    out = tmp_path / "impact.csv"
    md.write_csv(str(out))
    header = out.read_text().splitlines()[0]
    assert header.endswith("coco_status,coco_labels")


# ---- lint_sidecar_file integration -------------------------------------


def _lint(path, datatype, coco_path, *, ignore_orientation=True):
    return lint_sidecar_file(
        str(path),
        final=True,
        data_type=datatype,
        schema_path=None,
        input_files_path=None,
        ignore_missing_orientation=ignore_orientation,
        coco_path=str(coco_path) if coco_path else None,
    )


def test_lint_attaches_coco_impact_and_csv(tmp_path):
    sidecar = _write_sidecar(
        tmp_path,
        [
            ["DEFAULT", "", "", "", "", ""],
            ["a.jpg", "51", "-114", "100", "2024:01:01", "90"],
            ["c.jpg", "", "", "", "", ""],
        ],
    )
    coco = _write_coco(
        tmp_path,
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
            {"id": 3, "file_name": "c.jpg", "width": 10, "height": 10},
        ],
        annotations=[{"image_id": 1}, {"image_id": 3}, {"image_id": 3}],
    )
    report = _lint(sidecar, DataTypeEnum.oriented_image, coco)
    assert report.coco_impact is not None
    assert report.coco_impact.unusable == 1
    assert report.missing_data is not None and report.missing_data.include_coco
    # the unusable row carries its 2 labels in the CSV matrix
    crow = next(r for r in report.missing_data.rows if r["Filename"] == "c.jpg")
    assert crow["coco_status"] == "unusable"
    assert crow["coco_labels"] == "2"
    assert any("Label impact" in f.message for f in report.infos())


def test_lint_coco_ignored_for_non_image_datatype(tmp_path):
    sidecar = _write_sidecar(tmp_path, [["DEFAULT", "", "", "", "", ""]])
    coco = _write_coco(tmp_path, images=[{"id": 1, "file_name": "a.jpg"}], annotations=[])
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=DataTypeEnum.point_cloud,
        schema_path=None,
        input_files_path=None,
        coco_path=str(coco),
    )
    assert report.coco_impact is None
    assert any("COCO file ignored" in w.message for w in report.warnings())


def test_lint_coco_skipped_without_datatype(tmp_path):
    sidecar = _write_sidecar(
        tmp_path,
        [["DEFAULT", "", "", "", "", ""], ["a.jpg", "51", "-114", "100", "2024:01:01", "90"]],
    )
    coco = _write_coco(tmp_path, images=[{"id": 1, "file_name": "a.jpg"}], annotations=[])
    report = lint_sidecar_file(
        str(sidecar),
        final=False,
        data_type=None,
        schema_path=None,
        input_files_path=None,
        coco_path=str(coco),
    )
    assert report.coco_impact is None
    assert any("Skipped COCO label-impact" in f.message for f in report.infos())
