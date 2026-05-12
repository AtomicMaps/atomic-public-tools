"""Tests for `am-tools lint sidecar`."""

from __future__ import annotations

from pathlib import Path

from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.sidecar import lint_sidecar_file

# ---- helpers -----------------------------------------------------------


def _write_csv(tmp_path: Path, name: str, header: list[str], rows: list[list[str]]) -> Path:
    p = tmp_path / name
    lines = [",".join(header)] + [",".join(r) for r in rows]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _oriented_image_header() -> list[str]:
    return [
        "Filename",
        "CreateDate",
        "GPSAltitude",
        "GPSLatitude",
        "GPSLongitude",
        "Pitch",
        "Roll",
        "Heading",
    ]


def _lint_oriented(p, *, final=False):
    return lint_sidecar_file(
        str(p),
        final=final,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )


def _lint_pointcloud(p):
    return lint_sidecar_file(
        str(p),
        final=True,
        data_type=DataTypeEnum.point_cloud,
        schema_path=None,
        input_files_path=None,
    )


# ---- --final mode -------------------------------------------------------


def test_final_missing_required_column_errors(tmp_path):
    header = ["Filename", "CreateDate", "GPSAltitude", "GPSLatitude", "GPSLongitude"]
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["1.jpg", "2024:06:15 10:30:00", "1000", "51.0", "-114.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )
    assert report.has_errors()
    assert any("Pitch" in f.message for f in report.errors())


def test_final_blank_column_with_default_passes(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
        ["1.jpg", "", "", "", "", "", "", ""],
        ["2.jpg", "", "", "", "", "", "", ""],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )
    assert not report.has_errors(), [f.message for f in report.errors()]


def test_final_blank_column_without_default_errors(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
        ["2.jpg", "", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )
    assert report.has_errors()
    assert any("CreateDate" in f.message for f in report.errors())


def test_final_warns_when_schema_passed(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    schema = tmp_path / "schema.json"
    schema.write_text('{"column_name_mapping": {"x": "GPSLatitude"}}', encoding="utf-8")
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=schema,
        input_files_path=None,
    )
    assert any("--schema is ignored" in f.message for f in report.warnings())


# ---- input-files coverage ----------------------------------------------


def test_input_files_extras_with_default_warns(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "extra1.jpg").write_text("x")
    (pic_dir / "extra2.jpg").write_text("x")
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=str(pic_dir),
    )
    assert not report.has_errors(), [f.message for f in report.errors()]
    assert any("DEFAULT covers" in f.message for f in report.warnings())


def test_input_files_extras_without_default_errors(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["a.jpg", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "a.jpg").write_text("x")
    (pic_dir / "extra.jpg").write_text("x")
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=str(pic_dir),
    )
    assert report.has_errors()
    assert any("DEFAULT does not cover" in f.message for f in report.errors())


def test_basename_only_sidecar_row_matching_multiple_inputs_warns(tmp_path):
    """A sidecar row using just a basename matches every input file with that
    basename. Ambiguous — surface a warning so the operator can supply more
    specific paths."""
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
        ["dup.jpg", "2024:06:15 10:30:00", "1100", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    pic_dir = tmp_path / "pics"
    (pic_dir / "sub1").mkdir(parents=True)
    (pic_dir / "sub2").mkdir(parents=True)
    (pic_dir / "sub1" / "dup.jpg").write_text("x")
    (pic_dir / "sub2" / "dup.jpg").write_text("x")
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=str(pic_dir),
    )
    assert any("matches 2 input files" in f.message for f in report.warnings()), [
        f.message for f in report.warnings()
    ]


def test_disambiguated_paths_in_sidecar_match_unambiguously(tmp_path):
    """When the sidecar uses disambiguated paths (``sub1/dup.jpg``) instead of
    bare basenames, each input file matches exactly one row — no warning."""
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
        ["sub1/dup.jpg", "2024:06:15 10:30:00", "1100", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
        ["sub2/dup.jpg", "2024:06:15 10:31:00", "1200", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    pic_dir = tmp_path / "pics"
    (pic_dir / "sub1").mkdir(parents=True)
    (pic_dir / "sub2").mkdir(parents=True)
    (pic_dir / "sub1" / "dup.jpg").write_text("x")
    (pic_dir / "sub2" / "dup.jpg").write_text("x")
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=str(pic_dir),
    )
    assert not any(
        "matches" in f.message and "input files" in f.message for f in report.warnings()
    ), [f.message for f in report.warnings()]
    assert not any("no sidecar row" in f.message for f in report.warnings() + report.errors())


def test_multiple_sidecar_rows_for_one_file_warns(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
        ["a.jpg", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
        ["a.jpg", "2024:06:15 10:31:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "a.jpg").write_text("x")
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=str(pic_dir),
    )
    assert any("matches" in f.message and "sidecar rows" in f.message for f in report.warnings())


# ---- client mode (no --final) -----------------------------------------


def test_client_mode_missing_required_column_is_not_error(tmp_path):
    header = ["Filename", "CreateDate"]
    rows = [
        ["DEFAULT", ""],
        ["1.jpg", "2024:06:15 10:30:00"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p),
        final=False,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )
    assert not report.has_errors(), [f.message for f in report.errors()]


def test_client_mode_schema_renames_applied(tmp_path):
    header = ["Filename", "lat", "lon"]
    rows = [
        ["1.jpg", "200", "0"],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    schema = tmp_path / "schema.json"
    schema.write_text(
        '{"column_name_mapping": {"lat": "GPSLatitude", "lon": "GPSLongitude"}}',
        encoding="utf-8",
    )
    report = lint_sidecar_file(
        str(sidecar),
        final=False,
        data_type=DataTypeEnum.oriented_image,
        schema_path=schema,
        input_files_path=None,
    )
    assert any("latitude" in f.message for f in report.errors())


def test_client_mode_input_extras_warn(tmp_path):
    header = ["Filename", "GPSAltitude"]
    rows = [["1.jpg", "1000"]]
    sidecar = _write_csv(tmp_path, "s.csv", header, rows)
    pic_dir = tmp_path / "pics"
    pic_dir.mkdir()
    (pic_dir / "1.jpg").write_text("x")
    (pic_dir / "extra.jpg").write_text("x")
    report = lint_sidecar_file(
        str(sidecar),
        final=False,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=str(pic_dir),
    )
    assert not report.has_errors()
    assert any("informational" in (f.fix_hint or "") for f in report.warnings())


# ---- value-format checks ------------------------------------------------


def test_bad_latitude_string(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "2024:06:15 10:30:00", "1000", "abc", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert any("latitude" in f.message for f in report.errors())


def test_latitude_out_of_range(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "2024:06:15 10:30:00", "1000", "200", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert any("outside [-90" in f.message for f in report.errors())


def test_dms_latitude_accepted(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        [
            "1.jpg",
            "2024:06:15 10:30:00",
            "1000",
            "51 deg 2' 40.92\" N",
            "-114.0",
            "-90.0",
            "0.0",
            "0.0",
        ],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert not any("latitude" in f.message for f in report.errors())


def test_pitch_in_range_accepted(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "-179.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert not any("pitch" in f.message for f in report.errors())


def test_pitch_out_of_range(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "2024:06:15 10:30:00", "1000", "51.0", "-114.0", "200.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert any("pitch" in f.message for f in report.errors())


def test_iso_datetime_with_timezone_accepted(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "2024-06-15T10:30:00+00:00", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert not any("date" in f.message for f in report.errors())


def test_trailing_z_datetime_accepted(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "2024-06-15T10:30:00Z", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert not any("date" in f.message for f in report.errors())


def test_bogus_datetime_rejected(tmp_path):
    header = _oriented_image_header()
    rows = [
        ["DEFAULT", "", "", "", "", "", "", ""],
        ["1.jpg", "not a date", "1000", "51.0", "-114.0", "-90.0", "0.0", "0.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_oriented(p)
    assert any("date" in f.message for f in report.errors())


# ---- point cloud --------------------------------------------------------


def test_point_cloud_passes(tmp_path):
    header = [
        "Filename",
        "bounds.minx",
        "bounds.miny",
        "bounds.maxx",
        "bounds.maxy",
        "bounds.minz",
        "bounds.maxz",
        "num_points",
        "creation_year",
        "creation_doy",
    ]
    rows = [
        ["DEFAULT", "0", "0", "10", "10", "0", "5", "1.5e6", "2024", "200"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_pointcloud(p)
    assert not report.has_errors(), [f.message for f in report.errors()]


def test_point_cloud_num_points_zero_errors(tmp_path):
    header = [
        "Filename",
        "bounds.minx",
        "bounds.miny",
        "bounds.maxx",
        "bounds.maxy",
        "bounds.minz",
        "bounds.maxz",
        "num_points",
        "creation_year",
        "creation_doy",
    ]
    rows = [
        ["DEFAULT", "0", "0", "10", "10", "0", "5", "0", "2024", "200"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_pointcloud(p)
    assert any("num_points" in f.message for f in report.errors())


def test_point_cloud_min_gt_max_warns(tmp_path):
    header = [
        "Filename",
        "bounds.minx",
        "bounds.miny",
        "bounds.maxx",
        "bounds.maxy",
        "bounds.minz",
        "bounds.maxz",
        "num_points",
        "creation_year",
        "creation_doy",
    ]
    rows = [
        ["DEFAULT", "20", "0", "10", "10", "0", "5", "1000", "2024", "200"],
    ]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint_pointcloud(p)
    assert any("bounds.minx" in f.message and "bounds.maxx" in f.message for f in report.warnings())
