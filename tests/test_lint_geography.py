"""Tests for batch-level spatial outlier analysis in `am-tools lint sidecar`."""

from __future__ import annotations

from pathlib import Path

from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.sidecar import lint_sidecar_file


def _write_csv(tmp_path: Path, name: str, header: list[str], rows: list[list[str]]) -> Path:
    p = tmp_path / name
    lines = [",".join(header)] + [",".join(r) for r in rows]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _header() -> list[str]:
    return ["Filename", "CreateDate", "GPSAltitude", "GPSLatitude", "GPSLongitude"]


def _clustered_rows(n: int) -> list[list[str]]:
    """``n`` near-identical points around Salt Lake City."""
    rows = [["DEFAULT", "", "", "", ""]]
    for i in range(n):
        lat = 40.76 + i * 0.0005
        lon = -111.89 + i * 0.0005
        rows.append([f"img{i}.jpg", "2024:06:15 10:30:00", "1300", f"{lat:.5f}", f"{lon:.5f}"])
    return rows


def _lint(p: Path):
    return lint_sidecar_file(
        str(p),
        final=False,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )


def _messages(report) -> str:
    return " ".join(f.message for f in report.findings)


# ---- out-of-US ----------------------------------------------------------


def test_out_of_us_point_is_flagged(tmp_path):
    rows = _clustered_rows(5)
    rows.append(["paris.jpg", "2024:06:15 10:30:00", "1300", "48.85", "2.35"])
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    report = _lint(p)
    assert any(
        "outside the US" in f.message and "paris.jpg" in f.message for f in report.warnings()
    ), _messages(report)


def test_all_us_points_not_flagged_as_foreign(tmp_path):
    p = _write_csv(tmp_path, "s.csv", _header(), _clustered_rows(5))
    report = _lint(p)
    assert not any("outside the US" in f.message for f in report.findings), _messages(report)


def test_alaska_and_hawaii_count_as_us(tmp_path):
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["anchorage.jpg", "2024:06:15 10:30:00", "30", "61.22", "-149.90"],
        ["honolulu.jpg", "2024:06:15 10:30:00", "5", "21.31", "-157.86"],
    ]
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    report = _lint(p)
    assert not any("outside the US" in f.message for f in report.findings), _messages(report)


# ---- distance histogram + SD outliers -----------------------------------


def test_distance_histogram_rendered(tmp_path):
    p = _write_csv(tmp_path, "s.csv", _header(), _clustered_rows(5))
    report = _lint(p)
    assert any(
        "Batch center" in f.message and "Distance from center" in f.message
        for f in report.infos()
    ), _messages(report)


def test_distance_outlier_over_2sd_flagged(tmp_path):
    rows = _clustered_rows(10)
    # A point ~hundreds of miles away drags far past 2 SD.
    rows.append(["far.jpg", "2024:06:15 10:30:00", "1300", "45.00", "-120.00"])
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    report = _lint(p)
    assert any(
        "2 SD from the median distance" in f.message and "far.jpg" in f.message
        for f in report.warnings()
    ), _messages(report)


def test_single_point_skips_distance_distribution(tmp_path):
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["only.jpg", "2024:06:15 10:30:00", "1300", "40.76", "-111.89"],
    ]
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    report = _lint(p)
    assert any("Only 1 geolocated file" in f.message for f in report.infos()), _messages(report)
    assert not any("2 SD" in f.message for f in report.findings)


# ---- altitude -----------------------------------------------------------


def test_altitude_histogram_and_outlier(tmp_path):
    rows = _clustered_rows(10)
    # Override the last row with a wildly different altitude.
    rows[-1][2] = "50000"
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    report = _lint(p)
    assert any("Altitude distribution" in f.message for f in report.infos()), _messages(report)
    assert any(
        "2 SD from the median altitude" in f.message for f in report.warnings()
    ), _messages(report)


def test_spatial_findings_do_not_fail_lint(tmp_path):
    """Out-of-US / outlier findings are warnings, not errors."""
    rows = _clustered_rows(5)
    rows.append(["paris.jpg", "2024:06:15 10:30:00", "1300", "48.85", "2.35"])
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    report = _lint(p)
    assert not report.has_errors(), [f.message for f in report.errors()]
    assert report.exit_code() == 0


# ---- absent coordinates -------------------------------------------------


def test_no_coord_columns_is_noop(tmp_path):
    header = ["Filename", "CreateDate"]
    rows = [["DEFAULT", ""], ["1.jpg", "2024:06:15 10:30:00"]]
    p = _write_csv(tmp_path, "s.csv", header, rows)
    report = _lint(p)
    assert not any(
        "Batch center" in f.message or "Altitude distribution" in f.message
        for f in report.findings
    ), _messages(report)
