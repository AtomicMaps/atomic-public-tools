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


# ---- point cloud bounding-box distribution ------------------------------


# The batch CRS lives on the DEFAULT row's fallback_srs column; the point-cloud
# histogram transforms bbox centers into EPSG:3857 and errors without it.
_PC_CRS = "EPSG:32612"


def _pc_header(*, with_crs: bool = True) -> list[str]:
    cols = [
        "Filename",
        "bounds.minx",
        "bounds.maxx",
        "bounds.miny",
        "bounds.maxy",
        "bounds.minz",
        "bounds.maxz",
    ]
    return [*cols, "fallback_srs"] if with_crs else cols


def _pc_lint(p: Path):
    return lint_sidecar_file(
        str(p),
        final=False,
        data_type=DataTypeEnum.point_cloud,
        schema_path=None,
        input_files_path=None,
    )


def _pc_clustered_rows(n: int) -> list[list[str]]:
    """``n`` adjacent 100x100x10 tiles marching along X, CRS on the DEFAULT row."""
    rows = [["DEFAULT", "", "", "", "", "", "", _PC_CRS]]
    for i in range(n):
        minx = 1000.0 + i * 100
        rows.append([f"tile{i}.las", f"{minx}", f"{minx + 100}", "2000", "2100", "10", "20", ""])
    return rows


def test_point_cloud_distance_histogram_rendered(tmp_path):
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), _pc_clustered_rows(5))
    report = _pc_lint(p)
    assert any(
        "Batch center (median bbox center)" in f.message
        and "Distance from center (miles)" in f.message
        for f in report.infos()
    ), _messages(report)


def test_point_cloud_distance_outlier_over_2sd_flagged(tmp_path):
    rows = _pc_clustered_rows(10)
    # A tile far off in X drags well past 2 SD from the median center.
    rows.append(["far.las", "1000000", "1000100", "2000", "2100", "10", "20", ""])
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), rows)
    report = _pc_lint(p)
    assert any(
        "2 SD from the median distance from center" in f.message and "far.las" in f.message
        for f in report.warnings()
    ), _messages(report)


def test_point_cloud_elevation_distribution_and_outlier(tmp_path):
    rows = _pc_clustered_rows(10)
    # Override the last tile with a wildly different Z range.
    rows[-1][5] = "50000"
    rows[-1][6] = "50010"
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), rows)
    report = _pc_lint(p)
    assert any("Elevation distribution" in f.message for f in report.infos()), _messages(report)
    assert any(
        "2 SD from the median elevation" in f.message for f in report.warnings()
    ), _messages(report)


def test_point_cloud_histogram_reprojects_to_goal_crs(tmp_path):
    # The bbox X/Y centers are transformed into the goal CRS (EPSG:3857) for the
    # planar-distance histogram, and the report names that reprojection.
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), _pc_clustered_rows(5))
    report = _pc_lint(p)
    assert any(
        "Distance from center (miles)" in f.message and "EPSG:3857" in f.message
        for f in report.infos()
    ), _messages(report)
    # Elevation is converted from the source CRS's own vertical unit (not the
    # reprojection), so its distribution is *not* tagged with EPSG:3857.
    assert any("Elevation distribution" in f.message for f in report.infos()), _messages(report)
    assert not report.has_errors(), [f.message for f in report.errors()]


def test_point_cloud_elevation_uses_source_vertical_unit_not_inflated(tmp_path):
    # Regression: when the source CRS is in US survey feet, the Z-center must be
    # converted from feet (≈ identity into feet for display), NOT treated as
    # meters and multiplied by ~3.28. EPSG:2284 is NAD83 / Virginia South (ftUS).
    # Z bounds 990–1010 ⇒ a ~1,000 ft center; the buggy path reported ~3,280 ft.
    crs = "EPSG:2284"
    rows = [["DEFAULT", "", "", "", "", "", "", crs]]
    for i in range(10):
        minx = 11_500_000.0 + i * 100
        rows.append(
            [f"tile{i}.las", f"{minx}", f"{minx + 100}", "3300000", "3300100", "990", "1010", ""]
        )
    # One clear elevation outlier so a warning naming the median is emitted.
    rows.append(["high.las", "11500000", "11500100", "3300000", "3300100", "4990", "5010", ""])
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), rows)
    report = _pc_lint(p)
    warning = next(
        (f.message for f in report.warnings() if "median elevation" in f.message), None
    )
    assert warning is not None, _messages(report)
    # Median elevation center ≈ 1,000 ft (feet→meters→feet round-trip), and
    # crucially nowhere near the ~3,280 ft the unit-confusion bug produced.
    assert "median 1,0" in warning, warning
    assert "3,2" not in warning, warning


def test_point_cloud_center_reported_as_lat_lon(tmp_path):
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), _pc_clustered_rows(5))
    report = _pc_lint(p)
    assert any(
        "Batch center (median bbox center)" in f.message and "(lat/lon)" in f.message
        for f in report.infos()
    ), _messages(report)


def test_point_cloud_without_crs_errors(tmp_path):
    # No fallback_srs column → units are unknowable, so the analysis errors
    # rather than assuming meters.
    rows = [r[:-1] for r in _pc_clustered_rows(5)]  # drop the fallback_srs value
    p = _write_csv(tmp_path, "pc.csv", _pc_header(with_crs=False), rows)
    report = _pc_lint(p)
    assert any(
        "no" in f.message.lower() and "CRS" in f.message for f in report.errors()
    ), _messages(report)
    assert report.has_errors()


def _pc_file_srs_header() -> list[str]:
    return [*_pc_header(with_crs=False), "file_srs"]


def test_point_cloud_uses_per_file_srs(tmp_path):
    # Per-row file_srs (no batch fallback) drives the transform — no error.
    header = _pc_file_srs_header()
    rows = [["DEFAULT", "", "", "", "", "", "", ""]]
    for i in range(5):
        minx = 1000.0 + i * 100
        rows.append(
            [f"tile{i}.las", f"{minx}", f"{minx + 100}", "2000", "2100", "10", "20", _PC_CRS]
        )
    p = _write_csv(tmp_path, "pc.csv", header, rows)
    report = _pc_lint(p)
    assert any(
        "Distance from center (miles)" in f.message and "EPSG:3857" in f.message
        for f in report.infos()
    ), _messages(report)
    assert not report.has_errors(), [f.message for f in report.errors()]


def test_point_cloud_row_missing_both_srs_errors(tmp_path):
    # One row has neither file_srs nor a batch fallback → that row is flagged.
    header = _pc_file_srs_header()
    rows = [["DEFAULT", "", "", "", "", "", "", ""]]
    for i in range(4):
        minx = 1000.0 + i * 100
        rows.append(
            [f"tile{i}.las", f"{minx}", f"{minx + 100}", "2000", "2100", "10", "20", _PC_CRS]
        )
    rows.append(["nocrs.las", "1500", "1600", "2000", "2100", "10", "20", ""])
    p = _write_csv(tmp_path, "pc.csv", header, rows)
    report = _pc_lint(p)
    assert any(
        "nocrs.las" in f.message and "file_srs" in f.message and "fallback_srs" in f.message
        for f in report.errors()
    ), _messages(report)


def test_single_point_cloud_skips_distance_distribution(tmp_path):
    rows = [
        ["DEFAULT", "", "", "", "", "", "", _PC_CRS],
        ["only.las", "1000", "1100", "2000", "2100", "10", "20", ""],
    ]
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), rows)
    report = _pc_lint(p)
    assert any("Only 1 point cloud" in f.message for f in report.infos()), _messages(report)
    assert not any("2 SD" in f.message for f in report.findings)


def test_point_cloud_findings_do_not_fail_lint(tmp_path):
    rows = _pc_clustered_rows(10)
    rows.append(["far.las", "1000000", "1000100", "2000", "2100", "10", "20", ""])
    p = _write_csv(tmp_path, "pc.csv", _pc_header(), rows)
    report = _pc_lint(p)
    assert not report.has_errors(), [f.message for f in report.errors()]
    assert report.exit_code() == 0


def test_point_cloud_untransformable_crs_is_flagged(tmp_path):
    # A CRS is present but has no path to Web Mercator → error, reported as
    # "cannot be transformed" (not as the "no CRS" case). Written via pandas so
    # the comma-bearing WKT is quoted correctly.
    import pandas as pd

    local_crs = 'LOCAL_CS["unknown",UNIT["metre",1]]'
    df = pd.DataFrame(
        [
            {c: "" for c in _pc_header()} | {"Filename": "DEFAULT", "fallback_srs": local_crs},
            {
                "Filename": "a.las",
                "bounds.minx": "1000", "bounds.maxx": "1100",
                "bounds.miny": "2000", "bounds.maxy": "2100",
                "bounds.minz": "10", "bounds.maxz": "20",
                "fallback_srs": "",
            },
        ],
        columns=_pc_header(),
    )
    p = tmp_path / "pc.csv"
    df.to_csv(p, index=False)
    report = _pc_lint(p)
    assert any(
        "cannot be transformed" in f.message and "a.las" in f.message
        for f in report.errors()
    ), _messages(report)
