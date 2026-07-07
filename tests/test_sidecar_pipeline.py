"""Integration tests against the bundled example fixtures.

Uses synthetic EXIF-like metadata where the test only cares about the
post-extraction pipeline; for I/O paths (client-sidecar load, schema-driven
headerless rename) the real example CSVs and schema are read.
"""

from pathlib import Path

import pytest

from atomic_tools.client_sidecar import (
    load_and_clean_client_sidecar,
    merge_client_metadata,
)
from atomic_tools.commands.sidecar import (
    _ALL_SIDECAR_FIELD_GROUPS,
    _CRS_WEB_MERCATOR_COL,
    _add_crs_web_mercator_column,
    _add_file_srs_column,
    _add_spatial_reference_column,
    _disambiguate_filenames,
    _fill_missing_dates_from_filepath,
    _reproject_dataframe,
    _split_gps_position,
    _warn_missing_required_fields,
    build_sidecar_df,
)
from atomic_tools.utils.extractors import extract_pdal_crs
from atomic_tools.utils.utils import DataTypeEnum

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example-fake-data"
INPUT_SIDECAR = EXAMPLE_DIR / "input sidecar" / "input_sidecar.csv"
HEADERLESS_SIDECAR = EXAMPLE_DIR / "input sidecar" / "headerless_sidecar.csv"
EXAMPLE_SCHEMA = REPO_ROOT / "schemas" / "column_names_example.json"

ORIENTED_GROUPS = _ALL_SIDECAR_FIELD_GROUPS[DataTypeEnum.oriented_image]
PC_GROUPS = _ALL_SIDECAR_FIELD_GROUPS[DataTypeEnum.point_cloud]


def _build_single_type(metadata, groups, type_value, **kwargs):
    """build_sidecar_df helper for a single-type scan (all rows one type)."""
    return build_sidecar_df(
        metadata,
        field_groups_by_type={type_value: groups},
        types_by_label={label: type_value for label, _ in metadata},
        **kwargs,
    )


def _exif_like_metadata() -> list[tuple[str, dict]]:
    """Hand-built metadata mirroring what exiftool would produce for the 3
    images, after the GPSPosition noise filter and dict ordering match what
    `extract_exif_metadata` emits."""
    return [
        (
            "1.jpg",
            {
                "DateTimeOriginal": "2024:06:15 10:30:00",
                "CreateDate": "2024:06:15 10:30:00",
                "GPSAltitude": "1100 m Above Sea Level",
                "GPSLatitude": "51 deg 2' 40.92\" N",
                "GPSLongitude": "114 deg 4' 18.84\" W",
                "GimbalPitchDegree": -90.0,
                "GimbalYawDegree": 0.0,
                "GimbalRollDegree": 0.0,
            },
        ),
        (
            "2.jpeg",
            {
                "DateTimeOriginal": "2024:06:15 10:31:00",
                "CreateDate": "2024:06:15 10:31:00",
                "GPSAltitude": "1200 m Above Sea Level",
                "GPSLatitude": "51 deg 3' 0.00\" N",
                "GPSLongitude": "114 deg 4' 30.00\" W",
                "GimbalPitchDegree": -85.0,
                "GimbalYawDegree": 45.0,
                "GimbalRollDegree": 1.5,
            },
        ),
        (
            "3.jpg",
            {
                "DateTimeOriginal": "2024:06:15 10:32:00",
                "CreateDate": "2024:06:15 10:32:00",
                "GPSAltitude": "1150 m Above Sea Level",
                "GPSLatitude": "51 deg 2' 52.80\" N",
                "GPSLongitude": "114 deg 4' 12.00\" W",
                "GimbalPitchDegree": -88.0,
                "GimbalYawDegree": 90.0,
                "GimbalRollDegree": -0.5,
            },
        ),
    ]


def test_build_sidecar_df_layout_and_canonicalization():
    df = _build_single_type(_exif_like_metadata(), ORIENTED_GROUPS, "oriented_image")

    # DEFAULT first, then files sorted alphabetically.
    assert df["Filename"].tolist() == ["DEFAULT", "1.jpg", "2.jpeg", "3.jpg"]

    # DataType is column 2, filled per file and blank on DEFAULT.
    assert list(df.columns[:2]) == ["Filename", "DataType"]
    assert df["DataType"].tolist() == ["", "oriented_image", "oriented_image", "oriented_image"]

    cols = list(df.columns)
    # Aliases canonicalized: GimbalPitchDegree → Pitch, etc.
    for canonical in ("Pitch", "Roll", "Heading", "CreateDate"):
        assert canonical in cols, f"missing canonical {canonical!r}"
    for alias in ("GimbalPitchDegree", "GimbalYawDegree", "GimbalRollDegree", "DateTimeOriginal"):
        assert alias not in cols, f"alias {alias!r} should have been renamed"

    # The actual EXIF-derived values land on the correct rows.
    row_1 = df[df["Filename"] == "1.jpg"].iloc[0]
    assert row_1["Pitch"] == -90.0
    assert row_1["Heading"] == 0.0
    assert row_1["GPSAltitude"] == "1100 m Above Sea Level"


def test_build_sidecar_df_full_includes_extra_fields():
    metadata = _exif_like_metadata()
    metadata[0][1]["ExtraField"] = "kept-when-full"
    df = _build_single_type(metadata, ORIENTED_GROUPS, "oriented_image", full=True)
    assert "ExtraField" in df.columns
    df_filtered = _build_single_type(_exif_like_metadata(), ORIENTED_GROUPS, "oriented_image")
    assert "ExtraField" not in df_filtered.columns


def _projected_metadata() -> list[tuple[str, dict]]:
    """Metadata whose lat/lon columns hold UTM zone 12N (EPSG:32612) X/Y."""
    return [
        (
            "a.jpg",
            {
                "CreateDate": "2024:06:15 10:30:00",
                "GPSAltitude": "1100 m Above Sea Level",
                "GPSLongitude": "500000.0",  # X / easting (central meridian)
                "GPSLatitude": "5650300.0",  # Y / northing
                "GimbalPitchDegree": -90.0,
                "GimbalYawDegree": 0.0,
                "GimbalRollDegree": 0.0,
            },
        ),
        (
            "b.jpg",
            {
                "CreateDate": "2024:06:15 10:31:00",
                # No altitude → 2D transform, altitude untouched.
                "GPSLongitude": "501000.0",
                "GPSLatitude": "5651000.0",
                "GimbalPitchDegree": -85.0,
                "GimbalYawDegree": 45.0,
                "GimbalRollDegree": 1.5,
            },
        ),
    ]


def test_reproject_dataframe_image_coordinates():
    df = _build_single_type(_projected_metadata(), ORIENTED_GROUPS, "oriented_image")
    _reproject_dataframe(df, "EPSG:32612")

    # DEFAULT row stays blank.
    default = df[df["Filename"] == "DEFAULT"].iloc[0]
    assert default["GPSLongitude"] == ""
    assert default["GPSLatitude"] == ""

    # File rows are now decimal degrees in WGS84 range, with a cleaned altitude.
    row_a = df[df["Filename"] == "a.jpg"].iloc[0]
    assert -180.0 <= float(row_a["GPSLongitude"]) <= 180.0
    assert -90.0 <= float(row_a["GPSLatitude"]) <= 90.0
    assert float(row_a["GPSLongitude"]) == pytest.approx(-111.0, abs=0.5)
    assert 50.0 < float(row_a["GPSLatitude"]) < 52.0
    assert float(row_a["GPSAltitude"]) == pytest.approx(1100.0, abs=5.0)

    # Row without altitude: lon/lat reprojected, altitude left blank.
    row_b = df[df["Filename"] == "b.jpg"].iloc[0]
    assert -180.0 <= float(row_b["GPSLongitude"]) <= 180.0
    assert row_b["GPSAltitude"] == ""


def test_reproject_dataframe_skips_unparseable_rows():
    metadata = _projected_metadata()
    metadata[1][1]["GPSLongitude"] = ""  # blank → skipped
    metadata[1][1]["GPSLatitude"] = ""
    df = _build_single_type(metadata, ORIENTED_GROUPS, "oriented_image")
    _reproject_dataframe(df, "EPSG:32612")

    row_b = df[df["Filename"] == "b.jpg"].iloc[0]
    assert row_b["GPSLongitude"] == ""
    assert row_b["GPSLatitude"] == ""


def test_add_spatial_reference_column_point_cloud():
    metadata = [
        ("cloud.las", {"num_points": "1000", "bounds.minx": "0", "bounds.miny": "0"}),
    ]
    df = _build_single_type(metadata, PC_GROUPS, "point_cloud")
    _add_spatial_reference_column(df, "EPSG:2956")

    assert "fallback_srs" in df.columns
    default = df[df["Filename"] == "DEFAULT"].iloc[0]
    assert default["fallback_srs"] == "EPSG:2956"
    # File rows carry no fallback_srs value (it lives in DEFAULT).
    file_row = df[df["Filename"] == "cloud.las"].iloc[0]
    assert file_row["fallback_srs"] == ""


def test_add_file_srs_column_records_header_crs():
    metadata = [
        ("with_crs.las", {"num_points": "1", "bounds.minx": "0", "spatialreference": "EPSG:32612"}),
        ("no_crs.las", {"num_points": "1", "bounds.minx": "0", "spatialreference": ""}),
    ]
    df = _build_single_type(metadata, PC_GROUPS, "point_cloud")
    _add_file_srs_column(df, metadata)

    assert "file_srs" in df.columns
    assert df[df["Filename"] == "with_crs.las"].iloc[0]["file_srs"] == "EPSG:32612"
    # A file whose header has no CRS gets a blank cell (it relies on fallback_srs).
    assert df[df["Filename"] == "no_crs.las"].iloc[0]["file_srs"] == ""
    # The DEFAULT row is left blank too.
    assert df[df["Filename"] == "DEFAULT"].iloc[0]["file_srs"] == ""


# WKT PDAL emits for a file georeferenced to UTM zone 12N (EPSG:32612).
_UTM12N_WKT = (
    'PROJCS["WGS 84 / UTM zone 12N",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["central_meridian",-111],UNIT["metre",1],'
    'AUTHORITY["EPSG","32612"]]'
)


def test_extract_pdal_crs_prefers_top_level_spatialreference():
    meta = {"spatialreference": _UTM12N_WKT, "srs.wkt": "ignored"}
    assert extract_pdal_crs(meta) == _UTM12N_WKT


def test_extract_pdal_crs_prefers_compound_over_plain():
    # Aligns with atomicmapspy PointCloud.get_srs: the compound WKT (horizontal +
    # vertical) wins over the plain horizontal-only spatialreference, so a file
    # with a vertical datum keeps its full definition.
    compound = 'COMPD_CS["UTM12N + NAVD88",' + _UTM12N_WKT + ',VERT_CS["NAVD88"]]'
    meta = {"srs.compoundwkt": compound, "spatialreference": _UTM12N_WKT}
    assert extract_pdal_crs(meta) == compound


def test_extract_pdal_crs_falls_back_to_srs_keys():
    # PDAL leaves the top-level key empty but populates the srs.* block.
    meta = {"spatialreference": "", "comp_spatialreference": "", "srs.wkt": _UTM12N_WKT}
    assert extract_pdal_crs(meta) == _UTM12N_WKT


def test_extract_pdal_crs_none_when_header_has_no_crs():
    # PDAL emits empty strings for every CRS key when the header is unreferenced.
    meta = {"spatialreference": "", "comp_spatialreference": "", "srs.wkt": "  "}
    assert extract_pdal_crs(meta) is None


def test_meta_from_filters_info_maps_bounds_num_points_and_srs():
    from atomic_tools.utils.extractors import _meta_from_filters_info

    fi = {
        "bbox": {"minx": 0.0, "maxx": 1.0, "miny": 2.0, "maxy": 3.0, "minz": 4.0, "maxz": 5.0},
        "num_points": 42,
        "srs": {"compoundwkt": _UTM12N_WKT, "units": {"horizontal": "metre"}},
    }
    out = _meta_from_filters_info(fi)
    assert out["bounds.minx"] == 0.0 and out["bounds.maxz"] == 5.0
    assert out["num_points"] == 42
    # srs sub-keys flatten under the srs. prefix so extract_pdal_crs finds them.
    assert out["srs.compoundwkt"] == _UTM12N_WKT
    assert out["srs.units.horizontal"] == "metre"
    assert extract_pdal_crs(out) == _UTM12N_WKT


def test_extract_pdal_metadata_falls_back_to_filters_info_when_header_empty(monkeypatch):
    # E57 (and any reader whose header block is empty): `pdal info --metadata`
    # returns nothing, so extraction must fall back to a filters.info pipeline.
    from atomic_tools.utils import extractors

    fi = {
        "bbox": {"minx": 10.0, "maxx": 20.0, "miny": 30.0, "maxy": 40.0, "minz": 1.0, "maxz": 2.0},
        "num_points": 7,
        "srs": {"compoundwkt": _UTM12N_WKT},
    }
    monkeypatch.setattr(extractors, "find_pdal_bin", lambda: "/fake/pdal")
    monkeypatch.setattr(extractors, "_run_pdal_info", lambda *a, **k: {"metadata": {}})
    called = {}

    def fake_filters_info(pdal_bin, local_path):
        called["path"] = local_path
        return fi

    monkeypatch.setattr(extractors, "_run_pdal_filters_info", fake_filters_info)

    meta = extractors.extract_pdal_metadata("scan.e57", "scan.e57")
    assert called["path"] == "scan.e57"  # fallback actually ran
    assert meta["bounds.minx"] == 10.0 and meta["bounds.maxx"] == 20.0
    assert meta["num_points"] == 7
    assert extract_pdal_crs(meta) == _UTM12N_WKT


def test_extract_pdal_metadata_skips_filters_info_when_header_has_bounds(monkeypatch):
    # LAS/LAZ: the header block already carries bounds, so no pipeline fallback.
    from atomic_tools.utils import extractors

    monkeypatch.setattr(extractors, "find_pdal_bin", lambda: "/fake/pdal")
    monkeypatch.setattr(
        extractors,
        "_run_pdal_info",
        lambda *a, **k: {"metadata": {"minx": 1, "maxx": 2, "miny": 3, "maxy": 4, "count": 9}},
    )

    def boom(*a, **k):
        raise AssertionError("filters.info fallback should not run for a populated header")

    monkeypatch.setattr(extractors, "_run_pdal_filters_info", boom)

    meta = extractors.extract_pdal_metadata("cloud.las", "cloud.las")
    assert meta["bounds.minx"] == 1
    assert meta["num_points"] == 9  # count → num_points


def test_e57_seconds_to_datetime_valid_and_invalid():
    import datetime

    from atomic_tools.utils.extractors import _E57_EPOCH, _e57_seconds_to_datetime

    target = datetime.datetime(2024, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)
    secs = (target - _E57_EPOCH).total_seconds()
    got = _e57_seconds_to_datetime(secs)
    assert got == target
    assert got.year == 2024 and got.timetuple().tm_yday == 167  # leap-year DOY
    # Rejected values fall back to None (→ filename/client-sidecar date).
    assert _e57_seconds_to_datetime(float("nan")) is None
    assert _e57_seconds_to_datetime(float("inf")) is None
    assert _e57_seconds_to_datetime(-1) is None
    assert _e57_seconds_to_datetime("nope") is None
    assert _e57_seconds_to_datetime(10**15) is None  # beyond the ~2100 ceiling


def test_parse_e57_datetime_node_reads_and_missing():
    import datetime

    from atomic_tools.utils.extractors import _E57_EPOCH, _parse_e57_datetime_node

    target = datetime.datetime(2023, 3, 1, tzinfo=datetime.timezone.utc)
    secs = (target - _E57_EPOCH).total_seconds()

    class _Val:
        def value(self):
            return secs

    node = {"creationDateTime": {"dateTimeValue": _Val()}}
    assert _parse_e57_datetime_node(node, "creationDateTime") == target
    assert _parse_e57_datetime_node({}, "creationDateTime") is None  # absent → None


def test_extract_pdal_metadata_populates_e57_creation_date(monkeypatch):
    # E57 header/filters.info carry no date; the pye57-derived capture datetime
    # supplies creation_year / creation_doy.
    import datetime

    from atomic_tools.utils import extractors

    monkeypatch.setattr(extractors, "find_pdal_bin", lambda: "/fake/pdal")
    monkeypatch.setattr(extractors, "_run_pdal_info", lambda *a, **k: {"metadata": {}})
    monkeypatch.setattr(
        extractors,
        "_run_pdal_filters_info",
        lambda *a, **k: {
            "bbox": {"minx": 0, "maxx": 1, "miny": 0, "maxy": 1, "minz": 0, "maxz": 1},
            "num_points": 5,
        },
    )
    monkeypatch.setattr(
        extractors,
        "_extract_e57_capture_datetime",
        lambda p: datetime.datetime(2024, 6, 15, tzinfo=datetime.timezone.utc),
    )
    meta = extractors.extract_pdal_metadata("scan.e57", "scan.e57")
    assert meta["creation_year"] == 2024
    assert meta["creation_doy"] == 167
    assert meta["num_points"] == 5


def test_extract_pdal_metadata_e57_without_date_leaves_creation_fields_absent(monkeypatch):
    # When pye57 yields no date, creation fields stay absent (→ filename fallback),
    # not fabricated.
    from atomic_tools.utils import extractors

    monkeypatch.setattr(extractors, "find_pdal_bin", lambda: "/fake/pdal")
    monkeypatch.setattr(extractors, "_run_pdal_info", lambda *a, **k: {"metadata": {}})
    monkeypatch.setattr(
        extractors,
        "_run_pdal_filters_info",
        lambda *a, **k: {"bbox": {"minx": 0, "maxx": 1, "miny": 0, "maxy": 1}, "num_points": 5},
    )
    monkeypatch.setattr(extractors, "_extract_e57_capture_datetime", lambda p: None)
    meta = extractors.extract_pdal_metadata("scan.e57", "scan.e57")
    assert "creation_year" not in meta and "creation_doy" not in meta


def _crs_status(pc_metadata, spatial_reference=None):
    """Run _add_crs_web_mercator_column over a minimal df and return the per-row
    crs_web_mercator_ok value keyed by filename (DEFAULT included)."""
    import pandas as pd

    labels = [label for label, _ in pc_metadata]
    df = pd.DataFrame([{"Filename": "DEFAULT"}] + [{"Filename": lb} for lb in labels])
    _add_crs_web_mercator_column(df, pc_metadata, spatial_reference)
    return dict(zip(df["Filename"], df[_CRS_WEB_MERCATOR_COL], strict=True))


def test_crs_column_yes_with_embedded_crs():
    status = _crs_status([("cloud.las", {"spatialreference": _UTM12N_WKT})])
    assert status["cloud.las"] == "yes"
    # DEFAULT (and any non-point-cloud row) stays blank.
    assert status["DEFAULT"] == ""


def test_crs_column_yes_with_backup_when_header_missing():
    status = _crs_status(
        [("cloud.las", {"spatialreference": ""})], spatial_reference="EPSG:32612"
    )
    assert status["cloud.las"] == "yes"


def test_crs_column_yes_transforms_bbox_center():
    # Bounds present → the check transforms the actual bbox center to EPSG:3857.
    meta = {
        "spatialreference": _UTM12N_WKT,
        "bounds.minx": 500000.0,
        "bounds.maxx": 500100.0,
        "bounds.miny": 4500000.0,
        "bounds.maxy": 4500100.0,
        "bounds.minz": 10.0,
        "bounds.maxz": 20.0,
    }
    assert _crs_status([("cloud.las", meta)])["cloud.las"] == "yes"


def test_crs_column_no_when_center_unreachable_with_bounds():
    # Even with bounds present, a CRS with no path to Web Mercator is "no".
    meta = {
        "spatialreference": 'LOCAL_CS["unknown",UNIT["metre",1]]',
        "bounds.minx": 0.0,
        "bounds.maxx": 100.0,
        "bounds.miny": 0.0,
        "bounds.maxy": 100.0,
    }
    assert _crs_status([("cloud.las", meta)])["cloud.las"] == "no"


def test_crs_column_no_when_no_crs_and_no_backup():
    status = _crs_status([("a.las", {"spatialreference": ""}), ("b.las", {})])
    assert status["a.las"] == "no"
    assert status["b.las"] == "no"


def test_crs_column_no_when_crs_unreachable_from_3857():
    # A local engineering CRS parses but has no transform to Web Mercator.
    local_crs = 'LOCAL_CS["unknown",UNIT["metre",1]]'
    assert _crs_status([("cloud.las", {"spatialreference": local_crs})])["cloud.las"] == "no"


def test_crs_column_no_when_backup_unreachable_from_3857():
    status = _crs_status(
        [("cloud.las", {"spatialreference": ""})],
        spatial_reference='LOCAL_CS["x",UNIT["metre",1]]',
    )
    assert status["cloud.las"] == "no"


def test_load_input_sidecar_headered():
    df = load_and_clean_client_sidecar(
        url=str(INPUT_SIDECAR),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    # 4 rows: DEFAULT + 3 file rows
    assert len(df) == 4
    # First column is the filename column
    assert df.columns[0] == "Filename"
    # The required-group canonical fields are all present after alias rename.
    expected = {"GPSLatitude", "GPSLongitude", "GPSAltitude", "Pitch", "Roll", "Heading"}
    assert expected.issubset(set(df.columns))


def test_headerless_sidecar_with_schema_matches_headered():
    headered = load_and_clean_client_sidecar(
        url=str(INPUT_SIDECAR),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    headerless = load_and_clean_client_sidecar(
        url=str(HEADERLESS_SIDECAR),
        schema_path=EXAMPLE_SCHEMA,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )

    # Headerless input has no DEFAULT row, so 3 rows vs 4.
    assert len(headerless) == 3
    assert len(headered) == 4
    # Same column set after both pass through the cleaning pipeline.
    assert set(headered.columns) == set(headerless.columns)
    # Per-file values match between the two.
    file_col = headered.columns[0]
    for filename in ("1.jpg", "2.jpeg", "3.jpg"):
        h = headered[headered[file_col].str.strip() == filename].iloc[0]
        hl = headerless[headerless[file_col].str.strip() == filename].iloc[0]
        for col in headered.columns:
            assert str(h[col]).strip() == str(hl[col]).strip(), (
                f"mismatch on {filename}/{col}: {h[col]!r} vs {hl[col]!r}"
            )


@pytest.mark.parametrize(
    "client_csv,schema",
    [
        (INPUT_SIDECAR, None),
        (HEADERLESS_SIDECAR, EXAMPLE_SCHEMA),
    ],
)
def test_merge_file_metadata_wins_on_disagreement(client_csv, schema, caplog):
    """For 1.jpg the example sidecars set GPSAltitude=1000, EXIF=1100.
    File (EXIF) value wins; a warning is logged about the disagreement.
    """
    file_metadata = _exif_like_metadata()
    client_df = load_and_clean_client_sidecar(
        url=str(client_csv),
        schema_path=schema,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    with caplog.at_level("WARNING", logger="atomic_tools.client_sidecar"):
        merge_client_metadata(file_metadata, client_df)
    by_name = dict(file_metadata)
    assert by_name["1.jpg"]["GPSAltitude"] == "1100 m Above Sea Level"
    assert any("GPSAltitude" in rec.message and "1.jpg" in rec.message for rec in caplog.records)


def test_merge_client_empty_cell_preserves_exif():
    """1.jpg's CreateDate cell in the client CSV is empty — EXIF must survive."""
    file_metadata = _exif_like_metadata()
    client_df = load_and_clean_client_sidecar(
        url=str(INPUT_SIDECAR),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    merge_client_metadata(file_metadata, client_df)
    assert dict(file_metadata)["1.jpg"]["CreateDate"] == "2024:06:15 10:30:00"


def test_merge_default_row_fills_missing_and_summary_logged(caplog):
    """A DEFAULT row in the client CSV fills fields the file is missing, and
    the per-column summary lists what was added."""
    file_metadata = [
        ("1.jpg", {}),  # nothing — should pick up specific row + DEFAULT date
        ("2.jpeg", {"DateTimeOriginal": "2024:06:15 10:31:00"}),  # already set
        ("3.jpg", {}),
    ]
    client_df = load_and_clean_client_sidecar(
        url=str(INPUT_SIDECAR),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    with caplog.at_level("INFO", logger="atomic_tools.client_sidecar"):
        merge_client_metadata(file_metadata, client_df)

    by_name = dict(file_metadata)
    # 1.jpg's specific row has an empty date cell, so DEFAULT row fills it.
    # The date aliases canonicalize to CreateDate.
    assert by_name["1.jpg"]["CreateDate"] == "2024:07:15 10:30:00"
    # 1.jpg picks up its specific GPSAltitude.
    assert by_name["1.jpg"]["GPSAltitude"] == "1000 m Above Sea Level"

    summary = next(
        (rec.message for rec in caplog.records if rec.message.startswith("Added ")),
        "",
    )
    assert "columns of metadata" in summary
    assert "[CreateDate]" in summary
    assert "Default value on 1/3 files" in summary
    assert "[GPSAltitude]" in summary
    assert "File-specific value added to all files" in summary


def test_warn_missing_required_fields(caplog, capsys):
    """Files lacking any field from a required group should trigger a single
    summary warning log AND a detailed bright-red message on stderr."""
    file_metadata = [
        ("1.jpg", {"GPSLatitude": "51", "GPSLongitude": "-114"}),
        ("2.jpg", {"GPSLatitude": "51"}),  # missing GPSLongitude
    ]
    groups = [["GPSLatitude"], ["GPSLongitude"]]
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        _warn_missing_required_fields(file_metadata, groups)
    err = capsys.readouterr().err
    assert "MISSING REQUIRED METADATA" in err
    assert "\x1b[" in err  # ANSI escape: bright red
    assert "GPSLongitude" in err
    assert "2.jpg" in err
    warning_records = [rec for rec in caplog.records if rec.name == "atomic_tools.commands.sidecar"]
    assert len(warning_records) == 1
    assert "client sidecar merge" in warning_records[0].message.lower()


def test_fill_missing_dates_from_filepath_warns(caplog):
    """When a file has no date after merge, generation infers one from the
    filename and logs a warning saying so."""
    file_metadata = [("img.jpg", {"GPSLatitude": "51.0", "GPSLongitude": "-114.0"})]
    display_to_key = {"img.jpg": "trip/20240615_103000/img.jpg"}
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        _fill_missing_dates_from_filepath(file_metadata, ORIENTED_GROUPS, display_to_key)
    meta = dict(file_metadata)["img.jpg"]
    assert meta.get("CreateDate"), "expected a date inferred from the filename"
    assert any("from file path" in rec.message for rec in caplog.records)


def test_fill_missing_dates_skips_when_date_already_present():
    file_metadata = [("img.jpg", {"CreateDate": "2024:06:15 10:30:00"})]
    _fill_missing_dates_from_filepath(
        file_metadata, ORIENTED_GROUPS, {"img.jpg": "20240101/img.jpg"}
    )
    # Existing date untouched — no filename inference overrides it.
    assert dict(file_metadata)["img.jpg"]["CreateDate"] == "2024:06:15 10:30:00"


def test_warn_missing_required_fields_silent_when_all_satisfied(caplog, capsys):
    file_metadata = [("1.jpg", {"GPSLatitude": "51", "GPSLongitude": "-114"})]
    groups = [["GPSLatitude"], ["GPSLongitude"]]
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        _warn_missing_required_fields(file_metadata, groups)
    assert capsys.readouterr().err == ""
    assert not any("MISSING REQUIRED" in rec.message for rec in caplog.records)


def test_split_gps_position_string():
    out = _split_gps_position({"GPSPosition": "51 deg 3' 0.00\" N, 114 deg 4' 30.00\" W"})
    assert out == {
        "GPSLatitude": "51 deg 3' 0.00\" N",
        "GPSLongitude": "114 deg 4' 30.00\" W",
    }


def test_split_gps_position_list():
    assert _split_gps_position({"GPSPosition": [40.7, -74.0]}) == {
        "GPSLatitude": "40.7",
        "GPSLongitude": "-74.0",
    }


def test_split_gps_position_overwrites_existing():
    out = _split_gps_position(
        {
            "GPSPosition": "1.0, 2.0",
            "GPSLatitude": "old",
            "GPSLongitude": "old",
        }
    )
    assert out == {"GPSLatitude": "1.0", "GPSLongitude": "2.0"}


@pytest.mark.parametrize(
    "meta",
    [
        {"X": 1},  # absent
        {"GPSPosition": "no-comma"},  # unparseable string
        {"GPSPosition": ", "},  # empty halves
        {"GPSPosition": [1, 2, 3]},  # wrong list length
    ],
)
def test_split_gps_position_unchanged_on_bad_input(meta):
    assert _split_gps_position(meta) == meta


# ---- disambiguation -----------------------------------------------------


def test_disambiguate_unique_basenames_keep_basename():
    keys = ["a/1.jpg", "b/2.jpg", "c/3.jpg"]
    assert _disambiguate_filenames(keys) == {
        "a/1.jpg": "1.jpg",
        "b/2.jpg": "2.jpg",
        "c/3.jpg": "3.jpg",
    }


def test_disambiguate_colliding_basenames_add_one_parent():
    keys = ["a/1.jpg", "b/1.jpg"]
    assert _disambiguate_filenames(keys) == {
        "a/1.jpg": "a/1.jpg",
        "b/1.jpg": "b/1.jpg",
    }


def test_disambiguate_walks_up_until_parents_diverge():
    """Both files share the immediate parent ``x``, so the algorithm walks up
    one more level to ``a`` / ``b`` to make the labels unique."""
    keys = ["a/x/1.jpg", "b/x/1.jpg"]
    assert _disambiguate_filenames(keys) == {
        "a/x/1.jpg": "a/x/1.jpg",
        "b/x/1.jpg": "b/x/1.jpg",
    }


def test_disambiguate_mixed_depth_collision():
    """Three files with the same basename get a single round of extension —
    enough to make all three labels unique."""
    keys = ["a/x/1.jpg", "a/y/1.jpg", "b/x/1.jpg"]
    out = _disambiguate_filenames(keys)
    assert len(set(out.values())) == 3
    assert out["a/y/1.jpg"] == "y/1.jpg"  # uniquely identified at depth 1
    assert out["a/x/1.jpg"] == "a/x/1.jpg"
    assert out["b/x/1.jpg"] == "b/x/1.jpg"


def test_disambiguate_only_some_basenames_collide():
    """Only ``1.jpg`` collides; ``2.jpg`` keeps its bare basename."""
    keys = ["a/1.jpg", "b/1.jpg", "c/2.jpg"]
    assert _disambiguate_filenames(keys) == {
        "a/1.jpg": "a/1.jpg",
        "b/1.jpg": "b/1.jpg",
        "c/2.jpg": "2.jpg",
    }


def test_disambiguate_root_file_vs_nested():
    """A bare ``1.jpg`` at the scan root vs ``a/1.jpg`` deeper. The root file
    has no parent to add — its label stays as the basename, and the nested
    one gets prefixed."""
    keys = ["1.jpg", "a/1.jpg"]
    out = _disambiguate_filenames(keys)
    assert out == {"1.jpg": "1.jpg", "a/1.jpg": "a/1.jpg"}


# ---- merge with disambiguated labels ------------------------------------


def test_merge_uses_path_suffix_match_for_disambiguated_labels(tmp_path):
    """Files have disambiguated labels (``a/1.jpg``, ``b/1.jpg``); a client
    sidecar whose Filename column gives just ``a/1.jpg`` and ``b/1.jpg``
    must route to the right file."""
    csv = tmp_path / "client.csv"
    csv.write_text(
        "Filename,GPSAltitude\nDEFAULT,500\na/1.jpg,1100\nb/1.jpg,1200\n",
        encoding="utf-8",
    )
    file_metadata = [("a/1.jpg", {}), ("b/1.jpg", {})]
    client_df = load_and_clean_client_sidecar(
        url=str(csv),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    merge_client_metadata(file_metadata, client_df)
    by_label = dict(file_metadata)
    assert by_label["a/1.jpg"]["GPSAltitude"] == "1100"
    assert by_label["b/1.jpg"]["GPSAltitude"] == "1200"


def test_merge_basename_only_client_row_warns_on_ambiguity(tmp_path, caplog):
    """If the client provides a single row with a basename-only Filename, but
    multiple disambiguated files share that basename, the row matches both —
    log a clear warning."""
    csv = tmp_path / "client.csv"
    csv.write_text(
        "Filename,GPSAltitude\n1.jpg,9999\n",
        encoding="utf-8",
    )
    file_metadata = [("a/1.jpg", {}), ("b/1.jpg", {})]
    client_df = load_and_clean_client_sidecar(
        url=str(csv),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    with caplog.at_level("WARNING", logger="atomic_tools.client_sidecar"):
        merge_client_metadata(file_metadata, client_df)
    assert any("ambiguous" in rec.message.lower() for rec in caplog.records)


def test_merge_exact_label_wins_over_basename_match(tmp_path):
    """When client has both ``a/1.jpg`` and a generic ``1.jpg`` row, the file
    ``a/1.jpg`` must pick the exact-match row, not the basename row."""
    csv = tmp_path / "client.csv"
    csv.write_text(
        "Filename,GPSAltitude\n1.jpg,500\na/1.jpg,1100\n",
        encoding="utf-8",
    )
    file_metadata = [("a/1.jpg", {})]
    client_df = load_and_clean_client_sidecar(
        url=str(csv),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    merge_client_metadata(file_metadata, client_df)
    assert dict(file_metadata)["a/1.jpg"]["GPSAltitude"] == "1100"


def test_merge_does_not_match_unrelated_paths(tmp_path):
    """File ``a/1.jpg`` should not pick up a client ``b/1.jpg`` row — the
    parent components don't match, so it's a different file."""
    csv = tmp_path / "client.csv"
    csv.write_text(
        "Filename,GPSAltitude\nb/1.jpg,1200\n",
        encoding="utf-8",
    )
    file_metadata = [("a/1.jpg", {})]
    client_df = load_and_clean_client_sidecar(
        url=str(csv),
        schema_path=None,
        required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
    )
    merge_client_metadata(file_metadata, client_df)
    assert "GPSAltitude" not in dict(file_metadata)["a/1.jpg"]
