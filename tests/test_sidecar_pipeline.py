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
    _REQUIRED_SIDECAR_FIELD_GROUPS,
    _split_gps_position,
    _warn_missing_required_fields,
    build_sidecar_df,
)
from atomic_tools.utils.utils import DataTypeEnum

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example fake data"
INPUT_SIDECAR = EXAMPLE_DIR / "input sidecar" / "input_sidecar.csv"
HEADERLESS_SIDECAR = EXAMPLE_DIR / "input sidecar" / "headerless_sidecar.csv"
EXAMPLE_SCHEMA = REPO_ROOT / "schemas" / "example.json"

ORIENTED_GROUPS = _REQUIRED_SIDECAR_FIELD_GROUPS[DataTypeEnum.oriented_image]


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
    df = build_sidecar_df(_exif_like_metadata(), required_field_groups=ORIENTED_GROUPS)

    # DEFAULT first, then files sorted alphabetically.
    assert df["Filename"].tolist() == ["DEFAULT", "1.jpg", "2.jpeg", "3.jpg"]

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
    df = build_sidecar_df(metadata, required_field_groups=ORIENTED_GROUPS, full=True)
    assert "ExtraField" in df.columns
    df_filtered = build_sidecar_df(_exif_like_metadata(), required_field_groups=ORIENTED_GROUPS)
    assert "ExtraField" not in df_filtered.columns


def test_load_input_sidecar_headered():
    df = load_and_clean_client_sidecar(
        url=str(INPUT_SIDECAR),
        schema_path=None,
        required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
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
        required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
    )
    headerless = load_and_clean_client_sidecar(
        url=str(HEADERLESS_SIDECAR),
        schema_path=EXAMPLE_SCHEMA,
        required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
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
        required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
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
        required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
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
        required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
    )
    with caplog.at_level("INFO", logger="atomic_tools.client_sidecar"):
        merge_client_metadata(file_metadata, client_df)

    by_name = dict(file_metadata)
    # 1.jpg's specific row has an empty date cell, so DEFAULT row fills it.
    # CreateDate gets canonicalized to DateTimeOriginal by the alias map.
    assert by_name["1.jpg"]["DateTimeOriginal"] == "2024:07:15 10:30:00"
    # 1.jpg picks up its specific GPSAltitude.
    assert by_name["1.jpg"]["GPSAltitude"] == "1000 m Above Sea Level"

    summary = next(
        (rec.message for rec in caplog.records if rec.message.startswith("Added ")),
        "",
    )
    assert "columns of metadata" in summary
    assert "[DateTimeOriginal]" in summary
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
