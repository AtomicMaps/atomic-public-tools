from pathlib import Path

import pytest

from atomic_tools.client_sidecar import (
    ClientSchema,
    SidecarMergeError,
    _list_sidecar_csvs_below,
    load_and_clean_client_sidecar,
    load_client_schema,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_HEADER = "Filename,GPSLatitude,GPSLongitude\n"


def test_load_returns_empty_when_path_is_none():
    schema = load_client_schema(None)
    assert schema == ClientSchema()
    assert schema.column_names == ()
    assert dict(schema.column_name_mapping) == {}


def test_load_parses_example_json():
    schema = load_client_schema(REPO_ROOT / "schemas" / "column_names_example.json")
    assert schema.column_names == (
        "Filename",
        "CreateDate",
        "GPSAltitude",
        "GPSLatitude",
        "GPSLongitude",
        "Pitch",
        "Roll",
        "Heading",
    )
    assert dict(schema.column_name_mapping) == {}


def test_load_handles_missing_keys(tmp_path):
    p = tmp_path / "minimal.json"
    p.write_text("{}")
    schema = load_client_schema(p)
    assert schema.column_names == ()
    assert dict(schema.column_name_mapping) == {}


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + body)


def test_list_sidecar_csvs_below_skips_top_level(tmp_path):
    _write(tmp_path / "flightA" / "a.csv", "IMG_0001.jpg,40.1,-105.1\n")
    _write(tmp_path / "nested" / "deep" / "c.csv", "IMG_0003.jpg,42.3,-107.3\n")
    # A CSV directly in the top level (e.g. the generated sidecar) must be skipped.
    _write(tmp_path / "sidecar.csv", "DEFAULT,,\n")

    found = _list_sidecar_csvs_below(str(tmp_path))

    assert [Path(p).name for p in found] == ["a.csv", "c.csv"]


def test_directory_merge_combines_subfolder_csvs(tmp_path):
    _write(tmp_path / "flightA" / "a.csv", "IMG_0001.jpg,40.1,-105.1\n")
    _write(tmp_path / "flightB" / "b.csv", "IMG_0002.jpg,41.2,-106.2\n")
    _write(tmp_path / "sidecar.csv", "DEFAULT,,\n")  # top level — ignored

    df = load_and_clean_client_sidecar(str(tmp_path), schema_path=None, required_field_groups={})

    assert sorted(df["Filename"]) == ["IMG_0001.jpg", "IMG_0002.jpg"]


def test_directory_merge_dedupes_default_rows(tmp_path):
    _write(tmp_path / "flightA" / "a.csv", "DEFAULT,,99\nIMG_0001.jpg,40.1,-105.1\n")
    _write(tmp_path / "flightB" / "b.csv", "DEFAULT,,88\nIMG_0002.jpg,41.2,-106.2\n")

    df = load_and_clean_client_sidecar(str(tmp_path), schema_path=None, required_field_groups={})

    assert list(df["Filename"]).count("DEFAULT") == 1


def test_directory_merge_rejects_mismatched_schema(tmp_path):
    _write(tmp_path / "flightA" / "a.csv", "IMG_0001.jpg,40.1,-105.1\n")
    # Extra column -> different schema; the error must name the offending file.
    (tmp_path / "flightB").mkdir(parents=True)
    (tmp_path / "flightB" / "b.csv").write_text(
        "Filename,GPSLatitude,GPSLongitude,GPSAltitude\nIMG_0002.jpg,41.2,-106.2,1500\n"
    )

    with pytest.raises(SidecarMergeError) as excinfo:
        load_and_clean_client_sidecar(str(tmp_path), schema_path=None, required_field_groups={})

    assert "b.csv" in str(excinfo.value)


def test_directory_merge_errors_when_no_subfolder_csvs(tmp_path):
    # Only a top-level CSV exists, which is never scanned.
    _write(tmp_path / "sidecar.csv", "DEFAULT,,\n")

    with pytest.raises(SidecarMergeError, match="No sidecar CSVs"):
        load_and_clean_client_sidecar(str(tmp_path), schema_path=None, required_field_groups={})


def test_single_file_still_loads(tmp_path):
    csv = tmp_path / "client.csv"
    _write(csv, "IMG_0001.jpg,40.1,-105.1\n")

    df = load_and_clean_client_sidecar(str(csv), schema_path=None, required_field_groups={})

    assert list(df["Filename"]) == ["IMG_0001.jpg"]
