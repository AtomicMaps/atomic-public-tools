from pathlib import Path

from atomic_tools.client_sidecar import ClientSchema, load_client_schema

REPO_ROOT = Path(__file__).resolve().parent.parent


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
