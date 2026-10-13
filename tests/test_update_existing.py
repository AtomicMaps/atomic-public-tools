"""`sidecar --path <file> --update-existing`: add a single file's row to the
folder's existing sidecar instead of refusing to overwrite it."""

import pandas as pd
import pytest
from obstore.store import LocalStore
from typer.testing import CliRunner

from atomic_tools.cli import app
from atomic_tools.commands.sidecar import (
    _format_replay_command,
    _merge_into_existing,
)
from atomic_tools.errors import InvalidInputError
from atomic_tools.io.storage import LocalBackend, ObjectStoreBackend
from tests.test_auto_detect import patched_extractors  # noqa: F401

runner = CliRunner()


def _sidecar(tmp_path, *extra):
    return runner.invoke(app, ["sidecar", "--path", str(tmp_path), *extra])


def _single(tmp_path, name, *extra):
    return runner.invoke(app, ["sidecar", "--path", str(tmp_path / name), *extra])


def _rows(tmp_path):
    return pd.read_csv(tmp_path / "sidecar.csv", dtype=str, keep_default_na=False)


@pytest.fixture
def folder(tmp_path, patched_extractors):  # noqa: F811
    """A folder with a generated sidecar for a.jpg and b.jpg, plus a new c.jpg."""
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_text("x")
    _sidecar(tmp_path)
    assert (tmp_path / "sidecar.csv").exists()
    (tmp_path / "c.jpg").write_text("x")
    return tmp_path


# ---- CLI behaviour ---------------------------------------------------------------


def test_refusal_suggests_update_existing(folder):
    before = (folder / "sidecar.csv").read_bytes()
    result = _single(folder, "c.jpg")
    assert result.exit_code == 1
    assert (folder / "sidecar.csv").read_bytes() == before


def test_update_existing_adds_the_row(folder):
    before = _rows(folder)
    result = _single(folder, "c.jpg", "--update-existing")
    assert result.exit_code == 0, result.output
    after = _rows(folder)
    assert list(after["Filename"]) == ["DEFAULT", "a.jpg", "b.jpg", "c.jpg"]
    # Existing rows (and DEFAULT) are untouched.
    pd.testing.assert_frame_equal(after.iloc[:3][before.columns], before)


def test_already_listed_file_keeps_its_row(folder):
    before = (folder / "sidecar.csv").read_bytes()
    result = _single(folder, "a.jpg", "--update-existing")
    assert result.exit_code == 0, result.output
    assert (folder / "sidecar.csv").read_bytes() == before  # not even rewritten


def test_only_generated_sidecars_are_updated(tmp_path, patched_extractors):  # noqa: F811
    (tmp_path / "c.jpg").write_text("x")
    client_csv = "Filename,GPSAltitude\na.jpg,100\n"  # no DataType column, no DEFAULT row
    (tmp_path / "sidecar.csv").write_text(client_csv)
    result = _single(tmp_path, "c.jpg", "--update-existing")
    assert result.exit_code == 1
    assert (tmp_path / "sidecar.csv").read_text() == client_csv


def test_flag_requires_a_single_file(folder):
    before = (folder / "sidecar.csv").read_bytes()
    assert _sidecar(folder, "--update-existing").exit_code == 1
    assert (folder / "sidecar.csv").read_bytes() == before


def test_flag_without_existing_sidecar_writes_a_new_one(tmp_path, patched_extractors):  # noqa: F811
    (tmp_path / "c.jpg").write_text("x")
    result = _single(tmp_path, "c.jpg", "--update-existing")
    assert result.exit_code == 0, result.output
    assert list(_rows(tmp_path)["Filename"]) == ["DEFAULT", "c.jpg"]


def test_only_the_new_file_is_linted(tmp_path, patched_extractors):  # noqa: F811
    """A broken row elsewhere in the sidecar doesn't fail this file's run."""
    (tmp_path / "b.jpg").write_text("x")
    (tmp_path / "c.jpg").write_text("x")
    (tmp_path / "sidecar.csv").write_text(
        "Filename,DataType,GPSLatitude,GPSLongitude\n"
        "DEFAULT,,,\n"
        "b.jpg,oriented_image,,\n"  # missing required GPS
    )
    result = _single(tmp_path, "c.jpg", "--update-existing")
    assert result.exit_code == 0, result.output
    assert "b.jpg" not in result.output
    assert "c.jpg" in set(_rows(tmp_path)["Filename"])


def test_conflict_when_sidecar_changes_mid_run(folder, monkeypatch):
    before = (folder / "sidecar.csv").read_bytes()
    versions = iter(["v1", "v2"])  # read sees v1; the pre-write check sees v2
    monkeypatch.setattr(LocalBackend, "output_version", lambda self, name: next(versions))
    result = _single(folder, "c.jpg", "--update-existing")
    assert result.exit_code == 1  # SidecarConflictError, reported via _fail
    assert (folder / "sidecar.csv").read_bytes() == before


def test_replay_command_includes_flag():
    cmd = _format_replay_command(
        ["sidecar"], directory="./data/c.jpg", data_type=None, client_sidecar=None,
        client_schema=None, full=False, verbosity="default", spatial_reference=None,
        ignore_missing_orientation=False, update_existing=True,
    )
    assert cmd == "am-tools sidecar --path ./data/c.jpg --update-existing"


# ---- merge rules -----------------------------------------------------------------


def _df(rows):
    return pd.DataFrame(rows).fillna("")


def test_merge_unions_columns_and_sorts():
    existing = _df([
        {"Filename": "DEFAULT", "DataType": "", "GPSLatitude": ""},
        {"Filename": "z.jpg", "DataType": "oriented_image", "GPSLatitude": "45"},
    ])
    new = _df([
        {"Filename": "DEFAULT", "DataType": "", "bounds.minx": ""},
        {"Filename": "a.las", "DataType": "point_cloud", "bounds.minx": "500000"},
    ])
    merged, added, kept = _merge_into_existing(existing, new)
    assert added == ["a.las"] and kept == []
    assert list(merged.columns) == ["Filename", "DataType", "GPSLatitude", "bounds.minx"]
    assert list(merged["Filename"]) == ["DEFAULT", "z.jpg", "a.las"]  # oriented_image < point_cloud
    jpg = merged[merged["Filename"] == "z.jpg"].iloc[0]
    assert jpg["bounds.minx"] == "" and jpg["GPSLatitude"] == "45"


def test_merge_sorts_by_type_then_name():
    existing = _df([
        {"Filename": "DEFAULT", "DataType": ""},
        {"Filename": "b.jpg", "DataType": "oriented_image"},
        {"Filename": "x.las", "DataType": "point_cloud"},
    ])
    new = _df([
        {"Filename": "DEFAULT", "DataType": ""},
        {"Filename": "a.jpg", "DataType": "oriented_image"},
    ])
    merged, _, _ = _merge_into_existing(existing, new)
    assert list(merged["Filename"]) == ["DEFAULT", "a.jpg", "b.jpg", "x.las"]


def test_new_default_values_go_on_the_new_row():
    existing = _df([
        {"Filename": "DEFAULT", "DataType": "", "fallback_srs": ""},
        {"Filename": "old.las", "DataType": "point_cloud", "fallback_srs": ""},
    ])
    new = _df([
        {"Filename": "DEFAULT", "DataType": "", "fallback_srs": "EPSG:32612"},
        {"Filename": "new.las", "DataType": "point_cloud", "fallback_srs": ""},
    ])
    merged, _, _ = _merge_into_existing(existing, new)
    by_name = merged.set_index("Filename")["fallback_srs"]
    assert by_name["DEFAULT"] == ""
    assert by_name["old.las"] == ""
    assert by_name["new.las"] == "EPSG:32612"


def test_default_value_already_shared_is_not_copied():
    existing = _df([
        {"Filename": "DEFAULT", "DataType": "", "fallback_srs": "EPSG:32612"},
        {"Filename": "old.las", "DataType": "point_cloud", "fallback_srs": ""},
    ])
    new = _df([
        {"Filename": "DEFAULT", "DataType": "", "fallback_srs": "EPSG:32612"},
        {"Filename": "new.las", "DataType": "point_cloud", "fallback_srs": ""},
    ])
    merged, _, _ = _merge_into_existing(existing, new)
    assert merged.set_index("Filename")["fallback_srs"]["new.las"] == ""


def test_merge_rejects_non_generated_sidecars():
    with pytest.raises(InvalidInputError, match="wasn't generated by am-tools"):
        _merge_into_existing(_df([{"Filename": "a.jpg", "GPSAltitude": "1"}]),
                             _df([{"Filename": "DEFAULT", "DataType": ""}]))


# ---- object-store read/version -----------------------------------------------------


def test_object_store_read_output_and_version(monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.jpg").write_text("x")
    sidecar = tmp_path / "sub" / "sidecar.csv"
    sidecar.write_text("Filename,DataType\nDEFAULT,\na.jpg,oriented_image\n")
    monkeypatch.setattr(ObjectStoreBackend, "__init__", lambda self, **kwargs: None)
    backend = ObjectStoreBackend()
    backend._scheme, backend._bucket, backend._prefix = "s3", "bucket", "sub/c.jpg"
    backend._store = LocalStore(prefix=str(tmp_path))
    backend._single_key, backend._single_size, backend._listing = None, 0, None
    backend._folders = {}
    backend._modified = {}
    backend._listing_indexed = False
    backend.list_keys(include=[".jpg"], exclude=[])  # resolves the single object

    df, version = backend.read_output("sidecar.csv")
    assert list(df["Filename"]) == ["DEFAULT", "a.jpg"]
    assert version is not None and backend.output_version("sidecar.csv") == version
    assert backend.output_version("missing.csv") is None

    sidecar.write_text("Filename,DataType\nDEFAULT,\nb.jpg,oriented_image\n")
    assert backend.output_version("sidecar.csv") != version
