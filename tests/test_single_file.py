"""Single-file targets: `--path` may name one file/object instead of a folder."""

from __future__ import annotations

import json

import pytest
from obstore.store import LocalStore
from typer.testing import CliRunner

from atomic_tools import validate_path as _validate_path
from atomic_tools.cli import app
from atomic_tools.io.storage import LocalBackend, ObjectStoreBackend, from_path
from tests.test_auto_detect import patched_extractors  # noqa: F401

runner = CliRunner()


def validate_path(*args, **kwargs) -> dict:
    return _validate_path(*args, **kwargs).to_json_dict()


def _dir_with_siblings(tmp_path):
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "b.jpg").write_text("x")
    (tmp_path / "c.las").write_text("x")
    return tmp_path


# ---- local backend ------------------------------------------------------------


def test_from_path_local_file_lists_only_that_file(tmp_path):
    _dir_with_siblings(tmp_path)
    backend = from_path(str(tmp_path / "a.jpg"))
    assert isinstance(backend, LocalBackend)
    assert backend.single_file == "a.jpg"
    assert backend.list_keys(include=[".jpg", ".las"], exclude=[]) == ["a.jpg"]
    assert backend.display_root == str((tmp_path / "a.jpg").resolve())
    # open_local yields the real file; outputs land beside it.
    with backend.open_local("a.jpg") as local:
        assert local == str((tmp_path / "a.jpg").resolve())


def test_from_path_local_file_respects_include(tmp_path):
    _dir_with_siblings(tmp_path)
    assert from_path(str(tmp_path / "a.jpg")).list_keys(include=[".las"], exclude=[]) == []


def test_from_path_local_directory_unchanged(tmp_path):
    _dir_with_siblings(tmp_path)
    backend = from_path(str(tmp_path))
    assert backend.single_file is None
    assert sorted(backend.list_keys(include=[".jpg"], exclude=[])) == ["a.jpg", "b.jpg"]


def test_from_path_missing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_path(str(tmp_path / "nope.jpg"))


# ---- object-store backend (real obstore semantics via LocalStore) -------------


def _object_backend(monkeypatch, root, prefix):
    monkeypatch.setattr(ObjectStoreBackend, "__init__", lambda self, **kwargs: None)
    backend = ObjectStoreBackend()
    backend._scheme = "s3"
    backend._bucket = "bucket"
    backend._prefix = prefix
    backend._store = LocalStore(prefix=str(root))
    backend._single_key = None
    backend._single_size = 0
    return backend


def test_object_store_single_key_falls_back_to_head(monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    _dir_with_siblings(tmp_path / "sub")
    backend = _object_backend(monkeypatch, tmp_path, "sub/a.jpg")
    assert backend.list_keys(include=[".jpg"], exclude=[]) == ["sub/a.jpg"]
    assert backend.single_file == "sub/a.jpg"


def test_object_store_prefix_still_lists_folder(monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    _dir_with_siblings(tmp_path / "sub")
    backend = _object_backend(monkeypatch, tmp_path, "sub")
    assert sorted(backend.list_keys(include=[".jpg"], exclude=[])) == ["sub/a.jpg", "sub/b.jpg"]
    assert backend.single_file is None


def test_object_store_missing_key_lists_nothing(monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    backend = _object_backend(monkeypatch, tmp_path, "sub/nope.jpg")
    assert backend.list_keys(include=[".jpg"], exclude=[]) == []
    assert backend.single_file is None


def test_object_store_denied_head_is_not_an_auth_error(monkeypatch, tmp_path):
    """Policies scoped to ``folder/*`` deny HEAD on the bare key; that must read
    as "no such single object", not as bad credentials."""
    from obstore.exceptions import PermissionDeniedError

    import atomic_tools.io.storage as storage_mod

    (tmp_path / "sub").mkdir()
    backend = _object_backend(monkeypatch, tmp_path, "sub/a.jpg")

    def denied(store, key):
        raise PermissionDeniedError("AccessDenied")

    monkeypatch.setattr(storage_mod.obstore, "head", denied)
    assert backend.list_keys(include=[".jpg"], exclude=[]) == []
    assert backend.single_file is None


def test_object_store_folder_prefix_skips_head(monkeypatch, tmp_path):
    """An empty folder-like prefix (no extension) never triggers a HEAD."""
    import atomic_tools.io.storage as storage_mod

    (tmp_path / "flight1").mkdir()
    backend = _object_backend(monkeypatch, tmp_path, "flight1")
    calls = []
    monkeypatch.setattr(storage_mod.obstore, "head", lambda store, key: calls.append(key))
    assert backend.list_keys(include=[".jpg"], exclude=[]) == []
    assert calls == []


def test_object_store_single_key_writes_beside_object(monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    _dir_with_siblings(tmp_path / "sub")
    backend = _object_backend(monkeypatch, tmp_path, "sub/a.jpg")
    backend.list_keys(include=[".jpg"], exclude=[])
    uploaded = {}
    monkeypatch.setattr(
        "atomic_tools.io.storage.upload",
        lambda store, key, source: uploaded.setdefault("key", key),
    )
    assert backend.write_output("sidecar.csv", tmp_path / "x") == "s3://bucket/sub/sidecar.csv"
    assert uploaded["key"] == "sub/sidecar.csv"


# ---- validate end to end ------------------------------------------------------


def test_validate_path_single_file(tmp_path, patched_extractors):  # noqa: F811
    _dir_with_siblings(tmp_path)
    payload = validate_path(str(tmp_path / "a.jpg"), include="all")
    assert payload["status"] != "error", payload
    assert payload["summary"]["files_scanned"] == 1
    assert payload["summary"]["detected_types"] == ["oriented_image"]
    assert [r["Filename"] for r in payload["metadata"]] == ["DEFAULT", "a.jpg"]
    # The inventory check sees only the target file, so siblings aren't flagged.
    messages = " ".join(f["message"] for f in payload["findings"])
    assert "b.jpg" not in messages and "c.las" not in messages


def test_validate_cli_single_file(tmp_path, patched_extractors):  # noqa: F811
    _dir_with_siblings(tmp_path)
    result = runner.invoke(app, ["validate", "--path", str(tmp_path / "a.jpg"), "--api-mode"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["summary"]["files_scanned"] == 1
    # Nothing written beside the file.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.jpg", "b.jpg", "c.las"]


def test_validate_single_file_datatype_mismatch(tmp_path, patched_extractors):  # noqa: F811
    _dir_with_siblings(tmp_path)
    payload = validate_path(str(tmp_path / "a.jpg"), data_type="point_cloud")
    assert payload["error"]["type"] == "invalid_input"
    assert "is not a 'point_cloud' file" in payload["error"]["message"]


@pytest.mark.parametrize(
    "name,expected",
    [("notes.txt", "is not a supported file type"), ("layer.geojson", "is vector data")],
)
def test_validate_single_unsupported_file(tmp_path, patched_extractors, name, expected):  # noqa: F811
    (tmp_path / name).write_text("x")
    payload = validate_path(str(tmp_path / name))
    assert payload["error"]["type"] == "invalid_input"
    assert expected in payload["error"]["message"]


def test_validate_missing_path_is_invalid_input(tmp_path):
    payload = validate_path(str(tmp_path / "nope.jpg"))
    assert payload["error"]["type"] == "invalid_input"
    assert "does not exist" in payload["error"]["message"]


def test_cli_missing_path_exits_2(tmp_path):
    result = runner.invoke(app, ["validate", "--path", str(tmp_path / "nope.jpg"), "--api-mode"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["type"] == "invalid_input"


# ---- --path / --directory alias ------------------------------------------------


@pytest.mark.parametrize("flag", ["--path", "--directory"])
def test_validate_accepts_path_and_directory_alias(tmp_path, patched_extractors, flag):  # noqa: F811
    _dir_with_siblings(tmp_path)
    result = runner.invoke(app, ["validate", flag, str(tmp_path / "a.jpg"), "--api-mode"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["summary"]["files_scanned"] == 1


@pytest.mark.parametrize("command", ["sidecar", "validate"])
def test_help_lists_path_with_directory_alias(command):
    result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--path" in result.stdout
    assert "--directory" in result.stdout


@pytest.mark.parametrize("command", ["sidecar", "validate"])
def test_replay_command_uses_path(command):
    from atomic_tools.commands.sidecar import _format_replay_command

    cmd = _format_replay_command(
        [command],
        directory="./data/a.jpg",
        data_type=None,
        client_sidecar=None,
        client_schema=None,
        full=False,
        verbosity="default",
        spatial_reference=None,
        ignore_missing_orientation=False,
    )
    assert cmd == f"am-tools {command} --path ./data/a.jpg"


def test_sidecar_directory_alias_and_single_file(tmp_path, patched_extractors):  # noqa: F811
    """`sidecar --directory <file>` still works and writes the CSV beside the file."""
    _dir_with_siblings(tmp_path)
    result = runner.invoke(
        app,
        ["sidecar", "--directory", str(tmp_path / "a.jpg"), "--output-filename", "out.csv"],
    )
    assert (tmp_path / "out.csv").exists(), result.output
    rows = (tmp_path / "out.csv").read_text().splitlines()
    assert any(r.startswith("a.jpg") for r in rows)
    assert not any(r.startswith(("b.jpg", "c.las")) for r in rows)
