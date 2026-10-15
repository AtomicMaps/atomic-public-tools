"""Edge cases: single-file safety, uploads, sparse-read fallback, backend reuse."""

import io
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest
from obstore.store import LocalStore
from typer.testing import CliRunner

import atomic_tools.api as api_mod
import atomic_tools.io.storage as storage_mod
import atomic_tools.validators.sidecar as lint_mod
from atomic_tools import UploadedFile, validate_path, validate_upload
from atomic_tools.cli import app
from atomic_tools.commands.sidecar import _sparse_read_problem
from atomic_tools.io.partial_read import SparseStats
from atomic_tools.io.storage import LocalBackend, ObjectStoreBackend, from_path
from atomic_tools.validators.report import LintReport
from tests.test_auto_detect import patched_extractors  # noqa: F401

runner = CliRunner()


# ---- single-file sidecar never overwrites ----------------------------------------


def _folder(tmp_path):
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "b.jpg").write_text("x")
    (tmp_path / "sidecar.csv").write_text("Filename\nDEFAULT\na.jpg\nb.jpg\n")
    return tmp_path


def test_single_file_sidecar_refuses_to_overwrite(tmp_path, patched_extractors):  # noqa: F811
    _folder(tmp_path)
    before = (tmp_path / "sidecar.csv").read_text()
    result = runner.invoke(
        app, ["sidecar", "--path", str(tmp_path / "a.jpg"), "--output-filename", "sidecar.csv"]
    )
    assert result.exit_code == 1
    assert (tmp_path / "sidecar.csv").read_text() == before


def test_single_file_sidecar_defaults_to_its_own_name(tmp_path, patched_extractors):  # noqa: F811
    _folder(tmp_path)
    before = (tmp_path / "sidecar.csv").read_text()
    for _ in range(2):  # a rerun replaces its own default-named sidecar
        result = runner.invoke(app, ["sidecar", "--path", str(tmp_path / "a.jpg")])
        assert result.exit_code == 0, result.output
    assert (tmp_path / "sidecar_a_jpg.csv").exists()
    assert (tmp_path / "sidecar.csv").read_text() == before


def test_single_file_sidecar_writes_under_a_new_name(tmp_path, patched_extractors):  # noqa: F811
    _folder(tmp_path)
    before = (tmp_path / "sidecar.csv").read_text()
    runner.invoke(
        app, ["sidecar", "--path", str(tmp_path / "a.jpg"), "--output-filename", "a_sidecar.csv"]
    )
    assert (tmp_path / "a_sidecar.csv").exists()
    assert (tmp_path / "sidecar.csv").read_text() == before


def test_folder_sidecar_still_overwrites_its_own(tmp_path, patched_extractors):  # noqa: F811
    _folder(tmp_path)
    runner.invoke(app, ["sidecar", "--path", str(tmp_path)])
    assert "DataType" in (tmp_path / "sidecar.csv").read_text()  # regenerated


def test_object_store_output_exists(monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.jpg").write_text("x")
    (tmp_path / "sub" / "sidecar.csv").write_text("x")
    backend = _object_backend(monkeypatch, tmp_path, "sub/a.jpg")
    backend.list_keys(include=[".jpg"], exclude=[])
    assert backend.output_exists("sidecar.csv") is True
    assert backend.output_exists("other.csv") is False
    assert backend.output_location == "s3://bucket/sub"  # the folder, not the file


def test_local_output_location_is_the_folder(tmp_path):
    (tmp_path / "a.jpg").write_text("x")
    backend = from_path(str(tmp_path / "a.jpg"))
    assert isinstance(backend, LocalBackend)
    assert backend.output_location == str(tmp_path.resolve())


# ---- empty optional form fields ------------------------------------------------


def test_empty_optional_uploads_are_ignored(tmp_path, patched_extractors):  # noqa: F811
    empty = SimpleNamespace(file=io.BytesIO(b""), filename="")
    response = validate_upload(
        UploadedFile(b"x", "a.jpg"),
        client_sidecar=empty,
        client_schema=SimpleNamespace(file=io.BytesIO(b""), filename=""),
        coco=SimpleNamespace(file=io.BytesIO(b""), filename=""),
        work_dir=tmp_path,
    )
    assert response.status != "error", response.error


def test_nameless_upload_with_content_is_still_rejected(tmp_path):
    response = validate_upload(
        UploadedFile(b"x", "a.jpg"),
        coco=SimpleNamespace(file=io.BytesIO(b'{"images": []}'), filename=""),
        work_dir=tmp_path,
    )
    assert response.error.type == "invalid_input"


def test_fastapi_form_with_empty_file_inputs(patched_extractors):  # noqa: F811
    from tests.test_validate_upload import _make_app

    app_ = _make_app()
    from fastapi.testclient import TestClient

    resp = TestClient(app_).post(
        "/validate",
        files={
            "file": ("IMG_0001.jpg", b"jpeg", "image/jpeg"),
            "coco": ("", b"", "application/octet-stream"),
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] != "error", resp.json()


# ---- COCO scoped to a single-file run --------------------------------------------


def test_single_file_coco_ignores_other_images(tmp_path, patched_extractors):  # noqa: F811
    coco = {
        "images": [
            {"id": 1, "file_name": "IMG_0001.jpg", "width": 10, "height": 10},
            {"id": 2, "file_name": "IMG_0002.jpg", "width": 10, "height": 10},
            {"id": 3, "file_name": "IMG_0003.jpg", "width": 10, "height": 10},
        ],
        "annotations": [{"image_id": 1}, {"image_id": 2}, {"image_id": 3}],
    }
    payload = validate_upload(
        UploadedFile(b"x", "IMG_0001.jpg"),
        coco=UploadedFile(json.dumps(coco).encode(), "labels.coco.json"),
        include="all",
        work_dir=tmp_path,
    ).to_json_dict()
    details = payload["details"]["coco"]
    tiers = {t["tier"]: t["images"] for t in details["tiers"]}
    assert details["out_of_scope"] == 2
    assert tiers["not_on_disk"] == 0
    assert details["total_labels"] == 1
    assert not any("not_on_disk" in f["message"] and f["level"] != "info"
                   for f in payload["findings"])


# ---- include must be strings ---------------------------------------------------


@pytest.mark.parametrize("include", [["findings", 1], 5, [None]])
def test_bad_include_types_are_invalid_input(tmp_path, include):
    response = validate_path(str(tmp_path), include=include)
    assert response.error.type == "invalid_input"


# ---- temp dir errors don't leak paths --------------------------------------------


def test_temp_dir_failure_hides_server_path(tmp_path):
    missing = tmp_path / "no" / "such" / "dir"
    response = validate_upload(UploadedFile(b"x", "a.jpg"), work_dir=missing)
    assert response.error.type == "internal_error"
    assert str(tmp_path) not in response.error.message
    assert "temp dir" in response.error.message


# ---- which sparse-read warnings force a full download ----------------------------

_STATS = SparseStats(fmt="bmff", requests=3, fetched=3000, size=10**9,
                     ranges=((0, 1024), (0x337C0000, 0x337D0000)))


@pytest.mark.parametrize(
    "meta,expect_fallback",
    [
        ({"Make": "X"}, False),
        ({}, True),
        ({"Error": "File format error"}, True),
        # exiftool's own informational hint: the full file says the same.
        ({"Warning": "[minor] The ExtractEmbedded option may find more tags in the media data"},
         False),
        # [minor], but worded like a structural problem: don't trust it.
        ({"Warning": "[minor] Possibly incorrect maker notes offsets (fix by 1783?)"}, True),
        # Names an offset inside a fetched range: those bytes are real.
        ({"Warning": "Truncated 'RDT6' data at offset 0x337c2ff0"}, False),
        # Names an offset in a hole: the sparse copy may have caused it.
        ({"Warning": "Truncated 'mdat' data at offset 0x00100000"}, True),
        ({"Warning": "Truncated PNG image"}, True),
        ({"Warning": "Invalid EXIF text encoding for UserComment"}, True),
    ],
)
def test_sparse_read_problem_rules(meta, expect_fallback):
    assert (_sparse_read_problem(meta, _STATS) is not None) is expect_fallback


def test_offset_rule_needs_stats():
    assert _sparse_read_problem({"Warning": "Truncated data at offset 0x10"}, None) is not None


# ---- unsupported extensions skip the head request ---------------------------------


def test_png_metadata_view_skips_sparse_attempt(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "pano.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 100)
    backend = _object_backend(monkeypatch, tmp_path, "data")

    def no_get(*a, **k):
        raise AssertionError("a PNG must not cost a ranged GET before the full download")

    monkeypatch.setattr(storage_mod.obstore, "get", no_get)

    @contextmanager
    def fake_open_local(self, key):
        yield str(tmp_path / key)

    monkeypatch.setattr(ObjectStoreBackend, "open_local", fake_open_local)
    with backend.open_metadata_view("data/pano.png") as view:
        assert view.sparse is False


# ---- buffer uploads without an extra copy -----------------------------------------


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_buffer_uploads(tmp_path, patched_extractors, wrap):  # noqa: F811
    response = validate_upload(UploadedFile(wrap(b"jpeg"), "a.jpg"), work_dir=tmp_path)
    assert response.status != "error", response.error
    too_big = validate_upload(UploadedFile(wrap(b"x" * 11), "a.jpg"), max_bytes=10,
                              work_dir=tmp_path)
    assert too_big.error.type == "invalid_input"


# ---- the scan's backend is reused for the inventory check -------------------------


def test_run_validation_passes_its_backend_to_lint(monkeypatch, tmp_path):
    sentinel = object()
    seen = {}

    def fake_build(**kwargs):
        return pd.DataFrame([{"Filename": "DEFAULT"}]), sentinel, set(), []

    def fake_lint(df, **kwargs):
        seen.update(kwargs)
        return LintReport()

    monkeypatch.setattr(api_mod, "_build_sidecar", fake_build)
    monkeypatch.setattr(api_mod, "lint_sidecar_df", fake_lint)
    api_mod.run_validation(str(tmp_path))
    assert seen["input_backend"] is sentinel


def test_lint_uses_given_backend_instead_of_opening_one(monkeypatch, tmp_path):
    def boom(path):
        raise AssertionError("from_path must not be called when a backend is given")

    monkeypatch.setattr(lint_mod, "from_path", boom)
    (tmp_path / "a.jpg").write_text("x")
    backend = LocalBackend(tmp_path)
    df = pd.DataFrame(
        [
            {"Filename": "DEFAULT", "DataType": ""},
            {"Filename": "a.jpg", "DataType": "oriented_image"},
        ]
    )
    report = lint_mod.lint_sidecar_df(
        df, final=True, data_type=None, input_files_path="s3://unused", input_backend=backend
    )
    assert not any("Could not enumerate" in f.message for f in report.findings)


# ---- helpers --------------------------------------------------------------------------


def _object_backend(monkeypatch, root, prefix):
    monkeypatch.setattr(ObjectStoreBackend, "__init__", lambda self, **kwargs: None)
    backend = ObjectStoreBackend()
    backend._scheme, backend._bucket, backend._prefix = "s3", "bucket", prefix
    backend._store = LocalStore(prefix=str(root))
    backend._single_key, backend._single_size, backend._listing = None, 0, None
    backend._folders = {}
    backend._modified = {}
    backend._listing_indexed = False
    return backend
