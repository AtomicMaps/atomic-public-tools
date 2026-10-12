"""Library use from a web service: validate_upload, response models, health check."""

# No `from __future__ import annotations`: FastAPI must resolve the endpoint
# annotations in _make_app at definition time.

import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import pytest

import atomic_tools
import atomic_tools.api as api_mod
import atomic_tools.commands.sidecar as sidecar_mod
import atomic_tools.utils.utils as utils_mod
from atomic_tools import (
    DependencyReport,
    UploadedFile,
    ValidationResponse,
    check_dependencies,
    validate_upload,
)
from atomic_tools.utils.extractors import PdalNotFoundError, extract_exif_metadata
from atomic_tools.utils.utils import ExiftoolNotFoundError
from atomic_tools.validators.report import LintReport
from atomic_tools.validators.values import parse_creation_year
from tests.test_auto_detect import patched_extractors  # noqa: F401


def _starlette_like(data: bytes, filename: str):
    """Duck-types fastapi.UploadFile: a ``.file`` stream plus ``.filename``."""
    return SimpleNamespace(file=io.BytesIO(data), filename=filename)


def _upload(tmp_path, *args, **kwargs) -> dict:
    """validate_upload inside tmp_path (so leaks/cleanup are observable)."""
    kwargs.setdefault("work_dir", tmp_path)
    return validate_upload(*args, **kwargs).to_json_dict()


def _assert_no_leaks(payload: dict, tmp_path) -> None:
    text = json.dumps(payload)
    assert str(tmp_path) not in text
    assert str(tmp_path.resolve()) not in text
    assert "amtools-" not in text


# ---- accepted upload shapes ---------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: {"file": _starlette_like(b"jpeg", "IMG_0001.jpg")},
        lambda: {"file": UploadedFile(b"jpeg", "IMG_0001.jpg")},
        lambda: {"file": UploadedFile(io.BytesIO(b"jpeg"), "IMG_0001.jpg")},
        lambda: {"file": b"jpeg", "filename": "IMG_0001.jpg"},
        lambda: {"file": io.BytesIO(b"jpeg"), "filename": "IMG_0001.jpg"},
    ],
    ids=["starlette", "uploaded-bytes", "uploaded-stream", "raw-bytes", "raw-stream"],
)
def test_upload_shapes_keep_original_name(tmp_path, patched_extractors, make):  # noqa: F811
    payload = _upload(tmp_path, include="all", **make())
    assert payload["status"] != "error", payload
    assert [r["Filename"] for r in payload["metadata"]] == ["DEFAULT", "IMG_0001.jpg"]
    assert payload["summary"]["detected_types"] == ["oriented_image"]
    _assert_no_leaks(payload, tmp_path)
    assert list(tmp_path.iterdir()) == []  # temp dir cleaned up


def test_upload_stream_is_rewound(tmp_path, patched_extractors, monkeypatch):  # noqa: F811
    upload = _starlette_like(b"jpeg-bytes", "a.jpg")
    upload.file.read()  # e.g. a framework already peeked at it
    seen = {}

    def fake_exif(local_path, filename):
        with open(local_path, "rb") as fh:
            seen["bytes"] = fh.read()
        return {}

    monkeypatch.setattr(sidecar_mod, "extract_exif_metadata", fake_exif)
    _upload(tmp_path, upload)
    assert seen["bytes"] == b"jpeg-bytes"


# ---- filename handling --------------------------------------------------------


def test_path_traversal_name_is_reduced_to_basename(tmp_path, patched_extractors):  # noqa: F811
    payload = _upload(tmp_path, UploadedFile(b"x", "../../etc/evil.jpg"), include="metadata")
    assert payload["metadata"][1]["Filename"] == "evil.jpg"
    assert list(tmp_path.iterdir()) == []
    payload = _upload(tmp_path, UploadedFile(b"x", r"C:\Users\me\shot.jpg"), include="metadata")
    assert payload["metadata"][1]["Filename"] == "shot.jpg"


@pytest.mark.parametrize("name", [None, "", "  ", "..", "noextension"])
def test_bad_filenames_are_invalid_input(tmp_path, name):
    payload = _upload(tmp_path, SimpleNamespace(file=io.BytesIO(b"x"), filename=name))
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "invalid_input"
    assert list(tmp_path.iterdir()) == []


def test_raw_bytes_without_filename_is_invalid_input(tmp_path):
    payload = _upload(tmp_path, b"x")
    assert payload["error"]["type"] == "invalid_input"
    assert "filename" in payload["error"]["message"]


def test_max_bytes(tmp_path, patched_extractors):  # noqa: F811
    payload = _upload(tmp_path, UploadedFile(b"x" * 11, "a.jpg"), max_bytes=10)
    assert payload["error"]["type"] == "invalid_input"
    assert "10-byte limit" in payload["error"]["message"]
    assert list(tmp_path.iterdir()) == []
    assert _upload(tmp_path, UploadedFile(b"x" * 10, "a.jpg"), max_bytes=10)["status"] != "error"


def test_unsupported_upload_message_has_no_server_paths(tmp_path, patched_extractors):  # noqa: F811
    payload = _upload(tmp_path, UploadedFile(b"x", "notes.txt"))
    assert payload["error"]["type"] == "invalid_input"
    assert payload["error"]["message"].startswith("notes.txt is not a supported file type")
    _assert_no_leaks(payload, tmp_path)


# ---- companion uploads --------------------------------------------------------


@pytest.fixture
def exif_without_altitude(monkeypatch):
    """Client values only fill gaps (file metadata wins on conflict), so leave
    GPSAltitude out of the stubbed EXIF for the client upload to supply."""

    def fake_exif(local_path, filename):
        return {"CreateDate": "2024:06:15 10:30:00", "GPSLatitude": "51.0",
                "GPSLongitude": "-114.0", "Pitch": "-90", "Heading": "0", "Roll": "0"}

    monkeypatch.setattr(sidecar_mod, "extract_exif_metadata", fake_exif)


def test_client_sidecar_upload_is_merged(tmp_path, patched_extractors, exif_without_altitude):  # noqa: F811
    client = UploadedFile(b"Filename,GPSAltitude\nDEFAULT,\nIMG_0001.jpg,2222\n", "client.csv")
    payload = _upload(
        tmp_path, UploadedFile(b"x", "IMG_0001.jpg"), client_sidecar=client, include="all"
    )
    assert payload["status"] != "error", payload
    row = next(r for r in payload["metadata"] if r["Filename"] == "IMG_0001.jpg")
    assert row["GPSAltitude"] == "2222"
    _assert_no_leaks(payload, tmp_path)


def test_client_schema_upload_is_applied(tmp_path, patched_extractors, exif_without_altitude):  # noqa: F811
    # Headerless client CSV; the uploaded schema names its columns.
    client = UploadedFile(b"DEFAULT,\nIMG_0001.jpg,3333\n", "client.csv")
    schema = UploadedFile(
        json.dumps({"column_names": ["Filename", "GPSAltitude"]}).encode(), "schema.json"
    )
    payload = _upload(
        tmp_path,
        UploadedFile(b"x", "IMG_0001.jpg"),
        client_sidecar=client,
        client_schema=schema,
        include="metadata",
    )
    assert payload["status"] != "error", payload
    row = next(r for r in payload["metadata"] if r["Filename"] == "IMG_0001.jpg")
    assert row["GPSAltitude"] == "3333"


def test_coco_upload_matches_original_name(tmp_path, patched_extractors):  # noqa: F811
    coco = UploadedFile(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "IMG_0001.jpg", "width": 10, "height": 10}],
                "annotations": [{"image_id": 1}, {"image_id": 1}],
            }
        ).encode(),
        "labels.coco.json",
    )
    payload = _upload(tmp_path, UploadedFile(b"x", "IMG_0001.jpg"), coco=coco)
    assert payload["status"] != "error", payload
    coco_details = payload["details"]["coco"]
    assert coco_details["coco_path"] == "labels.coco.json"
    tiers = {t["tier"]: t for t in coco_details["tiers"]}
    # Matched by its original name: complete, not "not_on_disk" (as a uuid
    # temp name would have been).
    assert tiers["complete"]["images"] == 1
    assert tiers["complete"]["labels"] == 2
    assert tiers["not_on_disk"]["images"] == 0
    _assert_no_leaks(payload, tmp_path)


def test_companion_upload_needs_a_name(tmp_path, patched_extractors):  # noqa: F811
    payload = _upload(tmp_path, UploadedFile(b"x", "a.jpg"), coco=b"{}")
    assert payload["error"]["type"] == "invalid_input"
    assert list(tmp_path.iterdir()) == []


# ---- concurrency --------------------------------------------------------------


def test_parallel_uploads_are_isolated(tmp_path, patched_extractors):  # noqa: F811
    names = [f"IMG_{i:04d}.jpg" for i in range(8)]

    def run(name):
        return _upload(tmp_path, UploadedFile(b"x", name), include="metadata")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, names))
    for name, payload in zip(names, results, strict=True):
        assert [r["Filename"] for r in payload["metadata"]] == ["DEFAULT", name]
    assert list(tmp_path.iterdir()) == []


# ---- external tools -----------------------------------------------------------


def test_missing_exiftool_is_dependency_missing(tmp_path, patched_extractors, monkeypatch):  # noqa: F811
    def missing():
        raise ExiftoolNotFoundError("exiftool was not found.")

    monkeypatch.setattr(sidecar_mod, "find_exiftool_bin", missing)
    payload = _upload(tmp_path, UploadedFile(b"x", "a.jpg"))
    assert payload["error"] == {"type": "dependency_missing", "message": "exiftool was not found."}


def test_exiftool_missing_mid_run_is_not_swallowed(monkeypatch):
    def missing():
        raise ExiftoolNotFoundError("gone")

    monkeypatch.setattr(utils_mod, "find_exiftool_bin", missing)
    with pytest.raises(ExiftoolNotFoundError):
        extract_exif_metadata("/x/a.jpg", "a.jpg")


def test_exiftool_timeout_returns_empty(monkeypatch, caplog):
    monkeypatch.setattr(utils_mod, "find_exiftool_bin", lambda: "/fake/exiftool")
    monkeypatch.setenv("AM_TOOLS_EXIFTOOL_TIMEOUT", "1.5")
    seen = {}

    def hang(cmd, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(utils_mod.subprocess, "check_output", hang)
    with caplog.at_level("WARNING"):
        assert extract_exif_metadata("/x/a.jpg", "a.jpg") == {}
    assert seen["timeout"] == 1.5
    assert "timed out" in caplog.text


def test_exiftool_bin_override(monkeypatch, tmp_path):
    fake = tmp_path / "exiftool"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("EXIFTOOL_BIN", str(fake))
    assert utils_mod.find_exiftool_bin() == str(fake)
    monkeypatch.setenv("EXIFTOOL_BIN", str(tmp_path / "nope"))
    with pytest.raises(ExiftoolNotFoundError):
        utils_mod.find_exiftool_bin()


def _fake_run(pdal_result):
    def run(cmd, **kwargs):
        if cmd[1] == "-ver":
            return SimpleNamespace(returncode=0, stdout="13.0\n", stderr="")
        return pdal_result

    return run


def test_check_dependencies_all_found(monkeypatch):
    monkeypatch.setattr(api_mod, "find_exiftool_bin", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(api_mod, "find_pdal_bin", lambda: "/opt/conda/bin/pdal")
    banner = "----\npdal 2.8.1 (git-version: abc)\n----"
    ok = SimpleNamespace(returncode=0, stdout=banner, stderr="")
    monkeypatch.setattr(api_mod.subprocess, "run", _fake_run(ok))
    report = check_dependencies()
    assert isinstance(report, DependencyReport)
    assert report.ok is True
    assert report.exiftool.path == "/usr/bin/exiftool"
    assert report.exiftool.version == "13.0"
    assert report.pdal.version == "2.8.1"
    assert report.pye57.usable is True


def test_check_dependencies_flags_broken_binary(monkeypatch):
    """A binary that exists but can't run (e.g. missing dylib) is not usable."""
    monkeypatch.setattr(api_mod, "find_exiftool_bin", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(api_mod, "find_pdal_bin", lambda: "/opt/homebrew/bin/pdal")
    crash = SimpleNamespace(
        returncode=-6, stdout="", stderr="dyld: Library not loaded: libx265.dylib\n"
    )
    monkeypatch.setattr(api_mod.subprocess, "run", _fake_run(crash))
    report = check_dependencies()
    assert report.ok is False
    assert report.pdal.found is True and report.pdal.usable is False
    assert "Library not loaded" in report.pdal.detail
    assert report.exiftool.usable is True


def test_check_dependencies_reports_missing(monkeypatch):
    def no_exiftool():
        raise ExiftoolNotFoundError("exiftool was not found.")

    def no_pdal():
        raise PdalNotFoundError("pdal was not found.\nmore help")

    monkeypatch.setattr(api_mod, "find_exiftool_bin", no_exiftool)
    monkeypatch.setattr(api_mod, "find_pdal_bin", no_pdal)
    report = check_dependencies()
    assert report.ok is False
    assert report.exiftool.found is False and "not found" in report.exiftool.detail
    assert report.pdal.detail == "pdal was not found."


def test_probe_tool_unrunnable(monkeypatch):
    def boom(*a, **k):
        raise OSError("exec format error")

    monkeypatch.setattr(api_mod.subprocess, "run", boom)
    status = api_mod._probe_tool("/x/pdal", ["--version"])
    assert status.found and not status.usable and "exec format error" in status.detail


# ---- response model and text hygiene --------------------------------------------


def test_findings_are_plain_text():
    report = LintReport()
    report.add_info("\x1b[31m███\x1b[0m histogram")
    report.add_warning("ok", fix_hint="\x1b[1mbold\x1b[0m hint")
    result = api_mod.ValidationResult(
        df=__import__("pandas").DataFrame([{"Filename": "DEFAULT"}]),
        report=report,
        detected_types=set(),
        skipped_vector=[],
        crs_failures=[],
    )
    payload = api_mod.build_payload(result, {"findings"}).to_json_dict()
    assert "\x1b" not in json.dumps(payload)
    assert payload["findings"][0]["message"] == "███ histogram"
    assert payload["findings"][1]["fix_hint"] == "bold hint"


def test_unrequested_sections_are_unset(tmp_path, patched_extractors):  # noqa: F811
    response = validate_upload(UploadedFile(b"x", "a.jpg"), work_dir=tmp_path)
    assert isinstance(response, ValidationResponse)
    assert response.model_fields_set >= {"schema_version", "status", "summary", "error"}
    assert "metadata" not in response.to_json_dict()
    assert set(response.to_json_dict()) == {
        "schema_version", "status", "summary", "findings", "missing_data", "details", "error",
    }


def test_response_json_schema_builds():
    schema = ValidationResponse.model_json_schema()
    assert schema["properties"]["status"]["enum"] == [
        "passed", "passed_with_warnings", "failed", "error",
    ]


def test_lazy_exports():
    for name in atomic_tools.__all__:
        assert getattr(atomic_tools, name) is not None
    with pytest.raises(AttributeError):
        atomic_tools.nope  # noqa: B018


def test_creation_year_bound_tracks_the_clock():
    next_year = datetime.now().year + 1
    assert parse_creation_year(str(next_year)) == (True, None)
    ok, msg = parse_creation_year(str(next_year + 1))
    assert not ok and str(next_year) in msg


# ---- FastAPI end to end ---------------------------------------------------------


def _make_app():
    """The endpoint shape the README recommends."""
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi import FastAPI, Form, UploadFile

    app = FastAPI()

    @app.post(
        "/validate", response_model=ValidationResponse, response_model_exclude_unset=True
    )
    def validate(
        file: UploadFile,
        coco: UploadFile | None = None,
        include: str | None = Form(None),
    ):
        return validate_upload(file, coco=coco, include=include)

    @app.get("/health", response_model=DependencyReport)
    def health():
        return check_dependencies()

    return app


def test_fastapi_upload_round_trip(patched_extractors):  # noqa: F811
    app = _make_app()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/validate",
        files={"file": ("IMG_0001.jpg", b"jpeg-bytes", "image/jpeg")},
        data={"include": "findings,metadata"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in {"passed", "passed_with_warnings", "failed"}
    assert set(body) == {"schema_version", "status", "summary", "findings", "metadata", "error"}
    assert body["metadata"][1]["Filename"] == "IMG_0001.jpg"

    resp = client.post("/validate", files={"file": ("readme", b"x", "text/plain")})
    assert resp.status_code == 200
    assert resp.json()["error"]["type"] == "invalid_input"


def test_fastapi_health(monkeypatch):
    app = _make_app()
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_mod, "find_exiftool_bin", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(api_mod, "find_pdal_bin", lambda: "/usr/bin/pdal")
    monkeypatch.setattr(
        api_mod.subprocess,
        "run",
        _fake_run(SimpleNamespace(returncode=0, stdout="pdal 1.0", stderr="")),
    )
    body = TestClient(app).get("/health").json()
    assert body["ok"] is True and body["exiftool"]["path"] == "/usr/bin/exiftool"
