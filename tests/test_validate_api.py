"""Tests for `am-tools validate --api-mode` and `atomic_tools.validate_path`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import atomic_tools.api as api_mod
import atomic_tools.cli as cli_mod
import atomic_tools.commands.validate as validate_mod
from atomic_tools import validate_path as _validate_path
from atomic_tools.cli import app
from atomic_tools.client_sidecar import SidecarMergeError
from atomic_tools.commands.sidecar import NoSupportedFilesError, PointCloudCRSError
from atomic_tools.errors import InvalidInputError
from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.report import LintReport, MissingDataReport
from atomic_tools.validators.sidecar import lint_sidecar_df, lint_sidecar_file
from tests.test_auto_detect import _make_mixed_dir, patched_extractors  # noqa: F401

runner = CliRunner()


def validate_path(*args, **kwargs) -> dict:
    return _validate_path(*args, **kwargs).to_json_dict()


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Filename": "DEFAULT", "DataType": "", "GPSLatitude": ""},
            {"Filename": "a.jpg", "DataType": "oriented_image", "GPSLatitude": "51.0"},
            {"Filename": "b.jpg", "DataType": "oriented_image", "GPSLatitude": ""},
        ]
    )


@pytest.fixture
def stubbed(monkeypatch):
    """Stub extraction + lint. ``state`` lets a test shape the lint report."""
    state: dict = {"report": LintReport(), "build_exc": None}

    def fake_build_sidecar(**kwargs):
        state["build_kwargs"] = kwargs
        if state["build_exc"] is not None:
            raise state["build_exc"]
        kwargs["unclassified_out"].append("notes.txt")
        return _df(), object(), {DataTypeEnum.oriented_image}, ["layer.geojson"]

    def fake_lint(df, **kwargs):
        state["lint_kwargs"] = kwargs
        return state["report"]

    monkeypatch.setattr(api_mod, "_build_sidecar", fake_build_sidecar)
    monkeypatch.setattr(api_mod, "lint_sidecar_df", fake_lint)
    return state


def _invoke(*args: str):
    return runner.invoke(app, ["validate", *args])


def _json(result) -> dict:
    # stdout must hold exactly one JSON document and nothing else.
    return json.loads(result.stdout)


# ---- CLI ---------------------------------------------------------------------


def test_api_mode_passed_outputs_only_json(stubbed, tmp_path):
    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["schema_version"] == 2
    assert payload["status"] == "passed"
    assert payload["error"] is None
    assert payload["summary"]["files_scanned"] == 2
    assert payload["summary"]["detected_types"] == ["oriented_image"]
    # Default sections: findings, missing_data, details — metadata is opt-in.
    assert set(payload) >= {"findings", "missing_data", "details"}
    assert "metadata" not in payload
    assert payload["details"]["skipped_vector"] == ["layer.geojson"]
    assert payload["details"]["unclassified"] == ["notes.txt"]
    # Machine callers get the quiet build (no tqdm / red banners).
    assert stubbed["build_kwargs"]["quiet"] is True


def test_api_mode_failed_lint_still_exits_zero(stubbed, tmp_path):
    report = LintReport()
    report.add_error("boom", fix_hint="fix it", location="row 2")
    report.add_warning("meh")
    report.missing_data = MissingDataReport(
        filename_column="Filename",
        field_columns=["GPSLatitude"],
        rows=[{"Filename": "b.jpg", "GPSLatitude": "MISSING"}],
        row_types={"b.jpg": "oriented_image"},
    )
    stubbed["report"] = report

    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["status"] == "failed"
    assert payload["summary"]["errors"] == 1
    assert payload["summary"]["warnings"] == 1
    assert payload["findings"][0] == {
        "level": "error",
        "message": "boom",
        "fix_hint": "fix it",
        "location": "row 2",
    }
    assert payload["missing_data"] == {
        "oriented_image": [{"file": "b.jpg", "missing": ["GPSLatitude"]}]
    }


def test_missing_data_lists_only_missing_fields_by_type():
    table = MissingDataReport(
        filename_column="Filename",
        field_columns=["CreateDate", "GPSLatitude", "Pitch"],
        rows=[
            {"Filename": "v.avi", "CreateDate": "MISSING", "GPSLatitude": "", "Pitch": ""},
            {"Filename": "p.jpg", "CreateDate": "", "GPSLatitude": "MISSING", "Pitch": "MISSING"},
            # Added from COCO alone: no sidecar row, so no type.
            {"Filename": "gone.jpg", "coco_status": "not_on_disk", "coco_labels": "3"},
        ],
        include_coco=True,
        row_types={"v.avi": "full_motion_video", "p.jpg": "oriented_image"},
    )
    table.rows[0].update(coco_status="ok", coco_labels="0")
    table.rows[1].update(coco_status="unusable", coco_labels="2")
    assert table.to_dict({"p.jpg": "imgs/p.jpg"}) == {
        "full_motion_video": [
            {"file": "v.avi", "missing": ["CreateDate"], "coco_status": "ok", "coco_labels": 0}
        ],
        "oriented_image": [
            {
                "file": "imgs/p.jpg",
                "missing": ["GPSLatitude", "Pitch"],
                "coco_status": "unusable",
                "coco_labels": 2,
            }
        ],
        "unknown": [
            {"file": "gone.jpg", "missing": [], "coco_status": "not_on_disk", "coco_labels": 3}
        ],
    }


def test_api_mode_omits_cli_only_findings(stubbed, tmp_path):
    stubbed["report"].add_info("Batch center (median): ...\n  [0 - 1] mi  1", cli_only=True)
    stubbed["report"].add_info("kept")

    payload = _json(_invoke("--path", str(tmp_path), "--api-mode"))
    assert [f["message"] for f in payload["findings"]] == ["kept"]
    assert payload["summary"]["infos"] == 1


def test_files_section_sorts_files_by_outcome(stubbed, tmp_path):
    stubbed["report"].add_error("missing lat", files=["b.jpg"])
    stubbed["report"].add_warning("far away", files=["a.jpg", "b.jpg"])
    stubbed["report"].add_error("batch-level, names no file")

    files = validate_path(str(tmp_path))["files"]
    assert files["passed"] == []
    assert files["passed_with_warnings"] == ["a.jpg"]
    assert files["failed"] == ["b.jpg"]  # an error outranks a warning
    assert "skipped" not in files


def test_api_mode_warnings_only_status(stubbed, tmp_path):
    stubbed["report"].add_warning("meh")
    payload = _json(_invoke("--path", str(tmp_path), "--api-mode"))
    assert payload["status"] == "passed_with_warnings"


def test_api_mode_crs_failure_is_failed(stubbed, monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "_crs_failures", lambda df: ["a.las"])
    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == 0
    payload = _json(result)
    assert payload["status"] == "failed"
    assert payload["details"]["crs_failures"] == ["a.las"]


def test_api_mode_include_filters_sections(stubbed, tmp_path):
    payload = _json(
        _invoke("--path", str(tmp_path), "--api-mode", "--include", "metadata")
    )
    assert "metadata" in payload
    assert not {"findings", "missing_data", "details"} & set(payload)
    assert payload["metadata"][1] == {
        "Filename": "a.jpg",
        "DataType": "oriented_image",
        "GPSLatitude": "51.0",
    }


def test_api_mode_include_all(stubbed, tmp_path):
    payload = _json(_invoke("--path", str(tmp_path), "--api-mode", "--include", "all"))
    assert {"findings", "missing_data", "metadata", "details"} <= set(payload)


def test_api_mode_unknown_include_is_json_error(stubbed, tmp_path):
    result = _invoke("--path", str(tmp_path), "--api-mode", "--include", "bogus")
    assert result.exit_code == 2
    payload = _json(result)
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "invalid_input"
    assert "bogus" in payload["error"]["message"]


def test_api_mode_missing_directory_never_prompts(stubbed, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("wizard must not run in --api-mode")

    monkeypatch.setattr(validate_mod, "_run_interactive_wizard", boom)
    result = _invoke("--api-mode")
    assert result.exit_code == 2
    payload = _json(result)
    assert payload["status"] == "error"
    assert "--path" in payload["error"]["message"]


def test_api_mode_build_failure_is_json_error(stubbed, tmp_path):
    stubbed["build_exc"] = NoSupportedFilesError("No supported files found in /x")
    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == 1
    payload = _json(result)
    assert payload["status"] == "error"
    assert payload["error"] == {"type": "no_data", "message": "No supported files found in /x"}


@pytest.mark.parametrize(
    "exc,error_type,exit_code",
    [
        (NoSupportedFilesError("none here"), "no_data", 1),
        (SidecarMergeError("client CSVs differ in column count"), "invalid_input", 2),
        (PointCloudCRSError(["a.las"], Path("/x.csv")), "invalid_input", 2),
        (FileNotFoundError("client.csv"), "invalid_input", 2),
        # Storage/network failures are wrapped as plain RuntimeError: not "no data".
        (RuntimeError("Failed to list keys with prefix p: GenericError"), "internal_error", 1),
        (KeyError("boom"), "internal_error", 1),
    ],
)
def test_error_classification(stubbed, tmp_path, exc, error_type, exit_code):
    stubbed["build_exc"] = exc
    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == exit_code
    assert _json(result)["error"]["type"] == error_type


def test_response_building_failure_still_returns_json(stubbed, monkeypatch, tmp_path):
    def broken(result, include):
        raise TypeError("unserialisable cell")

    monkeypatch.setattr(api_mod, "build_payload", broken)
    payload = validate_path(str(tmp_path))
    assert payload["error"]["type"] == "internal_error"
    assert "unserialisable cell" in payload["error"]["message"]

    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == 1
    assert _json(result)["error"]["type"] == "internal_error"


def test_api_mode_stray_prints_go_to_stderr(stubbed, monkeypatch, tmp_path):
    def noisy_lint(df, **kwargs):
        print("chatty third-party library")
        return LintReport()

    monkeypatch.setattr(api_mod, "lint_sidecar_df", noisy_lint)
    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == 0
    assert _json(result)["status"] == "passed"
    assert "chatty" in result.stderr


def test_include_without_api_mode_is_rejected(stubbed, tmp_path):
    result = _invoke("--path", str(tmp_path), "--include", "all")
    assert result.exit_code == 2


def test_api_mode_skips_version_check(stubbed, monkeypatch, tmp_path):
    """sys.argv here is pytest's, so this checks the args click parsed."""
    called = []
    monkeypatch.setattr(cli_mod, "check_for_update", lambda: called.append(1))
    assert _invoke("--path", str(tmp_path), "--api-mode").exit_code == 0
    assert called == []

    _invoke("--path", str(tmp_path))
    assert called == [1]


# ---- Python API --------------------------------------------------------------


def test_validate_path_matches_cli(stubbed, tmp_path):
    from_cli = _json(_invoke("--path", str(tmp_path), "--api-mode", "--include", "all"))
    from_py = validate_path(str(tmp_path), include="all")
    assert from_py == from_cli


def test_validate_path_accepts_string_datatype(stubbed, tmp_path):
    payload = validate_path(str(tmp_path), data_type="oriented_image")
    assert payload["status"] == "passed"
    assert stubbed["build_kwargs"]["data_type"] == DataTypeEnum.oriented_image


def test_validate_path_never_raises(stubbed, tmp_path):
    stubbed["build_exc"] = InvalidInputError("bad CRS")
    payload = validate_path(str(tmp_path))
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "invalid_input"

    # A stray ValueError (pandas, pyproj, …) is a bug, not the caller's fault.
    stubbed["build_exc"] = ValueError("cannot reindex on an axis with duplicate labels")
    assert validate_path(str(tmp_path))["error"]["type"] == "internal_error"

    assert validate_path("")["error"]["type"] == "invalid_input"
    assert validate_path(str(tmp_path), data_type="nope")["status"] == "error"


def test_validate_path_end_to_end(tmp_path, patched_extractors):  # noqa: F811
    _make_mixed_dir(tmp_path)
    payload = validate_path(str(tmp_path), include="all")
    assert payload["status"] in {"passed", "passed_with_warnings", "failed"}
    json.dumps(payload)  # fully serialisable
    names = {row["Filename"] for row in payload["metadata"]}
    assert {"DEFAULT", "a.jpg", "b.las", "c.mp4"} <= names
    assert any(k.endswith("d.geojson") for k in payload["details"]["skipped_vector"])


def test_validate_path_files_end_to_end(tmp_path, patched_extractors):  # noqa: F811
    _make_mixed_dir(tmp_path)
    files = validate_path(str(tmp_path), limit=2)["files"]

    checked = files["passed"] + files["passed_with_warnings"] + files["failed"]
    # Keys as in ``details`` (relative to the scanned folder), only the files
    # the limit kept.
    assert sorted(checked) == ["a.jpg", "b.las"]


def test_missing_required_value_names_the_row():
    report = lint_sidecar_df(_df(), final=True, data_type=None, input_files_path=None)
    missing = [f for f in report.errors() if "'GPSLatitude'" in f.message]
    assert [f.files for f in missing] == [("b.jpg",)]


def test_bad_default_value_names_only_the_rows_that_inherit_it():
    df = pd.DataFrame(
        [
            {"Filename": "DEFAULT", "DataType": "", "GPSAltitude": "not a number"},
            {"Filename": "own.jpg", "DataType": "oriented_image", "GPSAltitude": "100"},
            {"Filename": "inherits.jpg", "DataType": "oriented_image", "GPSAltitude": ""},
            # Point clouds have no GPSAltitude field, so they don't inherit it.
            {"Filename": "cloud.las", "DataType": "point_cloud", "GPSAltitude": ""},
        ]
    )
    report = lint_sidecar_df(df, final=False, data_type=None, input_files_path=None)
    bad = [f for f in report.errors() if f.location == "DEFAULT, col 'GPSAltitude'"]
    assert [f.files for f in bad] == [("inherits.jpg",)]


def test_files_section_skips_untyped_rows_and_lists_rowless_files(stubbed, tmp_path):
    report = stubbed["report"]
    report.missing_data = MissingDataReport(  # b.jpg was untyped, so never checked
        filename_column="Filename", field_columns=[], row_types={"a.jpg": "oriented_image"}
    )
    report.add_error("no sidecar row", files=["extra/c.jpg"])

    files = validate_path(str(tmp_path))["files"]
    assert files == {
        "passed": ["a.jpg"],
        "passed_with_warnings": [],
        "failed": ["extra/c.jpg"],
    }


def test_inventory_findings_name_their_files(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    for rel in ("x/a.jpg", "y/a.jpg", "b.jpg"):
        (tmp_path / rel).write_text("x")
    df = pd.DataFrame(
        [
            {"Filename": "DEFAULT", "DataType": "", "GPSLatitude": ""},
            {"Filename": "a.jpg", "DataType": "oriented_image", "GPSLatitude": "51"},
        ]
    )
    report = lint_sidecar_df(df, final=True, data_type=None, input_files_path=str(tmp_path))
    ambiguous = [f for f in report.warnings() if "matches 2 input files" in f.message]
    assert [f.files for f in ambiguous] == [("a.jpg",)]
    rowless = [f for f in report.errors() if "have no sidecar row" in f.message]
    assert [f.files for f in rowless] == [("b.jpg",)]


# ---- in-memory lint parity ---------------------------------------------------


@pytest.mark.parametrize("data_type", [None, DataTypeEnum.oriented_image])
def test_lint_df_matches_lint_file(tmp_path, patched_extractors, data_type):  # noqa: F811
    """Linting the DataFrame directly gives the same report as the old
    write-CSV-then-lint path."""
    from atomic_tools.commands.sidecar import _build_sidecar

    scan = tmp_path / "scan"
    scan.mkdir()
    _make_mixed_dir(scan)
    df, _b, _d, _v = _build_sidecar(
        directory=str(scan),
        data_type=data_type,
        client_sidecar=None,
        client_schema=None,
    )
    # Mix in non-string cells to exercise normalisation.
    df["num"] = [1.5] + [None] * (len(df) - 1)

    csv_path = tmp_path / "sidecar.csv"
    df.to_csv(csv_path, index=False)
    kwargs = dict(
        final=True,
        data_type=data_type,
        input_files_path=str(scan),
        ignore_missing_orientation=False,
        coco_path=None,
        coco_not_on_disk_is_error=False,
    )
    from_file = lint_sidecar_file(str(csv_path), schema_path=None, **kwargs)
    from_df = lint_sidecar_df(df, **kwargs)

    assert from_df.findings == from_file.findings
    assert from_df.missing_data == from_file.missing_data
