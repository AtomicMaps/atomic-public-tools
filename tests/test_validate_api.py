"""Tests for `am-tools validate --api-mode` and `atomic_tools.validate_path`."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

import atomic_tools.api as api_mod
import atomic_tools.cli as cli_mod
import atomic_tools.commands.validate as validate_mod
from atomic_tools import validate_path as _validate_path
from atomic_tools.cli import app
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
    assert payload["schema_version"] == 1
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
    assert payload["missing_data"]["rows"] == [{"Filename": "b.jpg", "GPSLatitude": "MISSING"}]


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
    stubbed["build_exc"] = RuntimeError("No supported files found in /x")
    result = _invoke("--path", str(tmp_path), "--api-mode")
    assert result.exit_code == 1
    payload = _json(result)
    assert payload["status"] == "error"
    assert payload["error"] == {"type": "no_data", "message": "No supported files found in /x"}


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
    called = []
    monkeypatch.setattr(cli_mod, "check_for_update", lambda: called.append(1))
    monkeypatch.setattr(
        cli_mod.sys, "argv", ["amtools", "validate", "--path", str(tmp_path), "--api-mode"]
    )
    assert _invoke("--path", str(tmp_path), "--api-mode").exit_code == 0
    assert called == []

    monkeypatch.setattr(cli_mod.sys, "argv", ["amtools", "validate", "--path", str(tmp_path)])
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
    stubbed["build_exc"] = ValueError("bad CRS")
    payload = validate_path(str(tmp_path))
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "invalid_input"

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
