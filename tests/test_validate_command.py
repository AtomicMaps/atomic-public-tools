"""Tests for `am-tools validate` — lint without persisting a sidecar."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

import atomic_tools.api as api_mod
from atomic_tools.cli import app
from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.report import LintReport

runner = CliRunner()


@pytest.fixture
def captured_lint(monkeypatch):
    """Stub out extraction + lint so the command can run without exiftool/pdal.

    Records the kwargs passed to ``lint_sidecar_df`` and the DataFrame it
    received, and returns a benign passing report.
    """
    calls: dict = {}

    def fake_build_sidecar(**kwargs):
        calls["build_kwargs"] = kwargs
        df = pd.DataFrame(
            [
                {"Filename": "DEFAULT", "GPSLatitude": "", "GPSLongitude": ""},
                {"Filename": "a.jpg", "GPSLatitude": "51.0", "GPSLongitude": "-114.0"},
            ]
        )
        return df, object(), {DataTypeEnum.oriented_image}, []

    def fake_lint(df, **kwargs):
        calls["lint_df"] = df
        calls["lint_kwargs"] = kwargs
        return LintReport()

    monkeypatch.setattr(api_mod, "_build_sidecar", fake_build_sidecar)
    monkeypatch.setattr(api_mod, "lint_sidecar_df", fake_lint)
    return calls


def test_validate_runs_lint_in_final_mode(captured_lint, tmp_path):
    result = runner.invoke(
        app,
        [
            "validate",
            "--path",
            str(tmp_path),
            "--datatype",
            "oriented_image",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_lint["lint_kwargs"]["final"] is True
    assert captured_lint["lint_kwargs"]["data_type"] == DataTypeEnum.oriented_image
    assert captured_lint["lint_kwargs"]["input_files_path"] == str(tmp_path)
    # The assembled sidecar is linted in memory — no temp CSV round trip.
    assert list(captured_lint["lint_df"]["Filename"]) == ["DEFAULT", "a.jpg"]


def test_validate_without_datatype_runs_auto(captured_lint, tmp_path):
    """No --datatype: the command runs non-interactively (auto-detect) and lint
    receives data_type=None."""
    result = runner.invoke(app, ["validate", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert captured_lint["lint_kwargs"]["data_type"] is None
    assert captured_lint["build_kwargs"]["data_type"] is None


def test_validate_passes_coco_to_lint(captured_lint, tmp_path):
    result = runner.invoke(
        app,
        [
            "validate",
            "--path",
            str(tmp_path),
            "--datatype",
            "oriented_image",
            "--coco",
            str(tmp_path / "labels.coco.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_lint["lint_kwargs"]["coco_path"] == str(tmp_path / "labels.coco.json")
    # validate is informational: COCO not-on-disk is a warning here, not an error.
    assert captured_lint["lint_kwargs"]["coco_not_on_disk_is_error"] is False


def test_validate_passes_through_optional_args(captured_lint, tmp_path):
    result = runner.invoke(
        app,
        [
            "validate",
            "--path",
            str(tmp_path),
            "--datatype",
            "oriented_image",
            "--ignore-missing-orientation",
            "--full",
            "--spatial-reference",
            "EPSG:32612",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_lint["lint_kwargs"]["ignore_missing_orientation"] is True
    assert captured_lint["build_kwargs"]["full"] is True
    assert captured_lint["build_kwargs"]["spatial_reference"] == "EPSG:32612"


def test_validate_writes_no_sidecar(captured_lint, tmp_path):
    """The sidecar is linted in memory; the input directory is never written to."""
    result = runner.invoke(
        app,
        ["validate", "--path", str(tmp_path), "--datatype", "point_cloud"],
    )
    assert result.exit_code == 0, result.output
    # Nothing left behind in the scanned directory.
    assert list(tmp_path.iterdir()) == []


def test_validate_exits_nonzero_on_errors(monkeypatch, tmp_path):
    def fake_build_sidecar(**kwargs):
        return pd.DataFrame([{"Filename": "DEFAULT"}]), object(), set(), []

    def fake_lint(df, **kwargs):
        report = LintReport()
        report.add_error("boom")
        return report

    monkeypatch.setattr(api_mod, "_build_sidecar", fake_build_sidecar)
    monkeypatch.setattr(api_mod, "lint_sidecar_df", fake_lint)

    result = runner.invoke(
        app,
        ["validate", "--path", str(tmp_path), "--datatype", "point_cloud"],
    )
    assert result.exit_code == 1
