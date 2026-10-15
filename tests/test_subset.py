"""--limit / --sample: partial runs of validate and sidecar."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from atomic_tools import validate_path
from atomic_tools.cli import app
from atomic_tools.commands.sidecar import _build_sidecar, _select_keys
from atomic_tools.errors import InvalidInputError
from atomic_tools.utils.utils import DataTypeEnum
from tests.test_auto_detect import patched_extractors  # noqa: F401

runner = CliRunner()

OI = DataTypeEnum.oriented_image
PC = DataTypeEnum.point_cloud


def _types() -> dict[str, DataTypeEnum]:
    # Deliberately unsorted, like a filesystem listing.
    return {
        "z/9.jpg": OI,
        "a/1.jpg": OI,
        "m/5.las": PC,
        "b/2.jpg": OI,
        "c/3.las": PC,
        "a/0.jpg": OI,
    }


def _rows(df: pd.DataFrame) -> list[str]:
    return sorted(f for f in df["Filename"] if f != "DEFAULT")


def _make_dir(tmp_path):
    for name in ["c.jpg", "a.jpg", "b.jpg", "e.las", "d.las", "f.mp4"]:
        (tmp_path / name).write_text("x")
    return tmp_path


def _build(tmp_path, **kwargs):
    selection: list[tuple[int, int]] = []
    df, *_ = _build_sidecar(
        directory=str(tmp_path),
        data_type=None,
        client_sidecar=None,
        client_schema=None,
        quiet=True,
        selection_out=selection,
        **kwargs,
    )
    return df, selection


# ---- _select_keys ---------------------------------------------------------------


def test_limit_takes_first_n_by_path():
    assert list(_select_keys(_types(), limit=3)) == ["a/0.jpg", "a/1.jpg", "b/2.jpg"]


def test_limit_larger_than_folder_keeps_everything():
    assert set(_select_keys(_types(), limit=100)) == set(_types())


def test_sample_takes_n_per_type():
    picked = _select_keys(_types(), sample=1, seed=0)
    assert sorted(picked.values(), key=lambda t: t.value) == [OI, PC]


def test_sample_keeps_small_types_whole():
    picked = _select_keys(_types(), sample=3, seed=0)
    assert sum(t == PC for t in picked.values()) == 2
    assert sum(t == OI for t in picked.values()) == 3


def test_sample_is_repeatable_with_a_seed_regardless_of_listing_order():
    forward = _types()
    backward = dict(reversed(list(forward.items())))
    assert _select_keys(forward, sample=2, seed=7) == _select_keys(backward, sample=2, seed=7)


# ---- _build_sidecar ---------------------------------------------------------------


def test_build_with_limit(tmp_path, patched_extractors):  # noqa: F811
    df, selection = _build(_make_dir(tmp_path), limit=2)
    assert _rows(df) == ["a.jpg", "b.jpg"]
    assert selection == [(2, 6)]


def test_build_with_sample(tmp_path, patched_extractors):  # noqa: F811
    df, selection = _build(_make_dir(tmp_path), sample=1, seed=3)
    types = df.loc[df["Filename"] != "DEFAULT", "DataType"]
    assert sorted(types) == ["full_motion_video", "oriented_image", "point_cloud"]
    assert selection == [(3, 6)]


def test_build_without_selection_reports_none(tmp_path, patched_extractors):  # noqa: F811
    df, selection = _build(_make_dir(tmp_path))
    assert len(_rows(df)) == 6
    assert selection == []


def test_limit_that_cuts_nothing_reports_none(tmp_path, patched_extractors):  # noqa: F811
    df, selection = _build(_make_dir(tmp_path), limit=100)
    assert len(_rows(df)) == 6
    assert selection == []


def test_build_rejects_limit_and_sample(tmp_path, patched_extractors):  # noqa: F811
    with pytest.raises(InvalidInputError):
        _build(_make_dir(tmp_path), limit=1, sample=1)


def test_single_file_ignores_limit(tmp_path, patched_extractors):  # noqa: F811
    _make_dir(tmp_path)
    selection: list[tuple[int, int]] = []
    df, *_ = _build_sidecar(
        directory=str(tmp_path / "a.jpg"),
        data_type=None,
        client_sidecar=None,
        client_schema=None,
        limit=1,
        selection_out=selection,
    )
    assert _rows(df) == ["a.jpg"]
    assert selection == []


def test_dropped_video_takes_its_telemetry_along(tmp_path, patched_extractors, caplog):  # noqa: F811
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "z_flight.mp4").write_text("x")
    (tmp_path / "z_flight.srt").write_text("x")
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        df, _ = _build(tmp_path, limit=1)
    assert _rows(df) == ["a.jpg"]
    assert "match no video" not in caplog.text


# ---- validate / validate_path --------------------------------------------------------


def test_validate_report_says_it_was_partial(tmp_path, patched_extractors):  # noqa: F811
    result = runner.invoke(app, ["validate", "--path", str(_make_dir(tmp_path)), "--limit", "1"])
    assert "Checked 1 of 6 files (--limit 1)" in result.stdout


def test_validate_rejects_limit_with_sample(tmp_path):
    result = runner.invoke(
        app, ["validate", "--path", str(tmp_path), "--limit", "1", "--sample", "1"]
    )
    assert result.exit_code == 2


def test_validate_rejects_zero_limit(tmp_path):
    result = runner.invoke(app, ["validate", "--path", str(tmp_path), "--limit", "0"])
    assert result.exit_code == 2


def test_validate_rejects_seed_without_sample(tmp_path):
    result = runner.invoke(app, ["validate", "--path", str(tmp_path), "--seed", "1"])
    assert result.exit_code == 2


def test_validate_path_summary(tmp_path, patched_extractors):  # noqa: F811
    payload = validate_path(str(_make_dir(tmp_path)), sample=1, seed=1).to_json_dict()
    summary = payload["summary"]
    assert summary["files_total"] == 6
    assert summary["files_scanned"] == 3
    assert any("Checked 3 of 6" in f["message"] for f in payload["findings"])


def test_validate_path_summary_omits_selection_without_flags(tmp_path, patched_extractors):  # noqa: F811
    summary = validate_path(str(_make_dir(tmp_path))).to_json_dict()["summary"]
    assert "files_total" not in summary


@pytest.mark.parametrize(
    "kwargs", [{"limit": 1, "sample": 1}, {"seed": 1}, {"limit": 0}]
)
def test_validate_path_rejects_bad_selection(tmp_path, kwargs):
    payload = validate_path(str(tmp_path), **kwargs).to_json_dict()
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "invalid_input"


def test_api_mode_reports_both_as_json(tmp_path):
    result = runner.invoke(
        app,
        ["validate", "--api-mode", "--path", str(tmp_path), "--limit", "1", "--sample", "1"],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["type"] == "invalid_input"


# ---- sidecar ---------------------------------------------------------------------------


def test_sidecar_with_limit_writes_partial_csv(tmp_path, patched_extractors):  # noqa: F811
    _make_dir(tmp_path)
    result = runner.invoke(app, ["sidecar", "--path", str(tmp_path), "--limit", "1"])
    assert "covers only 1 of 6 files" in result.stderr
    df = pd.read_csv(tmp_path / "sidecar.csv", dtype=str)
    assert _rows(df) == ["a.jpg"]


def test_sidecar_rejects_limit_with_update_existing(tmp_path):
    result = runner.invoke(
        app, ["sidecar", "--path", str(tmp_path), "--limit", "1", "--update-existing"]
    )
    assert result.exit_code == 2
