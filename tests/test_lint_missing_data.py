"""Tests for the missing-data report (rows missing required fields → CSV)."""

from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from atomic_tools.cli import app
from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.report import MISSING_MARKER
from atomic_tools.validators.sidecar import lint_sidecar_file

runner = CliRunner()


def _write_csv(tmp_path: Path, name: str, header: list[str], rows: list[list[str]]) -> Path:
    p = tmp_path / name
    lines = [",".join(header)] + [",".join(r) for r in rows]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _header() -> list[str]:
    return ["Filename", "CreateDate", "GPSAltitude", "GPSLatitude", "GPSLongitude"]


def _lint_final(p: Path):
    return lint_sidecar_file(
        str(p),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )


# ---- builder ------------------------------------------------------------


def test_missing_data_lists_only_rows_with_gaps(tmp_path):
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["complete.jpg", "2024:06:15 10:30:00", "1000", "40.0", "-111.0"],
        ["nogps.jpg", "2024:06:15 10:30:00", "1000", "", ""],
        ["nodate.jpg", "", "1000", "40.0", "-111.0"],
    ]
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    table = _lint_final(p).missing_data

    assert table is not None
    assert table.filename_column == "Filename"
    # Complete + DEFAULT rows are excluded; only the two gappy rows remain.
    names = {r["Filename"] for r in table.rows}
    assert names == {"nogps.jpg", "nodate.jpg"}


def test_missing_data_marks_the_right_fields(tmp_path):
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["nogps.jpg", "2024:06:15 10:30:00", "1000", "", ""],
    ]
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    table = _lint_final(p).missing_data
    row = next(r for r in table.rows if r["Filename"] == "nogps.jpg")

    assert row["GPSLatitude"] == MISSING_MARKER
    assert row["GPSLongitude"] == MISSING_MARKER
    assert row["GPSAltitude"] == ""  # present
    assert row["CreateDate"] == ""  # present


def test_default_row_coverage_means_not_missing(tmp_path):
    """A value supplied on the DEFAULT row covers every data row."""
    rows = [
        ["DEFAULT", "2024:06:15 10:30:00", "1000", "40.0", "-111.0"],
        ["a.jpg", "", "", "", ""],
    ]
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    table = _lint_final(p).missing_data
    assert table.is_empty(), table.rows


def test_no_datatype_means_no_table(tmp_path):
    rows = [["Filename", "CreateDate"], ["1.jpg", ""]]
    p = _write_csv(tmp_path, "s.csv", rows[0], rows[1:])
    report = lint_sidecar_file(
        str(p),
        final=False,
        data_type=None,
        schema_path=None,
        input_files_path=None,
    )
    assert report.missing_data is None


# ---- CSV output ---------------------------------------------------------


def test_write_csv_roundtrip(tmp_path):
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["nogps.jpg", "2024:06:15 10:30:00", "1000", "", ""],
    ]
    p = _write_csv(tmp_path, "s.csv", _header(), rows)
    table = _lint_final(p).missing_data

    out = tmp_path / "report.csv"
    table.write_csv(str(out))

    with out.open(newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    assert parsed[0]["Filename"] == "nogps.jpg"
    assert parsed[0]["GPSLatitude"] == MISSING_MARKER
    assert parsed[0]["GPSAltitude"] == ""


# ---- CLI --report (non-interactive) ------------------------------------


def test_cli_report_flag_writes_csv(tmp_path):
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["nogps.jpg", "2024:06:15 10:30:00", "1000", "", ""],
    ]
    sidecar = _write_csv(tmp_path, "s.csv", _header(), rows)
    out = tmp_path / "missing.csv"
    result = runner.invoke(
        app,
        [
            "--silent",
            "lint",
            "sidecar",
            str(sidecar),
            "--final",
            "--datatype",
            "oriented_image",
            "--report",
            str(out),
        ],
    )
    assert result.exit_code == 1  # missing required values -> errors
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "nogps.jpg" in text
    assert MISSING_MARKER in text


def test_default_report_name_from_sidecar_path():
    """An empty prompt answer falls back to this name in the current directory."""
    from atomic_tools.commands.lint import _default_report_name

    assert _default_report_name("data/my_sidecar.csv") == "my_sidecar_lint_report.csv"
    assert _default_report_name("s3://bucket/key/cloud.csv") == "cloud_lint_report.csv"


def test_report_helper_warns_when_no_table(tmp_path, capsys):
    """--report with no datatype (so no required fields known) writes nothing."""
    from atomic_tools.commands.lint import _write_missing_data_report
    from atomic_tools.validators.report import LintReport

    report = LintReport()  # missing_data defaults to None
    out = tmp_path / "missing.csv"
    _write_missing_data_report(report, str(out))

    assert not out.exists()
    assert "datatype" in capsys.readouterr().err.lower()
