from typer.testing import CliRunner

from atomic_tools import __version__
from atomic_tools.cli import app

runner = CliRunner()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "metadata" in result.stdout
    assert "sidecar" in result.stdout


def test_sidecar_help_lists_generate():
    result = runner.invoke(app, ["sidecar", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.stdout


def test_sidecar_generate_help_lists_options():
    result = runner.invoke(app, ["sidecar", "generate", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--bucket",
        "--directory",
        "--data-type",
        "--output-filename",
        "--client-sidecar",
    ):
        assert flag in result.stdout


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_metadata_help_lists_subcommands():
    result = runner.invoke(app, ["metadata", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout
    assert "format" in result.stdout


def test_metadata_validate_detects_format(tmp_path):
    sample = tmp_path / "sample.json"
    sample.write_text("{}")
    result = runner.invoke(app, ["metadata", "validate", str(sample)])
    assert result.exit_code == 0
    assert "json" in result.stdout
    assert "not yet implemented" in result.stdout


def test_metadata_validate_rejects_unknown_extension(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello")
    result = runner.invoke(app, ["metadata", "validate", str(sample)])
    assert result.exit_code != 0
