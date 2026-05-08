from typer.testing import CliRunner

from atomic_tools import __version__
from atomic_tools.cli import app

runner = CliRunner()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sidecar" in result.stdout
    assert "lint" in result.stdout


def test_sidecar_help_lists_generate():
    result = runner.invoke(app, ["sidecar", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.stdout


def test_sidecar_generate_help_lists_options():
    result = runner.invoke(app, ["sidecar", "generate", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--directory",
        "--data-type",
        "--output-filename",
        "--client-sidecar",
        "--client-schema",
    ):
        assert flag in result.stdout
    assert "--bucket" not in result.stdout


def test_lint_help_lists_subcommands():
    result = runner.invoke(app, ["lint", "--help"])
    assert result.exit_code == 0
    assert "schema" in result.stdout
    assert "sidecar" in result.stdout


def test_lint_schema_help_shows_path_arg():
    result = runner.invoke(app, ["lint", "schema", "--help"])
    assert result.exit_code == 0
    assert "PATH" in result.stdout or "path" in result.stdout


def test_lint_sidecar_help_lists_options():
    result = runner.invoke(app, ["lint", "sidecar", "--help"])
    assert result.exit_code == 0
    for flag in ("--final", "--datatype", "--schema", "--input-files"):
        assert flag in result.stdout


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
