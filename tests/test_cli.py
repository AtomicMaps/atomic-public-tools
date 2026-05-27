import logging

import pytest
from typer.testing import CliRunner

from atomic_tools import __version__
from atomic_tools.cli import app

runner = CliRunner()


@pytest.fixture
def reset_root_logger():
    root = logging.getLogger()
    original_level = root.level
    yield
    root.setLevel(original_level)


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
        "--datatype",
        "--output-filename",
        "--client-sidecar",
        "--client-schema",
    ):
        assert flag in result.stdout
    assert "--bucket" not in result.stdout


def test_sidecar_generate_help_lists_spatial_reference():
    # Wide terminal so Rich doesn't truncate the long flag name.
    result = runner.invoke(app, ["sidecar", "generate", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--spatial-reference" in result.stdout


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


def test_root_help_lists_verbosity_flags():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.stdout
    assert "--silent" in result.stdout
    assert "-v" in result.stdout
    assert "-s" in result.stdout


@pytest.mark.parametrize(
    ("args", "expected_level"),
    [
        ([], logging.WARNING),
        (["--verbose"], logging.INFO),
        (["-v"], logging.INFO),
        (["--silent"], logging.ERROR),
        (["-s"], logging.ERROR),
        (["--verbose", "--silent"], logging.INFO),
        (["-v", "-s"], logging.INFO),
    ],
)
def test_verbosity_flags_set_root_logger_level(args, expected_level, reset_root_logger):
    # `lint schema --help` triggers the root callback (which configures logging)
    # before short-circuiting on the subcommand's --help.
    result = runner.invoke(app, [*args, "lint", "schema", "--help"])
    assert result.exit_code == 0
    assert logging.getLogger().level == expected_level
