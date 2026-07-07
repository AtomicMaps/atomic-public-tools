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


def test_sidecar_is_a_direct_command():
    # `am-tools sidecar` is the command itself now (no `generate` subcommand).
    result = runner.invoke(app, ["sidecar", "--help"])
    assert result.exit_code == 0
    assert "--directory" in result.stdout


def test_sidecar_help_lists_options():
    result = runner.invoke(app, ["sidecar", "--help"])
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


@pytest.mark.parametrize(
    "argv",
    [
        ["sidecar", "--help"],
        ["validate", "--help"],
        ["lint", "sidecar", "--help"],
    ],
)
def test_datatype_choices_exclude_non_scannable_types(argv):
    """imagery / cad / vector must never appear as --datatype choices; the five
    scannable filter types must."""
    result = runner.invoke(app, argv, env={"COLUMNS": "300"})
    assert result.exit_code == 0
    out = result.stdout
    for hidden in ("imagery", "cad", "vector"):
        assert hidden not in out, f"{hidden!r} should not be a --datatype choice"
    for shown in ("ortho_image", "oriented_image", "spherical_image", "point_cloud"):
        assert shown in out


def test_sidecar_generate_help_lists_spatial_reference():
    # Wide terminal so Rich doesn't truncate the long flag name.
    result = runner.invoke(app, ["sidecar", "--help"], env={"COLUMNS": "200"})
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


def test_root_help_lists_validate():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout


def test_validate_help_lists_shared_options(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["validate", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for flag in (
        "--directory",
        "--datatype",
        "--client-sidecar",
        "--client-schema",
        "--full",
        "--spatial-reference",
        "--ignore-missing-orientation",
    ):
        assert flag in result.stdout


def test_validate_help_omits_save_only_options():
    # validate never writes a sidecar, so the generate-only save options are gone.
    result = runner.invoke(app, ["validate", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--output-filename" not in result.stdout
    assert "--local-copy" not in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["sidecar", "--help"],
        ["validate", "--help"],
        ["lint", "sidecar", "--help"],
    ],
)
def test_coco_option_listed(argv):
    result = runner.invoke(app, argv, env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--coco" in result.stdout


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
