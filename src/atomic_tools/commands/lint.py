"""`am-tools lint` — validate schema and sidecar files locally."""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING, Annotated

import click
import questionary
import typer

from atomic_tools.commands.sidecar import (
    _ask_client_schema,
    _ask_data_type,
    _ask_verbosity,
    ask_schema_uri,
)
from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.schema import lint_schema_file
from atomic_tools.validators.sidecar import lint_sidecar_file

if TYPE_CHECKING:
    from atomic_tools.cli import VerbosityChoice

logger = logging.getLogger(__name__)

lint_app = typer.Typer(
    no_args_is_help=True,
    help="Validate schema/sidecar files before submission.",
)


def _ask_schema_path() -> str:
    return ask_schema_uri()


def _ask_sidecar_path() -> str:
    answer = questionary.text(
        "Path to sidecar CSV:",
        instruction="(Local path or s3://… URI)",
        validate=lambda v: bool(v.strip()) or "Required.",
    ).unsafe_ask()
    return answer.strip()


def _ask_final() -> bool:
    return questionary.confirm(
        "Treat as a generated (final) sidecar? (Yes enforces required-column coverage)",
        default=False,
    ).unsafe_ask()


def _ask_input_files() -> str | None:
    answer = questionary.text(
        "Optional directory of input files to compare against:",
        instruction="(Local path or s3://… URI; press Enter to skip)",
    ).unsafe_ask()
    answer = (answer or "").strip()
    return answer or None


def _echo_replay_command(command: str) -> None:
    click.secho(
        "\nEquivalent command (copy & paste to skip the wizard next time):",
        fg="cyan",
        err=True,
    )
    click.secho(f"  {command}\n", fg="bright_cyan", bold=True, err=True)


def _verbosity_prefix(verbosity: VerbosityChoice) -> list[str]:
    if verbosity == "verbose":
        return ["--verbose"]
    if verbosity == "silent":
        return ["--silent"]
    return []


def _format_schema_replay_command(path: str, verbosity: VerbosityChoice) -> str:
    parts = ["am-tools", *_verbosity_prefix(verbosity), "lint", "schema", shlex.quote(path)]
    return " ".join(parts)


def _format_sidecar_replay_command(
    path: str,
    final: bool,
    datatype: DataTypeEnum,
    schema: str | None,
    input_files: str | None,
    verbosity: VerbosityChoice,
) -> str:
    parts = [
        "am-tools",
        *_verbosity_prefix(verbosity),
        "lint",
        "sidecar",
        shlex.quote(path),
        "--datatype",
        datatype.value,
    ]
    if final:
        parts.append("--final")
    if schema is not None:
        parts += ["--schema", shlex.quote(schema)]
    if input_files:
        parts += ["--input-files", shlex.quote(input_files)]
    return " ".join(parts)


@lint_app.command("schema")
def lint_schema_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(
            help="Path or URI to a schema JSON file (local or s3://… / gs://… / az://…).",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Validate a client schema JSON file."""
    from atomic_tools.cli import Verbosity, level_for

    verbosity_state: Verbosity = ctx.ensure_object(Verbosity)
    verbosity_choice = verbosity_state.choice
    verbosity_provided = verbosity_state.verbose or verbosity_state.silent

    wizard_ran = path is None
    if path is None:
        try:
            path = _ask_schema_path()
            if not verbosity_provided:
                verbosity_choice = _ask_verbosity()
        except KeyboardInterrupt:
            raise typer.Exit(code=130) from None
        logging.getLogger().setLevel(
            level_for(
                verbose=verbosity_choice == "verbose",
                silent=verbosity_choice == "silent",
            )
        )

    if wizard_ran:
        _echo_replay_command(_format_schema_replay_command(path, verbosity_choice))

    report = lint_schema_file(path)
    typer.echo(report.render())
    raise typer.Exit(code=report.exit_code())


@lint_app.command("sidecar")
def lint_sidecar_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(
            help="Path to a sidecar CSV (local path or s3://… URI).",
            show_default=False,
        ),
    ] = None,
    final: Annotated[
        bool,
        typer.Option(
            "--final",
            help=(
                "Treat as a generated (final) sidecar. Enforces that every required "
                "column is present and has a value in DEFAULT or every row."
            ),
        ),
    ] = False,
    datatype: Annotated[
        DataTypeEnum | None,
        typer.Option(
            "--datatype",
            help="Data type of the input data (e.g. 'oriented_image', 'point_cloud').",
            case_sensitive=False,
        ),
    ] = None,
    schema: Annotated[
        str | None,
        typer.Option(
            "--schema",
            help=(
                "Optional client schema JSON (local path or s3://… / gs://… "
                "/ az://… URI). Used to apply positional column names + "
                "per-client renames before checking. Ignored when --final is set."
            ),
        ),
    ] = None,
    input_files: Annotated[
        str | None,
        typer.Option(
            "--input-files",
            help=(
                "Optional directory of input files to verify the sidecar covers "
                "(local path or s3://… URI)."
            ),
        ),
    ] = None,
) -> None:
    """Validate a sidecar CSV (client or generated)."""
    from atomic_tools.cli import Verbosity, level_for

    verbosity_state: Verbosity = ctx.ensure_object(Verbosity)
    verbosity_choice = verbosity_state.choice
    verbosity_provided = verbosity_state.verbose or verbosity_state.silent

    # check to see if they provided the fields
    final_provided = ctx.get_parameter_source("final") == click.core.ParameterSource.COMMANDLINE
    schema_provided = ctx.get_parameter_source("schema") == click.core.ParameterSource.COMMANDLINE
    input_files_provided = (
        ctx.get_parameter_source("input_files") == click.core.ParameterSource.COMMANDLINE
    )

    # If the user didn't provide required info via arguments, ask interactively
    wizard_ran = path is None or datatype is None
    if wizard_ran:
        try:
            if path is None:
                path = _ask_sidecar_path()
            if datatype is None:
                datatype = _ask_data_type()
            if not final_provided:
                final = _ask_final()
            if schema is None and not schema_provided:
                schema = _ask_client_schema()
            if input_files is None and not input_files_provided:
                input_files = _ask_input_files()
            if not verbosity_provided:
                verbosity_choice = _ask_verbosity()
        except KeyboardInterrupt:
            raise typer.Exit(code=130) from None
        logging.getLogger().setLevel(
            level_for(
                verbose=verbosity_choice == "verbose",
                silent=verbosity_choice == "silent",
            )
        )

    if not path:
        raise typer.BadParameter("Sidecar path is required.")
    if datatype is None:
        raise typer.BadParameter("--datatype is required.")

    if wizard_ran:
        _echo_replay_command(
            _format_sidecar_replay_command(
                path=path,
                final=final,
                datatype=datatype,
                schema=schema,
                input_files=input_files,
                verbosity=verbosity_choice,
            )
        )

    report = lint_sidecar_file(
        path,
        final=final,
        data_type=datatype,
        schema_path=schema,
        input_files_path=input_files,
    )
    typer.echo(report.render())
    raise typer.Exit(code=report.exit_code())
