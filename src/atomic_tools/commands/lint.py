"""`am-tools lint` — validate schema and sidecar files locally."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import click
import questionary
import typer

from atomic_tools.commands.sidecar import _ask_client_schema, _ask_data_type
from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.schema import lint_schema_file
from atomic_tools.validators.sidecar import lint_sidecar_file

logger = logging.getLogger(__name__)

lint_app = typer.Typer(
    no_args_is_help=True,
    help="Validate schema/sidecar files before submission.",
)


def _ask_schema_path() -> Path:
    answer = questionary.path(
        "Path to schema JSON:",
        validate=lambda v: Path(v).expanduser().is_file() or "File not found.",
    ).unsafe_ask()
    return Path(answer).expanduser().resolve()


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


@lint_app.command("schema")
def lint_schema_cmd(
    path: Annotated[
        Path | None,
        typer.Argument(help="Path to a schema JSON file.", show_default=False),
    ] = None,
) -> None:
    """Validate a client schema JSON file."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.WARNING,
    )

    if path is None:
        try:
            path = _ask_schema_path()
        except KeyboardInterrupt:
            raise typer.Exit(code=130) from None

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
        Path | None,
        typer.Option(
            "--schema",
            exists=True,
            dir_okay=False,
            readable=True,
            help=(
                "Optional client schema JSON. Used to apply headerless column "
                "names + per-client renames before checking. Ignored when --final is set."
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
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.WARNING,
    )

    final_provided = (
        ctx.get_parameter_source("final") == click.core.ParameterSource.COMMANDLINE
    )
    schema_provided = (
        ctx.get_parameter_source("schema") == click.core.ParameterSource.COMMANDLINE
    )
    input_files_provided = (
        ctx.get_parameter_source("input_files") == click.core.ParameterSource.COMMANDLINE
    )

    if path is None or datatype is None:
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
        except KeyboardInterrupt:
            raise typer.Exit(code=130) from None

    if not path:
        raise typer.BadParameter("Sidecar path is required.")
    if datatype is None:
        raise typer.BadParameter("--datatype is required.")

    report = lint_sidecar_file(
        path,
        final=final,
        data_type=datatype,
        schema_path=schema,
        input_files_path=input_files,
    )
    typer.echo(report.render())
    raise typer.Exit(code=report.exit_code())
