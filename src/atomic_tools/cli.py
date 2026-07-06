from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Literal

import typer

from atomic_tools import __version__
from atomic_tools.commands.lint import lint_app
from atomic_tools.commands.schema import build_schema_command
from atomic_tools.commands.sidecar import sidecar_app
from atomic_tools.commands.validate import validate as validate_command

VerbosityChoice = Literal["default", "verbose", "silent"]

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class Verbosity:
    verbose: bool = False
    silent: bool = False

    @property
    def level(self) -> int:
        return level_for(verbose=self.verbose, silent=self.silent)

    @property
    def choice(self) -> VerbosityChoice:
        if self.verbose:
            return "verbose"
        if self.silent:
            return "silent"
        return "default"


def level_for(*, verbose: bool, silent: bool) -> int:
    if verbose:
        return logging.INFO
    if silent:
        return logging.ERROR
    return logging.WARNING


app = typer.Typer(
    name="am-tools",
    help="Atomic public tools — format metadata for Flow intake.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(sidecar_app, name="sidecar", help="Generate a sidecar CSV from S3.")
app.add_typer(lint_app, name="lint", help="Validate schema/sidecar files before submission.")
app.command(
    name="build-schema",
    help="Build a client schema (column-name mapping) for a sidecar CSV.",
)(build_schema_command)
app.command(
    name="validate",
    help="Lint data without saving a sidecar (extract, build, and check in one step).",
)(validate_command)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"am-tools {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Increase log verbosity to INFO. Wins over --silent if both are passed.",
        ),
    ] = False,
    silent: Annotated[
        bool,
        typer.Option(
            "--silent",
            "-s",
            help="Reduce log output to errors only.",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Root command — see subcommands below."""
    verbosity = Verbosity(verbose=verbose, silent=silent)
    logging.basicConfig(format=_LOG_FORMAT, level=verbosity.level)
    logging.getLogger().setLevel(verbosity.level)
    ctx.obj = verbosity
