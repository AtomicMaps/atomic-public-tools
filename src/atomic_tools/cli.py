from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Literal

import click
import typer
from typer.core import TyperGroup

from atomic_tools import __version__
from atomic_tools.commands.lint import lint_app
from atomic_tools.commands.schema import build_schema_command
from atomic_tools.commands.sidecar import generate as sidecar_command
from atomic_tools.commands.update import update_command
from atomic_tools.commands.validate import validate as validate_command
from atomic_tools.version_check import check_for_update

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


_API_MODE_META = "atomic_tools.api_mode"


class _RootGroup(TyperGroup):
    """Notes whether ``--api-mode`` was passed before the root callback runs.

    The root callback (which runs the update check) executes before the
    subcommand parses its own options, and click no longer exposes the
    remaining args to it. ``parse_args`` sees exactly the args click was given,
    which — unlike ``sys.argv`` — is also right when the app is invoked
    programmatically (``app([...])``, ``CliRunner``).
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta[_API_MODE_META] = "--api-mode" in args
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_RootGroup,
    name="am-tools",
    help="Atomic public tools — format metadata for Flow intake.",
    no_args_is_help=True,
    add_completion=False,
)

app.command(
    name="sidecar",
    help="Generate a sidecar CSV by scanning a local or remote directory.",
)(sidecar_command)
app.add_typer(lint_app, name="lint", help="Validate schema/sidecar files before submission.")
app.command(
    name="build-schema",
    help="Build a client schema (column-name mapping) for a sidecar CSV.",
)(build_schema_command)
app.command(
    name="validate",
    help="Lint data without saving a sidecar (extract, build, and check in one step).",
)(validate_command)
app.command(
    name="update",
    help="Update am-tools to the latest version (git pull + reinstall).",
)(update_command)


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

    # First run of the day: nudge the user if the repo has a newer version.
    # Best-effort and silent-safe — never raises, and skips when --silent or
    # --api-mode (a server shouldn't make an outbound call per request).
    if not silent and not ctx.meta.get(_API_MODE_META, False):
        check_for_update()
