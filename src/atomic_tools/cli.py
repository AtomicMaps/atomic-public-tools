from typing import Annotated

import typer

from atomic_tools import __version__
from atomic_tools.commands.metadata import metadata_app

app = typer.Typer(
    name="am-tools",
    help="Atomic public tools — format metadata for Flow intake.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(metadata_app, name="metadata", help="Validate and format metadata files.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"am-tools {__version__}")
        raise typer.Exit()


@app.callback()
def main(
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
