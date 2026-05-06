from pathlib import Path
from typing import Annotated

import typer

metadata_app = typer.Typer(no_args_is_help=True, help="Validate and format metadata files.")

_SUPPORTED_FORMATS = {".json": "json", ".csv": "csv", ".xlsx": "excel"}


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_FORMATS:
        supported = ", ".join(sorted(_SUPPORTED_FORMATS))
        raise typer.BadParameter(
            f"Unsupported file extension '{suffix}'. Supported: {supported}."
        )
    return _SUPPORTED_FORMATS[suffix]


@metadata_app.command()
def validate(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Metadata file to check."),
    ],
) -> None:
    """Validate a metadata file against the Flow intake schema."""
    fmt = _detect_format(path)
    typer.echo(f"Detected format: {fmt} ({path.name})")
    typer.echo("validation not yet implemented")


@metadata_app.command("format")
def format_(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Metadata file to format."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the formatted output."),
    ] = None,
) -> None:
    """Format a metadata file into the canonical Flow intake shape."""
    fmt = _detect_format(path)
    typer.echo(f"Detected format: {fmt} ({path.name})")
    if output is not None:
        typer.echo(f"Would write to: {output}")
    typer.echo("formatting not yet implemented")
