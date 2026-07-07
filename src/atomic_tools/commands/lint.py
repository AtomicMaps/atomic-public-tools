"""`am-tools lint` — validate schema and sidecar files locally."""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import questionary
import typer

from atomic_tools.commands.sidecar import (
    _ask_client_schema,
    _ask_coco,
    _ask_ignore_missing_orientation,
    _ask_verbosity,
    ask_schema_uri,
)
from atomic_tools.utils.utils import DataTypeEnum, DataTypeFilter, uri_stem
from atomic_tools.validators.coco import IMAGE_DATA_TYPES
from atomic_tools.validators.report import LintReport
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


def _default_report_name(sidecar_path: str) -> str:
    return f"{uri_stem(sidecar_path) or 'sidecar'}_lint_report.csv"


def _write_missing_data_report(report: LintReport, report_path: str) -> None:
    table = report.missing_data
    if table is None:
        click.secho(
            "No missing-data report to write: the sidecar had no rows that could "
            "be classified to a data type (no DataType column and no inferable "
            "filenames), so no required fields apply.",
            fg="yellow",
            err=True,
        )
        return
    saved = Path(report_path).expanduser().resolve()
    try:
        table.write_csv(str(saved))
    except OSError as e:
        click.secho(f"Could not write report to {saved}: {e}", fg="red", err=True)
        return
    click.secho(
        f"Saved missing-data report ({len(table.rows)} row(s)) to {saved.as_uri()}",
        fg="green",
        err=True,
    )


def _maybe_offer_missing_data_report(report: LintReport, *, sidecar_path: str) -> None:
    """When a lint failed or warned, offer to save the missing-data rows to CSV.

    Only fires for an interactive TTY; scripted runs use the ``--report`` flag.
    """
    table = report.missing_data
    if table is None or table.is_empty():
        return
    if not (report.has_errors() or report.warnings()):
        return
    if not sys.stdin.isatty():
        return
    default_target = Path.cwd() / _default_report_name(sidecar_path)
    try:
        if not questionary.confirm(
            f"Save a CSV report of the {len(table.rows)} row(s) missing required data?",
            default=True,
        ).unsafe_ask():
            return
        report_path = questionary.text(
            "Path for the report CSV:",
            instruction=f"(Press Enter to save to the current directory: {default_target})",
            default="",
        ).unsafe_ask()
    except KeyboardInterrupt:
        return
    _write_missing_data_report(report, (report_path or "").strip() or str(default_target))


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
    datatype: DataTypeEnum | None,
    schema: str | None,
    input_files: str | None,
    verbosity: VerbosityChoice,
    ignore_missing_orientation: bool,
    coco: str | None,
) -> str:
    parts = [
        "am-tools",
        *_verbosity_prefix(verbosity),
        "lint",
        "sidecar",
        shlex.quote(path),
    ]
    if datatype is not None:
        parts += ["--datatype", datatype.value]
    if final:
        parts.append("--final")
    if schema is not None:
        parts += ["--schema", shlex.quote(schema)]
    if input_files:
        parts += ["--input-files", shlex.quote(input_files)]
    if ignore_missing_orientation:
        parts.append("--ignore-missing-orientation")
    if coco:
        parts += ["--coco", shlex.quote(coco)]
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
        DataTypeFilter | None,
        typer.Option(
            "--datatype",
            help=(
                "Optional filter: restrict checks to this data type. By default "
                "each row's type is read from the sidecar's DataType column (or "
                "inferred from its filename)."
            ),
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
    report_out: Annotated[
        str | None,
        typer.Option(
            "--report",
            help=(
                "Write a CSV of the rows missing required data (one column per "
                "required field) to this path. Skips the interactive prompt."
            ),
        ),
    ] = None,
    ignore_missing_orientation: Annotated[
        bool,
        typer.Option(
            "--ignore-missing-orientation",
            help=(
                "Only meaningful for --datatype oriented_image with --final. By "
                "default, missing orientation (Pitch/Heading/Roll) is an error; "
                "pass this to downgrade it to a warning (the images still process, "
                "appearing in Lens without orientation)."
            ),
        ),
    ] = False,
    coco: Annotated[
        str | None,
        typer.Option(
            "--coco",
            help=(
                "Optional COCO label file (local path, s3://… URI, or a directory "
                "containing one). Image data types only; --datatype required. "
                "Reports how many labels sit on images with missing/zero-size "
                "metadata (degraded/unusable/not_on_disk), and adds those tiers "
                "to the failed-rows CSV (--report)."
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
    ignore_orientation_provided = (
        ctx.get_parameter_source("ignore_missing_orientation")
        == click.core.ParameterSource.COMMANDLINE
    )
    coco_provided = ctx.get_parameter_source("coco") == click.core.ParameterSource.COMMANDLINE

    # --datatype is an optional filter; convert it to the full enum immediately.
    datatype_enum: DataTypeEnum | None = (
        DataTypeEnum(datatype.value) if datatype is not None else None
    )

    # The wizard now runs only when the sidecar path is missing; --datatype being
    # unset just means "detect per row", not "ask me".
    wizard_ran = path is None
    if wizard_ran:
        try:
            if path is None:
                path = _ask_sidecar_path()
            if not final_provided:
                final = _ask_final()
            # Show unless an explicit filter rules it out.
            if (
                final
                and datatype_enum in (None, DataTypeEnum.oriented_image)
                and not ignore_orientation_provided
            ):
                ignore_missing_orientation = _ask_ignore_missing_orientation()
            if schema is None and not schema_provided:
                schema = _ask_client_schema()
            if input_files is None and not input_files_provided:
                input_files = _ask_input_files()
            # COCO label impact applies to imagery; offer it in auto mode too.
            if not coco_provided and (
                datatype_enum is None or datatype_enum in IMAGE_DATA_TYPES
            ):
                coco = _ask_coco()
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
    # --datatype is optional now (auto per-row); --final no longer requires it.

    if wizard_ran:
        _echo_replay_command(
            _format_sidecar_replay_command(
                path=path,
                final=final,
                datatype=datatype_enum,
                schema=schema,
                input_files=input_files,
                verbosity=verbosity_choice,
                ignore_missing_orientation=ignore_missing_orientation,
                coco=coco,
            )
        )

    report = lint_sidecar_file(
        path,
        final=final,
        data_type=datatype_enum,
        schema_path=schema,
        input_files_path=input_files,
        ignore_missing_orientation=ignore_missing_orientation,
        coco_path=coco,
    )
    typer.echo(report.render())

    if report_out is not None:
        _write_missing_data_report(report, report_out)
    else:
        _maybe_offer_missing_data_report(report, sidecar_path=path)

    raise typer.Exit(code=report.exit_code())
