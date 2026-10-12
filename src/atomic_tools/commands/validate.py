"""``am-tools validate`` — lint data without saving a sidecar.

Does exactly what ``am-tools sidecar`` does — scan a directory,
extract per-file metadata, assemble a sidecar, and lint it — but never writes
the sidecar to the remote/local directory. It exists for the common case of
"I just want to know whether my data is clean" without leaving a file behind.

The metadata extraction, sidecar assembly, wizard prompts, and lint pathway are
all shared with ``sidecar`` (see :mod:`atomic_tools.commands.sidecar`);
this module only swaps the persist step for linting the in-memory DataFrame
(see :mod:`atomic_tools.api`, which also powers ``--api-mode``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from typing import Annotated

import click
import typer

from atomic_tools.api import (
    ValidationResult,
    build_payload,
    error_payload,
    parse_include,
    run_validation,
)
from atomic_tools.api import (
    _classify_exception as _classify_api_exception,
)
from atomic_tools.commands.sidecar import (
    _echo_replay_command,
    _fail,
    _format_replay_command,
    _raise_crs_failures_loudly,
    _run_interactive_wizard,
    _warn_skipped_vectors,
)
from atomic_tools.utils.utils import DataTypeEnum, DataTypeFilter

logger = logging.getLogger(__name__)


def _emit_json(payload: dict, stream) -> None:
    stream.write(json.dumps(payload, default=str))
    stream.write("\n")
    stream.flush()


def _run_api_mode(
    *,
    directory: str | None,
    data_type: DataTypeEnum | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool,
    spatial_reference: str | None,
    ignore_missing_orientation: bool,
    coco: str | None,
    include: str | None,
    verbose: bool,
) -> None:
    """Non-interactive validate: one JSON document on stdout, nothing else.

    Exit 0 whenever validation completed (pass/fail is in ``status``), 2 for
    bad input, 1 when the run itself errored.
    """
    real_stdout = sys.stdout
    if not verbose:
        # Logs already go to stderr; drop them to errors-only so a log
        # aggregator on the API host isn't flooded with per-file warnings.
        logging.getLogger().setLevel(logging.ERROR)

    try:
        sections = parse_include(include)
        if not directory:
            raise ValueError("--directory is required with --api-mode.")
    except ValueError as e:
        _emit_json(error_payload("invalid_input", str(e)), real_stdout)
        raise typer.Exit(code=2) from None

    # Anything that prints (third-party libraries, stray echoes) goes to stderr
    # so stdout carries exactly one JSON document.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run_validation(
                directory,
                data_type=data_type,
                client_sidecar=client_sidecar,
                client_schema=client_schema,
                full=full,
                spatial_reference=spatial_reference,
                ignore_missing_orientation=ignore_missing_orientation,
                coco=coco,
                quiet=True,
            )
    except Exception as e:  # noqa: BLE001 — always answer with JSON
        logger.error(f"Validation failed: {e}", exc_info=verbose)
        _emit_json(error_payload(_classify_api_exception(e), str(e)), real_stdout)
        raise typer.Exit(code=1) from None

    _emit_json(build_payload(result, sections), real_stdout)


def _validate(
    directory: str,
    data_type: DataTypeEnum | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool = False,
    spatial_reference: str | None = None,
    ignore_missing_orientation: bool = False,
    coco: str | None = None,
) -> ValidationResult:
    """Build the sidecar in memory and lint it without persisting anything."""
    return run_validation(
        directory,
        data_type=data_type,
        client_sidecar=client_sidecar,
        client_schema=client_schema,
        full=full,
        spatial_reference=spatial_reference,
        ignore_missing_orientation=ignore_missing_orientation,
        coco=coco,
    )


def validate(
    ctx: typer.Context,
    directory: Annotated[
        str | None,
        typer.Option(
            help=(
                "Directory to scan. Either an object-store URI "
                "(s3://bucket/prefix, gs://..., az://...) or a local filesystem path."
            ),
        ),
    ] = None,
    data_type: Annotated[
        DataTypeFilter | None,
        typer.Option(
            "--datatype",
            "--data-type",
            help=(
                "Optional filter: restrict to this data type. By default every "
                "file is auto-detected per file (recommended)."
            ),
            case_sensitive=False,
        ),
    ] = None,
    client_sidecar: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional path to client-supplied sidecar data. May be an "
                "object-store URI (s3://bucket/key/file.csv) or a local path. "
                "Point it at a single CSV, or at a directory: when it's a "
                "directory, every CSV in a subdirectory BELOW it is merged into "
                "one. All merged CSVs must share the same schema (column count); "
                "a mismatch aborts and names the bad file. Client values win on "
                "conflict."
            ),
        ),
    ] = None,
    client_schema: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional client-supplied JSON schema describing how to "
                "normalise the client sidecar CSV (positional column names, "
                "per-client renames). Local path or object-store URI "
                "(s3://bucket/key/schema.json, gs://..., az://...). See "
                "schemas/column_names_example.json. If omitted, the client "
                "CSV is used as-is with no renames."
            ),
        ),
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help=(
                "Include every metadata field extracted from each file. By "
                "default, only the canonical/required fields for the data type "
                "are kept (plus blank columns for any missing required fields)."
            ),
        ),
    ] = False,
    spatial_reference: Annotated[
        str | None,
        typer.Option(
            "--spatial-reference",
            "--spatial_reference",
            help=(
                "CRS of the source coordinates (e.g. 'EPSG:32612' or '32612'). "
                "For images, lat/lon (and altitude) are treated as X/Y/Z "
                "in this CRS and reprojected to EPSG:4326. For point clouds, a "
                "'fallback_srs' column is added with this value in the "
                "DEFAULT row."
            ),
        ),
    ] = None,
    ignore_missing_orientation: Annotated[
        bool,
        typer.Option(
            "--ignore-missing-orientation",
            help=(
                "Only meaningful for --datatype oriented_image. By default, "
                "missing orientation (Pitch/Heading/Roll) is an error; pass this "
                "to downgrade it to a warning (the images still process, "
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
                "containing one). Image data types only. Reports how many labels "
                "sit on images with missing/zero-size metadata "
                "(degraded/unusable/not_on_disk). With --api-mode, per-image "
                "tiers are returned under details.coco and missing_data."
            ),
        ),
    ] = None,
    api_mode: Annotated[
        bool,
        typer.Option(
            "--api-mode",
            help=(
                "Machine mode for API use: never prompts, skips the update "
                "check, and prints a single JSON document to stdout instead of "
                "the text report. Exit 0 when validation ran (see 'status'), "
                "2 for bad input, 1 if the run errored. Requires --directory."
            ),
        ),
    ] = False,
    include: Annotated[
        str | None,
        typer.Option(
            "--include",
            help=(
                "With --api-mode: comma-separated JSON sections to return — "
                "findings, missing_data, metadata, details, or 'all'. "
                "Default: findings,missing_data,details."
            ),
        ),
    ] = None,
) -> None:
    """Scan a directory, extract metadata, and lint it — without saving a sidecar.

    Identical to ``sidecar`` except nothing is written: no sidecar is
    saved to the input directory or locally. With no flags, prompts
    interactively for each value. Any flag passed on the command line is used
    as-is and not prompted for. ``--api-mode`` never prompts and prints JSON.
    """
    from atomic_tools.cli import Verbosity, level_for

    verbosity_state: Verbosity = ctx.ensure_object(Verbosity)
    verbosity_choice = verbosity_state.choice
    verbosity_provided = verbosity_state.verbose or verbosity_state.silent

    # --datatype is an optional filter; convert it to the full enum immediately.
    data_type_enum: DataTypeEnum | None = (
        DataTypeEnum(data_type.value) if data_type is not None else None
    )

    if api_mode:
        _run_api_mode(
            directory=directory,
            data_type=data_type_enum,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
            spatial_reference=spatial_reference,
            ignore_missing_orientation=ignore_missing_orientation,
            coco=coco,
            include=include,
            verbose=verbosity_state.verbose,
        )
        return
    if include is not None:
        raise typer.BadParameter("--include only applies with --api-mode.")

    # The wizard now only runs when the directory is missing; --datatype being
    # unset just means auto-detect, not "ask me".
    wizard_ran = directory is None
    if wizard_ran:
        full_provided = (
            ctx.get_parameter_source("full") == click.core.ParameterSource.COMMANDLINE
        )
        ignore_orientation_provided = (
            ctx.get_parameter_source("ignore_missing_orientation")
            == click.core.ParameterSource.COMMANDLINE
        )
        # output_filename / local_copy are irrelevant when nothing is saved, so
        # they're left at their defaults and ignored (hence save=False).
        result = _run_interactive_wizard(
            directory,
            data_type_enum,
            None,
            client_sidecar,
            client_schema,
            full,
            full_provided,
            verbosity_choice,
            verbosity_provided,
            False,
            True,
            spatial_reference,
            ignore_missing_orientation,
            ignore_orientation_provided,
            coco=coco,
            save=False,
        )
        directory = result.directory
        data_type_enum = result.data_type
        client_sidecar = result.client_sidecar
        client_schema = result.client_schema
        full = result.full
        verbosity_choice = result.verbosity
        spatial_reference = result.spatial_reference
        ignore_missing_orientation = result.ignore_missing_orientation
        coco = result.coco
        logging.getLogger().setLevel(
            level_for(
                verbose=verbosity_choice == "verbose",
                silent=verbosity_choice == "silent",
            )
        )

    if not directory:
        raise typer.BadParameter("Directory is required.")
    # data_type is an optional filter now — None means auto-detect.

    if wizard_ran:
        _echo_replay_command(
            _format_replay_command(
                ["validate"],
                directory=directory,
                data_type=data_type_enum,
                client_sidecar=client_sidecar,
                client_schema=client_schema,
                full=full,
                verbosity=verbosity_choice,
                spatial_reference=spatial_reference,
                ignore_missing_orientation=ignore_missing_orientation,
                coco=coco,
            )
        )

    try:
        result = _validate(
            directory=directory,
            data_type=data_type_enum,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
            spatial_reference=spatial_reference,
            ignore_missing_orientation=ignore_missing_orientation,
            coco=coco,
        )
    except Exception as e:
        _fail("Failed to validate data", e)

    typer.echo(result.report.render())
    _warn_skipped_vectors(result.skipped_vector)
    _raise_crs_failures_loudly(result.crs_failures)
    if result.failed:
        raise typer.Exit(code=1)
