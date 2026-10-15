"""``am-tools validate`` — lint data without saving a sidecar.

Does exactly what ``am-tools sidecar`` does — scan a directory (or one file),
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

import typer

from atomic_tools.api import error_payload, run_validation, validate_path
from atomic_tools.commands.sidecar import (
    LimitOption,
    SampleOption,
    SeedOption,
    _check_selection_options,
    _echo_replay_command,
    _fail,
    _format_replay_command,
    _raise_crs_failures_loudly,
    _run_interactive_wizard,
    _warn_skipped_vectors,
    passed_on_command_line,
)
from atomic_tools.models import ValidationResponse
from atomic_tools.utils.utils import DataTypeEnum, DataTypeFilter

logger = logging.getLogger(__name__)


def _emit_json(payload: ValidationResponse, stream) -> None:
    stream.write(json.dumps(payload.to_json_dict()))
    stream.write("\n")
    stream.flush()


def _run_api_mode(
    *,
    path: str | None,
    data_type: DataTypeEnum | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool,
    spatial_reference: str | None,
    ignore_missing_orientation: bool,
    coco: str | None,
    include: str | None,
    verbose: bool,
    limit: int | None = None,
    sample: int | None = None,
    seed: int | None = None,
) -> None:
    """Print one JSON document to stdout. Exit 0 if validation ran, else 2/1."""
    real_stdout = sys.stdout
    if not verbose:
        # Per-file warnings would flood a server's logs.
        logging.getLogger().setLevel(logging.ERROR)

    if not path:
        response = error_payload("invalid_input", "--path is required with --api-mode.")
    else:
        # Keep stray prints off stdout, which carries only the JSON.
        with contextlib.redirect_stdout(sys.stderr):
            response = validate_path(
                path,
                data_type=data_type,
                client_sidecar=client_sidecar,
                client_schema=client_schema,
                full=full,
                spatial_reference=spatial_reference,
                ignore_missing_orientation=ignore_missing_orientation,
                coco=coco,
                include=include,
                limit=limit,
                sample=sample,
                seed=seed,
            )

    _emit_json(response, real_stdout)
    if response.error is not None:
        raise typer.Exit(code=2 if response.error.type == "invalid_input" else 1)


def validate(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Option(
            "--path",
            "--directory",
            help=(
                "Directory or single file to validate (--directory is an "
                "alias). Either an object-store "
                "URI (s3://bucket/prefix, s3://bucket/key.jpg, gs://..., az://...) "
                "or a local filesystem path."
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
                "a mismatch aborts and names the bad file. Client values fill "
                "fields the files are missing; where both have a value, the "
                "file's own metadata is kept (with a warning)."
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
                "in this CRS and reprojected to EPSG:4326. For point clouds and "
                "orthos, a 'fallback_srs' column is added with this value in the "
                "DEFAULT row, used by files that carry no CRS of their own."
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
                "2 for bad input, 1 if the run errored. Requires --path."
            ),
        ),
    ] = False,
    include: Annotated[
        str | None,
        typer.Option(
            "--include",
            help=(
                "With --api-mode: comma-separated JSON sections to return — "
                "findings, missing_data, metadata, details, files, or 'all'. "
                "Default: findings,missing_data,details,files."
            ),
        ),
    ] = None,
    limit: LimitOption = None,
    sample: SampleOption = None,
    seed: SeedOption = None,
) -> None:
    """Scan a directory or file, extract metadata, and lint it — without saving a sidecar.

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
            path=path,
            data_type=data_type_enum,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
            spatial_reference=spatial_reference,
            ignore_missing_orientation=ignore_missing_orientation,
            coco=coco,
            include=include,
            verbose=verbosity_state.verbose,
            limit=limit,
            sample=sample,
            seed=seed,
        )
        return
    if include is not None:
        raise typer.BadParameter("--include only applies with --api-mode.")
    # After the api-mode branch, where validate_path reports these as JSON.
    _check_selection_options(limit, sample, seed)

    # The wizard now only runs when the path is missing; --datatype being
    # unset just means auto-detect, not "ask me".
    wizard_ran = path is None
    if wizard_ran:
        full_provided = passed_on_command_line(ctx, "full")
        ignore_orientation_provided = passed_on_command_line(ctx, "ignore_missing_orientation")
        # output_filename / local_copy are irrelevant when nothing is saved, so
        # they're left at their defaults and ignored (hence save=False).
        result = _run_interactive_wizard(
            path,
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
            limit=limit,
            sample=sample,
            seed=seed,
        )
        path = result.directory
        data_type_enum = result.data_type
        client_sidecar = result.client_sidecar
        client_schema = result.client_schema
        full = result.full
        verbosity_choice = result.verbosity
        spatial_reference = result.spatial_reference
        ignore_missing_orientation = result.ignore_missing_orientation
        coco = result.coco
        limit, sample, seed = result.limit, result.sample, result.seed
        logging.getLogger().setLevel(
            level_for(
                verbose=verbosity_choice == "verbose",
                silent=verbosity_choice == "silent",
            )
        )

    if not path:
        raise typer.BadParameter("--path is required.")
    # data_type is an optional filter now — None means auto-detect.

    if wizard_ran:
        _echo_replay_command(
            _format_replay_command(
                ["validate"],
                directory=path,
                data_type=data_type_enum,
                client_sidecar=client_sidecar,
                client_schema=client_schema,
                full=full,
                verbosity=verbosity_choice,
                spatial_reference=spatial_reference,
                ignore_missing_orientation=ignore_missing_orientation,
                coco=coco,
                limit=limit,
                sample=sample,
                seed=seed,
            )
        )

    try:
        result = run_validation(
            path,
            data_type=data_type_enum,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
            spatial_reference=spatial_reference,
            ignore_missing_orientation=ignore_missing_orientation,
            coco=coco,
            limit=limit,
            sample=sample,
            seed=seed,
        )
    except Exception as e:
        _fail("Failed to validate data", e)

    typer.echo(result.report.render())
    _warn_skipped_vectors(result.skipped_vector)
    _raise_crs_failures_loudly(result.crs_failures)
    if result.failed:
        raise typer.Exit(code=1)
