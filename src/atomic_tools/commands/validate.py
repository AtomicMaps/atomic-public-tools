"""``am-tools validate`` — lint data without saving a sidecar.

Does exactly what ``am-tools sidecar generate`` does — scan a directory,
extract per-file metadata, assemble a sidecar, and lint it — but never writes
the sidecar to the remote/local directory. It exists for the common case of
"I just want to know whether my data is clean" without leaving a file behind.

The metadata extraction, sidecar assembly, wizard prompts, and lint pathway are
all shared with ``sidecar generate`` (see :mod:`atomic_tools.commands.sidecar`);
this module only swaps the persist step for an in-memory temp file that is
linted and discarded.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

import click
import typer

from atomic_tools.commands.sidecar import (
    _build_sidecar,
    _crs_failures,
    _echo_replay_command,
    _fail,
    _format_replay_command,
    _raise_crs_failures_loudly,
    _run_interactive_wizard,
    _warn_skipped_vectors,
)
from atomic_tools.utils.utils import DataTypeEnum, DataTypeFilter
from atomic_tools.validators.sidecar import lint_sidecar_file

logger = logging.getLogger(__name__)


def _validate(
    directory: str,
    data_type: DataTypeEnum | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool = False,
    spatial_reference: str | None = None,
    ignore_missing_orientation: bool = False,
    coco: str | None = None,
):
    """Build the sidecar in memory and lint it without persisting anything."""
    logger.info(
        f"Starting validation — directory={directory!r} data_type={data_type} "
        f"client_sidecar={client_sidecar!r} client_schema={client_schema!r} "
        f"full={full} spatial_reference={spatial_reference!r} coco={coco!r}"
    )

    # validate never persists, so the backend is unused.
    df, _, _detected, skipped_vector = _build_sidecar(
        directory=directory,
        data_type=data_type,
        client_sidecar=client_sidecar,
        client_schema=client_schema,
        full=full,
        spatial_reference=spatial_reference,
    )
    crs_failures = _crs_failures(df)

    # Lint reads from a path, so stage the assembled sidecar in a throwaway temp
    # file rather than writing it back to the (possibly remote) input directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_csv = Path(tmp_dir) / "sidecar.csv"
        df.to_csv(local_csv, index=False)
        logger.info("Linting in-memory sidecar (not saved)…")
        report = lint_sidecar_file(
            str(local_csv),
            final=True,
            data_type=data_type,
            schema_path=None,
            input_files_path=directory,
            ignore_missing_orientation=ignore_missing_orientation,
            coco_path=coco,
            # validate is informational — a COCO referencing images absent from
            # the scanned directory is reported as a warning here, not a hard
            # failure (it blocks only when generating a sidecar that ships).
            coco_not_on_disk_is_error=False,
        )
        return report, skipped_vector, crs_failures


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
                "(degraded/unusable/not_on_disk), and adds those tiers to the "
                "failed-rows CSV (--report)."
            ),
        ),
    ] = None,
) -> None:
    """Scan a directory, extract metadata, and lint it — without saving a sidecar.

    Identical to ``sidecar generate`` except nothing is written: no sidecar is
    saved to the input directory or locally. With no flags, prompts
    interactively for each value. Any flag passed on the command line is used
    as-is and not prompted for.
    """
    from atomic_tools.cli import Verbosity, level_for

    verbosity_state: Verbosity = ctx.ensure_object(Verbosity)
    verbosity_choice = verbosity_state.choice
    verbosity_provided = verbosity_state.verbose or verbosity_state.silent

    # --datatype is an optional filter; convert it to the full enum immediately.
    data_type_enum: DataTypeEnum | None = (
        DataTypeEnum(data_type.value) if data_type is not None else None
    )

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
        report, skipped_vector, crs_failures = _validate(
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

    typer.echo(report.render())
    _warn_skipped_vectors(skipped_vector)
    _raise_crs_failures_loudly(crs_failures)
    if report.has_errors() or crs_failures:
        raise typer.Exit(code=1)
