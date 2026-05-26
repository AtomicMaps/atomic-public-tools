"""Generate a sidecar CSV by scanning a local or remote directory.

Originally `tasks/generate_sidecar_csv/generate_sidecar_csv.py`.

Requires the following external binaries on PATH at runtime (NOT pip-installable):

  * exiftool
  * pdal  (the conda binary at /opt/conda/envs/pdal/bin/pdal — see
          ``utils/extractors.py``; adjust if the client environment uses a
          different path)
"""

import logging
import shlex
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import click
import questionary
import typer

if TYPE_CHECKING:
    from atomic_tools.cli import VerbosityChoice

from atomic_tools.client_sidecar import (
    build_global_alias_map,
    load_and_clean_client_sidecar,
    merge_client_metadata,
)
from atomic_tools.io.storage import from_directory
from atomic_tools.utils.aws_errors import find_auth_error, print_help_block
from atomic_tools.utils.coordinates import transform_coordinates
from atomic_tools.utils.extractors import (
    extract_exif_metadata,
    extract_pdal_metadata,
    infer_date_from_filepath,
)
from atomic_tools.utils.utils import (
    DATA_TYPE_INFO,
    DataTypeEnum,
    _split_path_components,
    has_value,
    is_remote_uri,
)
from atomic_tools.validators.required_fields import (
    ALL_SIDECAR_FIELD_GROUPS as _ALL_SIDECAR_FIELD_GROUPS,
)
from atomic_tools.validators.required_fields import (
    REQUIRED_SIDECAR_FIELD_GROUPS as _REQUIRED_SIDECAR_FIELD_GROUPS,
)
from atomic_tools.validators.sidecar import lint_sidecar_file
from atomic_tools.validators.values import parse_elevation, to_decimal_degree

logger = logging.getLogger(__name__)

sidecar_app = typer.Typer(
    no_args_is_help=True,
    help="Generate a sidecar CSV by scanning a local or remote directory.",
)

_IMAGE_DATA_TYPES = {
    DataTypeEnum.ortho_image,
    DataTypeEnum.oriented_image,
    DataTypeEnum.spherical_image,
}
_VIDEO_DATA_TYPES = {DataTypeEnum.video}
_POINT_CLOUD_DATA_TYPES = {DataTypeEnum.point_cloud}


def _fail(message: str, exc: Exception) -> NoReturn:
    """Log a failure, surface AWS auth help if applicable, and exit non-zero."""
    logger.exception(message)
    auth_err = find_auth_error(exc)
    if auth_err is not None:
        print_help_block(auth_err)
    raise typer.Exit(code=1) from None


def _list_local_schemas() -> list[Path]:
    """Return sorted ``*.json`` files from ``./schemas/`` if that dir exists."""
    schemas_dir = Path.cwd() / "schemas"
    if not schemas_dir.is_dir():
        return []
    return sorted(schemas_dir.glob("*.json"))


def _ask_directory() -> str:
    return (
        questionary.text(
            "Directory to scan:",
            instruction="(Local folder or object-store URI like s3://bucket/prefix)",
            validate=lambda v: bool(v.strip()) or "Required.",
        )
        .unsafe_ask()
        .strip()
    )


def _ask_data_type() -> DataTypeEnum:
    choice = questionary.select(
        "Data type of the input data:",
        choices=[v.value for v in DataTypeEnum if v in _REQUIRED_SIDECAR_FIELD_GROUPS],
    ).unsafe_ask()
    return DataTypeEnum(choice)


def _ask_output_filename() -> str:
    return questionary.text(
        "Output filename:",
        default="sidecar.csv",
        instruction="(Press Enter to keep the default)",
    ).unsafe_ask()


def _ask_full() -> bool:
    return questionary.confirm(
        "Include every metadata field exiftool/pdal extracted? "
        "(No keeps only the canonical fields Flow needs)",
        default=False,
    ).unsafe_ask()


def _ask_local_copy() -> bool:
    return questionary.confirm(
        "Also save a local copy of the sidecar in this directory "
        "(so you can inspect it before submitting)?",
        default=True,
    ).unsafe_ask()


_VERBOSITY_DEFAULT = "Default (warnings only)"
_VERBOSITY_VERBOSE = "Verbose (info)"
_VERBOSITY_SILENT = "Silent (errors only)"


def _ask_verbosity() -> "VerbosityChoice":
    selection = questionary.select(
        "Logging verbosity:",
        choices=[_VERBOSITY_DEFAULT, _VERBOSITY_VERBOSE, _VERBOSITY_SILENT],
        default=_VERBOSITY_DEFAULT,
    ).unsafe_ask()
    if selection == _VERBOSITY_VERBOSE:
        return "verbose"
    if selection == _VERBOSITY_SILENT:
        return "silent"
    return "default"


def _ask_client_sidecar() -> str | None:
    answer = questionary.text(
        "Optional client-supplied sidecar CSV to merge in:",
        instruction=(
            "(Local path or s3:// URI to a CSV, or a directory whose "
            "subfolders hold the CSVs to merge; press Enter to skip)"
        ),
    ).unsafe_ask()
    answer = answer.strip()
    return answer or None


_SCHEMA_CUSTOM = "Custom path or URI…"
_SCHEMA_SKIP = "Skip"


def _validate_schema_input(v: str) -> bool | str:
    v = v.strip()
    if not v:
        return "Required."
    if is_remote_uri(v):
        return True
    return Path(v).expanduser().is_file() or "File not found (and not a remote URI)."


def ask_schema_uri(prompt: str = "Path or URI to schema JSON:") -> str:
    """Prompt for a schema path or URI; resolve local paths, pass URIs through."""
    answer = (
        questionary.text(
            prompt,
            instruction="(Local path or s3://… / gs://… / az://… URI)",
            validate=_validate_schema_input,
        )
        .unsafe_ask()
        .strip()
    )
    if is_remote_uri(answer):
        return answer
    return str(Path(answer).expanduser().resolve())


def _ask_client_schema() -> str | None:
    schemas = _list_local_schemas()
    choices = [str(p.name) for p in schemas] + [_SCHEMA_CUSTOM, _SCHEMA_SKIP]
    selection = questionary.select(
        "Optional client schema (normalises a client-supplied sidecar):",
        choices=choices,
        default=_SCHEMA_SKIP,
    ).unsafe_ask()
    if selection == _SCHEMA_SKIP:
        return None
    if selection == _SCHEMA_CUSTOM:
        return ask_schema_uri()
    return str(next(p for p in schemas if p.name == selection))


def _ask_spatial_reference() -> str | None:
    answer = questionary.text(
        "Optional spatial reference (CRS) of the source coordinates:",
        instruction=(
            "(e.g. 'EPSG:32612' or '32612'. For images/videos lat/lon/altitude "
            "are reprojected to EPSG:4326; for point clouds it's recorded as a "
            "column. Press Enter to skip)"
        ),
    ).unsafe_ask()
    answer = answer.strip()
    return answer or None


def _disambiguate_filenames(keys: list[str]) -> dict[str, str]:
    """Return a per-key display label, extended with the minimum number of
    parent directories needed to be unique across ``keys``.
    """
    parts_per_key: dict[str, tuple[str, ...]] = {key: _split_path_components(key) for key in keys}
    depths: dict[str, int] = {key: 1 for key in keys}

    while True:
        labels = {key: "/".join(parts[-depths[key] :]) for key, parts in parts_per_key.items()}
        counts: dict[str, int] = defaultdict(int)
        for label in labels.values():
            counts[label] += 1
        collisions = {label for label, n in counts.items() if n > 1}
        if not collisions:
            return labels
        bumped = False
        for key, label in labels.items():
            if label in collisions and depths[key] < len(parts_per_key[key]):
                depths[key] += 1
                bumped = True
        if not bumped:
            return labels


def _split_gps_position(meta: dict) -> dict:
    """Split a ``GPSPosition`` value into ``GPSLatitude``/``GPSLongitude``.

    exiftool emits ``GPSPosition`` as ``"<lat>, <lon>"`` (DMS or decimal),
    or as a 2-element list when the ``-n`` flag is in play. When a file or
    client sidecar provides ``GPSPosition``, treat it as authoritative and
    overwrite ``GPSLatitude``/``GPSLongitude`` with the split halves.
    Returns `meta` unchanged if ``GPSPosition`` is absent or unparseable.
    """
    raw = meta.get("GPSPosition")
    if raw is None:
        return meta
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            return meta
        lat, lon = (str(part).strip() for part in raw)
    else:
        text = str(raw)
        if "," not in text:
            return meta
        lat, lon = (part.strip() for part in text.split(",", 1))
    if not lat or not lon:
        return meta
    out = {k: v for k, v in meta.items() if k != "GPSPosition"}
    out["GPSLatitude"] = lat
    out["GPSLongitude"] = lon
    return out


_DATE_ALIASES = {"CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"}


def _date_group(required_field_groups: list[list[str]]) -> list[str] | None:
    for group in required_field_groups:
        if set(group) & _DATE_ALIASES:
            return group
    return None


def _fill_missing_dates_from_filepath(
    file_metadata: list[tuple[str, dict]],
    required_field_groups: list[list[str]],
    display_to_key: dict[str, str],
) -> None:
    """For each file with no date field after merge, infer one from its path."""
    date_group = _date_group(required_field_groups)
    if date_group is None:
        return
    canonical = date_group[0]
    for display_label, meta in file_metadata:
        if any(has_value(meta.get(field)) for field in date_group):
            continue
        original = display_to_key.get(display_label, display_label)
        lat = to_decimal_degree(meta.get("GPSLatitude"))
        lon = to_decimal_degree(meta.get("GPSLongitude"))
        inferred = infer_date_from_filepath(original, lat, lon)
        if inferred is None:
            continue
        formatted = inferred.isoformat()
        meta[canonical] = formatted
        logger.warning(
            f"Inferred {canonical} for {display_label!r} from file path: "
            f"{formatted}. No date was provided by EXIF or the client sidecar."
        )


def _warn_missing_required_fields(
    file_metadata: list[tuple[str, dict]],
    required_field_groups: list[list[str]],
) -> None:
    """Emit a loud warning when files lack required fields after the merge.

    A group is "satisfied" for a file if any field in the group is present
    (non-empty) in that file's metadata. For each unsatisfied group we log a
    structured WARNING and also print a bright-red message to stderr so the
    operator notices that the client sidecar needs to be updated.
    """
    if not file_metadata or not required_field_groups:
        return

    missing_by_group: list[tuple[str, list[str], list[str]]] = []
    for group in required_field_groups:
        canonical = group[0]
        missing_files = [
            filename
            for filename, meta in file_metadata
            if not any(has_value(meta.get(field)) for field in group)
        ]
        if missing_files:
            missing_by_group.append((canonical, list(group), missing_files))

    if not missing_by_group:
        return

    total = len(file_metadata)
    logger.warning("Missing required metadata after client sidecar merge — see details below.")

    header = (
        f"MISSING REQUIRED METADATA: {len(missing_by_group)} required field(s) "
        f"are not satisfied for every file. Update the client sidecar to "
        f"provide values for the listed files."
    )
    click.secho(header, fg="bright_red", bold=True, err=True, color=True)

    for canonical, group, missing_files in missing_by_group:
        accepted = ", ".join(group)
        sample = missing_files if len(missing_files) <= 5 else missing_files[:5] + ["…"]
        line = (
            f"  [{canonical}] missing on {len(missing_files)}/{total} file(s): "
            f"{sample} (accepted aliases: {accepted})"
        )
        click.secho(line, fg="bright_red", err=True, color=True)


def _canonicalize_keys(meta: dict, alias_to_canonical: dict[str, str]) -> dict:
    """Rename alias keys in `meta` to their canonical names. If the canonical
    is already present, the alias is dropped; among multiple aliases for the
    same canonical, the first-encountered one wins.
    """
    out: dict = {key: value for key, value in meta.items() if key not in alias_to_canonical}
    for key, value in meta.items():
        if key in alias_to_canonical:
            out.setdefault(alias_to_canonical[key], value)
    return out


def build_sidecar_df(
    file_metadata: list[tuple[str, dict]],
    required_field_groups: list[list[str]] | None = None,
    full: bool = False,
):
    """Assemble the sidecar DataFrame.

    Layout:
      - Row 0  : DEFAULT row — Filename="DEFAULT", all other columns empty
      - Row 1+ : one row per file with extracted metadata values, sorted by Filename
      - Col 0  : "Filename" (basename of the source key/path)

    Required field groups are checked per-file: if any file lacks all alternatives
    for a group, the canonical (first) field name is prepended as a blank column.

    When `full` is False, only columns belonging to `required_field_groups` are
    kept; everything else extracted from the source files is dropped.
    """
    import pandas as pd

    if not file_metadata:
        return pd.DataFrame(columns=["Filename"])

    alias_to_canonical = build_global_alias_map(required_field_groups or [])
    file_metadata = [
        (filename, _canonicalize_keys(_split_gps_position(meta), alias_to_canonical))
        for filename, meta in file_metadata
    ]

    all_cols: list[str] = []
    seen: set[str] = set()
    for _, meta in file_metadata:
        for col in meta:
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    if not full:
        allowed = {field for group in (required_field_groups or []) for field in group}
        all_cols = [c for c in all_cols if c in allowed]

    prepend_cols: list[str] = []
    if required_field_groups:
        for group in required_field_groups:
            canonical = group[0]
            all_covered = all(any(field in meta for field in group) for _, meta in file_metadata)
            if not all_covered and canonical not in prepend_cols:
                prepend_cols.append(canonical)

    all_cols = prepend_cols + [c for c in all_cols if c not in prepend_cols]
    columns = ["Filename", *all_cols]

    rows = [
        {"Filename": filename, **{col: meta.get(col, "") for col in all_cols}}
        for filename, meta in file_metadata
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values(by="Filename", kind="stable", ignore_index=True)

    default_row = pd.DataFrame([{"Filename": "DEFAULT", **{col: "" for col in all_cols}}])
    return pd.concat([default_row, df], ignore_index=True)


def _add_spatial_reference_column(df, spatial_reference: str) -> None:
    """Add a 'spatial_reference' column, populated only on the DEFAULT row."""
    df["spatial_reference"] = ""
    df.loc[df["Filename"] == "DEFAULT", "spatial_reference"] = spatial_reference


def _reproject_dataframe(df, in_srs: str) -> None:
    """Reproject GPSLongitude/GPSLatitude (and GPSAltitude as Z) from `in_srs`
    to EPSG:4326 in place, skipping the DEFAULT row and unparseable coordinates.
    """
    for idx in df.index:
        if df.at[idx, "Filename"] == "DEFAULT":
            continue
        lon = to_decimal_degree(df.at[idx, "GPSLongitude"])
        lat = to_decimal_degree(df.at[idx, "GPSLatitude"])
        if lon is None or lat is None:
            continue  # leave blank/unparseable rows untouched
        z = parse_elevation(df.at[idx, "GPSAltitude"]) if "GPSAltitude" in df.columns else None
        if z is None:
            nx, ny = transform_coordinates(lon, lat, in_srs, 4326)
        else:
            nx, ny, nz = transform_coordinates(lon, lat, in_srs, 4326, z=z)
            df.at[idx, "GPSAltitude"] = str(nz)
        df.at[idx, "GPSLongitude"] = str(nx)
        df.at[idx, "GPSLatitude"] = str(ny)


def _generate(
    directory: str,
    data_type: DataTypeEnum,
    output_filename: str,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool = False,
    local_copy: bool = False,
    spatial_reference: str | None = None,
) -> str:
    logger.info(
        f"Starting sidecar CSV generation — directory={directory!r} "
        f"data_type={data_type} output={output_filename!r} "
        f"client_sidecar={client_sidecar!r} client_schema={client_schema!r} "
        f"full={full} local_copy={local_copy} spatial_reference={spatial_reference!r}"
    )

    if spatial_reference:
        # Fail fast on an unparseable CRS before doing any extraction work.
        from pyproj import CRS
        from pyproj.exceptions import CRSError

        try:
            CRS(spatial_reference)
        except CRSError:
            raise ValueError(
                f"--spatial-reference value {spatial_reference!r} is not a valid "
                "spatial reference (try e.g. 'EPSG:32612' or '32612')."
            ) from None

    if data_type == DataTypeEnum.vector:
        raise NotImplementedError(
            "Vector data type is not yet supported for sidecar CSV generation."
        )

    if data_type not in DATA_TYPE_INFO:
        raise ValueError(
            f"Unsupported data_type '{data_type}'. Valid values: {list(DATA_TYPE_INFO.keys())}"
        )

    backend = from_directory(directory)
    info = DATA_TYPE_INFO[data_type]
    include = list(info.get("include") or [])
    exclude = list(info.get("exclude") or [])

    keys = backend.list_keys(include=include, exclude=exclude)
    if not keys:
        raise RuntimeError(f"No valid files found for '{data_type}' in {backend.display_root}")

    total = len(keys)
    logger.info(f"Found {total} file(s) to process in {backend.display_root}.")

    is_image_or_video = data_type in (_IMAGE_DATA_TYPES | _VIDEO_DATA_TYPES)
    is_point_cloud = data_type in _POINT_CLOUD_DATA_TYPES

    if not (is_image_or_video or is_point_cloud):
        raise ValueError(f"Unhandled data type for metadata extraction: {data_type}")

    display_labels = _disambiguate_filenames(keys)
    display_to_key = {v: k for k, v in display_labels.items()}
    n_disambiguated = sum(1 for k, v in display_labels.items() if v != Path(k).name)
    if n_disambiguated:
        logger.info(
            f"Disambiguated {n_disambiguated} file(s) whose basenames collided "
            f"by prepending parent directories."
        )

    file_metadata: list[tuple[str, dict]] = []
    empty_count = 0

    for i, key in enumerate(keys, 1):
        display_label = display_labels[key]
        tool = "EXIF" if is_image_or_video else "PDAL"
        logger.info(f"[{i}/{total}] Extracting {tool}: {key}")
        with backend.open_local(key) as local_path:
            if is_image_or_video:
                meta = extract_exif_metadata(local_path, filename=display_label)
            else:
                meta = extract_pdal_metadata(local_path, filename=display_label)
        if not meta:
            empty_count += 1
            logger.warning(f"No {tool} metadata returned for {key}")
        file_metadata.append((display_label, meta))

    if empty_count:
        logger.warning(
            f"{empty_count}/{total} file(s) produced empty metadata — "
            "those rows will have blank fields in the sidecar."
        )

    if client_sidecar:
        client_url = client_sidecar.strip()
        if not client_url:
            raise ValueError("--client-sidecar value cannot be empty.")
        logger.info(f"Merging client sidecar from {client_url}")
        client_df = load_and_clean_client_sidecar(
            url=client_url,
            schema_path=client_schema,
            required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
        )
        merge_client_metadata(file_metadata, client_df)

    # All groups (required + optional) drive canonicalization, column selection,
    # the DataFrame build, and date inference; only the loud "missing required
    # metadata" warning is restricted to truly-required groups.
    field_groups = _ALL_SIDECAR_FIELD_GROUPS.get(data_type, [])
    required_field_groups = list(_REQUIRED_SIDECAR_FIELD_GROUPS.get(data_type, []))
    _fill_missing_dates_from_filepath(file_metadata, field_groups, display_to_key)

    logger.info(f"Building sidecar DataFrame for {len(file_metadata)} file(s).")
    _warn_missing_required_fields(file_metadata, required_field_groups)
    df = build_sidecar_df(
        file_metadata,
        required_field_groups=field_groups,
        full=full,
    )
    logger.info(
        f"Sidecar DataFrame: {len(df)} row(s) (including DEFAULT), {len(df.columns)} column(s)."
    )

    if spatial_reference:
        if is_point_cloud:
            logger.info(
                f"Recording spatial_reference={spatial_reference!r} in the DEFAULT row."
            )
            _add_spatial_reference_column(df, spatial_reference)
        elif is_image_or_video:
            logger.info(f"Reprojecting coordinates from {spatial_reference!r} to EPSG:4326.")
            _reproject_dataframe(df, spatial_reference)

    local_copy_path: Path | None = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_csv = Path(tmp_dir) / output_filename
        df.to_csv(local_csv, index=False)
        logger.info(f"Writing sidecar CSV to {backend.display_root}/{output_filename}")
        sidecar_path = backend.write_output(output_filename, local_csv)
        if local_copy:
            local_copy_path = Path.cwd() / output_filename
            shutil.copy2(local_csv, local_copy_path)

    logger.info(f"Sidecar CSV written: {sidecar_path}")
    click.secho(f"\nSidecar written to: {sidecar_path}", fg="green", bold=True, err=True)
    if local_copy_path is not None:
        logger.info(f"Local copy written to: {local_copy_path}")
        click.secho(f"Local copy at: {local_copy_path}", fg="green", bold=True, err=True)

    return sidecar_path


def _format_replay_command(
    directory: str,
    data_type: DataTypeEnum,
    output_filename: str,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool,
    verbosity: "VerbosityChoice",
    local_copy: bool,
    spatial_reference: str | None,
) -> str:
    parts = ["am-tools"]
    if verbosity == "verbose":
        parts.append("--verbose")
    elif verbosity == "silent":
        parts.append("--silent")
    parts += [
        "sidecar",
        "generate",
        "--directory",
        shlex.quote(directory),
        "--datatype",
        data_type.value,
        "--output-filename",
        shlex.quote(output_filename),
    ]
    if client_sidecar:
        parts += ["--client-sidecar", shlex.quote(client_sidecar)]
    if client_schema is not None:
        parts += ["--client-schema", shlex.quote(client_schema)]
    if full:
        parts.append("--full")
    if local_copy:
        parts.append("--local-copy")
    if spatial_reference:
        parts += ["--spatial-reference", shlex.quote(spatial_reference)]
    return " ".join(parts)


def _echo_replay_command(command: str) -> None:
    click.secho(
        "\nEquivalent command (copy & paste to skip the wizard next time):",
        fg="cyan",
        err=True,
    )
    click.secho(f"  {command}\n", fg="bright_cyan", bold=True, err=True)


def _run_interactive_wizard(
    directory: str | None,
    data_type: DataTypeEnum | None,
    output_filename: str | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool,
    full_provided: bool,
    verbosity: "VerbosityChoice",
    verbosity_provided: bool,
    local_copy: bool,
    local_copy_provided: bool,
    spatial_reference: str | None,
) -> tuple[
    str, DataTypeEnum, str | None, str | None, str | None, bool, "VerbosityChoice", bool, str | None
]:
    """Prompt for any value the user didn't pass on the CLI. Returns the
    final (directory, data_type, output_filename, client_sidecar, client_schema, full,
    verbosity, local_copy, spatial_reference).
    """
    questionary.print(
        "Interactive mode — press Ctrl+C at any time to cancel.\n",
        style="fg:ansibrightblack",
    )
    try:
        if directory is None:
            directory = _ask_directory()
        if data_type is None:
            data_type = _ask_data_type()
        if output_filename is None:
            output_filename = _ask_output_filename()
        if client_sidecar is None:
            client_sidecar = _ask_client_sidecar()
        if client_schema is None and client_sidecar is not None:
            client_schema = _ask_client_schema()
        if spatial_reference is None:
            spatial_reference = _ask_spatial_reference()
        if not full_provided:
            full = _ask_full()
        if not local_copy_provided and any(
            is_remote_uri(p) for p in (directory, client_sidecar, client_schema)
        ):
            local_copy = _ask_local_copy()
        if not verbosity_provided:
            verbosity = _ask_verbosity()
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    return (
        directory,
        data_type,
        output_filename,
        client_sidecar,
        client_schema,
        full,
        verbosity,
        local_copy,
        spatial_reference,
    )


@sidecar_app.command()
def generate(
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
        DataTypeEnum | None,
        typer.Option(
            "--datatype",
            "--data-type",
            help="Data type of the input data (e.g. 'oriented_image', 'point_cloud').",
            case_sensitive=False,
        ),
    ] = None,
    output_filename: Annotated[
        str | None,
        typer.Option(help="Filename for the generated sidecar CSV."),
    ] = None,
    client_sidecar: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional path to client-supplied sidecar data. May be an "
                "object-store URI (s3://bucket/key/file.csv) or a local path. "
                "Point it at a single CSV, or at a directory: when it's a "
                "directory, every CSV in a subdirectory BELOW it is merged into "
                "one (the directory itself is not scanned, since the generated "
                "sidecar is written there). All merged CSVs must share the same "
                "schema (column count); a mismatch aborts and names the bad "
                "file. Values are merged into the generated sidecar; client "
                "values win on conflict."
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
    local_copy: Annotated[
        bool,
        typer.Option(
            "--local-copy/--no-local-copy",
            help=(
                "Also write a copy of the sidecar to the current working "
                "directory. Useful when the input directory or client sidecar "
                "is in object storage and you want a local copy to inspect. "
                "When unset, the wizard prompts only if a remote (s3://, "
                "gs://, az://) path was given."
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
                "For images/videos, lat/lon (and altitude) are treated as X/Y/Z "
                "in this CRS and reprojected to EPSG:4326. For point clouds, a "
                "'spatial_reference' column is added with this value in the "
                "DEFAULT row."
            ),
        ),
    ] = None,
) -> None:
    """Scan a directory, extract per-file metadata, and write a sidecar CSV.

    With no flags, prompts interactively for each value. Any flag that is
    passed on the command line is used as-is and not prompted for.
    """
    from atomic_tools.cli import Verbosity, level_for

    verbosity_state: Verbosity = ctx.ensure_object(Verbosity)
    verbosity_choice = verbosity_state.choice
    verbosity_provided = verbosity_state.verbose or verbosity_state.silent

    wizard_ran = directory is None or data_type is None
    if wizard_ran:
        full_provided = ctx.get_parameter_source("full") == click.core.ParameterSource.COMMANDLINE
        local_copy_provided = (
            ctx.get_parameter_source("local_copy") == click.core.ParameterSource.COMMANDLINE
        )
        (
            directory,
            data_type,
            output_filename,
            client_sidecar,
            client_schema,
            full,
            verbosity_choice,
            local_copy,
            spatial_reference,
        ) = _run_interactive_wizard(
            directory,
            data_type,
            output_filename,
            client_sidecar,
            client_schema,
            full,
            full_provided,
            verbosity_choice,
            verbosity_provided,
            local_copy,
            local_copy_provided,
            spatial_reference,
        )
        logging.getLogger().setLevel(
            level_for(
                verbose=verbosity_choice == "verbose",
                silent=verbosity_choice == "silent",
            )
        )

    output_filename = output_filename or "sidecar.csv"
    basename = Path(output_filename).name
    if basename != output_filename:
        logger.warning(
            f"--output-filename ignored directory portion of {output_filename!r}; "
            f"writing as {basename!r} in the input directory."
        )
        output_filename = basename

    if not directory:
        raise typer.BadParameter("Directory is required.")
    if data_type is None:
        raise typer.BadParameter("Data type is required.")

    if wizard_ran:
        _echo_replay_command(
            _format_replay_command(
                directory=directory,
                data_type=data_type,
                output_filename=output_filename,
                client_sidecar=client_sidecar,
                client_schema=client_schema,
                full=full,
                verbosity=verbosity_choice,
                local_copy=local_copy,
                spatial_reference=spatial_reference,
            )
        )

    try:
        sidecar_path = _generate(
            directory=directory,
            data_type=data_type,
            output_filename=output_filename,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
            local_copy=local_copy,
            spatial_reference=spatial_reference,
        )
    except Exception as e:
        _fail("Failed to generate sidecar CSV", e)

    logger.info("Linting generated sidecar…")
    try:
        report = lint_sidecar_file(
            sidecar_path,
            final=True,
            data_type=data_type,
            schema_path=None,
            input_files_path=directory,
        )
    except Exception as e:
        _fail("Failed to lint generated sidecar", e)

    typer.echo(report.render())
    if report.has_errors():
        raise typer.Exit(code=1)
