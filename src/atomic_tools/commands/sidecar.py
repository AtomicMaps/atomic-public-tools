"""Generate a sidecar CSV by scanning a local or remote directory.

Originally `tasks/generate_sidecar_csv/generate_sidecar_csv.py`.

Requires the following external binaries on PATH at runtime (NOT pip-installable):

  * exiftool
  * pdal  (the conda binary at /opt/conda/envs/pdal/bin/pdal — see
          ``utils/extractors.py``; adjust if the client environment uses a
          different path)
"""

import logging
import tempfile
from pathlib import Path
from typing import Annotated

import click
import questionary
import typer

from atomic_tools.client_sidecar import (
    build_global_alias_map,
    load_and_clean_client_sidecar,
    merge_client_metadata,
)
from atomic_tools.io.storage import from_directory
from atomic_tools.utils.extractors import (
    extract_exif_metadata,
    extract_pdal_metadata,
)
from atomic_tools.utils.utils import (
    DATA_TYPE_INFO,
    DataTypeEnum,
)

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

# Directory where the sidecar-path pointer file is written. None falls back to
# the current working directory. Holder for a future CLI option.
SIDECAR_PATH_OUTPUT_DIR: Path | None = None

# Each inner list: [canonical_name, *alternatives].
# A file "satisfies" a group if it has any field from the group in its metadata.
# If any file fails to satisfy a group, the canonical field is prepended as a
# blank column.
_REQUIRED_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    DataTypeEnum.oriented_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
        [
            "Pitch",
            "CameraPitch",
            "CameraPitchDegree",
            "GimbalPitchDegree",
            "PosePitchDegrees",
            "CameraOrientationNEDPitch",
            "GPSIMUPitch",
            "PitchAngle",
        ],
        [
            "Heading",
            "Yaw",
            "CameraYaw",
            "CameraYawDegree",
            "GimbalYawDegree",
            "PoseHeadingDegrees",
            "CameraOrientationNEDYaw",
            "GPSIMUYaw",
            "YawAngle",
            "GPSImgDirection",
            "imgDirection",
        ],
        [
            "Roll",
            "CameraRoll",
            "CameraRollDegree",
            "GimbalRollDegree",
            "PoseRollDegrees",
            "CameraOrientationNEDRoll",
            "GPSIMURoll",
            "RollAngle",
        ],
    ],
    DataTypeEnum.spherical_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
        ["Pitch", "CameraPitch", "GimbalPitchDegree", "PosePitchDegrees"],
        ["Heading", "Yaw", "GimbalYawDegree", "PoseHeadingDegrees", "GPSImgDirection"],
        ["Roll", "CameraRoll", "GimbalRollDegree", "PoseRollDegrees"],
    ],
    DataTypeEnum.ortho_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        ["DateTimeOriginal", "CreateDate", "ModifyDate", "GPSDateStamp"],
    ],
    DataTypeEnum.video: [
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
    ],
    DataTypeEnum.point_cloud: [
        ["bounds.minx"],
        ["bounds.miny"],
        ["bounds.maxx"],
        ["bounds.maxy"],
        ["bounds.minz"],
        ["bounds.maxz"],
        ["num_points"],
        ["creation_year"],
        ["creation_doy"],
    ],
}


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


def _ask_client_sidecar() -> str | None:
    answer = questionary.text(
        "Optional client-supplied sidecar CSV to merge in:",
        instruction="(Local path or s3:// URI; press Enter to skip)",
    ).unsafe_ask()
    answer = answer.strip()
    return answer or None


_SCHEMA_CUSTOM = "Custom path…"
_SCHEMA_SKIP = "Skip"


def _ask_client_schema() -> Path | None:
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
        custom = questionary.path(
            "Path to schema JSON:",
            validate=lambda v: Path(v).expanduser().is_file() or "File not found.",
        ).unsafe_ask()
        return Path(custom).expanduser().resolve()
    return next(p for p in schemas if p.name == selection)


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

    def _has_value(meta: dict, field: str) -> bool:
        s = str(meta.get(field, "")).strip()
        return bool(s) and s.lower() != "nan"

    missing_by_group: list[tuple[str, list[str], list[str]]] = []
    for group in required_field_groups:
        canonical = group[0]
        missing_files = [
            filename
            for filename, meta in file_metadata
            if not any(_has_value(meta, field) for field in group)
        ]
        if missing_files:
            missing_by_group.append((canonical, list(group), missing_files))

    if not missing_by_group:
        return

    total = len(file_metadata)
    logger.warning(
        "Missing required metadata after client sidecar merge — see details below."
    )

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
    out: dict = {
        key: value for key, value in meta.items() if key not in alias_to_canonical
    }
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
            all_covered = all(
                any(field in meta for field in group) for _, meta in file_metadata
            )
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

    default_row = pd.DataFrame(
        [{"Filename": "DEFAULT", **{col: "" for col in all_cols}}]
    )
    return pd.concat([default_row, df], ignore_index=True)


def _generate(
    directory: str,
    data_type: DataTypeEnum,
    output_filename: str,
    client_sidecar: str | None,
    client_schema: Path | None,
    full: bool = False,
) -> str:
    logger.info(
        f"Starting sidecar CSV generation — directory={directory!r} "
        f"data_type={data_type} output={output_filename!r} "
        f"client_sidecar={client_sidecar!r} client_schema={client_schema!r} "
        f"full={full}"
    )

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
        raise RuntimeError(
            f"No valid files found for '{data_type}' in {backend.display_root}"
        )

    total = len(keys)
    logger.info(f"Found {total} file(s) to process in {backend.display_root}.")

    is_image_or_video = data_type in (_IMAGE_DATA_TYPES | _VIDEO_DATA_TYPES)
    is_point_cloud = data_type in _POINT_CLOUD_DATA_TYPES

    if not (is_image_or_video or is_point_cloud):
        raise ValueError(f"Unhandled data type for metadata extraction: {data_type}")

    file_metadata: list[tuple[str, dict]] = []
    empty_count = 0

    for i, key in enumerate(keys, 1):
        basename = Path(key).name
        label = "EXIF" if is_image_or_video else "PDAL"
        logger.info(f"[{i}/{total}] Extracting {label}: {key}")
        with backend.open_local(key) as local_path:
            if is_image_or_video:
                meta = extract_exif_metadata(local_path, filename=basename)
            else:
                meta = extract_pdal_metadata(local_path, filename=basename)
        if not meta:
            empty_count += 1
            logger.warning(f"No {label} metadata returned for {key}")
        file_metadata.append((basename, meta))

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
            required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
        )
        merge_client_metadata(file_metadata, client_df)

    logger.info(f"Building sidecar DataFrame for {len(file_metadata)} file(s).")
    required_field_groups = list(_REQUIRED_SIDECAR_FIELD_GROUPS.get(data_type, []))
    _warn_missing_required_fields(file_metadata, required_field_groups)
    df = build_sidecar_df(
        file_metadata,
        required_field_groups=required_field_groups,
        full=full,
    )
    logger.info(
        f"Sidecar DataFrame: {len(df)} row(s) (including DEFAULT), {len(df.columns)} column(s)."
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_csv = Path(tmp_dir) / output_filename
        df.to_csv(local_csv, index=False)
        logger.info(f"Writing sidecar CSV to {backend.display_root}/{output_filename}")
        sidecar_path = backend.write_output(output_filename, local_csv)

    logger.info(f"Sidecar CSV written: {sidecar_path}")

    # Best-effort pointer file — failures only warn.
    pointer_dir = SIDECAR_PATH_OUTPUT_DIR or Path.cwd()
    pointer_path = pointer_dir / "sidecar_csv_path"
    try:
        pointer_dir.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(sidecar_path)
        logger.info(f"Wrote sidecar path pointer to {pointer_path}")
    except Exception:
        logger.warning(
            f"Could not write sidecar path pointer to {pointer_path}. "
            f"sidecar_csv_path={sidecar_path}"
        )

    return sidecar_path


def _run_interactive_wizard(
    directory: str | None,
    data_type: DataTypeEnum | None,
    output_filename: str | None,
    client_sidecar: str | None,
    client_schema: Path | None,
    full: bool,
    full_provided: bool,
) -> tuple[str, DataTypeEnum, str | None, str | None, Path | None, bool]:
    """Prompt for any value the user didn't pass on the CLI. Returns the
    final (directory, data_type, output_filename, client_sidecar, client_schema, full).
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
        if not full_provided:
            full = _ask_full()
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    return directory, data_type, output_filename, client_sidecar, client_schema, full


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
                "Optional path to a client-supplied sidecar CSV. May be an "
                "object-store URI (s3://bucket/key/file.csv) or a local path. "
                "Values are merged into the generated sidecar; client values "
                "win on conflict."
            ),
        ),
    ] = None,
    client_schema: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help=(
                "Optional path to a client-supplied JSON schema describing how to "
                "normalise the client sidecar CSV (headerless column names, "
                "per-client renames). See schemas/example.json. If omitted, the "
                "client CSV is used as-is with no renames."
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
) -> None:
    """Scan a directory, extract per-file metadata, and write a sidecar CSV.

    With no flags, prompts interactively for each value. Any flag that is
    passed on the command line is used as-is and not prompted for.
    """
    # Configure logging at command entry only — keeps the library importable
    # without imposing a global logging config on consumers.
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

    if directory is None or data_type is None:
        full_provided = (
            ctx.get_parameter_source("full") == click.core.ParameterSource.COMMANDLINE
        )
        directory, data_type, output_filename, client_sidecar, client_schema, full = (
            _run_interactive_wizard(
                directory,
                data_type,
                output_filename,
                client_sidecar,
                client_schema,
                full,
                full_provided,
            )
        )

    output_filename = output_filename or "sidecar.csv"

    if not directory:
        raise typer.BadParameter("Directory is required.")
    if data_type is None:
        raise typer.BadParameter("Data type is required.")

    try:
        _generate(
            directory=directory,
            data_type=data_type,
            output_filename=output_filename,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
        )
    except Exception:
        logger.exception("Failed to generate sidecar CSV")
        raise typer.Exit(code=1) from None
