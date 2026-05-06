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
        ["GPSLatitude", "GPSPosition"],
        ["GPSLongitude", "GPSPosition"],
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
        ["GPSLatitude", "GPSPosition"],
        ["GPSLongitude", "GPSPosition"],
        ["GPSAltitude"],
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
        ["Pitch", "CameraPitch", "GimbalPitchDegree", "PosePitchDegrees"],
        ["Heading", "Yaw", "GimbalYawDegree", "PoseHeadingDegrees", "GPSImgDirection"],
        ["Roll", "CameraRoll", "GimbalRollDegree", "PoseRollDegrees"],
    ],
    DataTypeEnum.ortho_image: [
        ["GPSLatitude", "GPSPosition"],
        ["GPSLongitude", "GPSPosition"],
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
        (filename, _canonicalize_keys(meta, alias_to_canonical))
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


@sidecar_app.command()
def generate(
    directory: Annotated[
        str,
        typer.Option(
            ...,
            help=(
                "Directory to scan. Either an object-store URI "
                "(s3://bucket/prefix, gs://..., az://...) or a local filesystem path."
            ),
        ),
    ],
    data_type: Annotated[
        DataTypeEnum,
        typer.Option(
            ...,
            help="Data type of the input data (e.g. 'oriented_image', 'point_cloud').",
            case_sensitive=False,
        ),
    ],
    output_filename: Annotated[
        str,
        typer.Option(help="Filename for the generated sidecar CSV."),
    ] = "sidecar.csv",
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
            "--full/--no-full",
            help=(
                "Include every metadata field extracted from each file. By "
                "default, only the canonical/required fields for the data type "
                "are kept (plus blank columns for any missing required fields)."
            ),
        ),
    ] = False,
) -> None:
    """Scan a directory, extract per-file metadata, and write a sidecar CSV."""
    # Configure logging at command entry only — keeps the library importable
    # without imposing a global logging config on consumers.
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

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
