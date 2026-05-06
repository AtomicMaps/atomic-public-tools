"""Generate a sidecar CSV by scanning an S3 directory and extracting metadata.

Originally `tasks/generate_sidecar_csv/generate_sidecar_csv.py`.

Requires the following external binaries on PATH at runtime (NOT pip-installable):

  * exiftool
  * pdal  (the conda binary at /opt/conda/envs/pdal/bin/pdal — see ``_PDAL_BIN``
          below; adjust if the client environment uses a different path)
"""

import contextlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import obstore
import typer

from atomic_tools.client_sidecar import (
    load_and_clean_client_sidecar,
    merge_client_metadata,
)
from atomic_tools.utils.object_store import ObjectStore, ObstoreBackend
from atomic_tools.utils.utils import (
    DATA_TYPE_INFO,
    DataTypeEnum,
    download,
    get_object_keys,
    normalize_object_store_path,
    run_exiftool,
    upload,
)

logger = logging.getLogger(__name__)

sidecar_app = typer.Typer(
    no_args_is_help=True,
    help="Generate a sidecar CSV by scanning an S3 directory.",
)

# exiftool fields that describe the tool/file-system rather than the asset
_EXIFTOOL_NOISE_FIELDS = {
    "SourceFile",
    "ExifToolVersion",
    "FileName",
    "Directory",
    "FileSize",
    "FilePermissions",
    "FileAccessDate",
    "FileInodeChangeDate",
    "FileModifyDate",
    "FileType",
    "FileTypeExtension",
    "MIMEType",
}

_IMAGE_DATA_TYPES = {
    DataTypeEnum.ortho_image,
    DataTypeEnum.oriented_image,
    DataTypeEnum.spherical_image,
}
_VIDEO_DATA_TYPES = {DataTypeEnum.video}
_POINT_CLOUD_DATA_TYPES = {DataTypeEnum.point_cloud}

_PDAL_BIN = "/opt/conda/envs/pdal/bin/pdal"
# LAS header stores bounds as flat keys; remap to bounds.X for downstream.
_LAS_BOUNDS_DIMS = frozenset({"minx", "miny", "maxx", "maxy", "minz", "maxz"})

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


def extract_exif_metadata(store: ObstoreBackend, key: str, filename: str) -> dict:
    """Extract EXIF/XMP/GPS metadata from an image or video via obstore.

    Downloads to a local temp file so exiftool has reliable random-access.
    Flags match BaseImage.run_exiftool so values are byte-identical to a
    re-read of the source image (the sidecar override step compares them).
    """
    try:
        suffix = Path(filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            logger.info(f"Downloading {key} from object store")
            data = obstore.get(store, key).bytes()
            logger.info(f"Downloaded {len(data):,} bytes for {filename}")
            with open(tmp_path, "wb") as f:
                f.write(data)
            logger.info(f"Running exiftool on {filename}")
            raw = run_exiftool(tmp_path, extra_args=["-fast", "-b", "-j"])
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        records = json.loads(raw.decode("utf-8", errors="replace"))
        if not records:
            logger.info(f"exiftool returned no records for {filename}")
            return {}
        record = records[0]
        cleaned = {}
        for tag, value in record.items():
            if tag in _EXIFTOOL_NOISE_FIELDS:
                continue
            cleaned[tag] = value if not isinstance(value, (dict, list)) else json.dumps(value)
        logger.info(f"Extracted {len(cleaned)} EXIF field(s) from {filename}")
        return cleaned
    except subprocess.CalledProcessError as exc:
        logger.warning(f"EXIF extraction failed (exiftool error) for {filename}: {exc}")
        return {}
    except Exception as exc:
        logger.warning(f"EXIF extraction failed for {filename}: {exc}")
        return {}


def _flatten_dict(data: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict, joining keys with '.'."""
    result = {}
    for k, v in data.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=full_key))
        elif isinstance(v, list):
            result[full_key] = json.dumps(v) if v else ""
        else:
            result[full_key] = v
    return result


def extract_pdal_metadata(store: ObstoreBackend, key: str, filename: str) -> dict:
    """Extract header metadata from a point cloud file via the pdal info CLI.

    Downloads to a local temp file first; the conda pdal build cannot read
    s3:// URIs directly. Calls the pdal binary rather than the Python bindings
    to avoid C++ ABI incompatibilities between the conda env and system libstdc++.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info(f"Downloading {key} from object store")
            local_path = download(store, key, destination_dir=tmp_dir)
            logger.info(f"Running pdal info for {filename}")
            proc = subprocess.run(
                [_PDAL_BIN, "info", "--metadata", local_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pdal info exited with code {proc.returncode}: {proc.stderr.strip()}"
                )
            data = json.loads(proc.stdout)
            meta = data.get("metadata", {})

            out: dict = {}
            remaining: dict = {}
            for k, v in meta.items():
                if k in _LAS_BOUNDS_DIMS:
                    out[f"bounds.{k}"] = v
                else:
                    remaining[k] = v
            out.update(_flatten_dict(remaining))

            # `pdal info --metadata` emits the LAS header point count as
            # metadata.count; the sidecar contract calls that field num_points.
            if "num_points" not in out and "count" in out:
                out["num_points"] = out["count"]

            logger.info(
                f"Extracted {len(out)} field(s) from {filename} — "
                f"bounds=({out.get('bounds.minx')}, {out.get('bounds.miny')}, "
                f"{out.get('bounds.maxx')}, {out.get('bounds.maxy')}), "
                f"num_points={out.get('num_points')}, "
                f"creation_year={out.get('creation_year')}, "
                f"creation_doy={out.get('creation_doy')}"
            )
            return out
    except Exception as exc:
        logger.warning(f"PDAL extraction failed for {filename}: {exc}")
        return {}


def build_sidecar_df(
    file_metadata: list[tuple[str, dict]],
    required_field_groups: list[list[str]] | None = None,
):
    """Assemble the sidecar DataFrame.

    Layout:
      - Row 0  : DEFAULT row — File="DEFAULT", all other columns empty
      - Row 1+ : one row per file with extracted metadata values
      - Col 0  : "Filename" (basename of the S3 key)

    Required field groups are checked per-file: if any file lacks all alternatives
    for a group, the canonical (first) field name is prepended as a blank column.
    """
    import pandas as pd

    if not file_metadata:
        return pd.DataFrame(columns=["Filename"])

    all_cols: list[str] = []
    seen_cols: set = set()
    for _, meta in file_metadata:
        for col in meta:
            if col not in seen_cols:
                all_cols.append(col)
                seen_cols.add(col)

    prepend_cols: list[str] = []
    prepend_set: set = set()
    if required_field_groups:
        for group in required_field_groups:
            canonical = group[0]
            all_covered = all(any(field in meta for field in group) for _, meta in file_metadata)
            if not all_covered and canonical not in prepend_set:
                prepend_cols.append(canonical)
                prepend_set.add(canonical)
                seen_cols.add(canonical)

    remaining = [c for c in all_cols if c not in prepend_set]
    all_cols = prepend_cols + remaining

    default_row: dict = {"Filename": "DEFAULT", **{col: "" for col in all_cols}}
    rows = [default_row]
    for filename, meta in file_metadata:
        row: dict = {"Filename": filename}
        for col in all_cols:
            row[col] = meta.get(col, "")
        rows.append(row)

    return pd.DataFrame(rows, columns=["Filename"] + all_cols)


def _generate(
    bucket: str,
    directory: str,
    data_type: DataTypeEnum,
    output_filename: str,
    client_sidecar: str | None,
) -> str:
    logger.info(
        f"Starting sidecar CSV generation — bucket={bucket!r} directory={directory!r} "
        f"data_type={data_type} output={output_filename!r} "
        f"client_sidecar={client_sidecar!r}"
    )

    if data_type == DataTypeEnum.vector:
        raise NotImplementedError(
            "Vector data type is not yet supported for sidecar CSV generation."
        )

    bucket = normalize_object_store_path(bucket.strip())
    directory = normalize_object_store_path(directory.strip())

    if not bucket:
        raise ValueError("Bucket name cannot be empty.")
    if not directory:
        raise ValueError("Directory path cannot be empty.")

    if data_type not in DATA_TYPE_INFO:
        raise ValueError(
            f"Unsupported data_type '{data_type}'. Valid values: {list(DATA_TYPE_INFO.keys())}"
        )

    info = DATA_TYPE_INFO[data_type]
    include = list(info.get("include") or [])
    exclude = list(info.get("exclude") or [])

    store = ObjectStore("s3").init_session(bucket=bucket)
    keys = get_object_keys(store=store, directory=directory, include=include, exclude=exclude)

    if not keys:
        raise RuntimeError(f"No valid files found for '{data_type}' in s3://{bucket}/{directory}")

    total = len(keys)
    logger.info(f"Found {total} file(s) to process in s3://{bucket}/{directory}.")

    is_image_or_video = data_type in (_IMAGE_DATA_TYPES | _VIDEO_DATA_TYPES)
    is_point_cloud = data_type in _POINT_CLOUD_DATA_TYPES

    file_metadata: list[tuple[str, dict]] = []
    empty_count = 0

    if is_image_or_video:
        for i, key in enumerate(keys, 1):
            basename = Path(key).name
            logger.info(f"[{i}/{total}] Extracting EXIF: {key}")
            meta = extract_exif_metadata(store, key, filename=basename)
            if not meta:
                empty_count += 1
                logger.warning(f"No EXIF metadata returned for {key}")
            file_metadata.append((basename, meta))

    elif is_point_cloud:
        for i, key in enumerate(keys, 1):
            basename = Path(key).name
            logger.info(f"[{i}/{total}] Extracting PDAL metadata: {key}")
            meta = extract_pdal_metadata(store, key, filename=basename)
            if not meta:
                empty_count += 1
                logger.warning(f"No PDAL metadata returned for {key}")
            file_metadata.append((basename, meta))

    else:
        raise ValueError(f"Unhandled data type for metadata extraction: {data_type}")

    if empty_count:
        logger.warning(
            f"{empty_count}/{total} file(s) produced empty metadata — "
            "those rows will have blank fields in the sidecar."
        )

    if client_sidecar:
        client_key = normalize_object_store_path(client_sidecar.strip())
        if not client_key:
            raise ValueError("--client-sidecar value cannot be empty.")
        client_url = f"s3://{bucket}/{client_key}"
        logger.info(f"Merging client sidecar from {client_url}")
        client_df = load_and_clean_client_sidecar(
            url=client_url,
            bucket=bucket,
            required_field_groups=_REQUIRED_SIDECAR_FIELD_GROUPS,
        )
        merge_client_metadata(file_metadata, client_df)

    logger.info(f"Building sidecar DataFrame for {len(file_metadata)} file(s).")
    required_field_groups = list(_REQUIRED_SIDECAR_FIELD_GROUPS.get(data_type, []))
    df = build_sidecar_df(file_metadata, required_field_groups=required_field_groups)
    logger.info(
        f"Sidecar DataFrame: {len(df)} row(s) (including DEFAULT), {len(df.columns)} column(s)."
    )

    output_key = f"{directory.rstrip('/')}/{output_filename}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / output_filename
        df.to_csv(local_path, index=False)
        logger.info(f"Uploading sidecar CSV to s3://{bucket}/{output_key}")
        upload(store, key=output_key, source=str(local_path))

    sidecar_s3_path = f"s3://{bucket}/{output_key}"
    logger.info(f"Sidecar CSV written: {sidecar_s3_path}")

    # Argo workflow integration: write the output S3 path to the parameters
    # file the workflow controller reads. Best-effort — non-Argo runs simply
    # skip this with a warning.
    try:
        argo_output_dir = "/var/run/argo/outputs/parameters"
        os.makedirs(argo_output_dir, exist_ok=True)
        with open(f"{argo_output_dir}/sidecar_csv_path", "w") as f:
            f.write(sidecar_s3_path)
    except Exception:
        logger.warning(f"Could not write Argo output parameter. sidecar_csv_path={sidecar_s3_path}")

    return sidecar_s3_path


@sidecar_app.command()
def generate(
    bucket: Annotated[
        str,
        typer.Option(..., help="S3 bucket name."),
    ],
    directory: Annotated[
        str,
        typer.Option(..., help="S3 directory/prefix to scan for files."),
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
                "Optional S3 key/prefix of a client-supplied sidecar CSV within the "
                "same bucket as --bucket. Values are merged into the generated "
                "sidecar; client values win on conflict."
            ),
        ),
    ] = None,
) -> None:
    """Scan an S3 directory, extract per-file metadata, and write a sidecar CSV."""
    # Configure logging at command entry only — keeps the library importable
    # without imposing a global logging config on consumers.
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

    try:
        _generate(
            bucket=bucket,
            directory=directory,
            data_type=data_type,
            output_filename=output_filename,
            client_sidecar=client_sidecar,
        )
    except Exception:
        logger.exception(
            f"Failed to generate sidecar CSV. "
            f"--bucket {bucket} --directory {directory} "
            f"--data-type {data_type} "
            f"--client-sidecar {client_sidecar}"
        )
        raise typer.Exit(code=1) from None
