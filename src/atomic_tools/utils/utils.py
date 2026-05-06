"""General utilities — file matching, exiftool, object-store I/O.

Originally `atomicmapspy/utils/utils.py` (only the functions actually used by
the sidecar generator and their transitive helpers — the source module is
~2200 lines of which we need ~250).
"""

import logging
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from enum import Enum
from fnmatch import fnmatch as _fnmatch  # noqa: F401  # kept for parity / future use
from fnmatch import translate as fnmatch_translate
from pathlib import Path
from typing import (
    Any,
    Optional,
)
from urllib.parse import urlparse

import obstore
import pandas as pd
from obstore.exceptions import (
    GenericError,
    InvalidPathError,
    NotSupportedError,
    PermissionDeniedError,
    PreconditionError,
    UnauthenticatedError,
)

from atomic_tools.utils.object_store import (
    REMOTE_SCHEME_TO_STORE_TYPE,
    ObjectStore,
    ObstoreBackend,
)

logger = logging.getLogger(__name__)

BYTES_TO_GB = 1e9
CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_RETRIES = 5

DATA_TYPE_INFO: dict = {
    "ortho_image": {
        "include": [".tif", ".tiff"],
        "exclude": ["_rgb.tif"],
        "sidecars": [".ecw"],
        "sub_types": ["rgb", "ir"],
        "footprint_feature_class": "footprints_ortho_image",
    },
    "spherical_image": {
        "include": [".jpg", ".jp2", "jpeg", ".png"],
        "exclude": ["PreviewImage.jpg", "ThumbnailImage.jpg", "annotation.json"],
        "sidecars": [],
        "sub_types": ["rgb"],
        "footprint_feature_class": "footprints_spherical_image",
    },
    "oriented_image": {
        "include": [".jpg", ".jp2", "jpeg", ".png"],
        "exclude": [
            ".tif",
            ".tiff",
            "PreviewImage.jpg",
            "ThumbnailImage.jpg",
            "annotation.json",
        ],
        "sidecars": [],
        "sub_types": ["rgb", "ir", "thermal"],
        "footprint_feature_class": "footprints_oriented_imagery",
    },
    "point_cloud": {
        "include": [".las", ".laz", "zlas"],
        "exclude": [".copc.las", ".copc.laz"],
        "sidecars": [],
        "sub_types": ["lidar", "photogrammetry", "terrestrial_lidar", "aerial_lidar"],
        "footprint_feature_class": "footprints_point_cloud",
    },
    "full_motion_video": {
        "include": [".mp4", ".mov", ".ts", ".avi", ".tts"],
        "exclude": [],
        "sidecars": [".gpx", ".kmz", ".srt"],
        "sub_types": ["rgb", "ir"],
        "footprint_feature_class": "footprints_video",
    },
    "vector": {
        "include": [".gdb", ".gdb.zip", ".gpkg", ".geojson", ".shp"],
        "exclude": [
            ".shx",
            ".dbf",
            ".prj",
            ".cpg",
            ".sbn",
            ".sbx",
            ".shp.xml",
            ".qpj",
            ".pmtiles",
            ".parquet",
            "_thumbnail.jpg",
        ],
        "sidecars": [],
        "sub_types": [],
    },
}


class DataTypeEnum(str, Enum):
    ortho_image = "ortho_image"
    oriented_image = "oriented_image"
    spherical_image = "spherical_image"
    point_cloud = "point_cloud"
    video = "full_motion_video"
    cad = "cad"
    vector = "vector"


def normalize_object_store_path(path: str) -> str:
    """Normalize an object store path by removing trailing slashes."""
    if not path:
        return path
    return path.rstrip("/")


def run_exiftool(
    path: str,
    extra_args: list[str],
    exiftool_config: str | None = None,
) -> bytes:
    """Run exiftool on a path and return raw stdout bytes.

    Strips a leading ``file://`` prefix. For HTTP/HTTPS URLs, pipes curl stdout
    into exiftool stdin (sequential read — suitable for images). Callers that
    require random-access seeking (e.g. video) should download to a local temp
    file and pass that path instead.
    """
    if path.startswith("file://"):
        path = path.replace("file://", "")

    cmd = ["exiftool"]
    if exiftool_config:
        cmd.extend(["-config", exiftool_config])
    cmd.extend(extra_args)

    if path.startswith("http"):
        cmd.append("-")
        curl_process = subprocess.Popen(["curl", "-s", path], stdout=subprocess.PIPE)
        try:
            output = subprocess.check_output(cmd, stdin=curl_process.stdout)
        except subprocess.CalledProcessError:
            curl_process.kill()
            curl_process.wait()
            raise
        finally:
            if curl_process.stdout:
                curl_process.stdout.close()
            curl_process.wait()
    else:
        cmd.append(path)
        output = subprocess.check_output(cmd)

    return output


# ---- Sidecar row matching --------------------------------------------------

_SIDECAR_INDEX_ATTR = "_sidecar_index"


def _build_sidecar_index(df: "pd.DataFrame") -> dict[str, Any]:
    file_col = df.columns[0]
    stripped = df[file_col].str.strip()
    candidates = df[stripped != "DEFAULT"]
    candidate_paths = candidates[file_col].str.strip()
    candidate_basenames = candidate_paths.str.split("/").str[-1]
    candidate_stems = candidate_basenames.str.rsplit(".", n=1).str[0]
    default_mask = stripped == "DEFAULT"
    default_row = df[default_mask].iloc[0] if default_mask.any() else None
    return {
        "file_col": file_col,
        "candidates": candidates,
        "candidate_paths": candidate_paths,
        "candidate_basenames": candidate_basenames,
        "candidate_stems": candidate_stems,
        "default_row": default_row,
    }


def _get_sidecar_index(df: "pd.DataFrame") -> dict[str, Any]:
    # Cache on df.attrs so repeated lookups against the same DataFrame instance
    # reuse the (relatively expensive) full-column .str ops.
    cached = df.attrs.get(_SIDECAR_INDEX_ATTR)
    if cached is None:
        cached = _build_sidecar_index(df)
        df.attrs[_SIDECAR_INDEX_ATTR] = cached
    return cached


def _flexible_match_vectorized(target: str, candidates: pd.Series) -> pd.Series:
    """Vectorized separator-agnostic / prefix-allowed match across many candidates.

    Replaces ``-`` and ``_`` with ``?`` (single-character wildcard) so separator
    differences are ignored, and allows the shorter side to act as a prefix
    pattern (trailing ``*``) for the longer.
    """
    if not isinstance(target, str) or not target:
        return pd.Series([False] * len(candidates), index=candidates.index)

    candidates_filled = candidates.fillna("").astype(str)
    n_target = target.replace("-", "?").replace("_", "?")
    target_len = len(target)

    cond_a = candidates_filled.str.fullmatch(fnmatch_translate(n_target))

    candidate_lens = candidates_filled.str.len()
    cond_c1 = (target_len <= candidate_lens) & candidates_filled.str.fullmatch(
        fnmatch_translate(f"{n_target}*")
    )

    n_candidates = candidates_filled.str.replace("-", "?", regex=False).str.replace(
        "_", "?", regex=False
    )
    needs_inverse = ~(cond_a | cond_c1)

    def _candidate_as_pattern(nc: str, suffix: str = "") -> bool:
        if not nc:
            return False
        return bool(re.fullmatch(fnmatch_translate(nc + suffix), target))

    cond_b = pd.Series(False, index=candidates.index)
    cond_c2 = pd.Series(False, index=candidates.index)
    if needs_inverse.any():
        n_candidates_remaining = n_candidates[needs_inverse]
        cond_b.loc[needs_inverse] = n_candidates_remaining.apply(_candidate_as_pattern)
        target_longer = target_len > candidate_lens
        c2_check_mask = needs_inverse & target_longer
        if c2_check_mask.any():
            cond_c2.loc[c2_check_mask] = n_candidates[c2_check_mask].apply(
                lambda nc: _candidate_as_pattern(nc, "*")
            )

    return cond_a | cond_b | cond_c1 | cond_c2


def find_sidecar_row(
    df: "pd.DataFrame",
    path: str,
) -> tuple[Optional["pd.Series"], Optional["pd.Series"]]:
    """Locate the file-specific and DEFAULT rows in a sidecar DataFrame.

    Match strategy: full path → basename → flexible (separator-agnostic / prefix)
    against either basename or stem. Returns (matched_row, default_row); either
    may be None.
    """
    idx = _get_sidecar_index(df)
    file_col = idx["file_col"]
    candidates = idx["candidates"]
    candidate_paths = idx["candidate_paths"]
    candidate_basenames = idx["candidate_basenames"]
    default_row = idx["default_row"]

    path_basename = Path(urlparse(path).path).name
    path_match = candidate_paths.eq(path) | candidate_basenames.eq(path_basename)

    if not path_match.any():
        target_stem = Path(path_basename).stem
        path_match = _flexible_match_vectorized(
            path_basename, candidate_basenames
        ) | _flexible_match_vectorized(target_stem, idx["candidate_stems"])
        if path_match.any():
            logger.info(
                f"Matched {path_basename} to sidecar row(s) via flexible matching: "
                f"{candidates[path_match][file_col].tolist()}"
            )

    matched: pd.Series | None = candidates[path_match].iloc[0] if path_match.any() else None
    return matched, default_row


def load_sidecar_df(
    sidecar_path: str,
    *,
    headerless_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a sidecar CSV (local or remote S3/GCS/Azure) into a DataFrame.

    When `headerless_columns` is given, the CSV is read with no header row and
    the supplied names are applied positionally — used for client sidecars that
    ship without column headers.
    """
    read_kwargs: dict[str, Any] = {
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8",
    }
    if headerless_columns is not None:
        read_kwargs["header"] = None
        read_kwargs["names"] = list(headerless_columns)

    parsed = urlparse(sidecar_path)
    store_type = REMOTE_SCHEME_TO_STORE_TYPE.get(parsed.scheme)
    if store_type:
        key = parsed.path.lstrip("/")
        # obstore's from_url("s3://bucket/key/...") roots the store at
        # bucket/key/..., which would prepend the key a second time on get.
        # So root the store at the bucket and pass `key` separately.
        bucket_url = f"{parsed.scheme}://{parsed.netloc}"
        store = ObjectStore(store_type).from_url(bucket_url)
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded_path = download(store, key, tmp_dir)
            return pd.read_csv(downloaded_path, **read_kwargs)
    return pd.read_csv(sidecar_path, **read_kwargs)


# ---- Object store I/O ------------------------------------------------------


def get_keys_and_metadata(store: ObstoreBackend, prefix: str) -> dict:
    """Recursively list object keys + metadata in a bucket under `prefix`."""
    try:
        chunks = list(obstore.list(store, prefix=prefix))
        return {
            item["path"]: {k: v for k, v in item.items() if k != "path"}
            for chunk in chunks
            for item in chunk
        }
    except (
        PermissionDeniedError,
        UnauthenticatedError,
        InvalidPathError,
        NotSupportedError,
        PreconditionError,
        GenericError,
    ) as e:
        raise RuntimeError(f"Failed to list keys with prefix {prefix}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during listing keys with prefix {prefix}: {e}") from e


def get_object_keys(
    store: ObstoreBackend,
    directory: str,
    include: list | tuple,
    exclude: list | tuple = (),
) -> list[str]:
    """List object keys whose suffixes match `include`, excluding any matching `exclude`."""
    if not directory.strip():
        raise ValueError("Directory path cannot be empty")
    if not include:
        raise ValueError("Must specify at least one file extension to include.")

    try:
        key_info = get_keys_and_metadata(store, directory)
    except Exception:
        logger.error("Error while filtering list of objects")
        raise

    if not key_info:
        logger.warning(f"No files found at {directory}")
        return []

    key_sizes = {k: key_info[k]["size"] for k in key_info}
    inc = tuple(e.lower() for e in include)
    excl = tuple(p.lower() for p in exclude) if exclude else ()

    logger.info(f"Filtering for extensions {inc}, excluding {excl}")

    gdb_roots: set = set()
    for k in key_info:
        kl = k.lower()
        if ".gdb/" in kl:
            try:
                gdb_roots.add(k[: kl.index(".gdb/")] + ".gdb")
            except Exception:
                gdb_roots.add(kl.split(".gdb/", 1)[0] + ".gdb")

    include_gdb = any(x.endswith(".gdb") for x in inc)
    filtered: list[str] = []

    for k in key_info:
        kl = k.lower()
        if ".gdb/" in kl:
            continue
        if kl.endswith(inc):
            if excl and kl.endswith(excl):
                continue
            if key_sizes.get(k, 0) == 0:
                logger.warning(f"Skipping zero-size file: {k}")
                continue
            filtered.append(k)

    if include_gdb and gdb_roots:
        filtered.extend(sorted(gdb_roots))

    result = list(dict.fromkeys(filtered))
    logger.info(f"Matched {len(result)} file(s) after filtering.")
    return result


def download(store: ObstoreBackend, key: str, destination_dir: str, buffer_size: int = 5) -> str:
    """Download an object from the store to `destination_dir` in chunks.

    Performs a free-space precheck (file_size + buffer_size GB) and a chunked,
    range-based download with exponential-backoff retries.
    """
    if not store:
        raise ValueError("Object store is not initialized. Initialize it first.")

    storage_path = Path(destination_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    try:
        _, _, free = shutil.disk_usage(storage_path)
        free_gb = free // BYTES_TO_GB
        logger.info(f"Disk space available in {destination_dir}: {free_gb} GB")

        meta = obstore.head(store, key)
        file_size = meta.get("size", 0)

        if file_size is None or file_size <= 0:
            raise ValueError(f"Invalid file size {file_size} for object {key}")

        required_space = file_size + (buffer_size * BYTES_TO_GB)
        if free < required_space:
            raise RuntimeError(
                f"Not enough disk space. Required: {required_space // BYTES_TO_GB} GB, "
                f"Available: {free_gb} GB"
            )

        destination_path = storage_path / Path(key).name
        logger.debug(f"Downloading {key} ({file_size / BYTES_TO_GB:.2f} GB) to {destination_path}")

        current_gb = 0
        with open(destination_path, "wb") as local_file:
            for start in range(0, file_size, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, file_size)

                if start >= end:
                    logger.error(f"Invalid byte range detected: start={start}, end={end} for {key}")
                    raise ValueError(f"Invalid byte range: start={start}, end={end}")

                for attempt in range(MAX_RETRIES):
                    try:
                        byte_range = obstore.get_range(store, key, start=start, end=end)
                        local_file.write(byte_range)

                        if int(end // BYTES_TO_GB) > current_gb:
                            current_gb = int(end // BYTES_TO_GB)
                            logger.info(f"Downloaded {current_gb} GB...")

                        break
                    except Exception as e:
                        logger.error(
                            f"Attempt {attempt + 1}: Failed to download bytes "
                            f"{start}-{end}. File size: {file_size}. Error: {e}"
                        )
                        time.sleep(2**attempt)
                else:
                    raise RuntimeError(
                        f"Failed to download range {start}-{end} after {MAX_RETRIES} attempts."
                    )
        logger.info(f"Successfully downloaded {key} to {destination_path}")
        return str(destination_path)

    except FileNotFoundError as not_found_error:
        logger.error(f"Object not found at {key}: {not_found_error}")
        raise

    except (
        PermissionDeniedError,
        UnauthenticatedError,
        InvalidPathError,
        NotSupportedError,
        PreconditionError,
        GenericError,
    ) as known_error:
        logger.error(f"Storage-related error while downloading {key}: {known_error}")
        raise RuntimeError(f"Storage error: {known_error}") from known_error

    except (ValueError, RuntimeError) as expected_error:
        logger.error(f"Download error for {key}: {expected_error}")
        raise expected_error

    except Exception as unexpected_error:
        logger.critical(f"Unexpected download error for {key}: {unexpected_error}")
        raise RuntimeError(
            f"Unexpected error during download {key}: {unexpected_error}"
        ) from unexpected_error


def upload(
    store: ObstoreBackend,
    key: str | Path,
    source: str | Path,
    chunk_size: int | None = 25 * 1024 * 1024,
    max_concurrency: int | None = 5,
) -> None:
    """Upload a local file to the object store via multipart upload."""
    if not store:
        raise ValueError("Object store is not initialized. Initialize it first.")

    if not Path(source).exists():
        raise FileNotFoundError(f"Local file {source} does not exist.")

    try:
        _chunk_size = chunk_size if chunk_size is not None else 25 * 1024 * 1024
        _max_concurrency = max_concurrency if max_concurrency is not None else 5
        obstore.put(
            store,
            path=str(key),
            file=Path(source),
            use_multipart=True,
            chunk_size=_chunk_size,
            max_concurrency=_max_concurrency,
        )
    except (
        PermissionDeniedError,
        UnauthenticatedError,
        InvalidPathError,
        NotSupportedError,
        PreconditionError,
        GenericError,
    ) as e:
        raise RuntimeError(f"Failed to upload {source} to {key}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during upload {source} to {key}: {e}") from e
