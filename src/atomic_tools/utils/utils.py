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
    ObstoreBackend,
    store_for_bucket,
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
    },
    "spherical_image": {
        "include": [".jpg", ".jp2", "jpeg", ".png"],
        "exclude": ["PreviewImage.jpg", "ThumbnailImage.jpg", "annotation.json"],
        "sidecars": [],
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
    },
    "point_cloud": {
        "include": [".las", ".laz", "zlas", ".e57"],
        "exclude": [".copc.las", ".copc.laz"],
        "sidecars": [],
    },
    "full_motion_video": {
        "include": [".mp4", ".mov", ".ts", ".avi", ".tts"],
        "exclude": [],
        "sidecars": [".gpx", ".kmz", ".srt"],
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


def _split_path_components(s: str) -> tuple[str, ...]:
    """Split a path-like string on '/' (and '\\') into non-empty components."""
    if not s:
        return ()
    return tuple(p for p in s.replace("\\", "/").split("/") if p)


def _build_sidecar_index(df: "pd.DataFrame") -> dict[str, Any]:
    file_col = df.columns[0]
    stripped = df[file_col].str.strip()
    candidates = df[stripped != "DEFAULT"]
    candidate_paths = candidates[file_col].str.strip()
    candidate_components = candidate_paths.apply(_split_path_components)
    candidate_last_components = candidate_components.apply(lambda p: p[-1] if p else "")
    candidate_stems = candidate_last_components.str.rsplit(".", n=1).str[0]
    default_mask = stripped == "DEFAULT"
    default_row = df[default_mask].iloc[0] if default_mask.any() else None
    return {
        "file_col": file_col,
        "candidates": candidates,
        "candidate_paths": candidate_paths,
        "candidate_basenames": candidate_last_components,
        "candidate_stems": candidate_stems,
        "candidate_components": candidate_components,
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


def _is_path_tail_suffix(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """True if one tuple's tail equals the other entirely.

    ``("a","1.jpg")`` and ``("1.jpg",)`` match; ``("a","1.jpg")`` and
    ``("b","1.jpg")`` do not. Empty inputs never match.
    """
    if not a or not b:
        return False
    if len(a) <= len(b):
        return b[-len(a) :] == a
    return a[-len(b) :] == b


def _path_suffix_match_vectorized(
    target_components: tuple[str, ...],
    candidate_components: pd.Series,
    candidate_basenames: pd.Series,
) -> pd.Series:
    """Vectorized path-component tail-suffix match.

    See ``_is_path_tail_suffix``. ``candidate_basenames`` (each candidate's
    last component) lets us prune to candidates sharing the target basename
    before the per-row tuple compare — tail-suffix requires equal last
    component, so anything else can't match.
    """
    result = pd.Series(False, index=candidate_components.index)
    if not target_components:
        return result
    basename_mask = candidate_basenames.eq(target_components[-1])
    if not basename_mask.any():
        return result
    for idx, parts in candidate_components[basename_mask].items():
        if _is_path_tail_suffix(target_components, parts):
            result.loc[idx] = True
    return result


def _parent_path_compatible(
    target_parents: tuple[str, ...],
    candidate_parents: tuple[str, ...],
) -> bool:
    """Like ``_is_path_tail_suffix`` but treats either side being empty as a match,
    since the fuzzy branch must accept rows without parent context.
    """
    if not target_parents or not candidate_parents:
        return True
    return _is_path_tail_suffix(target_parents, candidate_parents)


def find_sidecar_row(
    df: "pd.DataFrame",
    path: str,
) -> tuple[Optional["pd.Series"], Optional["pd.Series"]]:
    """Locate the file-specific and DEFAULT rows in a sidecar DataFrame.

    Match strategy, in priority order:
      1. Exact full-path match.
      2. Path-component tail-suffix match (one path's last components equal
         the other's, treating ``/`` as separator). Among multiple matches,
         the candidate with the most components (most specific) wins.
      3. Flexible (separator-agnostic / prefix) match on basename or stem.

    Returns (matched_row, default_row); either may be None.
    """
    idx = _get_sidecar_index(df)
    candidates = idx["candidates"]
    candidate_paths = idx["candidate_paths"]
    candidate_basenames = idx["candidate_basenames"]
    candidate_components = idx["candidate_components"]
    default_row = idx["default_row"]

    parsed_path = urlparse(path).path
    target_components = _split_path_components(parsed_path)

    exact_match = candidate_paths.eq(path)
    if exact_match.any():
        return candidates[exact_match].iloc[0], default_row

    suffix_match = _path_suffix_match_vectorized(
        target_components, candidate_components, candidate_basenames
    )
    if suffix_match.any():
        depths = candidate_components[suffix_match].apply(len)
        return candidates.loc[depths.idxmax()], default_row

    path_basename = Path(parsed_path).name
    target_stem = Path(path_basename).stem
    fuzzy_match = _flexible_match_vectorized(
        path_basename, candidate_basenames
    ) | _flexible_match_vectorized(target_stem, idx["candidate_stems"])
    if not fuzzy_match.any():
        return None, default_row

    target_parents = target_components[:-1]
    fuzzy_match &= candidate_components.apply(
        lambda parts: _parent_path_compatible(target_parents, parts[:-1])
    )
    if not fuzzy_match.any():
        return None, default_row

    logger.info(
        f"Matched {path_basename} to sidecar row(s) via flexible matching: "
        f"{candidates[fuzzy_match][idx['file_col']].tolist()}"
    )
    return candidates[fuzzy_match].iloc[0], default_row


def has_value(value: Any) -> bool:
    """Return True when `value` is non-empty and not the literal "nan"."""
    if value is None:
        return False
    s = str(value).strip()
    return bool(s) and s.lower() != "nan"


def is_remote_uri(uri: str | None) -> bool:
    """Return True if `uri` has a remote scheme (s3, gs, az, …)."""
    if not uri:
        return False
    return urlparse(uri).scheme.lower() in REMOTE_SCHEME_TO_STORE_TYPE


def read_text_uri(uri: str, *, encoding: str = "utf-8") -> str:
    """Read text content from a local path or remote URI (s3/gs/az)."""
    parsed = urlparse(uri)
    store_type = REMOTE_SCHEME_TO_STORE_TYPE.get(parsed.scheme.lower())
    if store_type:
        key = parsed.path.lstrip("/")
        store = store_for_bucket(parsed.scheme, parsed.netloc)
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded_path = download(store, key, tmp_dir)
            return Path(downloaded_path).read_text(encoding=encoding)
    return Path(uri).read_text(encoding=encoding)


def load_sidecar_df(
    sidecar_path: str,
    *,
    column_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a sidecar CSV (local or remote S3/GCS/Azure) into a DataFrame.

    When `column_names` is given, the CSV is read with no header row and
    the supplied names are applied positionally — used for client sidecars that
    ship without column headers.
    """
    read_kwargs: dict[str, Any] = {
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8",
    }
    if column_names is not None:
        read_kwargs["header"] = None
        read_kwargs["names"] = list(column_names)

    parsed = urlparse(sidecar_path)
    store_type = REMOTE_SCHEME_TO_STORE_TYPE.get(parsed.scheme)
    if store_type:
        key = parsed.path.lstrip("/")
        # Root the store at the bucket and pass `key` separately: obstore roots
        # a store at whatever URL it's given, so a bucket+key URL would prepend
        # the key a second time on get.
        store = store_for_bucket(parsed.scheme, parsed.netloc)
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


def _is_preview_image(key: str) -> bool:
    """True if `key`'s filename stem ends with ``_PreviewImage`` (any extension).

    Preview images are auxiliary renders that should never be treated as
    primary assets, regardless of their file extension (e.g.
    ``IMG_001_PreviewImage.jpg``, ``IMG_001_PreviewImage.png``).
    """
    stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.lower().endswith("_previewimage")


def filter_keys(
    key_sizes: dict[str, int],
    include: list | tuple,
    exclude: list | tuple = (),
) -> list[str]:
    """Filter a {key: size} mapping by suffix include/exclude.

    Skips zero-size files, files inside ``*.gdb/`` directories, and adds
    ``.gdb`` directory roots when ``.gdb`` is in `include`.
    """
    if not include:
        raise ValueError("Must specify at least one file extension to include.")

    inc = tuple(e.lower() for e in include)
    excl = tuple(p.lower() for p in exclude) if exclude else ()
    logger.info(f"Filtering for extensions {inc}, excluding {excl}")

    include_gdb = any(x.endswith(".gdb") for x in inc)
    gdb_roots: set[str] = set()
    if include_gdb:
        for k in key_sizes:
            kl = k.lower()
            if ".gdb/" in kl:
                try:
                    gdb_roots.add(k[: kl.index(".gdb/")] + ".gdb")
                except Exception:
                    gdb_roots.add(kl.split(".gdb/", 1)[0] + ".gdb")

    filtered: list[str] = []
    for k, size in key_sizes.items():
        kl = k.lower()
        if ".gdb/" in kl:
            continue
        if not kl.endswith(inc):
            continue
        if _is_preview_image(kl):
            continue
        if excl and kl.endswith(excl):
            continue
        if size == 0:
            logger.warning(f"Skipping zero-size file: {k}")
            continue
        filtered.append(k)

    if include_gdb and gdb_roots:
        filtered.extend(sorted(gdb_roots))

    result = list(dict.fromkeys(filtered))
    logger.info(f"Matched {len(result)} file(s) after filtering.")
    return result


def get_object_keys(
    store: ObstoreBackend,
    directory: str,
    include: list | tuple,
    exclude: list | tuple = (),
) -> list[str]:
    """List object keys whose suffixes match `include`, excluding any matching `exclude`."""
    if not directory.strip():
        raise ValueError("Directory path cannot be empty")

    try:
        key_info = get_keys_and_metadata(store, directory)
    except Exception:
        logger.error("Error while filtering list of objects")
        raise

    if not key_info:
        logger.warning(f"No files found at {directory}")
        return []

    key_sizes = {k: key_info[k]["size"] for k in key_info}
    return filter_keys(key_sizes, include, exclude)


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
