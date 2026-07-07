"""Per-file metadata extractors used by the sidecar generator.

These are deliberately storage-agnostic: callers pass a local filesystem
path (the storage backend is responsible for materialising remote files
locally first) and a display filename used in log messages.
"""

import datetime
import glob
import json
import logging
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from atomic_tools.utils.utils import has_value, run_exiftool
from atomic_tools.vendored import field_names

if TYPE_CHECKING:
    from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)

_DATE_FROM_PATH_PATTERN = re.compile(
    r"(?:[^0-9]|^)(19[0-9]{2}|20[0-9]{2}|[0-9]{2})(?:\W|_)?(1[0-2]|0[1-9])"
    r"(?:\W|_)?([012][0-9]|3[01])(?:\W|_)?(\d{6})?(?:\W|_|$)"
)

# Lazy singleton — TimezoneFinder() loads a binary KD-tree (~400ms) so we
# build it on first use and reuse across files.
_TIMEZONE_FINDER: "TimezoneFinder | None" = None


def _timezone_finder() -> "TimezoneFinder":
    global _TIMEZONE_FINDER
    if _TIMEZONE_FINDER is None:
        from timezonefinder import TimezoneFinder

        _TIMEZONE_FINDER = TimezoneFinder()
    return _TIMEZONE_FINDER


def infer_date_from_filepath(
    path: str, lat: float | None, lon: float | None
) -> datetime.datetime | None:
    """Extract an acquisition date from a file path as a fallback when EXIF/
    client-sidecar metadata is missing one.

    Looks for ``YYYYMMDD`` (or ``YYYY*MM*DD``, with optional ``HHMMSS``) in
    ``path``. When `lat`/`lon` are given, the naive datetime is localised
    using the file's timezone; otherwise UTC is assumed.
    """
    if not path:
        logger.warning("Path is not set; cannot infer date.")
        return None

    match = _DATE_FROM_PATH_PATTERN.search(path)
    if not match:
        logger.warning(f"No date found in path: {path}")
        return None

    year = match.group(1)
    if len(year) == 2:
        year = "20" + year
    month = match.group(2)
    day = match.group(3)
    date_str = year + month + day

    if match.group(4):
        date_str += match.group(4)
        try:
            naive = datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")
        except ValueError:
            logger.warning(f"Unable to parse date from path: {path}")
            return None
    else:
        naive = datetime.datetime.strptime(date_str, "%Y%m%d")

    if lat is None or lon is None:
        logger.warning(
            f"Latitude or longitude not provided (lat: {lat}, lon: {lon}); "
            "assuming UTC for date parsing."
        )
        return naive.replace(tzinfo=datetime.timezone.utc)

    import pytz

    try:
        tz_str = _timezone_finder().timezone_at(lng=lon, lat=lat)
        if not tz_str:
            logger.warning(
                f"Could not determine timezone for lat: {lat}, lon: {lon}; defaulting to UTC."
            )
            return pytz.utc.localize(naive).astimezone(datetime.timezone.utc)
        tz = pytz.timezone(tz_str)
        return tz.localize(naive).astimezone(datetime.timezone.utc)
    except Exception as e:
        raise ValueError(f"Error determining timezone for lat: {lat}, lon: {lon}: {e}") from e


# exiftool fields that describe the tool/file-system rather than the asset.
# GPSPosition is dropped because exiftool always auto-derives it from
# GPSLatitude/GPSLongitude; if a client sidecar later provides its own
# GPSPosition, _split_gps_position will reintroduce it as authoritative.
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
    "GPSPosition",
}

# LAS header stores bounds as flat keys; remap to bounds.X for downstream.
_LAS_BOUNDS_DIMS = frozenset({"minx", "miny", "maxx", "maxy", "minz", "maxz"})

# Keys in flattened `pdal info --metadata` output that carry the file's CRS as a
# WKT/PROJ string, in preference order. PDAL leaves all of these as "" when the
# point cloud header has no spatial reference. `spatialreference` is the
# top-level canonical value; the `srs.*` variants are fallbacks.
_PDAL_CRS_KEYS = (
    "spatialreference",
    "comp_spatialreference",
    "srs.wkt",
    "srs.compoundwkt",
    "srs.horizontal",
    "srs.proj4",
)

# How to get PDAL — surfaced to the user when it can't be located.
_PDAL_INSTALL_HELP = (
    "PDAL is required for point cloud metadata extraction but could not be found.\n"
    "Install it via conda (recommended):\n"
    "    conda install -c conda-forge pdal\n"
    "or, on macOS, via Homebrew:\n"
    "    brew install pdal\n"
    "If PDAL is already installed in a non-standard location, put its 'bin' "
    "directory on your PATH or set the PDAL_BIN environment variable to the "
    "full path of the pdal executable."
)

# Cached result of _find_pdal_bin() so the (filesystem-walking) search runs once.
_PDAL_BIN: "str | None" = None


class PdalNotFoundError(RuntimeError):
    """Raised when the pdal executable cannot be located on the system."""


def _candidate_pdal_paths() -> "list[str]":
    """Logical places a pdal executable might live, in priority order.

    Ordered: explicit override, then PATH, then the active conda env, then any
    conda env named 'pdal', then common conda/Homebrew install roots.
    """
    candidates: list[str] = []

    override = os.environ.get("PDAL_BIN")
    if override:
        candidates.append(override)

    on_path = shutil.which("pdal")
    if on_path:
        candidates.append(on_path)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "bin", "pdal"))

    # conda/mamba install roots; check a 'pdal'-named env first, then any env.
    conda_roots = [
        os.environ.get("CONDA_ROOT"),
        os.environ.get("MAMBA_ROOT_PREFIX"),
        "/opt/conda",
        "/opt/miniconda3",
        "/opt/anaconda3",
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
    ]
    for root in conda_roots:
        if not root:
            continue
        candidates.append(os.path.join(root, "envs", "pdal", "bin", "pdal"))
        candidates.extend(glob.glob(os.path.join(root, "envs", "*", "bin", "pdal")))
        candidates.append(os.path.join(root, "bin", "pdal"))

    # Homebrew / common system prefixes (PATH usually covers these, but not
    # always when invoked from a GUI or a stripped environment).
    candidates.extend(["/opt/homebrew/bin/pdal", "/usr/local/bin/pdal", "/usr/bin/pdal"])

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def find_pdal_bin() -> str:
    """Locate the pdal executable, caching the result for reuse.

    Searches PATH, the conda environment, and other logical install locations.
    Raises :class:`PdalNotFoundError` with install instructions if none of the
    candidates is an executable file.
    """
    global _PDAL_BIN
    if _PDAL_BIN is not None:
        return _PDAL_BIN

    for path in _candidate_pdal_paths():
        if os.path.isfile(path) and os.access(path, os.X_OK):
            logger.info(f"Using pdal executable at {path}")
            _PDAL_BIN = path
            return _PDAL_BIN

    raise PdalNotFoundError(_PDAL_INSTALL_HELP)


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


def extract_exif_metadata(local_path: str, filename: str) -> dict:
    """Extract EXIF/XMP/GPS metadata from a local image or video.

    Flags match BaseImage.run_exiftool so values are byte-identical to a
    re-read of the source image (the sidecar override step compares them).
    """
    try:
        logger.info(f"Running exiftool on {filename}")
        raw = run_exiftool(local_path, extra_args=["-fast", "-b", "-j"])
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


def extract_pdal_metadata(local_path: str, filename: str) -> dict:
    """Extract header metadata from a local point cloud file via the pdal info CLI.

    Calls the pdal binary rather than the Python bindings to avoid C++ ABI
    incompatibilities between the conda env and system libstdc++.
    """
    try:
        pdal_bin = find_pdal_bin()
        logger.info(f"Running pdal info for {filename}")
        proc = subprocess.run(
            [pdal_bin, "info", "--metadata", local_path],
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
    except PdalNotFoundError:
        # Missing pdal is a setup problem affecting every file, not a per-file
        # extraction failure — let it propagate so the caller can stop and tell
        # the user how to install it.
        raise
    except Exception as exc:
        logger.warning(f"PDAL extraction failed for {filename}: {exc}")
        return {}


_IMAGE_SIZE_COMBINED_RE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)")


def _positive_float(value: object) -> float | None:
    """Parse ``value`` as a float, returning it only when strictly positive."""
    try:
        num = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def _aspect_ratio_from_exif(meta: dict) -> float | None:
    """Resolve an image aspect ratio (width/height) from flat exiftool tags.

    Tries the ``IMAGE_SIZE_FIELDS`` (width_tag, height_tag) pairs in priority
    order, then falls back to parsing the combined ``ImageSize`` ("WxH") field.
    Returns None unless both dimensions parse as positive numbers.
    """
    for width_tag, height_tag in field_names.IMAGE_SIZE_FIELDS:
        width = _positive_float(meta.get(width_tag))
        height = _positive_float(meta.get(height_tag))
        if width is not None and height is not None:
            return width / height

    combined = meta.get(field_names.IMAGE_SIZE_COMBINED_FIELD)
    if combined:
        match = _IMAGE_SIZE_COMBINED_RE.match(str(combined))
        if match:
            width = float(match.group(1))
            height = float(match.group(2))
            if width > 0 and height > 0:
                return width / height
    return None


def spherical_signals_from_exif(meta: dict) -> dict:
    """Map flat exiftool ``-j`` tags onto ``infer_data_type``'s keyword signals.

    exiftool ``-j`` returns flat tags, not a raw XMP packet, so the ``xmp_packet``
    is synthesized from tags that only exist in the GPano/XMP namespace. Parity
    limitation vs the backend (which gets raw bytes): a GPano packet carrying
    none of ProjectionType/UsePanoramaViewer/FullPano*Pixels would be missed —
    acceptable; ProjectionType is mandatory per the GPano spec. We deliberately
    do NOT run a second exiftool invocation to fetch the raw packet.

    Returns a kwargs dict (subset of ``user_comment`` / ``aspect_ratio`` /
    ``xmp_packet``) suitable for ``infer_data_type(filename, **signals)``.
    """
    signals: dict = {}

    user_comment = meta.get("UserComment")
    if user_comment is not None:
        signals["user_comment"] = str(user_comment)

    aspect_ratio = _aspect_ratio_from_exif(meta)
    if aspect_ratio is not None:
        signals["aspect_ratio"] = aspect_ratio

    projection_type = meta.get("ProjectionType")
    if projection_type and has_value(projection_type):
        signals["xmp_packet"] = f'ProjectionType="{projection_type}"'.encode()
    elif any(
        tag in meta
        for tag in ("UsePanoramaViewer", "FullPanoWidthPixels", "FullPanoHeightPixels")
    ):
        # A GPano-namespace tag is present but ProjectionType wasn't surfaced;
        # signal spherical via a synthetic packet the XMP scanner recognises.
        signals["xmp_packet"] = b"GPano:present"

    return signals


def extract_pdal_crs(meta: dict) -> str | None:
    """Return the CRS PDAL read from a point cloud's header, or None if absent.

    Reads the flattened metadata produced by :func:`extract_pdal_metadata` and
    returns the first non-empty CRS string (see :data:`_PDAL_CRS_KEYS`). PDAL
    emits empty strings for these keys when the file carries no spatial
    reference, so a None result means "PDAL could not find a CRS".
    """
    for key in _PDAL_CRS_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
