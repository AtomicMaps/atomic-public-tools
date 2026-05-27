"""Per-file metadata extractors used by the sidecar generator.

These are deliberately storage-agnostic: callers pass a local filesystem
path (the storage backend is responsible for materialising remote files
locally first) and a display filename used in log messages.
"""

import datetime
import json
import logging
import re
import subprocess
from typing import TYPE_CHECKING

from atomic_tools.utils.utils import run_exiftool

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

_PDAL_BIN = "/opt/conda/envs/pdal/bin/pdal"
# LAS header stores bounds as flat keys; remap to bounds.X for downstream.
_LAS_BOUNDS_DIMS = frozenset({"minx", "miny", "maxx", "maxy", "minz", "maxz"})


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
