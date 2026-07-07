"""Per-file metadata extractors used by the sidecar generator.

These are deliberately storage-agnostic: callers pass a local filesystem
path (the storage backend is responsible for materialising remote files
locally first) and a display filename used in log messages.
"""

import datetime
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
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
# point cloud header has no spatial reference.
#
# The order mirrors the internal atomicmapspy PointCloud.get_srs (see
# data-engineering/atomicmapspy/atomicmapspy/point_cloud.py). `pdal info
# --metadata` surfaces the LAS *reader* metadata block, so we follow get_srs's
# reader-block candidate order: the compound WKT first — so a file carrying a
# vertical datum keeps its full horizontal+vertical definition instead of a
# horizontal-only string — then the plain spatial reference, the top-level
# compound reference, and the PROJ.4 string. `srs.wkt` and `srs.horizontal` are
# extra fallbacks (not in get_srs) retained for unusual PDAL outputs.
_PDAL_CRS_KEYS = (
    "srs.compoundwkt",
    "spatialreference",
    "comp_spatialreference",
    "srs.proj4",
    "srs.wkt",
    "srs.horizontal",
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


def _run_pdal_info(pdal_bin: str, local_path: str, *, nosrs: bool = False) -> dict:
    """Run ``pdal info --metadata`` and return the parsed JSON.

    When ``nosrs`` is set, the LAS reader is told to ignore the header SRS via
    the ``readers.las.nosrs`` option (used as a fallback for files whose SRS
    aborts the read). Raises ``RuntimeError`` on a non-zero exit.
    """
    cmd = [pdal_bin, "info", "--metadata", local_path]
    if nosrs:
        cmd.append("--readers.las.nosrs=true")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"pdal info exited with code {proc.returncode}: {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def _run_pdal_filters_info(pdal_bin: str, local_path: str) -> dict:
    """Return the ``filters.info`` stage metadata from a PDAL pipeline.

    ``pdal info --metadata`` only surfaces the LAS-style *header* metadata block,
    which some readers (notably E57) leave empty. A ``filters.info`` pipeline
    yields bbox / num_points / srs for any reader PDAL can open. PDAL writes
    pipeline metadata to a file rather than stdout, so route it through a temp
    file. Raises ``RuntimeError`` on a non-zero exit.
    """
    # A bare filename as the first stage lets PDAL infer the reader from the
    # extension (readers.e57 for .e57, etc.).
    pipeline = {"pipeline": [local_path, {"type": "filters.info"}]}
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipe_path = os.path.join(tmp_dir, "pipeline.json")
        meta_path = os.path.join(tmp_dir, "metadata.json")
        with open(pipe_path, "w") as f:
            json.dump(pipeline, f)
        proc = subprocess.run(
            [pdal_bin, "pipeline", pipe_path, f"--metadata={meta_path}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pdal pipeline exited with code {proc.returncode}: "
                f"{proc.stderr.strip()}"
            )
        with open(meta_path) as f:
            data = json.load(f)
    return data.get("stages", {}).get("filters.info", {})


def _meta_from_filters_info(filters_info: dict) -> dict:
    """Map a ``filters.info`` block onto the flat sidecar keys.

    Produces ``bounds.*`` (from bbox), ``num_points``, and ``srs.*`` (so
    :func:`extract_pdal_crs` finds the CRS the same way it does for LAS). Note
    filters.info carries no ``creation_year``/``creation_doy`` — E57 has no
    LAS-style header date, so those rows rely on filename/client-sidecar dates.
    """
    out: dict = {}
    bbox = filters_info.get("bbox")
    # Some PDAL versions nest the corner dict under an inner "bbox" key.
    if isinstance(bbox, dict) and isinstance(bbox.get("bbox"), dict):
        bbox = bbox["bbox"]
    if isinstance(bbox, dict):
        for dim in _LAS_BOUNDS_DIMS:
            if dim in bbox:
                out[f"bounds.{dim}"] = bbox[dim]
    if "num_points" in filters_info:
        out["num_points"] = filters_info["num_points"]
    srs = filters_info.get("srs")
    if isinstance(srs, dict):
        out.update(_flatten_dict({"srs": srs}))
    return out


# E57 dateTimeValue is seconds since the GPS epoch (1980-01-06 UTC), not the Unix
# epoch — mirrors atomicmapspy's point_cloud handling.
_E57_EPOCH = datetime.datetime(1980, 1, 6, 0, 0, 0, tzinfo=datetime.timezone.utc)
# Sanity ceiling (~year 2100) so a NaN/inf/absurd value can't overflow timedelta.
_E57_MAX_SECONDS = (
    datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc) - _E57_EPOCH
).total_seconds()


def _e57_seconds_to_datetime(seconds: object) -> datetime.datetime | None:
    """Convert an E57 ``dateTimeValue`` (GPS-epoch seconds) to a UTC datetime.

    Returns None for non-numeric, non-finite, or out-of-range values.
    """
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return None
    if not (0 <= seconds <= _E57_MAX_SECONDS):
        return None
    return _E57_EPOCH + datetime.timedelta(seconds=float(seconds))


def _parse_e57_datetime_node(node: object, field: str) -> datetime.datetime | None:
    """Extract a validated datetime from an E57 StructureNode ``field``.

    Returns None when the field is absent or its ``dateTimeValue`` is unusable.
    """
    try:
        seconds = node[field]["dateTimeValue"].value()
    except Exception:
        return None
    return _e57_seconds_to_datetime(seconds)


def _extract_e57_capture_datetime(local_path: str) -> datetime.datetime | None:
    """Read a capture datetime from an E57 file via pye57, or None if unavailable.

    Tries the root ``creationDateTime`` first, then per-scan ``acquisitionStart`` /
    ``acquisitionEnd`` — mirrors atomicmapspy's
    ``PointCloud._get_capture_date_from_e57``. E57 has no LAS-style header date, so
    this is how a point cloud's date is recovered. Any failure (pye57 missing,
    unreadable file, no timestamp) returns None and lets the date fall back to
    filename/client-sidecar inference.
    """
    try:
        import pye57
    except ImportError:
        logger.warning(
            "pye57 is not installed; cannot read E57 capture date. Install it, or "
            "provide the date via a client sidecar."
        )
        return None
    try:
        with pye57.E57(local_path) as e57:
            root = e57.image_file.root()
            dt = _parse_e57_datetime_node(root, "creationDateTime")
            if dt is not None:
                return dt
            try:
                data3d = root["data3D"]
            except Exception:
                data3d = None
            if data3d is not None:
                for scan in data3d:
                    for field in ("acquisitionStart", "acquisitionEnd"):
                        dt = _parse_e57_datetime_node(scan, field)
                        if dt is not None:
                            return dt
    except Exception as exc:
        logger.warning(f"Failed to read E57 capture date from {local_path}: {exc}")
    return None


def extract_pdal_metadata(local_path: str, filename: str) -> dict:
    """Extract header metadata from a local point cloud file via the pdal info CLI.

    Calls the pdal binary rather than the Python bindings to avoid C++ ABI
    incompatibilities between the conda env and system libstdc++.
    """
    try:
        pdal_bin = find_pdal_bin()
        logger.info(f"Running pdal info for {filename}")
        try:
            data = _run_pdal_info(pdal_bin, local_path)
        except (RuntimeError, json.JSONDecodeError) as first_err:
            # Mirror atomicmapspy's nosrs fallback: a malformed/unsupported
            # header SRS can make PDAL abort the read. Retry once ignoring the
            # header SRS so we still recover bounds/dates (the CRS then relies
            # on the --spatial-reference fallback). Only LAS/LAZ expose the
            # nosrs reader option, so limit the retry to them.
            if not local_path.lower().endswith((".las", ".laz")):
                raise
            logger.warning(
                f"pdal info failed for {filename} ({first_err}); "
                "retrying with nosrs (header SRS will be ignored)."
            )
            data = _run_pdal_info(pdal_bin, local_path, nosrs=True)
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

        # Some readers (E57) leave the header metadata block empty, so `--metadata`
        # yields no bounds. Fall back to a filters.info pipeline, which reports
        # bbox / num_points / srs for any reader PDAL can open.
        if "bounds.minx" not in out:
            logger.info(
                f"Header metadata empty for {filename}; "
                "reading bounds/srs via a filters.info pipeline."
            )
            out.update(_meta_from_filters_info(_run_pdal_filters_info(pdal_bin, local_path)))

        # E57 has no LAS-style header date, so filters.info can't supply
        # creation_year/creation_doy. Recover them from the E57's own timestamps
        # via pye57 (matching atomicmapspy). Left absent → the row falls back to
        # filename/client-sidecar dates.
        if local_path.lower().endswith(".e57") and "creation_year" not in out:
            captured = _extract_e57_capture_datetime(local_path)
            if captured is not None:
                out["creation_year"] = captured.year
                out["creation_doy"] = captured.timetuple().tm_yday
                logger.info(
                    f"Read E57 capture date for {filename}: {captured.date().isoformat()} "
                    f"(creation_year={captured.year}, creation_doy={captured.timetuple().tm_yday})"
                )

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
