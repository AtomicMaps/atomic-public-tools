"""Georeferencing for ortho images: GeoTIFF keys via exiftool, ECW via GDAL.

Mirrors data-engineering's ``OrthoImage``: the backend reads CRS and bounds with
rasterio, and reads ECW only through GDAL's (proprietary) ECW driver. Here GDAL
is optional; without its ECW driver an ECW's CRS comes from a ``.prj`` or
``.aux.xml`` beside it, in the same order the backend tries them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from xml.etree import ElementTree

from atomic_tools.schemas import field_registry
from atomic_tools.utils.utils import ExiftoolNotFoundError, run_exiftool

logger = logging.getLogger(__name__)

GEOTIFF_TAGS = (
    "GTModelType",
    "ModelTiePoint",
    "PixelScale",
    "ModelTransform",
    "ProjectedCSType",
    "GeographicType",
    "GTRasterType",
    "ImageWidth",
    "ImageHeight",
)
# GeoKey value for "user-defined": the CRS is spelled out in other keys, not a code.
_USER_DEFINED = 32767
_PIXEL_IS_POINT = 2
_MODEL_PROJECTED, _MODEL_GEOGRAPHIC = 1, 2
# Set on a row whose GeoTIFF spells its CRS out key by key instead of as an EPSG
# code; only GDAL can rebuild that, so without it lint warns instead of erroring.
CUSTOM_CRS_COLUMN = "file_srs_unreadable"
CUSTOM_CRS_NOTE = "custom GeoTIFF projection (install GDAL to read it)"

# Same order as the backend's _PROJECTION_SIDECAR_TEMPLATES.
PROJECTION_COMPANION_TEMPLATES = (
    "{stem}.prj",
    "{key}.aux.xml",
    "{stem}.aux.xml",
    "{key}.prj",
)
# Local/engineering CRSs can't be placed on a map, so they count as no CRS.
_UNSUPPORTED_CRS_WKT_MARKERS = (
    "EngineeringCRS",
    "LOCAL",
    "EDATUM",
    "Unknown engineering datum",
)

# ECW header items carried into the sidecar as their own columns.
ECW_HEADER_ITEMS = (
    *field_registry.ORTHO_ECW_FILE_METADATA_FIELDS,
    *field_registry.ORTHO_ECW_RASTER_PROPERTY_FIELDS,
    *field_registry.ORTHO_ECW_SRS_FIELDS,
)

_ECW_DATE_RE = re.compile(r"^(\d{4})[:/-](\d{2})[:/-](\d{2})(.*)$")
_GDAL_TIMEOUT = 120.0


def _num(value: float) -> str:
    return format(value, ".12g")


def _floats(value: object) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    try:
        return [float(v) for v in str(value).split()]
    except ValueError:
        return []


def _epsg(value: object) -> int | None:
    try:
        code = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return code if 0 < code < _USER_DEFINED else None


def bounds_from_corners(xs: list[float], ys: list[float]) -> dict[str, str]:
    return {
        "bounds.minx": _num(min(xs)),
        "bounds.miny": _num(min(ys)),
        "bounds.maxx": _num(max(xs)),
        "bounds.maxy": _num(max(ys)),
    }


def geotiff_georef(tags: dict) -> dict[str, str]:
    """``file_srs`` and ``bounds.*`` from numeric (``exiftool -n``) GeoTIFF tags."""
    out: dict[str, str] = {}
    model = _epsg(tags.get("GTModelType"))
    projected = _epsg(tags.get("ProjectedCSType"))
    geographic = _epsg(tags.get("GeographicType"))
    # A projected file's GeographicType is only its base datum, never its CRS.
    epsg = projected if model == _MODEL_PROJECTED else projected or geographic
    if epsg:
        out["file_srs"] = f"EPSG:{epsg}"
    elif model in (_MODEL_PROJECTED, _MODEL_GEOGRAPHIC):
        out[CUSTOM_CRS_COLUMN] = CUSTOM_CRS_NOTE

    try:
        width = int(tags["ImageWidth"])
        height = int(tags["ImageHeight"])
    except (KeyError, TypeError, ValueError):
        return out
    # Pixel-is-point coordinates name pixel centres; shift to the outer edges.
    half = 0.5 if str(tags.get("GTRasterType")) == str(_PIXEL_IS_POINT) else 0.0
    cols = (-half, width - half)
    rows = (-half, height - half)

    transform = _floats(tags.get("ModelTransform", ""))
    tie = _floats(tags.get("ModelTiePoint", ""))
    scale = _floats(tags.get("PixelScale", ""))
    if len(transform) == 16:
        a, b, _, d, e, f, _, h = transform[:8]
        xs = [a * c + b * r + d for c in cols for r in rows]
        ys = [e * c + f * r + h for c in cols for r in rows]
    elif len(tie) == 6 and len(scale) >= 2:
        # More than one tie point means ground control points, not an affine grid.
        i, j, _, x, y, _ = tie
        sx, sy = scale[0], scale[1]
        xs = [x + (c - i) * sx for c in cols]
        ys = [y - (r - j) * sy for r in rows]
    else:
        return out
    out.update(bounds_from_corners(xs, ys))
    return out


def extract_geotiff_georef(local_path: str, filename: str) -> dict[str, str]:
    try:
        raw = run_exiftool(
            local_path, extra_args=["-n", "-j", "-fast", *(f"-{t}" for t in GEOTIFF_TAGS)]
        )
        records = json.loads(raw.decode("utf-8", errors="replace"))
    except ExiftoolNotFoundError:
        raise
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.warning(f"GeoTIFF georeferencing could not be read for {filename}: {exc}")
        return {}
    out = geotiff_georef(records[0]) if records else {}
    if CUSTOM_CRS_COLUMN in out:
        crs = read_crs_with_gdal(local_path, filename)
        if crs:
            del out[CUSTOM_CRS_COLUMN]
            out["file_srs"] = crs
    return out


def read_crs_with_gdal(local_path: str, filename: str) -> str | None:
    """The raster's CRS as GDAL builds it, or None when GDAL isn't usable."""
    status = gdal_status()
    if not status.usable or status.path is None:
        return None
    info = _gdalinfo_json([status.path, "-json", "-nomd", local_path], filename)
    return normalize_crs((info.get("coordinateSystem") or {}).get("wkt")) if info else None


def _gdalinfo_json(cmd: list[str], filename: str) -> dict | None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_GDAL_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"exit {proc.returncode}")
        return json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        logger.warning(f"gdalinfo could not read {filename}: {exc}")
        return None


def normalize_crs(text: str | None) -> str | None:
    """``EPSG:<code>`` (or WKT) for a usable CRS; None if missing, unparseable or local."""
    if not text or not text.strip():
        return None
    from pyproj import CRS

    try:
        crs = CRS.from_user_input(text.strip())
    except Exception:  # noqa: BLE001 — pyproj raises several types for bad input
        return None
    try:
        epsg = crs.to_epsg()
    except Exception:  # noqa: BLE001
        epsg = None
    if epsg is not None:
        return f"EPSG:{epsg}"
    wkt = crs.to_wkt()
    if not wkt or any(marker in wkt for marker in _UNSUPPORTED_CRS_WKT_MARKERS):
        return None
    return wkt


def crs_from_projection_text(text: str) -> str | None:
    """CRS from a ``.prj`` (WKT or authority string) or a GDAL ``.aux.xml`` (``<SRS>``)."""
    payload = text.strip()
    if payload.startswith("<"):
        try:
            node = ElementTree.fromstring(payload).find(".//SRS")
        except ElementTree.ParseError:
            return None
        payload = (node.text or "").strip() if node is not None else ""
    return normalize_crs(payload)


def projection_companion_names(key: str) -> list[str]:
    stem = key.rsplit(".", 1)[0]
    return [t.format(stem=stem, key=key) for t in PROJECTION_COMPANION_TEMPLATES]


def normalize_ecw_date(value: str) -> str | None:
    """Only unambiguous year-first dates; a guessed day/month order could be wrong."""
    match = _ECW_DATE_RE.match(value.strip())
    if not match:
        return None
    year, month, day, rest = match.groups()
    return f"{year}-{month}-{day}{rest}"


# ---- GDAL --------------------------------------------------------------------------


@dataclass(frozen=True)
class GdalStatus:
    path: str | None
    version: str | None
    ecw: bool
    detail: str | None = None

    @property
    def usable(self) -> bool:
        return self.path is not None and self.version is not None


def find_gdalinfo_bin() -> str | None:
    override = os.environ.get("GDALINFO_BIN")
    if override:
        return override if os.path.isfile(override) and os.access(override, os.X_OK) else None
    return shutil.which("gdalinfo")


@lru_cache(maxsize=1)
def gdal_status() -> GdalStatus:
    """Whether gdalinfo runs and has the ECW driver. Cached for the process."""
    path = find_gdalinfo_bin()
    if path is None:
        return GdalStatus(None, None, False, "gdalinfo was not found (set GDALINFO_BIN).")
    try:
        version = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=20,
            stdin=subprocess.DEVNULL,
        )
        formats = subprocess.run(
            [path, "--formats"], capture_output=True, text=True, timeout=20,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return GdalStatus(path, None, False, f"Could not run gdalinfo: {e}")
    if version.returncode != 0 or formats.returncode != 0:
        err = (version.stderr or formats.stderr or "").strip().splitlines()
        return GdalStatus(path, None, False, f"gdalinfo failed: {err[0] if err else 'no output'}")
    has_ecw = any(line.strip().startswith("ECW ") for line in formats.stdout.splitlines())
    detail = None if has_ecw else "GDAL is installed but has no ECW driver."
    return GdalStatus(path, version.stdout.strip() or None, has_ecw, detail)


def read_ecw_header(local_path: str, filename: str) -> dict[str, str]:
    """Header fields GDAL's ECW driver reports, named as in the field registry."""
    status = gdal_status()
    if not status.ecw or status.path is None:
        return {}
    info = _gdalinfo_json([status.path, "-json", "-mdd", "ECW", local_path], filename)
    return ecw_fields_from_gdalinfo(info) if info else {}


def ecw_fields_from_gdalinfo(info: dict) -> dict[str, str]:
    metadata = info.get("metadata") or {}
    default = metadata.get("") or {}
    ecw_domain = metadata.get("ECW") or {}
    out: dict[str, str] = {}
    for name in (
        *field_registry.ORTHO_ECW_FILE_METADATA_FIELDS,
        *field_registry.ORTHO_ECW_RASTER_PROPERTY_FIELDS,
    ):
        if str(default.get(name, "")).strip():
            out[name] = str(default[name]).strip()
    for name in field_registry.ORTHO_ECW_SRS_FIELDS:
        if str(ecw_domain.get(name, "")).strip():
            out[name] = str(ecw_domain[name]).strip()

    raw_date = str(default.get(field_registry.ORTHO_ECW_ACQUISITION_DATE_FIELD, "")).strip()
    if raw_date:
        date = normalize_ecw_date(raw_date)
        if date:
            out[field_registry.DATETIME_FIELDS[0]] = date
        else:
            logger.warning(f"Ignoring ambiguous ECW acquisition date {raw_date!r}.")

    crs = normalize_crs((info.get("coordinateSystem") or {}).get("wkt"))
    if crs:
        out["file_srs"] = crs
    corners = info.get("cornerCoordinates") or {}
    points = [corners[k] for k in ("upperLeft", "lowerLeft", "upperRight", "lowerRight")
              if isinstance(corners.get(k), list) and len(corners[k]) >= 2]
    if len(points) == 4:
        out.update(bounds_from_corners([p[0] for p in points], [p[1] for p in points]))
    size = info.get("size")
    if isinstance(size, list) and len(size) == 2:
        out["ImageWidth"], out["ImageHeight"] = str(size[0]), str(size[1])
    return out
