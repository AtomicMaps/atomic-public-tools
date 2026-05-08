"""Per-field value parsers used by `lint sidecar`.

Each parser is a `Callable[[str], tuple[bool, str | None]]` returning
(ok, error_message). The validator looks up parsers by canonical column
name in `VALIDATORS`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import datetime

import pandas as pd

ParseResult = tuple[bool, str | None]
Validator = Callable[[str], ParseResult]


_DMS_RE = re.compile(
    r"""^\s*
    (-?\d+(?:\.\d+)?)\s*deg\s*       # degrees
    (\d+(?:\.\d+)?)['′]\s*            # minutes with apostrophe
    ([\d.]+)\s*["″]?\s*              # seconds with optional double-quote
    ([NSEWnsew])?                     # optional hemisphere
    \s*$""",
    re.VERBOSE,
)

_LEADING_NUM_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)")


def _try_float(value: str) -> float | None:
    s = value.strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def _parse_dms(value: str) -> float | None:
    m = _DMS_RE.match(value)
    if not m:
        return None
    deg = float(m.group(1))
    minutes = float(m.group(2))
    seconds = float(m.group(3))
    hemi = (m.group(4) or "").upper()
    sign = -1.0 if hemi in {"S", "W"} or deg < 0 else 1.0
    abs_deg = abs(deg)
    decimal = abs_deg + minutes / 60.0 + seconds / 3600.0
    return sign * decimal


def _parse_lat_or_lon(value: str, *, lo: float, hi: float, label: str) -> ParseResult:
    s = value.strip()
    if not s:
        return False, f"{label} is empty"
    f = _try_float(s)
    if f is None:
        f = _parse_dms(s)
    if f is None:
        return False, f"{label} not parseable as decimal or DMS: {value!r}"
    if not math.isfinite(f):
        return False, f"{label} is not a finite number"
    if not (lo <= f <= hi):
        return False, f"{label} {f} is outside [{lo}, {hi}]"
    return True, None


def parse_latitude(value: str) -> ParseResult:
    return _parse_lat_or_lon(value, lo=-90.0, hi=90.0, label="latitude")


def parse_longitude(value: str) -> ParseResult:
    return _parse_lat_or_lon(value, lo=-180.0, hi=180.0, label="longitude")


def parse_altitude(value: str) -> ParseResult:
    s = value.strip()
    if not s:
        return False, "altitude is empty"
    m = _LEADING_NUM_RE.match(s)
    if not m:
        return False, f"altitude has no leading number: {value!r}"
    f = float(m.group(1))
    if not math.isfinite(f):
        return False, "altitude is not a finite number"
    return True, None


_DATETIME_FORMATS = (
    "%Y:%m:%d %H:%M:%S%z",
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y:%m:%d",
    "%Y-%m-%d",
)


def parse_datetime(value: str) -> ParseResult:
    s = value.strip()
    if not s:
        return False, "date is empty"
    try:
        ts = pd.to_datetime(s, errors="raise", utc=False)
        if pd.isna(ts):
            raise ValueError("pandas returned NaT")
        return True, None
    except Exception:  # noqa: BLE001 — pandas raises a variety of types
        pass
    for fmt in _DATETIME_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return True, None
        except ValueError:
            continue
    return False, f"date not parseable (tried ISO-8601 + EXIF formats): {value!r}"


def _parse_float_in_range(label: str, lo: float, hi: float) -> Validator:
    def _parser(value: str) -> ParseResult:
        s = value.strip()
        if not s:
            return False, f"{label} is empty"
        f = _try_float(s)
        if f is None:
            return False, f"{label} not parseable as a number: {value!r}"
        if not math.isfinite(f):
            return False, f"{label} is not finite"
        if not (lo <= f <= hi):
            return False, f"{label} {f} is outside [{lo}, {hi}]"
        return True, None

    return _parser


parse_pitch = _parse_float_in_range("pitch", -180.0, 180.0)
parse_roll = _parse_float_in_range("roll", -180.0, 180.0)
parse_heading = _parse_float_in_range("heading", -360.0, 360.0)


def parse_bound(value: str) -> ParseResult:
    s = value.strip()
    if not s:
        return False, "bound is empty"
    f = _try_float(s)
    if f is None:
        return False, f"bound not parseable as a number: {value!r}"
    if not math.isfinite(f):
        return False, "bound is not finite"
    return True, None


_CURRENT_YEAR = datetime.now().year


def _int_range_parser(label: str, lo: int, hi: int) -> Validator:
    def _parser(value: str) -> ParseResult:
        s = value.strip()
        if not s:
            return False, f"{label} is empty"
        try:
            n = int(float(s))
        except ValueError:
            return False, f"{label} not parseable as an integer: {value!r}"
        if not (lo <= n <= hi):
            return False, f"{label} {n} is outside [{lo}, {hi}]"
        return True, None

    return _parser


parse_num_points = _int_range_parser("num_points", 1, 2**63 - 1)
parse_creation_year = _int_range_parser("creation_year", 1900, _CURRENT_YEAR + 1)
parse_creation_doy = _int_range_parser("creation_doy", 1, 366)


VALIDATORS: dict[str, Validator] = {
    "GPSLatitude": parse_latitude,
    "GPSLongitude": parse_longitude,
    "GPSAltitude": parse_altitude,
    "CreateDate": parse_datetime,
    "DateTimeOriginal": parse_datetime,
    "ModifyDate": parse_datetime,
    "GPSDateStamp": parse_datetime,
    "Pitch": parse_pitch,
    "Roll": parse_roll,
    "Heading": parse_heading,
    "bounds.minx": parse_bound,
    "bounds.miny": parse_bound,
    "bounds.maxx": parse_bound,
    "bounds.maxy": parse_bound,
    "bounds.minz": parse_bound,
    "bounds.maxz": parse_bound,
    "num_points": parse_num_points,
    "creation_year": parse_creation_year,
    "creation_doy": parse_creation_doy,
}
