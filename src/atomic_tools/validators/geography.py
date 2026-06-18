"""Batch-level spatial outlier analysis for `lint sidecar`.

The per-row value checks in ``sidecar.py`` catch coordinates that are
individually malformed (non-numeric, out of [-90, 90], etc.). This module
catches the subtler problem of coordinates that parse fine but are *wrong*
relative to the rest of the batch — a decimal point dropped, a sign flipped,
or an altitude in the wrong units — by looking at the batch as a whole:

* points that fall outside the US (approximate bounding boxes),
* a histogram of each file's distance (miles) from the batch median center,
* files lying more than 2 standard deviations from the median distance,
* the same distance/SD analysis applied to altitude.

Uses ``numpy`` only, which clients already have via pandas, so no extra
install is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from atomic_tools.validators.constants import DEFAULT_ROW_NAME, MAX_LISTED_FILES
from atomic_tools.validators.values import parse_elevation, to_decimal_degree

if TYPE_CHECKING:
    import pandas as pd

    from atomic_tools.validators.report import LintReport

_LAT_COL = "GPSLatitude"
_LON_COL = "GPSLongitude"
_ALT_COL = "GPSAltitude"

_EARTH_RADIUS_MILES = 3958.7613
_HISTOGRAM_BINS = 10
_BAR_WIDTH = 24
_SD_THRESHOLD = 2.0

# Approximate lat/lon bounding boxes for US territory. Deliberately generous —
# the goal is to catch gross outliers (wrong hemisphere, dropped sign), not to
# adjudicate borders. A point inside *any* box counts as "in the US".
_US_BBOXES: tuple[tuple[float, float, float, float], ...] = (
    (24.40, 49.40, -125.00, -66.90),  # contiguous 48 states
    (51.00, 71.50, -179.15, -129.00),  # Alaska (mainland + eastern Aleutians)
    (51.00, 53.00, 172.00, 180.00),  # Alaska — Aleutians west of the antimeridian
    (18.86, 22.24, -160.25, -154.80),  # Hawaii
    (17.62, 18.57, -67.30, -64.50),  # Puerto Rico & US Virgin Islands
)


def _in_us(lat: float, lon: float) -> bool:
    return any(
        lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
        for lat_min, lat_max, lon_min, lon_max in _US_BBOXES
    )


def _haversine_miles(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    """Great-circle distance (miles) from each ``(lat, lon)`` to the center point."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat, lon, center_lat, center_lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def _truncated_listing(items: list[str]) -> str:
    """Join display strings with ``, ``, collapsing the overflow to ``(+N more)``."""
    shown = items[:MAX_LISTED_FILES]
    tail = f" (+{len(items) - MAX_LISTED_FILES} more)" if len(items) > MAX_LISTED_FILES else ""
    return ", ".join(shown) + tail


def _render_histogram(values: np.ndarray, unit: str) -> str:
    """Return a text bar histogram of ``values`` using ``numpy.histogram``."""
    unit_suffix = f" {unit}" if unit else ""
    if np.ptp(values) == 0:
        # All identical — a histogram would just be degenerate near-zero-width
        # bins. Report the single shared value instead.
        return f"  all {values.size} files at {values[0]:,.1f}{unit_suffix}"
    counts, edges = np.histogram(values, bins=_HISTOGRAM_BINS)
    peak = int(counts.max()) or 1
    width = max(len(f"{e:,.0f}") for e in edges)
    lines = []
    for i, count in enumerate(counts):
        bar = "#" * round(_BAR_WIDTH * int(count) / peak)
        lines.append(
            f"  [{edges[i]:>{width},.0f} - {edges[i + 1]:>{width},.0f}] {unit}  "
            f"{bar} {int(count)}"
        )
    return "\n".join(lines)


def _non_default_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 0:
        return df
    first_col = df[df.columns[0]].astype(str).str.strip()
    return df[first_col != DEFAULT_ROW_NAME]


def analyze_spatial_distribution(df: pd.DataFrame, report: LintReport) -> None:
    """Add batch-level spatial outlier findings to ``report`` (no-op if no coords)."""
    rows = _non_default_rows(df)
    if rows.empty:
        return

    file_col = rows.columns[0]
    filenames = rows[file_col].astype(str).str.strip()

    _analyze_coordinates(rows, filenames, report)
    _analyze_altitude(rows, filenames, report)


def _analyze_coordinates(
    rows: pd.DataFrame,
    filenames: pd.Series,
    report: LintReport,
) -> None:
    if _LAT_COL not in rows.columns or _LON_COL not in rows.columns:
        return

    names: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    outside_us: list[str] = []
    for name, raw_lat, raw_lon in zip(filenames, rows[_LAT_COL], rows[_LON_COL], strict=True):
        lat = to_decimal_degree(raw_lat)
        lon = to_decimal_degree(raw_lon)
        # Individually-invalid coords are already flagged by the value checks;
        # only batch-analyze the ones that parsed into a sane range.
        if lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        names.append(name)
        lats.append(lat)
        lons.append(lon)
        if not _in_us(lat, lon):
            outside_us.append(name)

    if not names:
        return

    lat_arr = np.array(lats)
    lon_arr = np.array(lons)

    if outside_us:
        report.add_warning(
            f"{len(outside_us)} file(s) have coordinates outside the US: "
            f"{_truncated_listing([repr(n) for n in outside_us])}",
            fix_hint=(
                "Verify lat/lon aren't swapped or sign-flipped. (US check is an "
                "approximate bounding box — ignore if your data is legitimately abroad.)"
            ),
        )

    # Median rather than mean so a single wild outlier doesn't drag the center
    # toward itself and mask the very thing we're trying to surface.
    center_lat = float(np.median(lat_arr))
    center_lon = float(np.median(lon_arr))

    if len(names) < 2:
        report.add_info(
            f"Only 1 geolocated file; skipped distance distribution "
            f"(point at {center_lat:.5f}, {center_lon:.5f})."
        )
        return

    distances = _haversine_miles(lat_arr, lon_arr, center_lat, center_lon)
    report.add_info(
        f"Batch center (median): {center_lat:.5f}, {center_lon:.5f} "
        f"({len(names)} geolocated files). Distance from center (miles):\n"
        + _render_histogram(distances, "mi")
    )

    _flag_sd_outliers(
        names=names,
        values=distances,
        report=report,
        label="distance from center",
        unit="mi",
        high_only=True,
    )


def _analyze_altitude(
    rows: pd.DataFrame,
    filenames: pd.Series,
    report: LintReport,
) -> None:
    if _ALT_COL not in rows.columns:
        return

    names: list[str] = []
    alts: list[float] = []
    for name, raw_alt in zip(filenames, rows[_ALT_COL], strict=True):
        alt = parse_elevation(raw_alt)
        if alt is None:
            continue
        names.append(name)
        alts.append(alt)

    if len(names) < 2:
        return

    alt_arr = np.array(alts)
    report.add_info(
        f"Altitude distribution across {len(names)} files (units as provided):\n"
        + _render_histogram(alt_arr, "")
    )

    _flag_sd_outliers(
        names=names,
        values=alt_arr,
        report=report,
        label="altitude",
        unit="",
        high_only=False,
    )


def _flag_sd_outliers(
    *,
    names: list[str],
    values: np.ndarray,
    report: LintReport,
    label: str,
    unit: str,
    high_only: bool,
) -> None:
    """Warn about files more than 2 SD from the median of ``values``.

    The center is the median (not the mean) so a single wild value doesn't pull
    the reference toward itself and mask the outlier we're trying to surface.
    ``high_only`` flags only the upper tail (used for distance, which can't be
    negative); otherwise both tails are flagged (used for altitude).
    """
    center = float(np.median(values))
    std = float(values.std(ddof=1))
    if std == 0.0:
        return

    cutoff = _SD_THRESHOLD * std
    deviations = values - center if high_only else np.abs(values - center)
    outlier_idx = np.nonzero(deviations > cutoff)[0]
    if outlier_idx.size == 0:
        return

    unit_suffix = f" {unit}" if unit else ""
    detail = [f"{names[i]!r} ({values[i]:,.1f}{unit_suffix})" for i in outlier_idx]
    report.add_warning(
        f"{outlier_idx.size} file(s) are more than 2 SD from the median {label} "
        f"(median {center:,.1f}{unit_suffix}, SD {std:,.1f}{unit_suffix}): "
        + _truncated_listing(detail),
        fix_hint=(
            "Far outliers often indicate a malformed coordinate or altitude; double-check them."
        ),
    )
