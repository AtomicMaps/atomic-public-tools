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

For point clouds (which have no GPS columns) the same idea is applied to each
file's bounding-box center: the histogram shows each cloud's planar distance
(in miles) from the median bbox center, plus an elevation (Z-center, in feet)
distribution. Each bbox center is transformed into the goal CRS (Web Mercator)
first — using that row's effective CRS (its ``file_srs``, else the batch
``fallback_srs``) — so distance/elevation are reported in that CRS's units
(meters, converted to miles/feet) regardless of the source CRS; a row with no
CRS at all is an error rather than a guess at the source units.

Uses ``numpy`` only, which clients already have via pandas, so no extra
install is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
import numpy as np

from atomic_tools.utils.coordinates import (
    WEB_MERCATOR_EPSG,
    can_transform_to_web_mercator,
    transform_center_to_web_mercator,
    transform_coordinates,
    vertical_meters_per_unit,
)
from atomic_tools.validators.constants import DEFAULT_ROW_NAME, MAX_LISTED_FILES
from atomic_tools.validators.values import parse_elevation, to_decimal_degree

if TYPE_CHECKING:
    import pandas as pd

    from atomic_tools.validators.report import LintReport

_LAT_COL = "GPSLatitude"
_LON_COL = "GPSLongitude"
_ALT_COL = "GPSAltitude"

# Per-file header CRS (every point cloud row) and the batch fallback CRS (only
# the DEFAULT row, set via --spatial-reference). A row's effective CRS is its
# file_srs, else the batch fallback_srs (see sidecar._add_file_srs_column and
# sidecar._add_spatial_reference_column).
_FILE_SRS_COL = "file_srs"
_FALLBACK_SRS_COL = "fallback_srs"
_DATA_TYPE_COL = "DataType"
_POINT_CLOUD_TYPE = "point_cloud"

# Point-cloud bounding-box columns (see extractors.extract_pdal_metadata).
_BOUNDS_MINX = "bounds.minx"
_BOUNDS_MAXX = "bounds.maxx"
_BOUNDS_MINY = "bounds.miny"
_BOUNDS_MAXY = "bounds.maxy"
_BOUNDS_MINZ = "bounds.minz"
_BOUNDS_MAXZ = "bounds.maxz"

_EARTH_RADIUS_MILES = 3958.7613
# Point-cloud bounds come in source-CRS units, which for the projected CRSs
# these clouds use are meters. Convert distance to miles and elevation to feet
# so the histograms read in familiar units.
_METERS_PER_MILE = 1609.344
_FEET_PER_METER = 3.280839895
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


def _sd_outliers(values: np.ndarray, *, high_only: bool) -> tuple[np.ndarray, float, float]:
    """Return ``(mask, center, std)`` flagging files more than 2 SD from the median.

    Single source of truth for the outlier threshold and stats, shared by the
    histogram coloring, the per-histogram listing, and the warning. The center
    is the median (see module docstring). ``high_only`` flags only the upper
    tail (distance, which can't go negative); otherwise both tails are flagged.
    ``mask`` is all-False when the values don't vary.
    """
    center = float(np.median(values))
    std = float(values.std(ddof=1))
    if std == 0.0:
        return np.zeros(values.shape, dtype=bool), center, std
    cutoff = _SD_THRESHOLD * std
    deviations = values - center if high_only else np.abs(values - center)
    return deviations > cutoff, center, std


def _outlier_details(
    names: list[str], values: np.ndarray, value_mask: np.ndarray, unit: str
) -> list[str]:
    """Return ``"'name' (value unit)"`` for each file flagged in ``value_mask``."""
    unit_suffix = f" {unit}" if unit else ""
    return [
        f"{names[i]!r} ({values[i]:,.1f}{unit_suffix})" for i in np.nonzero(value_mask)[0]
    ]


def _outlier_bin_mask(values: np.ndarray, edges: np.ndarray, value_mask: np.ndarray) -> np.ndarray:
    """Return a per-bin bool mask of which bins hold any value flagged in ``value_mask``."""
    bin_mask = np.zeros(len(edges) - 1, dtype=bool)
    if value_mask.any():
        # Map each outlier value to its bin; clip so the topmost edge (a closed
        # bound in np.histogram) lands in the last bin rather than one past it.
        bins = np.clip(np.digitize(values[value_mask], edges) - 1, 0, len(edges) - 2)
        bin_mask[bins] = True
    return bin_mask


def _format_outlier_listing(details: list[str]) -> str:
    """Return a ``\\n``-prefixed block listing the outlier ``details``, or "" if none."""
    if not details:
        return ""
    lines = ["  outlier file(s) >2 SD from median:", *(f"    {d}" for d in details)]
    return "\n" + "\n".join(lines)


def _render_histogram(values: np.ndarray, unit: str, value_mask: np.ndarray) -> str:
    """Return a text bar histogram of ``values`` using ``numpy.histogram``.

    Bins containing files flagged in ``value_mask`` (>2 SD from the median) are
    printed in red.
    """
    if np.ptp(values) == 0:
        # All identical — a histogram would just be degenerate near-zero-width
        # bins. Report the single shared value instead.
        unit_suffix = f" {unit}" if unit else ""
        return f"  all {values.size} files at {values[0]:,.1f}{unit_suffix}"
    counts, edges = np.histogram(values, bins=_HISTOGRAM_BINS)
    outlier_bins = _outlier_bin_mask(values, edges, value_mask)
    peak = int(counts.max()) or 1
    # Use the fewest decimals that keep every bound label distinct, so adjacent
    # buckets don't print identical bounds after rounding.
    precision = 0
    while len({f"{e:,.{precision}f}" for e in edges}) < len(edges):
        precision += 1
    width = max(len(f"{e:,.{precision}f}") for e in edges)
    lines = []
    for i, count in enumerate(counts):
        bar = "█" * round(_BAR_WIDTH * int(count) / peak)
        label = f"{bar} {int(count)}" if count else ""
        line = (
            f"  [{edges[i]:>{width},.{precision}f} - "
            f"{edges[i + 1]:>{width},.{precision}f}] {unit}  "
            f"{label}"
        )
        if outlier_bins[i]:
            line = click.style(line, fg="red")
        lines.append(line)
    return "\n".join(lines)


def _non_default_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 0:
        return df
    first_col = df[df.columns[0]].astype(str).str.strip()
    return df[first_col != DEFAULT_ROW_NAME]


def _cell_crs(value: object) -> str | None:
    """Return a non-empty CRS string from a cell, or None for blank/NaN."""
    crs = str(value).strip()
    return crs if crs and crs.lower() != "nan" else None


def _fallback_crs(df: pd.DataFrame) -> str | None:
    """Return the batch fallback CRS from the DEFAULT row's ``fallback_srs``.

    Populated only when the sidecar was built with ``--spatial-reference``; a row
    with no ``file_srs`` of its own falls back to this batch value.
    """
    if df.shape[1] == 0 or _FALLBACK_SRS_COL not in df.columns:
        return None
    is_default = df[df.columns[0]].astype(str).str.strip() == DEFAULT_ROW_NAME
    for value in df.loc[is_default, _FALLBACK_SRS_COL]:
        crs = _cell_crs(value)
        if crs is not None:
            return crs
    return None


def _row_crs(row: pd.Series, fallback: str | None) -> str | None:
    """Return a row's effective CRS: its ``file_srs``, else the batch ``fallback``."""
    if _FILE_SRS_COL in row.index:
        crs = _cell_crs(row[_FILE_SRS_COL])
        if crs is not None:
            return crs
    return fallback


def _is_point_cloud_sidecar(rows: pd.DataFrame) -> bool:
    """True if these rows look like point clouds (carry bounds or a file_srs col)."""
    cols = set(rows.columns)
    return _FILE_SRS_COL in cols or {_BOUNDS_MINX, _BOUNDS_MAXX} <= cols


def _check_point_cloud_crs(
    rows: pd.DataFrame,
    filenames: pd.Series,
    fallback: str | None,
    report: LintReport,
) -> None:
    """Error on point-cloud rows whose CRS is missing or can't reach Web Mercator.

    Flow renders point clouds in Web Mercator, so every point-cloud row must have
    a CRS (its ``file_srs``, else the batch ``fallback_srs``) that pyproj can
    transform to EPSG:3857. Two failure modes are reported separately: no CRS at
    all, and a CRS present but not transformable (these are the rows the generated
    sidecar flags ``no`` in ``crs_web_mercator_ok``).

    In a mixed sidecar the ``file_srs`` column exists for every row, but only
    point-cloud rows require a CRS — image/video rows legitimately have none. So
    when a ``DataType`` column is present, restrict the check to point-cloud rows;
    older sidecars with no ``DataType`` column fall back to checking every row.
    """
    if not _is_point_cloud_sidecar(rows):
        return
    if _DATA_TYPE_COL in rows.columns:
        types = rows[_DATA_TYPE_COL].astype(str).str.strip().tolist()
    else:
        types = [_POINT_CLOUD_TYPE] * len(rows)

    missing: list[str] = []
    untransformable: list[str] = []
    for name, (_, row), dtype in zip(filenames, rows.iterrows(), types, strict=True):
        if dtype != _POINT_CLOUD_TYPE:
            continue
        crs = _row_crs(row, fallback)
        if crs is None:
            missing.append(name)
        elif not _row_crs_reaches_web_mercator(row, crs):
            untransformable.append(name)

    if missing:
        report.add_error(
            f"{len(missing)} point cloud row(s) have no CRS in either "
            f"'{_FILE_SRS_COL}' or '{_FALLBACK_SRS_COL}': "
            f"{_truncated_listing([repr(n) for n in missing])}",
            fix_hint=(
                "Regenerate the sidecar so PDAL records each file's CRS in "
                f"'{_FILE_SRS_COL}', or pass --spatial-reference to set a "
                f"'{_FALLBACK_SRS_COL}'."
            ),
        )
    if untransformable:
        report.add_error(
            f"{len(untransformable)} point cloud row(s) have a CRS that cannot be "
            f"transformed to {WEB_MERCATOR_EPSG}: "
            f"{_truncated_listing([repr(n) for n in untransformable])}",
            fix_hint=(
                "Flow renders point clouds in Web Mercator; the CRS in "
                f"'{_FILE_SRS_COL}' (or the '{_FALLBACK_SRS_COL}' fallback) must be "
                "convertible to it. Correct the CRS or pass a valid "
                "--spatial-reference."
            ),
        )


def _row_crs_reaches_web_mercator(row: pd.Series, crs: str) -> bool:
    """True if ``crs`` transforms to Web Mercator for this row.

    Prefers transforming the row's bbox center (when ``bounds.*`` are present) —
    stricter than merely building a transformer, since it also catches a CRS that
    maps real coordinates to non-finite values — and falls back to a CRS-only
    check otherwise.
    """
    cx = _bounds_centroid(row, _BOUNDS_MINX, _BOUNDS_MAXX)
    cy = _bounds_centroid(row, _BOUNDS_MINY, _BOUNDS_MAXY)
    if cx is not None and cy is not None:
        return transform_center_to_web_mercator(cx, cy, None, crs) is not None
    return can_transform_to_web_mercator(crs)


def analyze_spatial_distribution(df: pd.DataFrame, report: LintReport) -> None:
    """Add batch-level spatial outlier findings to ``report`` (no-op if no coords)."""
    fallback = _fallback_crs(df)
    rows = _non_default_rows(df)
    if rows.empty:
        return

    file_col = rows.columns[0]
    filenames = rows[file_col].astype(str).str.strip()

    _analyze_coordinates(rows, filenames, report)
    _analyze_altitude(rows, filenames, report)
    _check_point_cloud_crs(rows, filenames, fallback, report)
    _analyze_point_cloud_bounds(rows, filenames, report, fallback)


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
    outlier_mask, center, std = _sd_outliers(distances, high_only=True)
    details = _outlier_details(names, distances, outlier_mask, "mi")
    report.add_info(
        f"Batch center (median): {center_lat:.5f}, {center_lon:.5f} "
        f"({len(names)} geolocated files). Distance from center (miles):\n"
        + _render_histogram(distances, "mi", outlier_mask)
        + _format_outlier_listing(details)
    )

    _flag_sd_outliers(
        details=details,
        center=center,
        std=std,
        report=report,
        label="distance from center",
        unit="mi",
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
    outlier_mask, center, std = _sd_outliers(alt_arr, high_only=False)
    details = _outlier_details(names, alt_arr, outlier_mask, "")
    report.add_info(
        f"Altitude distribution across {len(names)} files (units as provided):\n"
        + _render_histogram(alt_arr, "", outlier_mask)
        + _format_outlier_listing(details)
    )

    _flag_sd_outliers(
        details=details,
        center=center,
        std=std,
        report=report,
        label="altitude",
        unit="",
    )


def _bounds_centroid(
    row: pd.Series, lo_col: str, hi_col: str
) -> float | None:
    """Return the midpoint of ``[lo_col, hi_col]`` for one row, or None if unparseable.

    Reuses :func:`parse_elevation`, which extracts a leading signed number, so it
    copes with the plain numeric bounds PDAL emits (including negatives).
    """
    if lo_col not in row.index or hi_col not in row.index:
        return None
    lo = parse_elevation(row[lo_col])
    hi = parse_elevation(row[hi_col])
    if lo is None or hi is None:
        return None
    return (lo + hi) / 2.0


def _centers_to_web_mercator(
    cx_list: list[float],
    cy_list: list[float],
    crs_list: list[str],
) -> tuple[list[float], list[float]] | None:
    """Transform every bbox center's X/Y from its own CRS into Web Mercator.

    Each center is transformed with its row's effective CRS (``crs_list``), so a
    batch spanning multiple source CRSs still lands in one comparable frame.
    All-or-nothing: returns None if any center fails to transform (an unusable
    CRS, or a point outside the projection's valid domain).

    Z is deliberately *not* transformed here — pyproj's 3D transform leaves Z
    untouched when the source vertical datum is ``unknown`` (so a feet value
    would survive uncorrected). Elevation is converted from source units
    separately via :func:`vertical_meters_per_unit`.
    """
    tx: list[float] = []
    ty: list[float] = []
    for cx, cy, crs in zip(cx_list, cy_list, crs_list, strict=True):
        result = transform_center_to_web_mercator(cx, cy, None, crs)
        if result is None:
            return None
        tx.append(result[0])
        ty.append(result[1])
    return tx, ty


def _format_center(center_x: float, center_y: float) -> str:
    """Describe a Web Mercator batch center as lat/lon (with the raw EPSG:3857 X/Y).

    The histograms work in Web Mercator, but a center in EPSG:3857 meters is
    unreadable as a location, so lead with the lat/lon (EPSG:4326) a human can
    drop into a map. Falls back to just the X/Y if the back-transform fails.
    """
    try:
        lon, lat = transform_coordinates(center_x, center_y, WEB_MERCATOR_EPSG, "EPSG:4326")
    except Exception:
        return f"{center_x:,.3f}, {center_y:,.3f} (EPSG:3857)"
    return (
        f"{lat:.6f}, {lon:.6f} (lat/lon) "
        f"[EPSG:3857 {center_x:,.3f}, {center_y:,.3f}]"
    )


def _analyze_point_cloud_bounds(
    rows: pd.DataFrame,
    filenames: pd.Series,
    report: LintReport,
    fallback: str | None,
) -> None:
    """Distance-from-centroid histogram for point clouds, from each file's bbox.

    Mirrors :func:`_analyze_coordinates` but for point clouds: the "location" of
    each file is the center of its bounding box (midpoint of min/max in X and Y),
    the batch center is the median of those centers, and distance is the planar
    (Euclidean) distance in miles. Each bbox center's X/Y is transformed into the
    goal CRS (Web Mercator, meters) using that row's effective CRS (its
    ``file_srs``, else the batch ``fallback``) so distances are comparable
    regardless of the source CRS's units, and the batch center is reported in
    lat/lon. Rows with no CRS are skipped here (and flagged by
    :func:`_check_point_cloud_crs`). The Z-center is converted to feet from the
    source CRS's own vertical unit (not the reprojection, which can't be trusted
    to put Z in meters — see :func:`vertical_meters_per_unit`) and fed to a
    separate elevation distribution mirroring :func:`_analyze_altitude`. No-op
    when bounds absent.
    """
    if not {_BOUNDS_MINX, _BOUNDS_MAXX, _BOUNDS_MINY, _BOUNDS_MAXY} <= set(rows.columns):
        return

    names: list[str] = []
    cx_list: list[float] = []
    cy_list: list[float] = []
    cz_feet_list: list[float | None] = []
    crs_list: list[str] = []
    for name, (_, row) in zip(filenames, rows.iterrows(), strict=True):
        cx = _bounds_centroid(row, _BOUNDS_MINX, _BOUNDS_MAXX)
        cy = _bounds_centroid(row, _BOUNDS_MINY, _BOUNDS_MAXY)
        crs = _row_crs(row, fallback)
        # Individually-malformed bounds are already flagged by the value checks,
        # and CRS-less rows by _check_point_cloud_crs; only histogram the files
        # whose X/Y bounds parsed and that carry a CRS to reach the goal CRS.
        if cx is None or cy is None or crs is None:
            continue
        names.append(name)
        cx_list.append(cx)
        cy_list.append(cy)
        crs_list.append(crs)
        # Convert the Z-center from the source CRS's *own* vertical unit (often
        # US survey feet for state-plane data) to feet for display. Done from the
        # source unit rather than from a 3D reprojection because pyproj leaves Z
        # untouched when the vertical datum is unknown — so reprojected Z can't
        # be trusted to be in meters.
        cz = _bounds_centroid(row, _BOUNDS_MINZ, _BOUNDS_MAXZ)
        vfactor = vertical_meters_per_unit(crs)
        if cz is None or vfactor is None:
            cz_feet_list.append(None)
        else:
            cz_feet_list.append(cz * vfactor * _FEET_PER_METER)

    if not names:
        return

    # Transform the centers' X/Y into the goal CRS so planar distances are in
    # Web Mercator meters; this doubles as a transformability check at lint time.
    transformed = _centers_to_web_mercator(cx_list, cy_list, crs_list)
    if transformed is None:
        report.add_error(
            "Could not transform the point-cloud bounding-box centers to the goal "
            "CRS (EPSG:3857), so their distance/elevation units are unknown.",
            fix_hint=(
                f"Ensure each row's '{_FILE_SRS_COL}' (or the batch "
                f"'{_FALLBACK_SRS_COL}') is a CRS pyproj can transform to EPSG:3857."
            ),
        )
        return
    cx_list, cy_list = transformed
    crs_note = " (reprojected to EPSG:3857)"

    cx_arr = np.array(cx_list)
    cy_arr = np.array(cy_list)

    # Median rather than mean so a single wild outlier doesn't drag the center
    # toward itself and mask the very thing we're trying to surface.
    center_x = float(np.median(cx_arr))
    center_y = float(np.median(cy_arr))
    center_label = _format_center(center_x, center_y)

    if len(names) < 2:
        report.add_info(
            f"Only 1 point cloud; skipped distance distribution "
            f"(bbox center at {center_label})."
        )
    else:
        distances = np.hypot(cx_arr - center_x, cy_arr - center_y) / _METERS_PER_MILE
        outlier_mask, center, std = _sd_outliers(distances, high_only=True)
        details = _outlier_details(names, distances, outlier_mask, "mi")
        report.add_info(
            f"Batch center (median bbox center): {center_label} "
            f"({len(names)} point clouds). Distance from center (miles){crs_note}:\n"
            + _render_histogram(distances, "mi", outlier_mask)
            + _format_outlier_listing(details)
        )
        _flag_sd_outliers(
            details=details,
            center=center,
            std=std,
            report=report,
            label="distance from center",
            unit="mi",
        )

    _analyze_point_cloud_elevation(names, cz_feet_list, report)


def _analyze_point_cloud_elevation(
    names: list[str], cz_feet_list: list[float | None], report: LintReport
) -> None:
    """Z-center distribution across point clouds (analog of :func:`_analyze_altitude`).

    ``cz_feet_list`` is already in feet, converted from each file's own source
    vertical unit (see :func:`_analyze_point_cloud_bounds`).
    """
    z_names = [n for n, z in zip(names, cz_feet_list, strict=True) if z is not None]
    z_vals = [z for z in cz_feet_list if z is not None]
    if len(z_names) < 2:
        return

    z_arr = np.array(z_vals)
    outlier_mask, center, std = _sd_outliers(z_arr, high_only=False)
    details = _outlier_details(z_names, z_arr, outlier_mask, "ft")
    report.add_info(
        f"Elevation distribution across {len(z_names)} point clouds "
        f"(bbox Z center, feet):\n"
        + _render_histogram(z_arr, "ft", outlier_mask)
        + _format_outlier_listing(details)
    )
    _flag_sd_outliers(
        details=details,
        center=center,
        std=std,
        report=report,
        label="elevation",
        unit="ft",
    )


def _flag_sd_outliers(
    *,
    details: list[str],
    center: float,
    std: float,
    report: LintReport,
    label: str,
    unit: str,
) -> None:
    """Warn about the outlier ``details`` built by :func:`_outlier_details`.

    The same ``details`` feed the per-histogram listing and the red histogram
    bins, so the warning, listing, and coloring all reference the same set.
    """
    if not details:
        return

    unit_suffix = f" {unit}" if unit else ""
    report.add_warning(
        f"{len(details)} file(s) are more than 2 SD from the median {label} "
        f"(median {center:,.1f}{unit_suffix}, SD {std:,.1f}{unit_suffix}): "
        + _truncated_listing(details),
        fix_hint=(
            "Far outliers often indicate a malformed coordinate or altitude; double-check them."
        ),
    )
