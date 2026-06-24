"""Coordinate reference system (CRS) transforms used by sidecar generation."""

from __future__ import annotations

import math

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError, ProjError

# Web Mercator — the projection Flow renders point clouds in, so every source
# CRS must be reachable from it.
WEB_MERCATOR_EPSG = "EPSG:3857"


def can_transform_to_web_mercator(crs: int | str | CRS) -> bool:
    """Return True if ``crs`` parses and pyproj can build a transform to EPSG:3857.

    Catches both an unparseable CRS (``CRSError``) and a parseable-but-isolated
    CRS with no coordinate operation to Web Mercator (``ProjError``, e.g. a local
    engineering CRS), so the caller can treat both as "unusable".
    """
    try:
        Transformer.from_crs(CRS(crs), CRS(WEB_MERCATOR_EPSG), always_xy=True)
    except (CRSError, ProjError):
        return False
    return True


def transform_center_to_web_mercator(
    x: float,
    y: float,
    z: float | None,
    crs: int | str | CRS,
) -> tuple[float, ...] | None:
    """Transform a bounding-box center from ``crs`` into Web Mercator (the goal CRS).

    A stricter transformability check than :func:`can_transform_to_web_mercator`:
    it runs an actual coordinate through the transform, so it also catches a CRS
    that builds a transformer but maps real points to non-finite values (the
    ``inf`` pyproj returns for coordinates outside a projection's valid domain).

    Returns ``(x, y)`` (or ``(x, y, z)`` when ``z`` is given) in EPSG:3857 — whose
    units are meters — or ``None`` if the CRS is unparseable, has no operation to
    Web Mercator, or yields non-finite output.
    """
    try:
        transformer = Transformer.from_crs(
            CRS(crs).to_3d(), CRS(WEB_MERCATOR_EPSG).to_3d(), always_xy=True
        )
        result = transformer.transform(x, y) if z is None else transformer.transform(x, y, z)
    except (CRSError, ProjError):
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in result):
        return None
    return result


def vertical_meters_per_unit(crs: int | str | CRS) -> float | None:
    """Return the factor that converts ``crs``'s vertical (Z) unit to meters.

    Point-cloud Z bounds are stored in the source CRS's vertical unit, which for
    US state-plane data is frequently US survey feet. Converting Z via a 3D
    transform to Web Mercator is unsafe: when the file's vertical datum is
    ``unknown`` (common in LAS headers), pyproj passes Z through untouched — it
    does not even apply the foot→meter unit conversion — so a feet value
    silently survives as if it were already meters. This reads the unit straight
    off the CRS axes instead, so the elevation conversion is correct regardless
    of whether the vertical datum is known.

    Prefers the vertical ("up") axis' ``unit_conversion_factor``; for a 2D CRS
    with no vertical axis, falls back to a horizontal *linear* axis (a projected
    CRS in feet almost always stores Z in feet too). Returns ``1.0`` (assume
    meters) when no linear unit can be determined, or ``None`` if ``crs`` is
    unparseable.
    """
    try:
        parsed = CRS(crs)
    except CRSError:
        return None
    up = [a for a in parsed.axis_info if (a.direction or "").lower() == "up"]
    if up:
        return up[0].unit_conversion_factor
    horizontal = [
        a
        for a in parsed.axis_info
        if (a.direction or "").lower() in ("north", "south", "east", "west")
        and a.unit_name
        and any(token in a.unit_name.lower() for token in ("metre", "meter", "foot", "feet"))
    ]
    if horizontal:
        return horizontal[0].unit_conversion_factor
    return 1.0


def transform_coordinates(
    x: float,
    y: float,
    in_srs: int | str | CRS,
    out_srs: int | str | CRS,
    z: float | None = None,
) -> tuple[float, ...]:
    """Transform an ``(x, y[, z])`` coordinate between coordinate systems.

    Uses ``always_xy=True`` so inputs and outputs are in lon/lat (x/y) order
    regardless of the CRS axis convention. Returns ``(x, y)`` when ``z`` is
    None, otherwise ``(x, y, z)``.
    """
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError(f"X and Y values must be numbers: {type(x)}, {type(y)}")
    try:
        in_srs = CRS(in_srs).to_3d()
    except CRSError:
        raise ValueError(
            f"Input srs value `{in_srs}` is not parsable into a valid spatial reference"
        ) from None
    try:
        out_srs = CRS(out_srs).to_3d()
    except CRSError:
        raise ValueError(
            f"Output srs value `{out_srs}` is not parsable into a valid spatial reference"
        ) from None

    transformer = Transformer.from_crs(in_srs, out_srs, always_xy=True)
    if z is None:
        return transformer.transform(x, y)
    return transformer.transform(x, y, z)
