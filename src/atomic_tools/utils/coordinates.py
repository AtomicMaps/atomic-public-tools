"""Coordinate reference system (CRS) transforms used by sidecar generation."""

from __future__ import annotations

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError


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
