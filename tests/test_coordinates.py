"""Unit tests for CRS transforms and the altitude parser."""

import pytest

from atomic_tools.utils.coordinates import transform_coordinates
from atomic_tools.validators.values import parse_elevation


def test_transform_utm_to_wgs84_2d():
    # On the central meridian of UTM zone 12N (EPSG:32612): easting 500000 → -111°.
    x, y = 500000.0, 5650300.0
    lon, lat = transform_coordinates(x, y, "EPSG:32612", 4326)
    assert lon == pytest.approx(-111.0, abs=0.01)
    assert 50.0 < lat < 52.0


def test_transform_3d_returns_z():
    out = transform_coordinates(500000.0, 5650300.0, "EPSG:32612", 4326, z=1100.0)
    assert len(out) == 3
    lon, lat, z = out
    assert -180.0 <= lon <= 180.0
    assert -90.0 <= lat <= 90.0
    # Between geographic CRSs the ellipsoidal height is essentially unchanged.
    assert z == pytest.approx(1100.0, abs=5.0)


def test_transform_accepts_int_epsg_code():
    lon, lat = transform_coordinates(500000.0, 5650300.0, 32612, 4326)
    assert -180.0 <= lon <= 180.0


def test_transform_invalid_input_srs_raises_valueerror():
    with pytest.raises(ValueError, match="Input srs"):
        transform_coordinates(1.0, 2.0, "NOT_A_CRS", 4326)


def test_transform_invalid_output_srs_raises_valueerror():
    with pytest.raises(ValueError, match="Output srs"):
        transform_coordinates(1.0, 2.0, 4326, "NOT_A_CRS")


def test_transform_non_numeric_raises_typeerror():
    with pytest.raises(TypeError):
        transform_coordinates("a", 2.0, 4326, 4326)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1000 m Above Sea Level", 1000.0),
        ("1234.5", 1234.5),
        ("  42 ft  ", 42.0),
        ("", None),
        ("   ", None),
        (None, None),
        ("no number here", None),
    ],
)
def test_parse_elevation(value, expected):
    assert parse_elevation(value) == expected
