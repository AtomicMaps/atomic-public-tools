"""Unit tests for CRS transforms and the altitude parser."""

import pytest

from atomic_tools.utils.coordinates import (
    can_transform_to_web_mercator,
    transform_center_to_web_mercator,
    transform_coordinates,
    vertical_meters_per_unit,
)
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
    "crs",
    [
        "EPSG:3857",
        "EPSG:4326",
        "EPSG:32612",
        32612,
        # WKT for a geographic/projected CRS resolves and reaches Web Mercator.
        'PROJCS["WGS 84 / UTM zone 12N",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        'PARAMETER["central_meridian",-111],UNIT["metre",1],'
        'AUTHORITY["EPSG","32612"]]',
    ],
)
def test_can_transform_to_web_mercator_true(crs):
    assert can_transform_to_web_mercator(crs) is True


@pytest.mark.parametrize(
    "crs",
    [
        "NOT_A_CRS",  # unparseable → CRSError
        # Local engineering CRS with no relation to WGS84 → ProjError.
        'LOCAL_CS["unknown",UNIT["metre",1]]',
    ],
)
def test_can_transform_to_web_mercator_false(crs):
    assert can_transform_to_web_mercator(crs) is False


def test_transform_center_to_web_mercator_returns_meters():
    # UTM 12N center → finite EPSG:3857 (meter) coordinates, Z preserved.
    out = transform_center_to_web_mercator(500000.0, 5650300.0, 1100.0, "EPSG:32612")
    assert out is not None
    x, y, z = out
    assert all(isinstance(v, float) for v in (x, y, z))
    # On the UTM 12N central meridian this is roughly -111° → ~ -12.36 million m east.
    assert x == pytest.approx(-12356463.0, abs=1.0)
    assert z == pytest.approx(1100.0, abs=5.0)


def test_transform_center_to_web_mercator_non_meter_source():
    # A US-survey-foot State Plane CRS still resolves to EPSG:3857 meters.
    out = transform_center_to_web_mercator(1500000.0, 500000.0, 30.0, "EPSG:2225")
    assert out is not None and len(out) == 3


def test_transform_center_to_web_mercator_2d_when_z_none():
    out = transform_center_to_web_mercator(500000.0, 5650300.0, None, "EPSG:32612")
    assert out is not None and len(out) == 2


@pytest.mark.parametrize("crs", ["NOT_A_CRS", 'LOCAL_CS["unknown",UNIT["metre",1]]'])
def test_transform_center_to_web_mercator_unusable_crs_returns_none(crs):
    assert transform_center_to_web_mercator(1.0, 2.0, 3.0, crs) is None


def test_vertical_meters_per_unit_metre_crs():
    # UTM 12N is in meters: factor is 1.0.
    assert vertical_meters_per_unit("EPSG:32612") == pytest.approx(1.0)


def test_vertical_meters_per_unit_ftus_crs():
    # A US survey foot state-plane CRS (2D, no vertical axis) falls back to its
    # horizontal linear unit — feet — so Z bounds convert correctly to meters.
    assert vertical_meters_per_unit("EPSG:2284") == pytest.approx(0.3048006, abs=1e-6)


def test_vertical_meters_per_unit_compound_ftus_vertical():
    # Compound CRS with an explicit US-survey-foot vertical axis: the up-axis
    # factor wins. Mirrors the LAS files that motivated this (NAD83 Virginia
    # South ftUS + a feet vertical with an unknown datum).
    wkt = (
        'COMPD_CS["x",PROJCS["NAD83 / Virginia South (ftUS)",'
        'GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
        'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Lambert_Conformal_Conic_2SP"],'
        'PARAMETER["latitude_of_origin",36.3333333333333],'
        'PARAMETER["central_meridian",-78.5],'
        'PARAMETER["standard_parallel_1",36.7666666666667],'
        'PARAMETER["standard_parallel_2",37.9666666666667],'
        'PARAMETER["false_easting",11482916.667],'
        'PARAMETER["false_northing",3280833.333],'
        'UNIT["US survey foot",0.304800609601219],AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH]],'
        'VERT_CS["unknown",VERT_DATUM["unknown",2005],'
        'UNIT["US survey foot",0.304800609601219],AXIS["Up",UP]]]'
    )
    assert vertical_meters_per_unit(wkt) == pytest.approx(0.3048006, abs=1e-6)


def test_vertical_meters_per_unit_unparseable_returns_none():
    assert vertical_meters_per_unit("NOT_A_CRS") is None


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
