"""File types data-engineering accepts: ECW, GeoTIFF orthos, video telemetry, and
the formats intake classifies but skips."""

from __future__ import annotations

import json
import struct
import subprocess
from types import SimpleNamespace

import pytest
from obstore.store import LocalStore

from atomic_tools.api import check_dependencies, validate_path
from atomic_tools.commands import sidecar as sidecar_mod
from atomic_tools.commands.sidecar import _build_sidecar
from atomic_tools.io.storage import LocalBackend, ObjectStoreBackend
from atomic_tools.utils import raster
from atomic_tools.utils.companions import (
    drop_ecw_shadowed_by_tif,
    drop_mesh_textures,
    pair_telemetry,
)
from atomic_tools.utils.raster import (
    GdalStatus,
    crs_from_projection_text,
    ecw_fields_from_gdalinfo,
    extract_geotiff_georef,
    geotiff_georef,
    normalize_ecw_date,
)
from atomic_tools.utils.utils import DataTypeEnum, ExiftoolNotFoundError, find_exiftool_bin
from atomic_tools.validators.sidecar import lint_sidecar_df
from tests.test_auto_detect import patched_extractors  # noqa: F401


def _have_exiftool() -> bool:
    try:
        find_exiftool_bin()
    except ExiftoolNotFoundError:
        return False
    return True


HAVE_EXIFTOOL = _have_exiftool()
UTM12_WKT = (
    'PROJCS["WGS 84 / UTM zone 12N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
    '6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",-111],PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1],'
    'AUTHORITY["EPSG","32612"]]'
)
NO_GDAL = GdalStatus(None, None, False, "gdalinfo was not found")
GDAL_ECW = GdalStatus("/fake/gdalinfo", "GDAL 3.9.3", True)


def geotiff(
    width: int = 200,
    height: int = 100,
    epsg: int = 32612,
    origin: tuple[float, float] = (500000.0, 4100000.0),
    scale: tuple[float, float] = (1.0, 1.0),
    geographic: bool = False,
) -> bytes:
    """A minimal uncompressed GeoTIFF with tie point, pixel scale and GeoKeys."""
    bo = "<"
    pixels = b"\x80" * (width * height)

    def short(*v):
        return 3, len(v), struct.pack(bo + "H" * len(v), *v)

    def long_(*v):
        return 4, len(v), struct.pack(bo + "I" * len(v), *v)

    def double(*v):
        return 12, len(v), struct.pack(bo + "d" * len(v), *v)

    keys = [(1024, 0, 1, 2 if geographic else 1), (1025, 0, 1, 1),
            (2048 if geographic else 3072, 0, 1, epsg)]
    tags = {
        256: short(width), 257: short(height), 258: short(8), 259: short(1),
        262: short(1), 273: long_(0), 277: short(1), 278: short(height),
        279: long_(len(pixels)), 33550: double(scale[0], scale[1], 0.0),
        33922: double(0, 0, 0, origin[0], origin[1], 0),
        34735: short(1, 1, 0, len(keys), *(x for k in keys for x in k)),
    }
    values_off = 8 + 2 + len(tags) * 12 + 4
    body, extra = struct.pack(bo + "H", len(tags)), b""
    for tag in sorted(tags):
        typ, count, raw = tags[tag]
        if len(raw) > 4:
            field = struct.pack(bo + "I", values_off + len(extra))
            extra += raw + (b"\0" if len(raw) & 1 else b"")
        else:
            field = raw.ljust(4, b"\0")
        body += struct.pack(bo + "HHI", tag, typ, count) + field
    body += struct.pack(bo + "I", 0)
    data = bytearray(b"II" + struct.pack(bo + "HI", 42, 8) + body + extra + pixels)
    strip_entry = 8 + 2 + sorted(tags).index(273) * 12
    struct.pack_into(bo + "I", data, strip_entry + 8, values_off + len(extra))
    return bytes(data)


@pytest.fixture
def no_gdal(monkeypatch):
    monkeypatch.setattr(sidecar_mod, "gdal_status", lambda: NO_GDAL)


def _scan(path, data_type=None, **kwargs):
    unsupported: list[tuple[str, str]] = []
    kwargs.setdefault("client_sidecar", None)
    df, backend, detected, vector = _build_sidecar(
        directory=str(path), data_type=data_type, client_schema=None,
        unsupported_out=unsupported, **kwargs,
    )
    return df, sorted(key for key, _ in unsupported)


def _row(df, name):
    return df[df["Filename"] == name].iloc[0].to_dict()


# ---- GeoTIFF georeferencing ------------------------------------------------------


def test_geotiff_georef_from_tie_point_and_scale():
    out = geotiff_georef({
        "ProjectedCSType": 32612, "ImageWidth": 200, "ImageHeight": 100,
        "ModelTiePoint": "0 0 0 500000 4100000 0", "PixelScale": "0.5 0.25 0",
    })
    assert out == {
        "file_srs": "EPSG:32612", "bounds.minx": "500000", "bounds.miny": "4099975",
        "bounds.maxx": "500100", "bounds.maxy": "4100000",
    }


def test_geotiff_georef_pixel_is_point_shifts_half_a_pixel():
    out = geotiff_georef({
        "GeographicType": 4326, "GTRasterType": 2, "ImageWidth": 10, "ImageHeight": 10,
        "ModelTiePoint": "0 0 0 10 20 0", "PixelScale": "1 1 0",
    })
    assert (out["bounds.minx"], out["bounds.maxy"]) == ("9.5", "20.5")
    assert out["file_srs"] == "EPSG:4326"


def test_geotiff_georef_model_transform():
    transform = "2 0 0 1000 0 -2 0 5000 0 0 0 0 0 0 0 1"
    out = geotiff_georef({"ImageWidth": 10, "ImageHeight": 5, "ModelTransform": transform})
    assert out == {"bounds.minx": "1000", "bounds.miny": "4990",
                   "bounds.maxx": "1020", "bounds.maxy": "5000"}


def test_geotiff_georef_skips_user_defined_crs_and_gcps():
    out = geotiff_georef({
        "ProjectedCSType": 32767, "ImageWidth": 10, "ImageHeight": 10,
        "ModelTiePoint": "0 0 0 1 1 0 10 10 0 2 2 0", "PixelScale": "1 1 0",
    })
    assert out == {}


@pytest.mark.skipif(not HAVE_EXIFTOOL, reason="exiftool not installed")
def test_extract_geotiff_georef_with_exiftool(tmp_path):
    path = tmp_path / "ortho.tif"
    path.write_bytes(geotiff(geographic=True, epsg=4326, origin=(-112.5, 40.25),
                             scale=(0.0001, 0.0001)))
    out = extract_geotiff_georef(str(path), "ortho.tif")
    assert out["file_srs"] == "EPSG:4326"
    assert float(out["bounds.minx"]) == pytest.approx(-112.5)
    assert float(out["bounds.miny"]) == pytest.approx(40.24)


# ---- ECW --------------------------------------------------------------------------


def test_ecw_fields_from_gdalinfo():
    info = {
        "metadata": {
            "": {"VERSION": "3", "FILE_METADATA_ACQUISITION_DATE": "2023:05:01",
                 "FILE_METADATA_COMPANY": "Acme", "UNREGISTERED": "x"},
            "ECW": {"PROJ": "NUTM12", "DATUM": "WGS84", "UNITS": "METERS"},
        },
        "coordinateSystem": {"wkt": UTM12_WKT},
        "cornerCoordinates": {"upperLeft": [500000, 4100000], "lowerLeft": [500000, 4099000],
                              "upperRight": [501000, 4100000], "lowerRight": [501000, 4099000]},
        "size": [1000, 1000],
    }
    out = ecw_fields_from_gdalinfo(info)
    assert out["DateTimeOriginal"] == "2023-05-01"
    assert out["file_srs"] == "EPSG:32612"
    assert (out["bounds.minx"], out["bounds.maxy"]) == ("500000", "4100000")
    assert (out["VERSION"], out["PROJ"], out["FILE_METADATA_COMPANY"]) == ("3", "NUTM12", "Acme")
    assert "UNREGISTERED" not in out
    assert (out["ImageWidth"], out["ImageHeight"]) == ("1000", "1000")


@pytest.mark.parametrize("raw,expected", [
    ("2023-05-01", "2023-05-01"), ("2023/05/01 10:00", "2023-05-01 10:00"),
    ("05/01/2023", None), ("", None),
])
def test_normalize_ecw_date(raw, expected):
    assert normalize_ecw_date(raw) == expected


def test_crs_from_projection_files():
    assert crs_from_projection_text(UTM12_WKT) == "EPSG:32612"
    assert crs_from_projection_text("EPSG:2284") == "EPSG:2284"
    aux = f"<PAMDataset><SRS>{UTM12_WKT}</SRS></PAMDataset>"
    assert crs_from_projection_text(aux) == "EPSG:32612"
    assert crs_from_projection_text('LOCAL_CS["arbitrary",UNIT["metre",1]]') is None
    assert crs_from_projection_text("<PAMDataset><Metadata/></PAMDataset>") is None


def test_ecw_without_driver_reads_crs_from_prj(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    (tmp_path / "tile_20230501.ecw").write_text("x")
    (tmp_path / "TILE_20230501.PRJ").write_text(UTM12_WKT)
    df, _ = _scan(tmp_path)
    row = _row(df, "tile_20230501.ecw")
    assert row["DataType"] == "ortho_image"
    assert row["file_srs"] == "EPSG:32612"
    assert row["CreateDate"].startswith("2023-05-01")


def test_ecw_projection_file_order(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    (tmp_path / "a.ecw").write_text("x")
    (tmp_path / "a.ecw.aux.xml").write_text("<PAMDataset><SRS>EPSG:2284</SRS></PAMDataset>")
    (tmp_path / "a.aux.xml").write_text("<PAMDataset><SRS>EPSG:4326</SRS></PAMDataset>")
    df, _ = _scan(tmp_path)
    assert _row(df, "a.ecw")["file_srs"] == "EPSG:2284"


def test_ecw_with_driver_uses_the_header(tmp_path, patched_extractors, monkeypatch):  # noqa: F811
    (tmp_path / "a.ecw").write_text("x")
    (tmp_path / "a.prj").write_text("EPSG:4326")
    monkeypatch.setattr(sidecar_mod, "gdal_status", lambda: GDAL_ECW)
    seen = []

    def fake_header(path, label):
        seen.append(path)
        return {"file_srs": "EPSG:32612", "bounds.minx": "1", "bounds.miny": "2",
                "bounds.maxx": "3", "bounds.maxy": "4", "DateTimeOriginal": "2023-05-01"}

    monkeypatch.setattr(sidecar_mod, "read_ecw_header", fake_header)
    df, _ = _scan(tmp_path)
    row = _row(df, "a.ecw")
    assert row["file_srs"] == "EPSG:32612"  # header beats the .prj
    assert row["bounds.maxx"] == "3"
    assert seen == [str(tmp_path / "a.ecw")]


def test_read_ecw_header_runs_gdalinfo(monkeypatch):
    monkeypatch.setattr(raster, "gdal_status", lambda: GDAL_ECW)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        info = {"metadata": {"": {"VERSION": "2"}}, "size": [4, 2]}
        return SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr="")

    monkeypatch.setattr(raster.subprocess, "run", fake_run)
    assert raster.read_ecw_header("/x/a.ecw", "a.ecw") == {
        "VERSION": "2", "ImageWidth": "4", "ImageHeight": "2"}
    assert calls == [["/fake/gdalinfo", "-json", "-mdd", "ECW", "/x/a.ecw"]]


def test_gdal_status_detects_the_ecw_driver(monkeypatch, tmp_path):
    fake = tmp_path / "gdalinfo"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("GDALINFO_BIN", str(fake))

    def fake_run(cmd, **kwargs):
        out = "GDAL 3.9.3" if "--version" in cmd else (
            "Supported Formats:\n  GTiff -raster- (rw+vs): GeoTIFF\n"
            "  ECW -raster- (rov): ERDAS Compressed Wavelets (SDK 5.5)\n")
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(raster.subprocess, "run", fake_run)
    raster.gdal_status.cache_clear()
    try:
        assert raster.gdal_status() == GdalStatus(str(fake), "GDAL 3.9.3", True, None)
    finally:
        raster.gdal_status.cache_clear()


def test_ecw_on_object_store_reads_prj_without_downloading(
    tmp_path, patched_extractors, no_gdal, monkeypatch  # noqa: F811
):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.ecw").write_text("x")
    (tmp_path / "sub" / "a.prj").write_text(UTM12_WKT)
    monkeypatch.setattr(ObjectStoreBackend, "__init__", lambda self, **kwargs: None)
    backend = ObjectStoreBackend()
    backend._scheme, backend._bucket, backend._prefix = "s3", "bucket", "sub"
    backend._store = LocalStore(prefix=str(tmp_path))
    backend._single_key, backend._single_size, backend._listing = None, 0, None
    backend._folders = {}
    backend._modified = {}
    backend._listing_indexed = False
    monkeypatch.setattr(sidecar_mod, "from_path", lambda path: backend)
    monkeypatch.setattr(
        ObjectStoreBackend, "open_local", lambda self, key: pytest.fail("downloaded")
    )
    df, _ = _scan("s3://bucket/sub")
    assert _row(df, "a.ecw")["file_srs"] == "EPSG:32612"


def test_single_ecw_file_is_supported(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    (tmp_path / "a.ecw").write_text("x")
    (tmp_path / "a.tif").write_text("x")
    df, _ = _scan(tmp_path / "a.ecw")
    assert list(df["Filename"]) == ["DEFAULT", "a.ecw"]


def test_explicit_ortho_datatype_lists_ecw(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    (tmp_path / "a.ecw").write_text("x")
    df, _ = _scan(tmp_path, data_type=DataTypeEnum.ortho_image)
    assert "a.ecw" in set(df["Filename"])


# ---- intake rules -----------------------------------------------------------------


def test_tif_takes_precedence_over_same_name_ecw():
    kept, dropped = drop_ecw_shadowed_by_tif(["a/x.tif", "a/x.ecw", "a/y.ecw", "b/x.ecw"])
    assert kept == ["a/x.tif", "a/y.ecw", "b/x.ecw"] and dropped == ["a/x.ecw"]


def test_mesh_textures_are_dropped():
    present = {"m/tex.obj"}
    keys = ["m/tex.jpg", "m/photo.jpg", "m/tex.las"]
    kept, dropped = drop_mesh_textures(keys, present.__contains__)
    assert kept == ["m/photo.jpg", "m/tex.las"] and dropped == ["m/tex.jpg"]


def test_telemetry_pairs_by_similar_name():
    pairs, unmatched = pair_telemetry(
        sorted(["v/DJI_0001.MP4", "v/DJI_0002.MP4", "v/DJI_0001.SRT", "v/flight.gpx"])
    )
    assert pairs == {"v/DJI_0001.MP4": ["v/DJI_0001.SRT"]}
    assert unmatched == ["v/flight.gpx"]


def test_scan_applies_intake_rules(tmp_path, patched_extractors, no_gdal, caplog):  # noqa: F811
    for name in ("x.tif", "x.ecw", "tex.jpg", "tex.mtl", "photo.jpg", "clip.mp4",
                 "clip.srt", "lonely.gpx", "scene.ply", "old.zlas", "b.las"):
        (tmp_path / name).write_text("x")
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        df, unclassified = _scan(tmp_path)
    names = set(df["Filename"])
    assert {"x.tif", "photo.jpg", "clip.mp4", "b.las"} <= names
    assert names.isdisjoint({"x.ecw", "tex.jpg", "clip.srt", "lonely.gpx", "scene.ply", "old.zlas"})
    assert _row(df, "clip.mp4")["telemetry_sidecar"] == "clip.srt"
    assert unclassified == ["old.zlas", "scene.ply"]
    messages = " ".join(r.message for r in caplog.records)
    assert "match no video" in messages and "convert it to .laz" in messages


def test_single_unsupported_file_is_invalid_input(tmp_path, patched_extractors):  # noqa: F811
    (tmp_path / "scene.ply").write_text("x")
    result = validate_path(str(tmp_path / "scene.ply"))
    assert result.error is not None and result.error.type == "invalid_input"


def test_inventory_ignores_companions(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    for name in ("clip.mp4", "clip.srt", "x.tif", "x.ecw", "tex.jpg", "tex.obj"):
        (tmp_path / name).write_text("x")
    result = validate_path(str(tmp_path), include="findings")
    messages = " ".join(f.message for f in result.findings or [])
    assert "clip.srt" not in messages and "x.ecw" not in messages and "tex.jpg" not in messages


# ---- ortho lint -------------------------------------------------------------------


def _ortho_df(**row):
    import pandas as pd

    base = {"Filename": "o.tif", "DataType": "ortho_image", "CreateDate": "2024-06-15",
            "bounds.minx": "500000", "bounds.miny": "4099900", "bounds.maxx": "500200",
            "bounds.maxy": "4100000", "file_srs": "EPSG:32612", "fallback_srs": ""}
    base.update(row)
    default = {k: "" for k in base} | {"Filename": "DEFAULT"}
    return pd.DataFrame([default, base])


def _lint(df):
    return lint_sidecar_df(df, final=True, data_type=None, input_files_path=None)


def test_ortho_passes_on_crs_bounds_and_date_without_gps():
    report = _lint(_ortho_df())
    assert not report.errors(), report.render()


def test_ortho_without_crs_errors():
    report = _lint(_ortho_df(file_srs=""))
    assert any("ortho image row(s) have no CRS" in f.message for f in report.errors())


def test_fallback_srs_satisfies_ortho_crs():
    df = _ortho_df(file_srs="")
    df.loc[0, "fallback_srs"] = "EPSG:32612"
    assert not _lint(df).errors()


def test_ortho_missing_bounds_errors_but_ecw_only_warns():
    errors = _lint(_ortho_df(**{"bounds.minx": ""})).errors()
    assert any("'bounds.minx'" in f.message for f in errors)
    report = _lint(_ortho_df(Filename="o.ecw", **{"bounds.minx": ""}))
    assert not report.errors()
    assert any("ECW row(s)" in f.message for f in report.warnings())


@pytest.mark.skipif(not HAVE_EXIFTOOL, reason="exiftool not installed")
def test_validate_real_geotiff(tmp_path):
    (tmp_path / "ortho_20240615.tif").write_bytes(geotiff())
    result = validate_path(str(tmp_path), include="all")
    assert result.status == "passed", result.findings
    row = result.metadata[1]
    assert row["file_srs"] == "EPSG:32612" and row["bounds.maxx"] == "500200"


# ---- API --------------------------------------------------------------------------


@pytest.mark.parametrize("data_type", ["imagery", "gaussian_splat", "vector", "cad"])
def test_validate_path_rejects_unscannable_types(tmp_path, data_type):
    result = validate_path(str(tmp_path), data_type=data_type)
    assert result.error is not None and result.error.type == "invalid_input"
    assert "can't be validated" in result.error.message


def test_health_reports_ecw_as_optional(monkeypatch):
    monkeypatch.setenv("GDALINFO_BIN", "/nonexistent/gdalinfo")
    report = check_dependencies()
    assert report.ecw.optional and not report.ecw.usable
    assert report.ok == (report.exiftool.usable and report.pdal.usable and report.pye57.usable)
    raster.gdal_status.cache_clear()


def test_local_backend_sibling_lookup(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "A.PRJ").write_text("EPSG:4326")
    backend = LocalBackend(tmp_path)
    assert backend.find_sibling("d/a.prj") == "d/A.PRJ"
    assert not backend.sibling_exists("d/a.prj")
    assert backend.sibling_exists("d/A.PRJ")
    assert backend.read_text("d/A.PRJ") == "EPSG:4326"


def test_subprocess_is_not_used_for_gdal_when_missing(monkeypatch):
    monkeypatch.setenv("GDALINFO_BIN", "/nonexistent/gdalinfo")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran gdalinfo"))
    raster.gdal_status.cache_clear()
    try:
        assert raster.gdal_status().ecw is False
    finally:
        raster.gdal_status.cache_clear()


# ---- custom GeoTIFF CRS -----------------------------------------------------------


def test_projected_file_never_reports_its_base_datum_as_crs():
    out = geotiff_georef({"GTModelType": 1, "ProjectedCSType": 32767, "GeographicType": 4326})
    assert "file_srs" not in out
    assert out[raster.CUSTOM_CRS_COLUMN] == raster.CUSTOM_CRS_NOTE


def test_geographic_file_uses_geographic_type():
    assert geotiff_georef({"GTModelType": 2, "GeographicType": 4269}) == {"file_srs": "EPSG:4269"}


def test_custom_crs_is_read_with_gdal_when_available(monkeypatch):
    gdal = GdalStatus("/fake/gdalinfo", "GDAL 3.9", False)
    monkeypatch.setattr(raster, "gdal_status", lambda: gdal)
    tags = [{"GTModelType": 1, "ProjectedCSType": 32767}]
    monkeypatch.setattr(raster, "run_exiftool", lambda path, extra_args: json.dumps(tags).encode())

    def fake_run(cmd, **kwargs):
        assert cmd[:3] == ["/fake/gdalinfo", "-json", "-nomd"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {"coordinateSystem": {"wkt": UTM12_WKT}}), stderr="")

    monkeypatch.setattr(raster.subprocess, "run", fake_run)
    assert extract_geotiff_georef("/x/o.tif", "o.tif") == {"file_srs": "EPSG:32612"}


def test_custom_crs_without_gdal_is_a_warning():
    df = _ortho_df(file_srs="", **{raster.CUSTOM_CRS_COLUMN: raster.CUSTOM_CRS_NOTE})
    report = _lint(df)
    assert not report.errors(), report.render()
    assert any("custom CRS" in f.message for f in report.warnings())


# ---- dates: Flow's fallback order -------------------------------------------------


def test_undated_image_uses_file_modification_time(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    import os
    from datetime import datetime, timezone

    (tmp_path / "survey_20200101.tif").write_text("x")
    (tmp_path / "tile_20230501.ecw").write_text("x")
    stamp = datetime(2021, 3, 4, 5, 6, 7, tzinfo=timezone.utc).timestamp()
    for name in ("survey_20200101.tif", "tile_20230501.ecw"):
        os.utime(tmp_path / name, (stamp, stamp))
    df, _ = _scan(tmp_path)
    # Images: modification time before the path date, as BaseImage.get_datetime.
    assert _row(df, "survey_20200101.tif")["CreateDate"] == "2021-03-04T05:06:07+00:00"
    # ECW: never the modification time.
    assert _row(df, "tile_20230501.ecw")["CreateDate"].startswith("2023-05-01")


def test_object_store_modified_time_comes_from_the_listing(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.tif").write_text("x")
    monkeypatch.setattr(ObjectStoreBackend, "__init__", lambda self, **kwargs: None)
    backend = ObjectStoreBackend()
    backend._scheme, backend._bucket, backend._prefix = "s3", "bucket", "sub"
    backend._store = LocalStore(prefix=str(tmp_path))
    backend._single_key, backend._single_size, backend._listing = None, 0, None
    backend._folders, backend._modified = {}, {}
    backend._listing_indexed = False
    backend.list_keys(include=[".tif"], exclude=[])
    assert backend.modified_time("sub/a.tif") is not None


# ---- review fixes -----------------------------------------------------------------


def test_client_sidecar_crs_reaches_ecw_rows(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.ecw").write_text("x")
    client = tmp_path / "client.csv"
    client.write_text("Filename,file_srs\na.ecw,EPSG:2284\n")
    df, _ = _scan(tmp_path / "data", client_sidecar=str(client))
    assert _row(df, "a.ecw")["file_srs"] == "EPSG:2284"


def test_uploads_never_use_the_saved_copys_mtime(patched_extractors, no_gdal):  # noqa: F811
    from atomic_tools.api import UploadedFile, validate_upload

    result = validate_upload(UploadedFile(b"x", "plain.tif"), include="all")
    assert result.metadata is not None
    assert not result.metadata[1].get("CreateDate")


def test_validate_path_can_skip_file_dates(tmp_path, patched_extractors, no_gdal):  # noqa: F811
    (tmp_path / "plain.tif").write_text("x")
    with_dates = validate_path(str(tmp_path), include="metadata")
    without = validate_path(str(tmp_path), include="metadata", use_file_dates=False)
    assert with_dates.metadata[1].get("CreateDate")
    assert not without.metadata[1].get("CreateDate")


def test_ecw_only_sidecar_without_bounds_columns_only_warns():
    import pandas as pd

    df = pd.DataFrame([
        {"Filename": "DEFAULT", "DataType": "", "CreateDate": "", "file_srs": ""},
        {"Filename": "a.ecw", "DataType": "ortho_image", "CreateDate": "2024-06-15",
         "file_srs": "EPSG:32612"},
    ])
    report = _lint(df)
    assert not report.errors(), report.render()
    assert any("every row is an ECW" in f.message for f in report.warnings())


def test_mixed_tif_and_ecw_without_bounds_columns_errors():
    import pandas as pd

    rows = [{"Filename": n, "DataType": "ortho_image", "CreateDate": "2024-06-15",
             "file_srs": "EPSG:32612"} for n in ("a.ecw", "b.tif")]
    df = pd.DataFrame([{"Filename": "DEFAULT", "DataType": "", "CreateDate": "",
                        "file_srs": ""}, *rows])
    assert any("bounds.minx" in f.message for f in _lint(df).errors())


def test_excluded_tif_does_not_hide_its_ecw():
    kept, dropped = drop_ecw_shadowed_by_tif(["tile_rgb.tif", "tile_rgb.ecw"])
    assert "tile_rgb.ecw" in kept and dropped == []


def test_single_video_finds_its_telemetry(tmp_path, patched_extractors):  # noqa: F811
    (tmp_path / "flight1.mp4").write_text("x")
    (tmp_path / "flight1.srt").write_text("x")
    df, _ = _scan(tmp_path / "flight1.mp4")
    assert _row(df, "flight1.mp4")["telemetry_sidecar"] == "flight1.srt"


def test_single_image_ignores_telemetry_beside_it(tmp_path, patched_extractors, caplog):  # noqa: F811
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "flight.gpx").write_text("x")
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        _scan(tmp_path / "a.jpg")
    assert "match no video" not in caplog.text


def test_unsupported_files_are_reported_with_reasons(tmp_path, patched_extractors):  # noqa: F811
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "scene.ply").write_text("x")
    result = validate_path(str(tmp_path), include="details")
    assert result.summary.skipped_unsupported == 1 and result.summary.unclassified == 0
    skipped = result.details.skipped_unsupported
    assert skipped[0].file == "scene.ply" and "splat" in skipped[0].reason.lower()


def test_ecw_header_items_become_columns(tmp_path, patched_extractors, monkeypatch):  # noqa: F811
    (tmp_path / "a.ecw").write_text("x")
    monkeypatch.setattr(sidecar_mod, "gdal_status", lambda: GDAL_ECW)
    monkeypatch.setattr(
        sidecar_mod, "read_ecw_header",
        lambda path, label: {"file_srs": "EPSG:32612", "PROJ": "NUTM12", "VERSION": "3"},
    )
    df, _ = _scan(tmp_path)
    row = _row(df, "a.ecw")
    assert (row["PROJ"], row["VERSION"]) == ("NUTM12", "3")


def test_object_store_folder_index_is_built_in_one_pass(tmp_path, monkeypatch):
    for folder in ("f1", "f2"):
        (tmp_path / "sub" / folder).mkdir(parents=True)
        (tmp_path / "sub" / folder / "a.jpg").write_text("x")
    (tmp_path / "sub" / "f1" / "a.obj").write_text("x")
    monkeypatch.setattr(ObjectStoreBackend, "__init__", lambda self, **kwargs: None)
    backend = ObjectStoreBackend()
    backend._scheme, backend._bucket, backend._prefix = "s3", "bucket", "sub"
    backend._store = LocalStore(prefix=str(tmp_path))
    backend._single_key, backend._single_size, backend._listing = None, 0, None
    backend._folders, backend._modified = {}, {}
    backend._listing_indexed = False
    backend.list_keys(include=[".jpg"], exclude=[])
    monkeypatch.setattr(
        "obstore.list_with_delimiter", lambda *a, **k: pytest.fail("listed again")
    )
    assert backend.sibling_exists("sub/f1/a.obj")
    assert not backend.sibling_exists("sub/f2/a.obj")
    assert set(backend._folders) == {"sub/f1", "sub/f2"}
