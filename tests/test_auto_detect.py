"""Auto-detect data-type classification: vendored classifier, EXIF signal
adapter, mixed-directory scan, filter/override semantics, and lint round-trips."""

from __future__ import annotations

import pytest

from atomic_tools.commands import sidecar as sidecar_mod
from atomic_tools.commands.sidecar import _build_sidecar
from atomic_tools.utils.extractors import spherical_signals_from_exif
from atomic_tools.utils.utils import DataTypeEnum, infer_data_type
from atomic_tools.validators.sidecar import lint_sidecar_file

# ---- vendored infer_data_type (offline) --------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("a.jpg", "oriented_image"),  # ambiguous → default oriented
        ("a.jpeg", "oriented_image"),
        ("a.png", "oriented_image"),
        ("cloud.las", "point_cloud"),
        ("cloud.laz", "point_cloud"),
        ("cloud.e57", "point_cloud"),
        ("clip.mp4", "full_motion_video"),
        ("clip.mov", "full_motion_video"),
        ("map.tif", "ortho_image"),
        ("map.tiff", "ortho_image"),
        ("layer.geojson", "vector"),
        ("layer.shp", "vector"),
        ("foo.copc.las", None),  # excluded from point_cloud, claimed by nothing
        ("x_rgb.tif", None),  # excluded from ortho
        ("_thumbnail.jpg", None),  # thumbnail excluded
        ("PreviewImage.jpg", None),
        ("mystery.dat", None),
    ],
)
def test_infer_data_type_extension_routing(filename, expected):
    assert infer_data_type(filename) == expected


def test_infer_data_type_gdb_inner_file_is_vector():
    assert infer_data_type("dataset.gdb/a00000001.gdbtable") == "vector"


def test_infer_spherical_via_xmp_packet():
    packet = b'<x:xmpmeta><rdf:Description GPano:ProjectionType="equirectangular"/></x:xmpmeta>'
    assert infer_data_type("pano.jpg", xmp_packet=packet) == "spherical_image"


def test_infer_spherical_via_aspect_ratio():
    assert infer_data_type("pano.jpg", aspect_ratio=2.0) == "spherical_image"
    assert infer_data_type("pano.jpg", aspect_ratio=1.5) == "oriented_image"


def test_infer_spherical_via_user_comment():
    uc = 'ASCII{"e57_representation": "spherical"}'
    assert infer_data_type("pano.jpg", user_comment=uc) == "spherical_image"


def test_infer_no_signal_jpg_defaults_oriented():
    assert infer_data_type("plain.jpg") == "oriented_image"


# ---- spherical_signals_from_exif ---------------------------------------------


def test_signals_projection_type_becomes_packet():
    signals = spherical_signals_from_exif({"ProjectionType": "equirectangular"})
    assert signals["xmp_packet"] == b'ProjectionType="equirectangular"'


def test_signals_full_pano_width_only_becomes_packet():
    signals = spherical_signals_from_exif({"FullPanoWidthPixels": 8000})
    assert signals["xmp_packet"] == b"GPano:present"


def test_signals_image_size_combined_aspect_ratio():
    signals = spherical_signals_from_exif({"ImageSize": "8000x4000"})
    assert signals["aspect_ratio"] == pytest.approx(2.0)


def test_signals_image_size_pair_aspect_ratio():
    signals = spherical_signals_from_exif({"ImageWidth": 8000, "ImageHeight": 4000})
    assert signals["aspect_ratio"] == pytest.approx(2.0)


def test_signals_user_comment_passthrough():
    signals = spherical_signals_from_exif({"UserComment": "hello"})
    assert signals["user_comment"] == "hello"


def test_signals_empty_meta_is_empty():
    assert spherical_signals_from_exif({}) == {}


def test_signals_end_to_end_projection_type_flips_spherical():
    meta = {"ProjectionType": "equirectangular", "ImageWidth": 8000, "ImageHeight": 4000}
    assert infer_data_type("pano.jpg", **spherical_signals_from_exif(meta)) == "spherical_image"


# ---- mixed-directory scan integration ----------------------------------------

_IMAGE_META = {
    "CreateDate": "2024:06:15 10:30:00",
    "GPSLatitude": "51.0",
    "GPSLongitude": "-114.0",
    "GPSAltitude": "1000",
    "Pitch": "-90",
    "Heading": "0",
    "Roll": "0",
}
_PC_META = {
    "num_points": "1000",
    "bounds.minx": "500000",
    "bounds.maxx": "500100",
    "bounds.miny": "4500000",
    "bounds.maxy": "4500100",
    "bounds.minz": "10",
    "bounds.maxz": "20",
    "creation_year": "2024",
    "creation_doy": "100",
    "spatialreference": "EPSG:32612",
}


@pytest.fixture
def patched_extractors(monkeypatch):
    """Stub exiftool/pdal so scans need no external binaries. Records whether
    find_pdal_bin was called."""
    calls = {"find_pdal": 0}

    def fake_exif(local_path, filename):
        low = filename.lower()
        if low.endswith((".jpg", ".jpeg", ".png", ".jp2")):
            return dict(_IMAGE_META)
        if low.endswith((".mp4", ".mov", ".ts", ".avi", ".tts")):
            return {"CreateDate": "2024:06:15 10:30:00"}
        return {}

    def fake_pdal(local_path, filename):
        return dict(_PC_META)

    def fake_find_pdal():
        calls["find_pdal"] += 1
        return "/fake/pdal"

    monkeypatch.setattr(sidecar_mod, "extract_exif_metadata", fake_exif)
    monkeypatch.setattr(sidecar_mod, "extract_pdal_metadata", fake_pdal)
    monkeypatch.setattr(sidecar_mod, "find_pdal_bin", fake_find_pdal)
    return calls


def _make_mixed_dir(tmp_path):
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "b.las").write_text("x")
    (tmp_path / "c.mp4").write_text("x")
    (tmp_path / "d.geojson").write_text("x")
    (tmp_path / "x_rgb.tif").write_text("x")
    return tmp_path


def test_mixed_directory_scan(tmp_path, patched_extractors, caplog):
    _make_mixed_dir(tmp_path)
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        df, _backend, detected, vector = _build_sidecar(
            directory=str(tmp_path),
            data_type=None,
            client_sidecar=None,
            client_schema=None,
        )

    assert detected == {DataTypeEnum.oriented_image, DataTypeEnum.point_cloud, DataTypeEnum.video}

    # DataType is column 2 and holds each file's detected type.
    assert list(df.columns[:2]) == ["Filename", "DataType"]
    by_name = {row["Filename"]: row["DataType"] for _, row in df.iterrows()}
    assert by_name["a.jpg"] == "oriented_image"
    assert by_name["b.las"] == "point_cloud"
    assert by_name["c.mp4"] == "full_motion_video"
    assert by_name["DEFAULT"] == ""

    # Vector files are returned for the caller to report at the end of the run
    # (not warned about mid-scan); unclassified files still warn mid-scan.
    assert any(name.endswith("d.geojson") for name in vector)
    messages = " ".join(rec.message for rec in caplog.records)
    assert "vector sidecar generation is not supported" not in messages
    assert "matched no known data type" in messages

    # PC-only column present because point-cloud rows exist.
    assert "file_srs" in df.columns

    # find_pdal_bin ran because a point cloud was detected.
    assert patched_extractors["find_pdal"] == 1

    # Per-type blank prepending: image rows carry blank bounds cells (not their
    # requirement), and PC rows carry blank GPS cells — neither type pollutes the
    # other with a spuriously-required blank column.
    row_a = df[df["Filename"] == "a.jpg"].iloc[0]
    assert row_a["bounds.minx"] == ""
    row_b = df[df["Filename"] == "b.las"].iloc[0]
    assert row_b["GPSLatitude"] == ""


def test_preview_and_thumbnail_files_skipped_silently(tmp_path, patched_extractors, caplog):
    # Camera-generated derivatives (the exact exclude patterns that make
    # infer_data_type return None) must be dropped quietly — not counted in the
    # "no known data type" warning.
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "R0010013_ThumbnailImage.jpg").write_text("x")
    (tmp_path / "R0010013_PreviewImage.jpg").write_text("x")
    (tmp_path / "shot_thumbnail.png").write_text("x")
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        df, _backend, _detected, _vector = _build_sidecar(
            directory=str(tmp_path),
            data_type=None,
            client_sidecar=None,
            client_schema=None,
        )
    names = [n for n in df["Filename"] if n != "DEFAULT"]
    assert names == ["a.jpg"]  # only the real image became a row
    messages = " ".join(rec.message for rec in caplog.records)
    assert "matched no known data type" not in messages


def test_scan_only_pdal_when_point_cloud_present(tmp_path, patched_extractors):
    (tmp_path / "a.jpg").write_text("x")
    _df, _backend, detected, _vector = _build_sidecar(
        directory=str(tmp_path),
        data_type=None,
        client_sidecar=None,
        client_schema=None,
    )
    assert detected == {DataTypeEnum.oriented_image}
    # No point cloud → find_pdal_bin never called.
    assert patched_extractors["find_pdal"] == 0


def test_no_supported_files_raises(tmp_path, patched_extractors):
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "layer.geojson").write_text("x")  # vector → skipped
    with pytest.raises(RuntimeError, match="No supported files"):
        _build_sidecar(
            directory=str(tmp_path),
            data_type=None,
            client_sidecar=None,
            client_schema=None,
        )


# ---- filter / override semantics ---------------------------------------------


def test_filter_point_cloud_scans_only_las(tmp_path, patched_extractors):
    _make_mixed_dir(tmp_path)
    df, _backend, detected, _vector = _build_sidecar(
        directory=str(tmp_path),
        data_type=DataTypeEnum.point_cloud,
        client_sidecar=None,
        client_schema=None,
    )
    assert detected == {DataTypeEnum.point_cloud}
    names = [n for n in df["Filename"] if n != "DEFAULT"]
    assert names == ["b.las"]


def test_filter_spherical_stays_spherical_no_refinement(tmp_path, patched_extractors):
    # Plain jpgs with no GPano signals. With an explicit spherical filter they
    # must NOT be reclassified to oriented.
    (tmp_path / "1.jpg").write_text("x")
    (tmp_path / "2.jpg").write_text("x")
    df, _backend, detected, _vector = _build_sidecar(
        directory=str(tmp_path),
        data_type=DataTypeEnum.spherical_image,
        client_sidecar=None,
        client_schema=None,
    )
    assert detected == {DataTypeEnum.spherical_image}
    assert set(df[df["Filename"] != "DEFAULT"]["DataType"]) == {"spherical_image"}


# ---- lint round-trips --------------------------------------------------------


def _write_csv(path, header, rows):
    lines = [",".join(header)] + [",".join(map(str, r)) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_lint_generated_csv_per_type_required_checks(tmp_path):
    # A mixed final sidecar: an oriented row (complete) and a PC row missing a
    # required bounds column. The PC-required error must fire only for the PC row.
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude", "Pitch", "Heading", "Roll", "bounds.minx", "bounds.miny",
              "bounds.maxx", "bounds.maxy", "bounds.minz", "bounds.maxz", "num_points",
              "creation_year", "creation_doy"]
    rows = [
        ["DEFAULT"] + [""] * (len(header) - 1),
        ["a.jpg", "oriented_image", "2024:06:15 10:30:00", "51", "-114", "1000",
         "-90", "0", "0", "", "", "", "", "", "", "", "", ""],
        ["b.las", "point_cloud", "", "", "", "", "", "", "",
         "", "1", "1", "1", "1", "1", "1000", "2024", "100"],  # bounds.minx missing
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None, input_files_path=None
    )
    errors = " ".join(f.message for f in report.errors())
    assert "bounds.minx" in errors
    assert "point_cloud" in errors  # message names the type (multi-type scan)
    # The oriented row is complete → no oriented required error.
    assert "GPSLatitude" not in errors


def test_lint_csv_without_datatype_column_infers(tmp_path):
    header = ["Filename", "CreateDate", "GPSLatitude", "GPSLongitude", "GPSAltitude"]
    rows = [
        ["DEFAULT", "", "", "", ""],
        ["1.jpg", "", "", "", ""],  # oriented (inferred) missing everything
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None, input_files_path=None
    )
    assert report.has_errors()


def test_lint_unknown_datatype_value_warns_and_skips(tmp_path):
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude", "Pitch", "Heading", "Roll"]
    rows = [
        ["DEFAULT"] + [""] * (len(header) - 1),
        ["a.jpg", "oriented_image", "2024:06:15 10:30:00", "51", "-114", "1000",
         "-90", "0", "0"],
        ["weird.bin", "not_a_type"] + [""] * (len(header) - 2),
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None, input_files_path=None
    )
    assert any("unrecognized DataType" in w.message for w in report.warnings())
    # The bad row was skipped for required checks; the good row still validates.
    assert not report.has_errors()


def test_lint_final_errors_when_no_row_classifies(tmp_path):
    """Final mode must not report PASSED when it had nothing to check."""
    header = ["Filename", "DataType", "CreateDate"]
    rows = [["DEFAULT", "", ""], ["weird.bin", "not_a_type", ""]]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None, input_files_path=None
    )
    assert any(
        "No sidecar row could be classified" in e.message for e in report.errors()
    )


def test_lint_filter_excludes_other_type_rows(tmp_path):
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude", "Pitch", "Heading", "Roll"]
    rows = [
        ["DEFAULT"] + [""] * (len(header) - 1),
        ["a.jpg", "oriented_image", "2024:06:15 10:30:00", "51", "-114", "1000",
         "-90", "0", "0"],
        ["b.las", "point_cloud", "", "", "", "", "", "", ""],
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p),
        final=True,
        data_type=DataTypeEnum.oriented_image,
        schema_path=None,
        input_files_path=None,
    )
    # b.las filtered out → its missing bounds don't error; an info records it.
    assert not report.has_errors(), [f.message for f in report.errors()]
    assert any("Filtered to oriented_image" in f.message for f in report.infos())


def test_lint_final_without_datatype_works(tmp_path):
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude", "Pitch", "Heading", "Roll"]
    rows = [
        ["DEFAULT"] + [""] * (len(header) - 1),
        ["a.jpg", "oriented_image", "2024:06:15 10:30:00", "51", "-114", "1000",
         "-90", "0", "0"],
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None, input_files_path=None
    )
    assert not report.has_errors(), [f.message for f in report.errors()]


def test_mixed_missing_data_report_non_applicable_cells_blank(tmp_path):
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude", "Pitch", "Heading", "Roll", "bounds.minx", "bounds.miny",
              "bounds.maxx", "bounds.maxy", "bounds.minz", "bounds.maxz", "num_points",
              "creation_year", "creation_doy"]
    rows = [
        ["DEFAULT"] + [""] * (len(header) - 1),
        ["a.jpg", "oriented_image", "", "", "", "", "", "", "",
         "", "", "", "", "", "", "", "", ""],  # oriented missing its required fields
        ["b.las", "point_cloud", "", "", "", "", "", "", "",
         "", "", "", "", "", "", "", "", ""],  # PC missing its required fields
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None, input_files_path=None
    )
    table = report.missing_data
    assert table is not None
    by_name = {row[table.filename_column]: row for row in table.rows}
    # The oriented row is missing GPSLatitude but bounds.minx is not applicable.
    a_row = by_name["a.jpg"]
    assert a_row["GPSLatitude"] == "MISSING"
    assert a_row["bounds.minx"] == ""  # not applicable to an image row
    # The PC row is missing bounds.minx but GPSLatitude is not applicable.
    b_row = by_name["b.las"]
    assert b_row["bounds.minx"] == "MISSING"
    assert b_row["GPSLatitude"] == ""


def test_auto_mode_inventory_check(tmp_path):
    # Sidecar covers a.jpg; disk also has b.jpg (oriented) with no row and DEFAULT
    # covers nothing → an inventory error naming the missing file.
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude", "Pitch", "Heading", "Roll"]
    rows = [
        ["DEFAULT"] + [""] * (len(header) - 1),
        ["a.jpg", "oriented_image", "2024:06:15 10:30:00", "51", "-114", "1000",
         "-90", "0", "0"],
    ]
    sidecar = _write_csv(tmp_path / "s.csv", header, rows)
    pics = tmp_path / "pics"
    pics.mkdir()
    (pics / "a.jpg").write_text("x")
    (pics / "b.jpg").write_text("x")
    report = lint_sidecar_file(
        str(sidecar),
        final=True,
        data_type=None,
        schema_path=None,
        input_files_path=str(pics),
    )
    errors = " ".join(f.message for f in report.errors())
    assert "b.jpg" in errors
    assert "a.jpg" not in errors  # a.jpg has a row


def test_client_datatype_column_dropped(tmp_path, patched_extractors, caplog):
    (tmp_path / "a.jpg").write_text("x")
    client = tmp_path / "client.csv"
    client.write_text(
        "Filename,DataType,GPSAltitude\nDEFAULT,,\na.jpg,point_cloud,2222\n",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="atomic_tools.commands.sidecar"):
        df, _backend, _detected, _vector = _build_sidecar(
            directory=str(tmp_path),
            data_type=None,
            client_sidecar=str(client),
            client_schema=None,
        )
    # Detected type wins; the client DataType column is dropped with a warning.
    assert df[df["Filename"] == "a.jpg"].iloc[0]["DataType"] == "oriented_image"
    assert any("DataType" in rec.message for rec in caplog.records)


# ---- lint-side regressions ---------------------------------------------------


def test_inventory_check_covers_spherical_image_rows(tmp_path):
    """Filename-only inference always answers oriented_image for a .jpg, so a
    spherical batch must still have its --input-files inventory checked."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (data / name).write_bytes(b"x" * 10)
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude"]
    rows = [
        ["DEFAULT", "", "", "", "", ""],
        ["a.jpg", "spherical_image", "2024:06:15 10:30:00", "51", "-114", "1000"],
        ["b.jpg", "spherical_image", "2024:06:15 10:30:00", "51", "-114", "1000"],
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None,
        input_files_path=str(data),
    )
    assert "c.jpg" in report.render()


def test_inventory_check_ignores_preview_tifs(tmp_path):
    """Generation drops preview .tif files, so lint must not report them as
    input files with no sidecar row."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("R001.JPG", "R001.tif", "site.tif"):
        (data / name).write_bytes(b"x" * 10)
    header = ["Filename", "DataType", "CreateDate", "GPSLatitude", "GPSLongitude",
              "GPSAltitude", "Pitch", "Heading", "Roll"]
    rows = [
        ["DEFAULT"] + [""] * 8,
        ["R001.JPG", "oriented_image", "2024:06:15 10:30:00", "51", "-114", "1000",
         "0", "0", "0"],
        ["site.tif", "ortho_image", "2024:06:15 10:30:00", "51", "-114", "1000",
         "", "", ""],
    ]
    p = _write_csv(tmp_path / "s.csv", header, rows)
    report = lint_sidecar_file(
        str(p), final=True, data_type=None, schema_path=None,
        input_files_path=str(data),
    )
    assert "R001.tif" not in report.render()


def test_bare_lint_sidecar_prompts_for_final(tmp_path, monkeypatch):
    """`am-tools lint sidecar <path>` with no flags must ask which mode to lint
    in — defaulting to client mode silently skips every required-field check."""
    from typer.testing import CliRunner

    from atomic_tools.cli import app
    from atomic_tools.commands import lint as lint_mod

    header = ["Filename", "DataType"]
    p = _write_csv(tmp_path / "s.csv", header, [["DEFAULT", ""], ["a.jpg", "oriented_image"]])

    asked: list[str] = []
    monkeypatch.setattr(lint_mod, "_ask_final", lambda: (asked.append("final"), True)[1])
    monkeypatch.setattr(lint_mod, "_ask_ignore_missing_orientation", lambda: False)
    monkeypatch.setattr(lint_mod, "_ask_client_schema", lambda: None)
    monkeypatch.setattr(lint_mod, "_ask_input_files", lambda: None)
    monkeypatch.setattr(lint_mod, "_ask_coco", lambda: None)
    monkeypatch.setattr(lint_mod, "_ask_verbosity", lambda: "default")

    result = CliRunner().invoke(app, ["lint", "sidecar", str(p)])
    assert asked == ["final"]
    # Final mode ran: the sidecar is missing every required column.
    assert result.exit_code == 1

    asked.clear()
    CliRunner().invoke(app, ["lint", "sidecar", str(p), "--final"])
    assert asked == []
