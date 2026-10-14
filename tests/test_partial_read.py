"""Sparse metadata reads (atomic_tools.io.partial_read) and the full-download fallback."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from obstore.store import LocalStore

import atomic_tools.commands.sidecar as sidecar_mod
from atomic_tools.io.partial_read import (
    SparseBudgetExceeded,
    SparseReadError,
    UnsupportedFormat,
    build_sparse_copy,
)
from atomic_tools.io.storage import MetadataView, ObjectStoreBackend

PIXELS = 4 * 1024 * 1024  # stand-in image/media payload that must never be fetched
# Space between IFD0 and the Exif/GPS IFDs, wider than any read-ahead, so those
# IFDs are only fetched if their pointers are followed correctly.
SUB_IFD_GAP = 1024 * 1024
HAVE_EXIFTOOL = shutil.which("exiftool") is not None
FS_TAGS = {
    "SourceFile", "FileName", "Directory", "FileModifyDate", "FileAccessDate",
    "FileInodeChangeDate", "FilePermissions",
}


# ---- synthetic file builders -------------------------------------------------------


def _ascii(s: str) -> tuple[int, bytes]:
    return 2, s.encode() + b"\0"


def _tiff(
    bo: str = "<",
    big: bool = False,
    ifd_at_end: bool = True,
    long_pointers: bool = False,
    gap: int = SUB_IFD_GAP,
) -> bytes:
    """A TIFF/BigTIFF with Exif + GPS sub-IFDs, out-of-line values, and the
    directories either before or after a large pixel block."""
    off_fmt, off_sz = ("Q", 8) if big else ("I", 4)
    cnt_fmt, cnt_sz, ent_sz = ("Q", 8, 20) if big else ("H", 2, 12)
    # BigTIFF writers may use LONG8 or plain 4-byte LONG for offsets/sub-IFDs.
    long_t = 16 if big and not long_pointers else 4
    header_sz = 16 if big else 8

    def rational(*vals: float) -> tuple[int, bytes]:
        return 5, b"".join(struct.pack(bo + "II", int(v * 1000), 1000) for v in vals)

    def short(v: int) -> tuple[int, bytes]:
        return 3, struct.pack(bo + "H", v)

    def ptr(v: int) -> tuple[int, bytes]:
        return long_t, struct.pack(bo + ("I" if long_t == 4 else "Q"), v)

    meta_base = header_sz + (PIXELS if ifd_at_end else 0)
    pixel_off = header_sz if ifd_at_end else None  # filled below for front layout

    # IFD contents; sub-IFD pointers patched after layout.
    gps = [(0, (1, b"\x02\x03\x00\x00")), (1, (2, b"N\0")),
           (2, rational(45, 26, 51.75)), (3, (2, b"W\0")), (4, rational(89, 38, 14.54))]
    exif = [(0x9003, _ascii("2024:06:15 10:30:00")), (0xA002, short(320)), (0xA003, short(200))]
    ifd0 = [(256, short(320)), (257, short(200)), (271, _ascii("Atomic Test Optics")),
            (272, _ascii("SparseCam 9000")), (273, ptr(0)), (279, ptr(PIXELS)),
            (306, _ascii("2024:06:15 10:30:00")), (34665, ptr(0)), (34853, ptr(0))]

    def ifd_size(entries: list) -> int:
        out_of_line = sum(len(v) + (len(v) & 1) for _, (_, v) in entries if len(v) > off_sz)
        return cnt_sz + len(entries) * ent_sz + off_sz + out_of_line

    ifd0_off = meta_base
    exif_off = ifd0_off + ifd_size(ifd0) + gap
    gps_off = exif_off + ifd_size(exif)
    meta_end = gps_off + ifd_size(gps)
    if pixel_off is None:
        pixel_off = meta_end
    patch = {273: pixel_off, 34665: exif_off, 34853: gps_off}
    ifd0 = [(t, ptr(patch[t]) if t in patch else v) for t, v in ifd0]

    def ser(entries: list, at: int) -> bytes:
        entries = sorted(entries)
        values_at = at + cnt_sz + len(entries) * ent_sz + off_sz
        body, extra = struct.pack(bo + cnt_fmt, len(entries)), b""
        for tag, (typ, raw) in entries:
            size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 16: 8}[typ]
            count = len(raw) // size
            if len(raw) > off_sz:
                field = struct.pack(bo + off_fmt, values_at + len(extra))
                extra += raw + (b"\0" if len(raw) & 1 else b"")
            else:
                field = raw.ljust(off_sz, b"\0")
            body += struct.pack(bo + "HH", tag, typ) + struct.pack(bo + off_fmt, count) + field
        return body + struct.pack(bo + off_fmt, 0) + extra

    meta = ser(ifd0, ifd0_off) + b"\0" * gap + ser(exif, exif_off) + ser(gps, gps_off)
    if big:
        header = (b"II" if bo == "<" else b"MM") + struct.pack(bo + "HHHQ", 43, 8, 0, ifd0_off)
    else:
        header = (b"II" if bo == "<" else b"MM") + struct.pack(bo + "HI", 42, ifd0_off)
    pixels = bytes(range(256)) * (PIXELS // 256)
    return header + (pixels + meta if ifd_at_end else meta + pixels)


def _jpeg() -> bytes:
    # metadata only; APP1 segments are capped at 64 KiB, so no gap
    exif = b"Exif\0\0" + _tiff(ifd_at_end=False, gap=0)[: -PIXELS]
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    sos = b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00"
    return b"\xff\xd8" + app1 + sos + b"\x55" * PIXELS + b"\xff\xd9"


def _mp4() -> bytes:
    ftyp = struct.pack(">I4s", 20, b"ftyp") + b"isom" + struct.pack(">I", 512) + b"isom"
    mdat = struct.pack(">I4s", 8 + PIXELS, b"mdat") + b"\x00" * PIXELS
    created = 3_800_000_000  # seconds since 1904 (2024)
    mvhd_body = (b"\x00\x00\x00\x00" + struct.pack(">IIII", created, created, 1000, 5000)
                 + struct.pack(">IH", 0x10000, 0x100) + b"\0" * 10
                 + struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
                 + b"\0" * 24 + struct.pack(">I", 2))
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    return ftyp + mdat + moov  # moov at the end, like non-"faststart" files


def _avi() -> bytes:
    avih = b"avih" + struct.pack("<I", 56) + struct.pack(
        "<14I", 33333, 0, 0, 0x10, 150, 0, 1, 0, 320, 200, 0, 0, 0, 0
    )
    hdrl = b"LIST" + struct.pack("<I", 4 + len(avih)) + b"hdrl" + avih
    movi = b"LIST" + struct.pack("<I", 4 + PIXELS) + b"movi" + b"\x00" * PIXELS
    idx1 = b"idx1" + struct.pack("<I", 16) + b"\x00" * 16
    body = b"AVI " + hdrl + movi + idx1
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _png() -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + b"\0\0\0\0"
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"x" * 100)
            + chunk(b"IEND", b""))


# ---- helpers ---------------------------------------------------------------------


class Recorder:
    """fetch(start, end) over in-memory bytes, recording each range."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.ranges: list[tuple[int, int]] = []

    def __call__(self, start: int, end: int) -> bytes:
        self.ranges.append((start, end))
        return self.data[start:end]

    def touched(self, start: int, end: int) -> bool:
        return any(a < end and b > start for a, b in self.ranges)


def _sparse(tmp_path: Path, data: bytes, name: str, **kw):
    rec = Recorder(data)
    path = tmp_path / f"sparse_{name}"
    stats = build_sparse_copy(str(path), len(data), rec, name=name, **kw)
    return path, stats, rec


def _exif(path: Path) -> dict:
    out = subprocess.run(["exiftool", "-fast", "-b", "-j", str(path)], capture_output=True)
    return {k: v for k, v in json.loads(out.stdout)[0].items() if k not in FS_TAGS}


def _assert_same_as_full(tmp_path: Path, data: bytes, sparse_path: Path, name: str) -> dict:
    full = tmp_path / f"full_{name}"
    full.write_bytes(data)
    if not HAVE_EXIFTOOL:
        pytest.skip("exiftool not installed")
    full_tags, sparse_tags = _exif(full), _exif(sparse_path)
    assert sparse_tags == full_tags
    assert "Warning" not in sparse_tags and "Error" not in sparse_tags
    return full_tags


# ---- formats ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "bo,big,ifd_at_end,long_pointers",
    [
        ("<", False, True, False), ("<", False, False, False), (">", False, True, False),
        ("<", True, True, False), (">", True, True, False),
        # 4-byte LONG pointers left-justified in BigTIFF's 8-byte value field.
        (">", True, True, True), ("<", True, True, True),
    ],
    ids=[
        "le-ifd-at-end", "le-ifd-first", "be-ifd-at-end", "bigtiff-le", "bigtiff-be",
        "bigtiff-be-long-pointers", "bigtiff-le-long-pointers",
    ],
)
def test_tiff_sparse_matches_full(tmp_path, bo, big, ifd_at_end, long_pointers):
    data = _tiff(bo, big, ifd_at_end, long_pointers)
    path, stats, rec = _sparse(tmp_path, data, "ortho.tif")
    assert stats.fmt == "tiff"
    assert stats.fetched < len(data) // 10
    header = 16 if big else 8
    pixel_start = header if ifd_at_end else len(data) - PIXELS
    # Image data in the middle is never fetched (head/readahead may clip its edges).
    middle = pixel_start + 1024 * 1024
    assert not rec.touched(middle, middle + 1024 * 1024)
    tags = _assert_same_as_full(tmp_path, data, path, "ortho.tif")
    assert tags["Make"] == "Atomic Test Optics"
    assert tags["DateTimeOriginal"] == "2024:06:15 10:30:00"
    # Every metadata byte (IFDs, sub-IFDs, out-of-line values) is real in the
    # sparse copy. (exiftool -j omits BigTIFF GPS tags in full reads too, so
    # check the bytes rather than a tag there.)
    meta_start, meta_end = (pixel_start + PIXELS, len(data)) if ifd_at_end else (0, pixel_start)
    assert path.read_bytes()[meta_start:meta_end] == data[meta_start:meta_end]
    if not big:
        assert "14.54" in str(tags["GPSLongitude"])


def test_jpeg_sparse_stops_at_scan(tmp_path):
    data = _jpeg()
    path, stats, rec = _sparse(tmp_path, data, "IMG_0001.jpg")
    assert stats.fmt == "jpeg"
    assert stats.fetched <= 256 * 1024  # the head request alone covers it
    assert not rec.touched(1024 * 1024, len(data))
    tags = _assert_same_as_full(tmp_path, data, path, "IMG_0001.jpg")
    assert tags["Model"] == "SparseCam 9000"


def test_mp4_sparse_skips_mdat_and_reads_trailing_moov(tmp_path):
    data = _mp4()
    path, stats, rec = _sparse(tmp_path, data, "clip.mp4")
    assert stats.fmt == "bmff"
    mdat_middle = 28 + PIXELS // 2
    assert not rec.touched(mdat_middle, mdat_middle + 1024)
    tags = _assert_same_as_full(tmp_path, data, path, "clip.mp4")
    assert tags["CreateDate"].startswith("2024")


def test_avi_sparse_skips_frames(tmp_path):
    data = _avi()
    path, stats, rec = _sparse(tmp_path, data, "clip.avi")
    assert stats.fmt == "riff"
    assert not rec.touched(1024 * 1024, 2 * 1024 * 1024)
    tags = _assert_same_as_full(tmp_path, data, path, "clip.avi")
    assert tags["ImageWidth"] == 320


def test_png_is_unsupported(tmp_path):
    with pytest.raises(UnsupportedFormat):
        _sparse(tmp_path, _png(), "pano.png")


def test_unknown_format_is_unsupported(tmp_path):
    with pytest.raises(UnsupportedFormat):
        _sparse(tmp_path, b"\x00" * 4096, "mystery.bin")


def test_budget_exceeded(tmp_path):
    # The first read alone (header + read-ahead) wants more than this budget.
    with pytest.raises(SparseBudgetExceeded):
        _sparse(tmp_path, _tiff(), "ortho.tif", budget=16 * 1024)


def test_malformed_jpeg_is_a_sparse_read_error(tmp_path):
    data = b"\xff\xd8" + b"\x00" * 4096  # SOI, then no marker sync
    with pytest.raises(SparseReadError):
        _sparse(tmp_path, data, "broken.jpg")


def test_head_bytes_save_a_request(tmp_path):
    data = _jpeg()
    rec = Recorder(data)
    head = data[: 256 * 1024]
    build_sparse_copy(str(tmp_path / "s.jpg"), len(data), rec, name="s.jpg", head=head)
    assert rec.ranges == []


# ---- object-store backend --------------------------------------------------------


def _backend(monkeypatch, root: Path) -> ObjectStoreBackend:
    monkeypatch.setattr(ObjectStoreBackend, "__init__", lambda self, **kwargs: None)
    backend = ObjectStoreBackend()
    backend._scheme, backend._bucket, backend._prefix = "s3", "bucket", "data"
    backend._store = LocalStore(prefix=str(root))
    backend._single_key, backend._single_size = None, 0
    backend._listing = None
    backend._folders = {}
    backend._modified = {}
    backend._listing_indexed = False
    return backend


def test_backend_metadata_view_is_sparse_for_tiff(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ortho.tif").write_bytes(_tiff())
    backend = _backend(monkeypatch, tmp_path)
    with backend.open_metadata_view("data/ortho.tif") as view:
        assert view.sparse is True
        assert view.stats.fetched < view.stats.size // 10
        assert os.path.getsize(view.path) == view.stats.size
        assert Path(view.path).name == "ortho.tif"
        path = view.path
    assert not os.path.exists(path)  # cleaned up


def test_backend_metadata_view_downloads_png_in_full(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "pano.png").write_bytes(_png())
    backend = _backend(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ObjectStoreBackend, "open_local", _fake_open_local(tmp_path / "data" / "pano.png")
    )
    with backend.open_metadata_view("data/pano.png") as view:
        assert view.sparse is False


def test_sparse_reads_can_be_disabled(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ortho.tif").write_bytes(_tiff())
    backend = _backend(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ObjectStoreBackend, "open_local", _fake_open_local(tmp_path / "data" / "ortho.tif")
    )
    monkeypatch.setenv("AM_TOOLS_SPARSE_READS", "0")
    with backend.open_metadata_view("data/ortho.tif") as view:
        assert view.sparse is False


def _fake_open_local(path: Path):
    from contextlib import contextmanager

    @contextmanager
    def open_local(self, key):
        yield str(path)

    return open_local


# ---- fallback in the extraction loop ---------------------------------------------


class _FakeBackend:
    def __init__(self, sparse: bool) -> None:
        self.sparse = sparse
        self.full_reads = 0

    def open_metadata_view(self, key):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            yield MetadataView(path="/sparse/" + key, sparse=self.sparse)

        return cm()

    def open_local(self, key):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            self.full_reads += 1
            yield "/full/" + key

        return cm()


@pytest.mark.parametrize(
    "sparse_meta",
    [{"Warning": "Truncated PNG image", "Make": "X"}, {"Error": "File format error"}, {}],
    ids=["warning", "error", "empty"],
)
def test_problem_on_sparse_read_falls_back_to_full(monkeypatch, sparse_meta):
    calls = []

    def fake_extract(path, filename):
        calls.append(path)
        return sparse_meta if path.startswith("/sparse/") else {"Make": "Real"}

    monkeypatch.setattr(sidecar_mod, "extract_exif_metadata", fake_extract)
    backend = _FakeBackend(sparse=True)
    meta = sidecar_mod._extract_exif_with_fallback(backend, "a.png", "a.png")
    assert meta == {"Make": "Real"}
    assert calls == ["/sparse/a.png", "/full/a.png"]
    assert backend.full_reads == 1


def test_clean_sparse_read_is_used_as_is(monkeypatch):
    monkeypatch.setattr(sidecar_mod, "extract_exif_metadata", lambda path, filename: {"Make": "X"})
    backend = _FakeBackend(sparse=True)
    assert sidecar_mod._extract_exif_with_fallback(backend, "a.jpg", "a.jpg") == {"Make": "X"}
    assert backend.full_reads == 0


def test_full_read_with_warning_is_not_retried(monkeypatch):
    monkeypatch.setattr(
        sidecar_mod, "extract_exif_metadata", lambda path, filename: {"Warning": "real"}
    )
    backend = _FakeBackend(sparse=False)
    assert sidecar_mod._extract_exif_with_fallback(backend, "a.jpg", "a.jpg") == {"Warning": "real"}
    assert backend.full_reads == 0
