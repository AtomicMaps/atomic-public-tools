"""Sparse local copies of remote files: fetch only the bytes exiftool reads.

exiftool needs a local file, but for most formats it reads only a small part
of it — the JPEG header, the TIFF directories, the MP4 ``moov`` box. Downloading
a 1.5 GB ortho to read 400 KB of tags is the dominant cost of validating remote
data. :func:`build_sparse_copy` creates a local file of the object's full size
(sparse, so it takes no disk space), fetches just the byte ranges exiftool will
read for the detected format, and leaves everything else as zeros. exiftool then
runs on it unchanged — same flags, same output — because every byte it reads is
real.

Formats whose metadata can't be located cheaply (PNG walks every chunk; unknown
formats) raise :class:`UnsupportedFormat`; malformed structure raises
:class:`SparseReadError`; an unexpectedly large fetch raises
:class:`SparseBudgetExceeded`. Callers fall back to a full download in all three
cases, and also when exiftool reports a Warning/Error on the sparse copy (see
``commands.sidecar._extract_exif_with_fallback``).

Validated against 82 real files (TIFF orthos with the IFD 74% into a 1.5 GB
file, JPEG, MP4, AVI, TTS): exiftool output identical to a full download, ~92x
less data fetched.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable
from dataclasses import dataclass

# fetch(start, end) -> bytes for [start, end); may return fewer bytes at EOF.
Fetch = Callable[[int, int], bytes]

HEAD_BYTES = 256 * 1024
_READAHEAD = 64 * 1024
# Abort (and fall back to a full download) past this much fetched data.
DEFAULT_BUDGET = 128 * 1024 * 1024
_MAX_VALUE_BYTES = 64 * 1024 * 1024
_MAX_IFDS = 4000
_MAX_ENTRIES = 10_000
_TS_EDGE_BYTES = 4 * 1024 * 1024


class SparseReadError(Exception):
    """The file's structure couldn't be walked (truncated/malformed)."""


class UnsupportedFormat(SparseReadError):
    """No cheap way to locate this format's metadata; read the whole file."""


class SparseBudgetExceeded(SparseReadError):
    """The walk wanted more data than the budget allows."""


@dataclass(frozen=True)
class SparseStats:
    fmt: str
    requests: int
    fetched: int
    size: int


class SparseCopy:
    """A local file the size of the remote object, filled in only where fetched."""

    def __init__(
        self,
        path: str,
        size: int,
        fetch: Fetch,
        *,
        head: bytes = b"",
        budget: int = DEFAULT_BUDGET,
    ) -> None:
        self.size = size
        self._fetch = fetch
        self._budget = budget
        self._have: list[tuple[int, int]] = []  # merged [start, end) intervals
        self.requests = 0
        self.fetched = 0
        self._fh = open(path, "w+b")  # noqa: SIM115 — closed in close()
        self._fh.truncate(size)
        if head:
            self._store(0, head[:size])

    def close(self) -> None:
        self._fh.close()

    def _store(self, start: int, data: bytes) -> None:
        self._fh.seek(start)
        self._fh.write(data)
        merged: list[tuple[int, int]] = []
        for a, b in sorted([*self._have, (start, start + len(data))]):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self._have = merged

    def _gaps(self, start: int, end: int) -> list[tuple[int, int]]:
        gaps, cur = [], start
        for a, b in self._have:
            if b <= cur:
                continue
            if a >= end:
                break
            if a > cur:
                gaps.append((cur, a))
            cur = max(cur, b)
            if cur >= end:
                break
        if cur < end:
            gaps.append((cur, end))
        return gaps

    def ensure(self, offset: int, length: int, readahead: int = 0) -> None:
        """Make bytes [offset, offset+length) real, fetching what's missing."""
        if length <= 0 or offset < 0 or offset >= self.size:
            return
        end = min(self.size, offset + length)
        for a, b in self._gaps(offset, end):
            b = min(self.size, max(b, a + readahead))
            if self.fetched + (b - a) > self._budget:
                raise SparseBudgetExceeded(
                    f"sparse read would exceed {self._budget} bytes"
                )
            data = bytes(self._fetch(a, b))
            self.requests += 1
            self.fetched += len(data)
            if not data:
                raise SparseReadError(f"empty read at offset {a}")
            self._store(a, data)

    def read(self, offset: int, length: int, readahead: int = 0) -> bytes:
        self.ensure(offset, length, readahead)
        self._fh.seek(offset)
        return self._fh.read(max(0, min(length, self.size - offset)))

    def flush(self) -> None:
        self._fh.flush()


# ---- TIFF / BigTIFF (also the Exif block inside JPEG APP1) -----------------------

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4,
              12: 8, 13: 4, 16: 8, 17: 8, 18: 8}
_SUB_IFD_TAGS = {34665, 34853, 40965, 330}  # Exif, GPS, Interop, SubIFDs
_MAKERNOTE = 37500
# Offsets at which maker notes commonly start an IFD (bare, or after a
# vendor header such as "Nikon\0\2\x10\0\0" / "OLYMPUS\0II").
_MAKERNOTE_IFD_SKIPS = (0, 8, 10, 12, 18)
_THUMB_OFFSET, _THUMB_LENGTH = 513, 514


def _walk_tiff(sp: SparseCopy, base: int = 0) -> None:
    """Fetch every IFD (chain + sub-IFDs + maker-note IFDs) and every tag value
    stored out of line. ``base`` is where the TIFF header starts."""
    hdr = sp.read(base, 16, readahead=_READAHEAD)
    if len(hdr) < 8 or hdr[:2] not in (b"II", b"MM"):
        raise SparseReadError("not a TIFF header")
    bo = "<" if hdr[:2] == b"II" else ">"
    big = struct.unpack(bo + "H", hdr[2:4])[0] == 43
    off_fmt, off_sz = ("Q", 8) if big else ("I", 4)
    cnt_fmt, cnt_sz = ("Q", 8) if big else ("H", 2)
    ent_sz = 20 if big else 12
    first = struct.unpack(bo + off_fmt, hdr[8:16] if big else hdr[4:8])[0]

    queue: list[tuple[int, bool]] = [(first, True)]  # (offset, strict)
    seen: set[int] = set()
    while queue:
        ifd, strict = queue.pop(0)
        if not ifd or ifd in seen or base + ifd + cnt_sz > sp.size:
            continue
        if len(seen) >= _MAX_IFDS:
            raise SparseReadError("too many IFDs")
        seen.add(ifd)
        n = struct.unpack(bo + cnt_fmt, sp.read(base + ifd, cnt_sz, _READAHEAD))[0]
        if n == 0 or n > _MAX_ENTRIES:
            if strict:
                raise SparseReadError(f"implausible IFD entry count {n}")
            continue  # a maker-note guess that wasn't an IFD
        raw = sp.read(base + ifd + cnt_sz, n * ent_sz + off_sz, _READAHEAD)
        thumb_off = thumb_len = None
        for i in range(n):
            e = raw[i * ent_sz:(i + 1) * ent_sz]
            if len(e) < ent_sz:
                break
            tag, typ = struct.unpack(bo + "HH", e[:4])
            count = struct.unpack(bo + off_fmt, e[4:12] if big else e[4:8])[0]
            value = e[12:20] if big else e[8:12]
            nbytes = _TYPE_SIZE.get(typ, 1) * count
            out_of_line = nbytes > off_sz
            voff = struct.unpack(bo + off_fmt, value)[0] if out_of_line else None
            if out_of_line:
                sp.ensure(base + voff, min(nbytes, _MAX_VALUE_BYTES))
            if tag in _SUB_IFD_TAGS:
                if out_of_line:
                    data = sp.read(base + voff, nbytes)
                    width = 8 if (big or typ in (16, 18)) else 4
                    fmt = "Q" if width == 8 else "I"
                    ptrs = [struct.unpack(bo + fmt, data[j:j + width])[0]
                            for j in range(0, len(data) - width + 1, width)]
                else:
                    ptrs = [struct.unpack(bo + off_fmt, value)[0]] if count == 1 else []
                queue += [(p, False) for p in ptrs]
            elif tag == _MAKERNOTE and out_of_line:
                queue += [(voff + skip, False) for skip in _MAKERNOTE_IFD_SKIPS]
            elif tag == _THUMB_OFFSET and not out_of_line:
                thumb_off = struct.unpack(bo + off_fmt, value)[0]
            elif tag == _THUMB_LENGTH and not out_of_line:
                thumb_len = struct.unpack(bo + off_fmt, value)[0]
        if thumb_off and thumb_len and thumb_len <= _MAX_VALUE_BYTES:
            sp.ensure(base + thumb_off, thumb_len)  # exiftool -b extracts it
        nxt = raw[n * ent_sz:n * ent_sz + off_sz]
        if len(nxt) == off_sz:
            queue.append((struct.unpack(bo + off_fmt, nxt)[0], strict))


# ---- JPEG ------------------------------------------------------------------------


def _walk_jpeg(sp: SparseCopy) -> None:
    """Everything before the first scan (exiftool -fast stops there), plus any
    secondary images listed in an MPF index (exiftool -b extracts them)."""
    off = 2
    while off + 4 <= sp.size:
        hdr = sp.read(off, 4, _READAHEAD)
        if hdr[0] != 0xFF:
            raise SparseReadError(f"lost JPEG marker sync at {off}")
        marker = hdr[1]
        if marker == 0xDA:  # start of scan
            return
        if marker == 0xFF:  # fill byte
            off += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            off += 2
            continue
        seg_len = struct.unpack(">H", hdr[2:4])[0]
        if seg_len < 2:
            raise SparseReadError("bad JPEG segment length")
        body = sp.read(off + 4, seg_len - 2)
        if marker == 0xE2 and body.startswith(b"MPF\x00"):
            _fetch_mpf_images(sp, off + 8, body[4:])
        off += 2 + seg_len
    raise SparseReadError("no JPEG start-of-scan found")


def _fetch_mpf_images(sp: SparseCopy, mpf_base: int, tiff: bytes) -> None:
    try:
        bo = "<" if tiff[:2] == b"II" else ">"
        ifd = struct.unpack(bo + "I", tiff[4:8])[0]
        n = struct.unpack(bo + "H", tiff[ifd:ifd + 2])[0]
        for i in range(n):
            tag, _typ, count, val = struct.unpack(
                bo + "HHII", tiff[ifd + 2 + i * 12: ifd + 14 + i * 12]
            )
            if tag != 0xB002:  # MPEntry
                continue
            for j in range(count // 16):
                _attr, size, offset = struct.unpack(
                    bo + "III", tiff[val + j * 16: val + j * 16 + 12]
                )
                if offset and size <= _MAX_VALUE_BYTES:
                    sp.ensure(mpf_base + offset, size)
    except struct.error as e:
        raise SparseReadError(f"bad MPF index: {e}") from e


# ---- MP4/MOV (ISO-BMFF), AVI (RIFF), MPEG-TS ---------------------------------------


def _walk_bmff(sp: SparseCopy) -> None:
    """Every top-level box except media data (``mdat``)."""
    off = 0
    while off + 8 <= sp.size:
        hdr = sp.read(off, 16, _READAHEAD)
        size, typ = struct.unpack(">I4s", hdr[:8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", hdr[8:16])[0]
            header = 16
        elif size == 0:
            size = sp.size - off
        if size < header:
            raise SparseReadError(f"bad box size at {off}")
        if typ != b"mdat":
            sp.ensure(off, size)
        off += size


def _walk_riff(sp: SparseCopy) -> None:
    """Every chunk except frame data (``LIST movi``) and the index (``idx1``)."""
    off = 0
    while off + 8 <= sp.size:
        cid, size = struct.unpack("<4sI", sp.read(off, 12, _READAHEAD)[:8])
        if cid in (b"RIFF", b"LIST"):
            form = sp.read(off + 8, 4)
            if form == b"movi":
                off += 8 + size + (size & 1)
                continue
            if cid == b"RIFF":  # descend: RIFF/AVIX bodies hold the chunks
                off += 12
                continue
        if cid != b"idx1":
            sp.ensure(off, 8 + size)
        off += 8 + size + (size & 1)


def _walk_mpegts(sp: SparseCopy) -> None:
    """exiftool reads the start for streams and the end for duration."""
    sp.ensure(0, _TS_EDGE_BYTES)
    sp.ensure(max(0, sp.size - _TS_EDGE_BYTES), _TS_EDGE_BYTES)


def _detect(head: bytes, name: str) -> str:
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if head[:2] == b"\xff\xd8":
        return "jpeg"
    if head[4:8] in (b"ftyp", b"moov", b"wide", b"free", b"mdat", b"skip", b"pnot"):
        return "bmff"
    if head[:4] == b"RIFF" and head[8:12] in (b"AVI ", b"AVIX"):
        return "riff"
    if os.path.splitext(name)[1].lower() in (".ts", ".tts", ".m2ts", ".mts") and (
        head[:1] == b"\x47" or head[4:5] == b"\x47"
    ):
        return "mpegts"
    return "unsupported"


_WALKERS = {
    "tiff": _walk_tiff,
    "jpeg": _walk_jpeg,
    "bmff": _walk_bmff,
    "riff": _walk_riff,
    "mpegts": _walk_mpegts,
}


def build_sparse_copy(
    path: str,
    size: int,
    fetch: Fetch,
    *,
    name: str,
    head: bytes = b"",
    budget: int = DEFAULT_BUDGET,
) -> SparseStats:
    """Write a sparse copy of a ``size``-byte remote object to ``path``.

    ``head`` is the object's first bytes if the caller already has them (saves a
    request); ``name`` is the object's key (its extension disambiguates some
    formats). Raises :class:`SparseReadError` (or a subclass) when the caller
    should download the whole file instead; ``path`` may then be partial.
    """
    sp = SparseCopy(path, size, fetch, head=head, budget=budget)
    try:
        fmt = _detect(sp.read(0, 16, HEAD_BYTES), name)
        walker = _WALKERS.get(fmt)
        if walker is None:
            raise UnsupportedFormat(f"no sparse reader for {name!r}")
        try:
            walker(sp)
        except (struct.error, IndexError, ValueError) as e:
            raise SparseReadError(f"could not walk {fmt} structure: {e}") from e
        sp.flush()
        return SparseStats(fmt=fmt, requests=sp.requests, fetched=sp.fetched, size=size)
    finally:
        sp.close()
