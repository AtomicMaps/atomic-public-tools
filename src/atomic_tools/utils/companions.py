"""Which listed files become sidecar rows, mirroring data-engineering's intake.

Shared by sidecar generation and the lint inventory check so both agree on
what counts as a data file.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from difflib import SequenceMatcher

from atomic_tools.utils.utils import DATA_TYPE_INFO, DataTypeEnum, infer_data_type

# Video telemetry companions (write_video_stac.SIDECAR_EXTS). The registry lists
# them as video "sidecars", but they are never rows of their own.
TELEMETRY_EXTENSIONS = (".gpx", ".kmz", ".srt")
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".ts", ".tts", ".avi")
_TELEMETRY_SIMILARITY = 0.9

# write_image_stac._MESH_COMPANION_EXTENSIONS
_MESH_COMPANION_EXTENSIONS = (".obj", ".mtl")
_MESH_TEXTURE_EXTENSIONS = (".jpg", ".jpeg", ".jp2", ".png")

# Formats intake classifies but can't process; skipped with this reason.
UNSUPPORTED_EXTENSIONS = {
    ".zlas": "Esri .zlas can't be read by PDAL or Flow; convert it to .laz",
}
UNSUPPORTED_TYPES = {
    DataTypeEnum.gaussian_splat: "Gaussian splats are not yet processed by Flow intake",
}


def listing_extensions(types: Iterable[DataTypeEnum]) -> list[str]:
    """``include`` plus ``sidecars`` for each type, order-stable."""
    out: list[str] = []
    for data_type in types:
        info = DATA_TYPE_INFO.get(data_type) or {}
        for ext in [*(info.get("include") or []), *(info.get("sidecars") or [])]:
            if ext not in out:
                out.append(ext)
    return out


def is_telemetry(key: str) -> bool:
    return key.lower().endswith(TELEMETRY_EXTENSIONS)


def unsupported_reason(key: str, data_type: DataTypeEnum | None) -> str | None:
    lower = key.lower()
    for ext, reason in UNSUPPORTED_EXTENSIONS.items():
        if lower.endswith(ext):
            return reason
    return UNSUPPORTED_TYPES.get(data_type) if data_type is not None else None


def drop_ecw_shadowed_by_tif(keys: list[str]) -> tuple[list[str], list[str]]:
    """Drop an ``.ecw`` when an ortho ``.tif``/``.tiff`` shares its path stem
    (write_ortho_image_stac.filter_tif_ecw).

    Only tifs intake keeps count: the backend applies the ortho excludes (such
    as ``_rgb.tif``) before this rule, so an excluded tif never hides an ECW.
    """
    tif_stems = {
        k.rsplit(".", 1)[0]
        for k in keys
        if k.lower().endswith((".tif", ".tiff"))
        and infer_data_type(k) == DataTypeEnum.ortho_image.value
    }
    kept: list[str] = []
    dropped: list[str] = []
    for key in keys:
        if key.lower().endswith(".ecw") and key.rsplit(".", 1)[0] in tif_stems:
            dropped.append(key)
        else:
            kept.append(key)
    return kept, dropped


def drop_mesh_textures(
    keys: list[str], sibling_exists: Callable[[str], bool]
) -> tuple[list[str], list[str]]:
    """Drop images with a same-stem ``.obj``/``.mtl``: they are mesh texture
    atlases, not photos (write_image_stac._is_mesh_texture)."""
    kept: list[str] = []
    dropped: list[str] = []
    for key in keys:
        if key.lower().endswith(_MESH_TEXTURE_EXTENSIONS):
            stem = key.rsplit(".", 1)[0]
            if any(sibling_exists(stem + ext) for ext in _MESH_COMPANION_EXTENSIONS):
                dropped.append(key)
                continue
        kept.append(key)
    return kept, dropped


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= _TELEMETRY_SIMILARITY


def telemetry_near(videos: Iterable[str], telemetry: list[str]) -> list[str]:
    """The telemetry files whose names are similar enough to pair with one of ``videos``."""
    stems = [os.path.splitext(os.path.basename(v))[0] for v in videos]
    return [
        t for t in telemetry
        if any(_similar(os.path.splitext(os.path.basename(t))[0], s) for s in stems)
    ]


def pair_telemetry(keys: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Attach telemetry files to videos the way the backend groups them.

    The backend groups keys whose basenames (without extension) are at least 90%
    similar, then gives every telemetry file in a group to every video in it.
    The grouping is greedy, so ``keys`` (videos and telemetry together) must be
    in the backend's listing order. Returns ``({video: [telemetry]}, unmatched)``.
    """
    stems = [os.path.splitext(os.path.basename(k))[0] for k in keys]
    used: set[int] = set()
    pairs: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for i in range(len(keys)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(keys)):
            if j not in used and _similar(stems[i], stems[j]):
                group.append(j)
                used.add(j)
        used.add(i)
        group_videos = [keys[g] for g in group if keys[g].lower().endswith(_VIDEO_EXTENSIONS)]
        group_telemetry = [keys[g] for g in group if is_telemetry(keys[g])]
        if group_videos:
            for video in group_videos:
                if group_telemetry:
                    pairs[video] = group_telemetry
        else:
            unmatched.extend(group_telemetry)
    return pairs, unmatched
