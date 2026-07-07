# VENDORED from data-engineering@unified_date_fields
#   atomicmapspy/atomicmapspy/utils/utils.py (selected definitions)
# Do not edit by hand. Re-vendor with `am-tools update` (dev machines) —
# drift is detected by tests/test_vendored_drift.py.

import json
import re
from enum import Enum
from typing import Dict, List, Optional


DATA_TYPE_INFO: Dict = {
    "ortho_image": {
        "include": [".tif", ".tiff"],
        "exclude": ["_rgb.tif"],
        "sidecars": [".ecw"],
        "sub_types": [
            "rgb",
            "ir",
        ],
        "footprint_feature_class": "footprints_ortho_image",
    },
    "spherical_image": {
        "include": [".jpg", ".jp2", ".jpeg", ".png"],
        "exclude": [
            "PreviewImage.jpg",
            "ThumbnailImage.jpg",
            "annotation.json",
            "_thumbnail.jpg",
            "_thumbnail.jpeg",
            "_thumbnail.png",
        ],
        "sidecars": [],
        "sub_types": ["rgb"],
        "footprint_feature_class": "footprints_spherical_image",
    },
    "oriented_image": {
        "include": [".jpg", ".jp2", ".jpeg", ".png"],
        "exclude": [
            ".tif",
            ".tiff",
            "PreviewImage.jpg",
            "ThumbnailImage.jpg",
            "annotation.json",
            "_thumbnail.jpg",
            "_thumbnail.jpeg",
            "_thumbnail.png",
        ],
        "sidecars": [],
        "sub_types": [
            "rgb",
            "ir",
            "thermal",
        ],
        "footprint_feature_class": "footprints_oriented_imagery",
    },
    # Unified gather type for the create_image_stac task: collects every image
    # extension shared by oriented and spherical imagery. The task classifies
    # each file as oriented_image or spherical_image at runtime, so per-item
    # data_type, collections, and footprints stay split by image type.
    "imagery": {
        "include": [".jpg", ".jp2", ".jpeg", ".png"],
        "exclude": [
            ".tif",
            ".tiff",
            "PreviewImage.jpg",
            "ThumbnailImage.jpg",
            "annotation.json",
            "_thumbnail.jpg",
            "_thumbnail.jpeg",
            "_thumbnail.png",
        ],
        "sidecars": [],
        "sub_types": [
            "rgb",
            "ir",
            "thermal",
        ],
    },
    "point_cloud": {
        "include": [".las", ".laz", ".zlas", ".e57"],
        "exclude": [".copc.las", ".copc.laz"],
        "sidecars": [],
        "sub_types": ["lidar", "photogrammetry"],
        "footprint_feature_class": "footprints_point_cloud",
    },
    "full_motion_video": {
        "include": [".mp4", ".mov", ".ts", ".avi", ".tts"],
        "exclude": [],
        "sidecars": [".gpx", ".kmz", ".srt"],
        "sub_types": ["rgb", "ir"],
        "footprint_feature_class": "footprints_video",
    },
    "vector": {
        "include": [".gdb", ".gdb.zip", ".gpkg", ".geojson", ".shp"],
        "exclude": [
            ".shx",
            ".dbf",
            ".prj",
            ".cpg",
            ".sbn",
            ".sbx",
            ".shp.xml",
            ".qpj",
            ".pmtiles",
            ".parquet",
            "_thumbnail.jpg",
        ],
        "sidecars": [],
        "sub_types": [],
    },
}


class DataTypeEnum(str, Enum):
    ortho_image = "ortho_image"
    oriented_image = "oriented_image"
    spherical_image = "spherical_image"
    imagery = "imagery"
    point_cloud = "point_cloud"
    video = "full_motion_video"
    cad = "cad"
    vector = "vector"


class ImageDataTypeEnum(str, Enum):
    """Subset of DataTypeEnum restricted to image-based data types."""

    oriented_image = "oriented_image"
    spherical_image = "spherical_image"
    ortho_image = "ortho_image"


def get_valid_subtypes(data_type: str) -> List[str]:
    """Get list of valid subtypes for a data type"""
    return DATA_TYPE_INFO.get(data_type, {}).get("sub_types", [])


_INFER_DATA_TYPE_ORDER = [
    "oriented_image",
    "spherical_image",
    "ortho_image",
    "point_cloud",
    "full_motion_video",
    "vector",
]


_AMBIGUOUS_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".jp2", ".png")


_SPHERICAL_ASPECT_RATIO_MIN = 1.95


_SPHERICAL_ASPECT_RATIO_MAX = 2.05


def _xmp_indicates_spherical(xmp_packet: bytes) -> bool:
    """Return True if an XMP packet indicates an equirectangular spherical image.

    Looks for either the Google Photo Sphere `GPano:` namespace or an explicit
    `ProjectionType="equirectangular"` attribute. Match is case-insensitive and
    does not require well-formed XML parsing — XMP is conventionally bounded by
    `<x:xmpmeta>` markers but we scan the raw bytes for robustness against
    minor format variation across cameras/stitchers.
    """
    if not xmp_packet:
        return False
    lower = xmp_packet.lower()
    if b"gpano:" in lower or b"xmlns:gpano" in lower:
        return True
    # Match `ProjectionType="equirectangular"` (with or without quotes,
    # tolerant of whitespace) as a real attribute pairing rather than two
    # tokens appearing anywhere in the packet.
    if re.search(rb'projectiontype\s*=\s*["\']?\s*equirectangular', lower):
        return True
    return False


def _aspect_ratio_indicates_spherical(aspect_ratio: float) -> bool:
    """Return True if width/height ratio is in the equirectangular band (~2:1)."""
    return _SPHERICAL_ASPECT_RATIO_MIN <= aspect_ratio <= _SPHERICAL_ASPECT_RATIO_MAX


def _user_comment_indicates_spherical(user_comment: Optional[str]) -> bool:
    """Return True if an EXIF UserComment marks an e57-derived spherical image.

    Spherical images extracted from e57 point clouds carry a JSON UserComment
    with `e57_representation` set to `spherical` (written by the
    create_point_cloud_stac task). Parsing is tolerant of an exiftool charset
    prefix (e.g. a leading "ASCII" tag before the JSON) and of non-JSON comments.
    """
    if not user_comment:
        return False
    try:
        start = user_comment.index("{")
        data = json.loads(user_comment[start:])
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    return str(data.get("e57_representation", "")).lower() == "spherical"


def infer_data_type(
    filename: str,
    *,
    xmp_packet: Optional[bytes] = None,
    aspect_ratio: Optional[float] = None,
    user_comment: Optional[str] = None,
) -> Optional[str]:
    """Infer the DATA_TYPE_INFO key for a single file from its filename.

    Returns the data-type name (e.g. "oriented_image", "point_cloud", "vector")
    or None if the file does not match any data type's include rules.

    Rules:
    - Files inside an unzipped FileGDB directory (path contains `.gdb/`) are
      classified as "vector". The per-type workflow's batch_generator will
      collapse them to the .gdb root before processing.
    - Files matching any data type's `exclude` patterns are rejected for that
      type. Excluded but otherwise unmatched files return None
      (e.g. `foo.copc.las` → None; the exclude on point_cloud rules it out and
      no other type claims `.las`).
    - Sidecar extensions (e.g. `.gpx` for video, `.ecw` for ortho) are
      classified as their primary data type — sidecars travel with primaries.
    - For ambiguous extensions (`.jpg`/`.jp2`/`.png`, listed under both
      oriented_image and spherical_image), the result depends on the optional
      `user_comment`, `xmp_packet` and `aspect_ratio` kwargs, checked in order:
        - UserComment JSON with `e57_representation=spherical` (written by the
          create_point_cloud_stac task for e57-derived panoramas)
          → spherical_image
        - XMP containing `GPano:` namespace or `ProjectionType=equirectangular`
          → spherical_image
        - Aspect ratio in ~[1.95, 2.05] (equirectangular projection is 2:1)
          → spherical_image
        - No signal present → oriented_image (the default)
      Callers without byte access (e.g. workflow-template inference from
      filenames alone) get the oriented_image default.

    Args:
        filename: Path or filename to classify (case-insensitive).
        xmp_packet: Optional raw XMP bytes from the file's metadata.
        aspect_ratio: Optional image width/height ratio. Floats.
        user_comment: Optional EXIF UserComment string from the file's metadata.

    Returns:
        Data-type name from DATA_TYPE_INFO, or None if no type matches.
    """
    if not filename:
        return None

    lower = filename.lower()

    # FileGDB inner files collapse to the vector pipeline.
    if ".gdb/" in lower:
        return "vector"

    # JPG/JP2/PNG disambiguation: check spherical signals before falling
    # through to the default oriented_image classification below.
    if lower.endswith(_AMBIGUOUS_IMAGE_EXTENSIONS):
        # Exclude previews/thumbnails for ambiguous images using both types' excludes.
        excludes: set = set()
        for dt in ("oriented_image", "spherical_image"):
            excludes.update(
                e.lower() for e in (DATA_TYPE_INFO[dt].get("exclude") or [])
            )
        if excludes and lower.endswith(tuple(excludes)):
            return None
        if _user_comment_indicates_spherical(user_comment):
            return "spherical_image"
        if xmp_packet is not None and _xmp_indicates_spherical(xmp_packet):
            return "spherical_image"
        if aspect_ratio is not None and _aspect_ratio_indicates_spherical(aspect_ratio):
            return "spherical_image"
        return "oriented_image"

    for data_type in _INFER_DATA_TYPE_ORDER:
        info = DATA_TYPE_INFO.get(data_type, {})
        excludes = tuple((info.get("exclude") or []))
        if excludes and lower.endswith(tuple(e.lower() for e in excludes)):
            continue
        patterns = list(info.get("include") or []) + list(info.get("sidecars") or [])
        if not patterns:
            continue
        if lower.endswith(tuple(p.lower() for p in patterns)):
            return data_type

    return None
