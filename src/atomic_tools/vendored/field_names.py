# VENDORED from data-engineering@unified_date_fields
#   atomicmapspy/atomicmapspy/utils/field_names.py
# Do not edit by hand. Re-vendor with `am-tools update` (dev machines) —
# drift is detected by tests/test_vendored_drift.py.

"""Central registry of canonical metadata fields and their source-key fallbacks.

Each "canonical" concept (acquisition datetime, GPS position, camera
orientation, ...) can be populated from one of several source keys depending on
the camera/vendor that produced the file. Historically those ordered fallback
lists lived inline in each image/video class, which made them hard to audit and
easy to let drift apart. This module collects them in one place so the parsing
methods can import the lists and the *only* thing that varies per class is the
parsing logic, not the key names.

Three source namespaces are represented here:

* **Image EXIF** — ``exiftool -j`` returns flat tag names with no namespace
  prefix (e.g. ``"DateTimeOriginal"``, ``"GPSLatitude"``). Consumed by
  ``base_image`` / ``oriented_image`` / ``spherical_image``.
* **Video** — per-frame metadata columns from ffprobe / exiftool / KLV,
  normalized to lowercase (e.g. ``"platform_pitch_angle"``). Consumed by the
  ``video`` subpackage.
* **Point cloud** — PDAL pipeline metadata is a *nested* dict keyed by stage
  name (``filters.info`` / ``readers.las`` / ...) whose values contain further
  nested ``srs`` objects, plus e57 node-tree fields. These don't fit the flat
  "ordered list of keys" shape, so they're expressed as *key paths* (and, for
  the SRS fallback, ``(label, source, key_path)`` tuples) that ``point_cloud.py``
  walks against the stage dicts it supplies.

The friendly sidecar-column → EXIF-tag aliases (``Latitude`` → ``GPSLatitude``
etc.) are owned by :mod:`atomicmapspy.utils.sidecar` (``_SIDECAR_COLUMN_ALIASES``)
and are deliberately not duplicated here.
"""

from typing import List, Literal, Tuple

# ---------------------------------------------------------------------------
# Image EXIF fallback chains (flat exiftool tag names)
# ---------------------------------------------------------------------------

# Canonical: acquisition_datetime (int epoch). GPS-derived and file-modify-time
# fallbacks are handled separately in base_image.get_datetime().
DATETIME_FIELDS: List[str] = [
    "DateTimeOriginal",
    "CreateDate",
    "ModifyDate",
    "FirstPhotoDate",
    "LastPhotoDate",
]

# Canonical: img_width / img_height (get_img_size returns them as a (width,
# height) tuple). Each source tuple is (width_tag, height_tag).
IMAGE_SIZE_FIELDS: List[Tuple[str, str]] = [
    ("ImageWidth", "ImageHeight"),
    ("ExifImageWidth", "ExifImageHeight"),
    ("FullPanoWidthPixels", "FullPanoHeightPixels"),
    ("CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels"),
]
# Combined "WxH" string fallback for the pairs above.
IMAGE_SIZE_COMBINED_FIELD: str = "ImageSize"

# Canonical: focal_length / focal_length_35mm.
FOCAL_LENGTH_FIELD: str = "FocalLength"
FOCAL_LENGTH_35MM_FIELD: str = "FocalLengthIn35mmFormat"
# Combined field parsed for both values when the two above are missing.
FOCAL_LENGTH_COMBINED_FIELD: str = "FocalLength35efl"

# Canonical: lat / lon.
GPS_LATITUDE_FIELD: str = "GPSLatitude"
GPS_LONGITUDE_FIELD: str = "GPSLongitude"
# Combined "<lat>, <lon>" fallback.
GPS_POSITION_COMBINED_FIELD: str = "GPSPosition"

# Canonical: cam_pitch / cam_heading / cam_roll (OrientedImage).
# Each tuple is (pitch_tag, yaw_tag, roll_tag), tried in priority order.
CAM_ORIENTATION_PYR_FIELDS: List[Tuple[str, str, str]] = [
    # Tried first: most common EXIF field set
    ("CameraPitch", "CameraYaw", "CameraRoll"),
    ("Pitch", "Yaw", "Roll"),  # General fallback
    ("Pitch", "Heading", "Roll"),  # Canonical sidecar triple
    ("CameraPitchDegree", "CameraYawDegree", "CameraRollDegree"),
    ("GimbalPitchDegree", "GimbalYawDegree", "GimbalRollDegree"),
    ("PosePitchDegrees", "PoseHeadingDegrees", "PoseRollDegrees"),
    (
        "CameraOrientationNEDPitch",
        "CameraOrientationNEDYaw",
        "CameraOrientationNEDRoll",
    ),  # Skydio
    ("GPSIMUPitch", "GPSIMUYaw", "GPSIMURoll"),  # Phase One
    ("PitchAngle", "YawAngle", "RollAngle"),  # Nikon
]
# Heading-only fallback when a full pitch/yaw/roll triple is unavailable.
CAM_HEADING_ONLY_FIELDS: List[str] = ["GPSImgDirection", "imgDirection"]

# Canonical (SphericalImage): orientation (quaternion) + cam_pitch/heading/roll.
# Quaternion sources first; values are space-separated "w x y z".
SPHERICAL_QUATERNION_FIELDS: List[str] = ["CameraOrientation", "UserComment"]
# Euler-angle fallbacks (pitch_tag, yaw_tag, roll_tag).
SPHERICAL_EULER_FIELDS: List[Tuple[str, str, str]] = [
    ("PosePitchDegrees", "PoseHeadingDegrees", "PoseRollDegrees"),
    ("Pitch", "Heading", "Roll"),  # Canonical sidecar triple
]

# ---------------------------------------------------------------------------
# Video fallback chains (normalized lowercase per-frame metadata columns)
# ---------------------------------------------------------------------------

# Canonical: start_time / end_time (derived from the first/last frame).
VIDEO_DATETIME_FIELDS: List[str] = ["precision_time_stamp", "gpsdatetime"]

# Canonical: cam_lat_field / cam_lon_field. Each tuple is (lat_tag, lon_tag).
VIDEO_CAMERA_LOCATION_FIELDS: List[Tuple[str, str]] = [
    ("gpslatitude", "gpslongitude"),
    ("sensor_latitude", "sensor_longitude"),
]

# Canonical: frame_lat_field / frame_lon_field. (lat_tag, lon_tag).
VIDEO_FRAME_CENTER_LOCATION_FIELDS: List[Tuple[str, str]] = [
    ("frame_center_latitude", "frame_center_longitude"),
]

# Canonical: frame_elev_field.
VIDEO_FRAME_CENTER_ELEVATION_FIELDS: List[str] = [
    "frame_center_elevation",
    "frame_center_height_above_ellipsoid",
]

# Canonical: cam_elev_field (+ cam_elev_datum). Each tuple is (field, datum);
# order is priority. MISB sensor_true_altitude is already orthometric; GPS/EXIF
# altitude and MISB ellipsoid_height are WGS84 ellipsoidal and need geoid
# conversion.
VIDEO_CAMERA_ELEVATION_FIELDS: List[
    Tuple[str, Literal["ellipsoidal", "orthometric"]]
] = [
    ("gpsaltitude", "ellipsoidal"),
    ("sensor_true_altitude", "orthometric"),
    ("sensor_ellipsoid_height", "ellipsoidal"),
]

# Canonical: height_above_ground_field (single candidate).
VIDEO_HEIGHT_ABOVE_GROUND_FIELD: str = "relaltitude"

# Canonical: pitch_field / yaw_field / roll_field. (pitch_tag, yaw_tag, roll_tag).
VIDEO_ORIENTATION_PYR_FIELDS: List[Tuple[str, str, str]] = [
    ("platform_pitch_angle", "platform_heading_angle", "platform_roll_angle"),
    (
        "platform_pitch_angle_full",
        "platform_heading_angle",
        "platform_roll_angle_full",
    ),
    ("pitch", "yaw", "roll"),
]

# Canonical: hfov_field / vfov_field. (hfov_tag, vfov_tag).
VIDEO_FOV_FIELDS: List[Tuple[str, str]] = [
    ("sensor_horizontal_field_of_view", "sensor_vertical_field_of_view"),
]

# ---------------------------------------------------------------------------
# Point cloud (nested PDAL pipeline metadata + e57 node tree)
# ---------------------------------------------------------------------------
# PDAL metadata is a nested dict keyed by stage name; the constants below name
# the stages/keys and the key paths to walk. PointCloud supplies the actual
# stage dicts (filters_info / reader_metadata) and the e57 nodes.

# PDAL stage names.
PC_FILTERS_INFO_STAGE: str = "filters.info"
PC_HEXBIN_STAGE: str = "filters.hexbin"
# PDAL reader stage name for E57 files.
PC_E57_READER: str = "readers.e57"

# Keys read off the filters.info / hexbin stage dicts.
PC_NUM_POINTS_KEY: str = "num_points"  # filters.info.num_points
PC_DIMENSIONS_KEY: str = "dimensions"  # filters.info.dimensions
PC_BBOX_KEY: str = "bbox"  # filters.info.bbox
PC_HEXBIN_BOUNDARY_KEY: str = "boundary"  # filters.hexbin.boundary

# Canonical: capture_date. LAS/LAZ headers expose it as a (year, day-of-year)
# pair on the reader metadata.
PC_CAPTURE_DATE_LAS_FIELDS: Tuple[str, str] = ("creation_year", "creation_doy")

# E57 capture-date fallback, most specific first. The root field is a file-level
# timestamp; the scan fields live under data3D[i]. The numeric GPS-epoch seconds
# sit on a "dateTimeValue" sub-node of each field.
PC_E57_DATA3D_KEY: str = "data3D"
PC_E57_ROOT_DATETIME_FIELD: str = "creationDateTime"
PC_E57_SCAN_DATETIME_FIELDS: List[str] = ["acquisitionStart", "acquisitionEnd"]
PC_E57_DATETIME_VALUE_KEY: str = "dateTimeValue"

# Canonical: srs (CRS). Ordered (label, from_reader, key_path) candidates.
# `from_reader` selects which stage dict point_cloud reads against — True = the
# active reader metadata, False = filters.info — and `key_path` is walked into
# nested sub-dicts. First non-empty string wins.
PC_SRS_CANDIDATES: List[Tuple[str, bool, Tuple[str, ...]]] = [
    ("filters.comp_spatialreference", False, ("comp_spatialreference",)),
    ("filters.srs.compoundwkt", False, ("srs", "compoundwkt")),
    ("filters.spatialreference", False, ("spatialreference",)),
    ("reader.srs.compoundwkt", True, ("srs", "compoundwkt")),
    ("reader.spatialreference", True, ("spatialreference",)),
    ("reader.comp_spatialreference", True, ("comp_spatialreference",)),
    ("reader.srs.proj4", True, ("srs", "proj4")),
]
# Nested key path for the horizontal linear unit; checked on reader then filters.
PC_SRS_HORIZONTAL_UNIT_PATH: Tuple[str, ...] = ("srs", "units", "horizontal")

# ---------------------------------------------------------------------------
# Vector attribute columns
# ---------------------------------------------------------------------------

# Canonical vector attribute column for feature timestamps.
VECTOR_DATETIME_FIELD: str = "capture_time"
