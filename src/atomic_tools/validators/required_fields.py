"""Field groups by data type — single source of truth.

Each inner list is ``[canonical_name, *aliases]``. A file/row "satisfies" a
group if it has any field from the group present (non-empty). The canonical
(first) entry is the name written into the final sidecar.

Tiers (lint behaviour differs; ``ALL_*`` merges the ones that drive
canonicalization/column-selection):

* ``REQUIRED_*``    — mandatory: the linter *errors* when none of a group's
  fields are present in a final sidecar.
* ``OPTIONAL_*``    — best-effort: the linter only *warns* when they're absent.
* ``REFERENCED_*``  — non-required EXIF fields the data-engineering pipeline
  reads (focal length, make/model, accuracy, dimensions, …). The linter
  produces *no* signal for them, but they are offered as suggestions in
  ``am-tools schema build`` and are canonicalized / carried into generated
  sidecars when present (they are folded into ``ALL_*``).
* ``REFERENCED_FULL_*`` — comprehensive/advanced referenced fields (calibration
  internals, quaternion orientation, view angles, …). Offered only under
  ``schema build --full`` and *not* folded into ``ALL_*``, so they don't bloat
  default generated sidecars.

Invariant: each canonical name appears in exactly one tier per data type
(``canonical_candidates`` keys on the canonical, so a duplicate across tiers
would silently overwrite), and a given alias maps to the same canonical across
all data types (``build_global_alias_map`` flattens every group into one
``{alias: canonical}`` map and warns on collisions).

Note on dates: ``CreateDate`` is documented to clients as optional (generation
infers a date from the filename when EXIF / the client sidecar don't provide
one — see ``_fill_missing_dates_from_filepath``). It is kept in the *required*
groups, though, because the final sidecar must still end up with a date — if
filename inference also fails, the final lint should error.
"""

from atomic_tools.utils.utils import DataTypeEnum

# Date group, shared across image/video data types. Aliases beyond the first
# four (FirstPhotoDate/LastPhotoDate/GPSTimeStamp) are extra date sources the
# data-engineering pipeline reads; listing them lets a client column with one of
# those names canonicalize to the date field. Still one group -> no new
# obligation.
_DATE_GROUP = [
    "CreateDate",
    "DateTimeOriginal",
    "ModifyDate",
    "GPSDateStamp",
    "FirstPhotoDate",
    "LastPhotoDate",
    "GPSTimeStamp",
]

REQUIRED_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    DataTypeEnum.oriented_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        _DATE_GROUP,
    ],
    DataTypeEnum.spherical_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        _DATE_GROUP,
        ["Pitch", "CameraPitch", "GimbalPitchDegree", "PosePitchDegrees"],
        ["Heading", "Yaw", "GimbalYawDegree", "PoseHeadingDegrees", "GPSImgDirection"],
        ["Roll", "CameraRoll", "GimbalRollDegree", "PoseRollDegrees"],
    ],
    DataTypeEnum.ortho_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        _DATE_GROUP,
    ],
    DataTypeEnum.video: [
        _DATE_GROUP,
    ],
    DataTypeEnum.point_cloud: [
        ["bounds.minx"],
        ["bounds.miny"],
        ["bounds.maxx"],
        ["bounds.maxy"],
        ["bounds.minz"],
        ["bounds.maxz"],
        ["num_points"],
        ["creation_year"],
        ["creation_doy"],
    ],
}

# Best-effort groups: present on most files but legitimately missing on some.
# The linter warns (never errors) when an optional group is absent.
OPTIONAL_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    DataTypeEnum.oriented_image: [
        [
            "Pitch",
            "CameraPitch",
            "CameraPitchDegree",
            "GimbalPitchDegree",
            "PosePitchDegrees",
            "CameraOrientationNEDPitch",
            "GPSIMUPitch",
            "PitchAngle",
        ],
        [
            "Heading",
            "Yaw",
            "CameraYaw",
            "CameraYawDegree",
            "GimbalYawDegree",
            "PoseHeadingDegrees",
            "CameraOrientationNEDYaw",
            "GPSIMUYaw",
            "YawAngle",
            "GPSImgDirection",
            "imgDirection",
        ],
        [
            "Roll",
            "CameraRoll",
            "CameraRollDegree",
            "GimbalRollDegree",
            "PoseRollDegrees",
            "CameraOrientationNEDRoll",
            "GPSIMURoll",
            "RollAngle",
        ],
    ],
}


# Non-required EXIF fields the data-engineering pipeline reads. The linter emits
# no signal for these, but they're offered as suggestions in `schema build` and
# (because they're folded into ALL_* below) canonicalized and carried into
# generated sidecars when present. Curated to fields a client is plausibly able
# to supply in a sidecar CSV.
REFERENCED_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    DataTypeEnum.oriented_image: [
        ["FocalLength", "FocalLengthIn35mmFormat", "FocalLength35efl"],
        ["GPSXYAccuracy"],
        ["GPSZAccuracy"],
        ["Make"],
        ["Model"],
        ["CameraSource"],
        ["ImageWidth", "ExifImageWidth"],
        ["ImageHeight", "ExifImageHeight"],
    ],
    DataTypeEnum.spherical_image: [
        ["FocalLength", "FocalLengthIn35mmFormat", "FocalLength35efl"],
        ["ProjectionType"],
        ["Make"],
        ["Model"],
        ["CaptureSoftware"],
        ["ImageWidth", "ExifImageWidth", "FullPanoWidthPixels", "CroppedAreaImageWidthPixels"],
        ["ImageHeight", "ExifImageHeight", "FullPanoHeightPixels", "CroppedAreaImageHeightPixels"],
    ],
    DataTypeEnum.ortho_image: [
        ["FocalLength", "FocalLengthIn35mmFormat"],
        ["Make"],
        ["Model"],
        ["ImageWidth", "ExifImageWidth"],
        ["ImageHeight", "ExifImageHeight"],
    ],
    DataTypeEnum.video: [
        ["Make"],
        ["Model"],
        ["SensorName"],
        ["CameraSource"],
    ],
    # point_cloud: none — its referenced fields are PDAL/E57-derived, not
    # client-suppliable EXIF.
}

# Comprehensive/advanced referenced fields: calibration internals, quaternion
# orientation, initial-view angles, etc. Offered only under `schema build
# --full`. Each group adds a *new* canonical (never re-lists one already in a
# required/optional/curated-referenced group for the same data type) and is NOT
# folded into ALL_*.
REFERENCED_FULL_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    DataTypeEnum.oriented_image: [
        ["CalibratedFocalLength", "CalibratedFocalLengthX", "CalibratedFocalLengthY"],
        ["CalibratedOpticalCenterX"],
        ["CalibratedOpticalCenterY"],
        ["XResolution"],
        ["YResolution"],
        ["Orientation"],
        ["ImageSize"],
    ],
    DataTypeEnum.spherical_image: [
        ["ScaleFactor35efl"],
        ["CameraOrientation", "UserComment"],  # quaternion orientation
        ["InitialViewHeadingDegrees"],
        ["InitialViewPitchDegrees"],
        ["InitialViewRollDegrees"],
        ["MIMEType"],
        ["ImageSize"],
    ],
    DataTypeEnum.ortho_image: [
        ["Orientation"],
        ["ImageSize"],
    ],
}


# Required + optional + curated-referenced groups per data type. Used anywhere we
# canonicalize aliases, select columns, or build the sidecar DataFrame — i.e.
# everywhere except the required-vs-optional error/warning decision. Including
# optional/referenced groups here is what keeps their aliases (e.g. ``Yaw`` ->
# ``Heading``, ``FocalLengthIn35mmFormat`` -> ``FocalLength``) canonicalizing.
ALL_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    dt: REQUIRED_SIDECAR_FIELD_GROUPS.get(dt, [])
    + OPTIONAL_SIDECAR_FIELD_GROUPS.get(dt, [])
    + REFERENCED_SIDECAR_FIELD_GROUPS.get(dt, [])
    for dt in (
        REQUIRED_SIDECAR_FIELD_GROUPS.keys()
        | OPTIONAL_SIDECAR_FIELD_GROUPS.keys()
        | REFERENCED_SIDECAR_FIELD_GROUPS.keys()
    )
}

# ALL_* plus the comprehensive referenced tier. Backs `schema build --full`;
# deliberately not used by sidecar generation/linting. Every REFERENCED_FULL_*
# data type is already a key in ALL_* (it unions in REQUIRED_*), so iterating
# ALL_*'s keys covers them all.
ALL_SIDECAR_FIELD_GROUPS_FULL: dict[str, list[list[str]]] = {
    dt: groups + REFERENCED_FULL_SIDECAR_FIELD_GROUPS.get(dt, [])
    for dt, groups in ALL_SIDECAR_FIELD_GROUPS.items()
}
