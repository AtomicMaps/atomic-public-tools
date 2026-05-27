"""Required- and optional-field groups by data type — single source of truth.

Each inner list is ``[canonical_name, *aliases]``. A file/row "satisfies" a
group if it has any field from the group present (non-empty). The canonical
(first) entry is the name written into the final sidecar.

``REQUIRED_*`` groups are mandatory: the linter errors when none of a group's
fields are present in a final sidecar. ``OPTIONAL_*`` groups are best-effort:
the linter only warns when they're absent.

Note on dates: ``CreateDate`` is documented to clients as optional (generation
infers a date from the filename when EXIF / the client sidecar don't provide
one — see ``_fill_missing_dates_from_filepath``). It is kept in the *required*
groups, though, because the final sidecar must still end up with a date — if
filename inference also fails, the final lint should error.
"""

from atomic_tools.utils.utils import DataTypeEnum

REQUIRED_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    DataTypeEnum.oriented_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
    ],
    DataTypeEnum.spherical_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
        ["Pitch", "CameraPitch", "GimbalPitchDegree", "PosePitchDegrees"],
        ["Heading", "Yaw", "GimbalYawDegree", "PoseHeadingDegrees", "GPSImgDirection"],
        ["Roll", "CameraRoll", "GimbalRollDegree", "PoseRollDegrees"],
    ],
    DataTypeEnum.ortho_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
    ],
    DataTypeEnum.video: [
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
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


# Required + optional groups per data type. Used anywhere we canonicalize
# aliases, select columns, or build the sidecar DataFrame — i.e. everywhere
# except the required-vs-optional error/warning decision. Including optional
# groups here is what keeps optional aliases (e.g. ``Yaw`` -> ``Heading``)
# canonicalizing.
ALL_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    dt: REQUIRED_SIDECAR_FIELD_GROUPS.get(dt, []) + OPTIONAL_SIDECAR_FIELD_GROUPS.get(dt, [])
    for dt in REQUIRED_SIDECAR_FIELD_GROUPS.keys() | OPTIONAL_SIDECAR_FIELD_GROUPS.keys()
}
