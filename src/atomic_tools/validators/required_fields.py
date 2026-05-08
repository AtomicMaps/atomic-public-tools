"""Required-field groups by data type — single source of truth.

Each inner list is ``[canonical_name, *aliases]``. A file/row "satisfies" a
group if it has any field from the group present (non-empty). The canonical
(first) entry is the name written into the final sidecar.
"""

from atomic_tools.utils.utils import DataTypeEnum

REQUIRED_SIDECAR_FIELD_GROUPS: dict[str, list[list[str]]] = {
    DataTypeEnum.oriented_image: [
        ["GPSLatitude"],
        ["GPSLongitude"],
        ["GPSAltitude"],
        ["CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"],
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
        ["DateTimeOriginal", "CreateDate", "ModifyDate", "GPSDateStamp"],
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
