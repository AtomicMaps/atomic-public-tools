"""Consistency between the vendored field registry and amtools' field groups.

amtools keeps its tiered field groups (required / optional / referenced) as
hand-authored *policy* in :mod:`atomic_tools.validators.required_fields` rather
than deriving them programmatically from the field registry — tier assignment,
canonical-first ordering, and amtools-only extras are policy the canonical
registry does not encode. These subset assertions enforce that whenever the
canonical field lists mention a source key, the corresponding amtools group
still lists it, so the two cannot silently drift apart.

Offline: imports only :mod:`atomic_tools.schemas.field_registry` and the group
dicts; needs no data-engineering access.

Out of scope: the ``VIDEO_*`` constants are per-frame lowercase KLV/ffprobe
metadata columns — a different namespace from amtools' file-level exiftool
sidecar columns — so they are deliberately not checked here.
"""

from __future__ import annotations

from atomic_tools.schemas import field_registry
from atomic_tools.utils.utils import DataTypeEnum
from atomic_tools.validators.required_fields import (
    _DATE_GROUP,
    OPTIONAL_SIDECAR_FIELD_GROUPS,
    REFERENCED_SIDECAR_FIELD_GROUPS,
    REQUIRED_SIDECAR_FIELD_GROUPS,
)


def _group_containing(groups, member):
    """Return the group (list) that contains ``member``, or None."""
    for group in groups:
        if member in group:
            return group
    return None


def test_datetime_fields_subset_of_date_group():
    date_group = set(_DATE_GROUP)
    missing = set(field_registry.DATETIME_FIELDS) - date_group
    assert not missing, f"add {missing} to required_fields._DATE_GROUP"


def test_oriented_orientation_fields_subset_of_optional_groups():
    optional = OPTIONAL_SIDECAR_FIELD_GROUPS[DataTypeEnum.oriented_image]
    pitch_group, yaw_group, roll_group = optional[0], optional[1], optional[2]

    for pitch, yaw, roll in field_registry.CAM_ORIENTATION_PYR_FIELDS:
        assert pitch in pitch_group, f"add {pitch!r} to oriented optional pitch group"
        assert yaw in yaw_group, f"add {yaw!r} to oriented optional yaw/heading group"
        assert roll in roll_group, f"add {roll!r} to oriented optional roll group"

    for heading in field_registry.CAM_HEADING_ONLY_FIELDS:
        assert heading in yaw_group, (
            f"add heading-only field {heading!r} to oriented optional yaw group"
        )


def test_image_size_fields_subset_of_spherical_size_groups():
    referenced = REFERENCED_SIDECAR_FIELD_GROUPS[DataTypeEnum.spherical_image]
    width_group = _group_containing(referenced, "ImageWidth")
    height_group = _group_containing(referenced, "ImageHeight")
    assert width_group is not None and height_group is not None

    for width_tag, height_tag in field_registry.IMAGE_SIZE_FIELDS:
        assert width_tag in width_group, f"add {width_tag!r} to spherical width group"
        assert height_tag in height_group, f"add {height_tag!r} to spherical height group"


def test_focal_length_fields_subset_of_each_image_focal_group():
    focal_fields = {
        field_registry.FOCAL_LENGTH_FIELD,
        field_registry.FOCAL_LENGTH_35MM_FIELD,
        field_registry.FOCAL_LENGTH_COMBINED_FIELD,
    }
    for image_type in (
        DataTypeEnum.oriented_image,
        DataTypeEnum.spherical_image,
        DataTypeEnum.ortho_image,
    ):
        referenced = REFERENCED_SIDECAR_FIELD_GROUPS[image_type]
        focal_group = _group_containing(referenced, field_registry.FOCAL_LENGTH_FIELD)
        assert focal_group is not None, f"{image_type.value} has no FocalLength group"
        missing = focal_fields - set(focal_group)
        assert not missing, f"add {missing} to the {image_type.value} FocalLength group"


def test_pc_capture_date_fields_subset_of_point_cloud_required():
    required = REQUIRED_SIDECAR_FIELD_GROUPS[DataTypeEnum.point_cloud]
    required_canonicals = {group[0] for group in required}
    missing = set(field_registry.PC_CAPTURE_DATE_LAS_FIELDS) - required_canonicals
    assert not missing, f"add {missing} to point_cloud required groups"
