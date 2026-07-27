"""Typed access to the vendored data-engineering field registry.

The canonical source keys amtools checks (acquisition datetime, GPS position,
camera orientation, ...) live in ``atomic_tools/vendored/field_registry.json``,
copied verbatim from data-engineering
(``atomicmapspy/atomicmapspy/schemas/field_registry.json``). JSON can't carry a
Python "do not edit" banner, so the JSON is refreshed by ``am-tools update`` and
drift-checked by ``tests/test_vendored_drift.py``; this module is amtools' own
hand-written, typed view over it.

The registry is a superset of what amtools consumes — the JSON keeps every chain
so it stays a faithful mirror of the backend, while this module surfaces only the
handful of chains amtools actually reads, as typed module constants. Need another
chain? Add it here with the matching ``field_*`` helper (or pull it ad hoc via the
generic helpers below); the JSON already carries it.

Each registry "chain" declares a ``kind`` describing its shape:

* ``list`` — an ordered fallback list of source keys (``fields``).
* ``scalar`` — a single source key (``field``).
* ``tuple_list`` — a list of key groups tried as a unit (``fields`` = list of
  lists), e.g. ``(width_tag, height_tag)`` pairs.
* ``tuple`` — a single fixed key group (``fields`` = a flat list).
* ``annotated_list`` — a list of ``{field, ...}`` objects (video elevation).
* ``srs_candidate_list`` / ``key_path`` — point-cloud SRS shapes.
"""

from __future__ import annotations

import importlib.resources
import json
from functools import lru_cache

# The vendored JSON lives alongside the other verbatim data-engineering copies.
_REGISTRY_PACKAGE = "atomic_tools.vendored"
_REGISTRY_FILENAME = "field_registry.json"


@lru_cache(maxsize=1)
def load_registry() -> dict:
    """Return the parsed field registry, read once and cached."""
    text = (
        importlib.resources.files(_REGISTRY_PACKAGE)
        .joinpath(_REGISTRY_FILENAME)
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


def _chain(name: str) -> dict:
    """Return the chain spec for ``name``, raising ``KeyError`` if it's absent."""
    return load_registry()["chains"][name]


# --- Generic, kind-aware accessors ------------------------------------------
# Callers pick the accessor matching the chain's declared ``kind``; a mismatch
# surfaces as a KeyError on the missing ``fields``/``field`` key rather than
# silently returning the wrong shape.


def field_list(name: str) -> list[str]:
    """Ordered fallback list of source keys for a ``kind: list`` chain."""
    return list(_chain(name)["fields"])


def field_scalar(name: str) -> str:
    """The single source key for a ``kind: scalar`` chain."""
    return _chain(name)["field"]


def field_tuples(name: str) -> list[tuple[str, ...]]:
    """List of key groups (each tried as a unit) for a ``kind: tuple_list`` chain."""
    return [tuple(group) for group in _chain(name)["fields"]]


def field_tuple(name: str) -> tuple[str, ...]:
    """The single fixed key group for a ``kind: tuple`` chain."""
    return tuple(_chain(name)["fields"])


# ---------------------------------------------------------------------------
# Chains amtools consumes (see extractors.py and required_fields policy tests).
# ---------------------------------------------------------------------------

# Canonical: acquisition_datetime — ordered exiftool date-tag fallback.
DATETIME_FIELDS: list[str] = field_list("DATETIME_FIELDS")

# Canonical: image_shape [height, width]. Each group is (width_tag, height_tag);
# IMAGE_SIZE_COMBINED_FIELD is the "WxH" string fallback.
IMAGE_SIZE_FIELDS: list[tuple[str, ...]] = field_tuples("IMAGE_SIZE_FIELDS")
IMAGE_SIZE_COMBINED_FIELD: str = field_scalar("IMAGE_SIZE_COMBINED_FIELD")

# Canonical: focal_length / focal_length_35mm (+ combined-string fallback).
FOCAL_LENGTH_FIELD: str = field_scalar("FOCAL_LENGTH_FIELD")
FOCAL_LENGTH_35MM_FIELD: str = field_scalar("FOCAL_LENGTH_35MM_FIELD")
FOCAL_LENGTH_COMBINED_FIELD: str = field_scalar("FOCAL_LENGTH_COMBINED_FIELD")

# Canonical: cam_pitch / cam_heading / cam_roll. Each group is
# (pitch_tag, yaw_tag, roll_tag), tried in priority order; heading-only is the
# fallback when no full triple is present.
CAM_ORIENTATION_PYR_FIELDS: list[tuple[str, ...]] = field_tuples(
    "CAM_ORIENTATION_PYR_FIELDS"
)
CAM_HEADING_ONLY_FIELDS: list[str] = field_list("CAM_HEADING_ONLY_FIELDS")

# Canonical: capture_date — LAS/LAZ (creation_year, creation_doy) header pair.
PC_CAPTURE_DATE_LAS_FIELDS: tuple[str, ...] = field_tuple("PC_CAPTURE_DATE_LAS_FIELDS")
