"""Per-client sidecar normalisation rules.

FUTURE WORK
-----------
The hardcoded ``CLIENT_SCHEMAS`` mapping below is a temporary placeholder. It
will be replaced by per-client JSON schema files that the client provides
alongside their data (e.g. dropped into the same S3 bucket, or shipped in this
repo under ``client_schemas/<bucket>.json``). When that lands:

  * ``client_schema_for_bucket`` should look up and parse the JSON file rather
    than read from the in-memory dict.
  * The ``ClientSchema`` dataclass can stay as the in-memory representation —
    just hydrate it from the JSON.
  * Drop the ``__PLACEHOLDER__`` entries below; they exist only so an
    un-filled-in schema fails loudly downstream.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClientSchema:
    """Per-client sidecar normalisation rules.

    headerless_columns: positional column names applied when the client CSV
        ships without a header row. Empty tuple means "the CSV has a header".
    column_renames: client-specific column renames applied AFTER positional
        naming and BEFORE the global alias canonicalisation in
        `_apply_global_aliases`.
    """

    headerless_columns: tuple[str, ...] = ()
    column_renames: Mapping[str, str] = field(default_factory=dict)


# Bucket → ClientSchema. Bucket name is matched after stripping `s3://` and
# any trailing slash (see `client_schema_for_bucket`).
#
# PLACEHOLDER: ATC. Real column names to be filled in by the data team.
# `__PLACEHOLDER__` tags are deliberately invalid — they will not silently
# match a real EXIF/canonical column, so a sidecar generated against an
# un-filled-in schema fails loudly downstream.
CLIENT_SCHEMAS: Mapping[str, ClientSchema] = {
    "atomic-saas-atc-production": ClientSchema(
        headerless_columns=(
            "Filename",
            "__PLACEHOLDER__col2",
            "__PLACEHOLDER__col3",
            "__PLACEHOLDER__col4",
        ),
        column_renames={
            "M21": "__PLACEHOLDER__M21_target",
            "M31": "__PLACEHOLDER__M31_target",
            "M41": "__PLACEHOLDER__M41_target",
        },
    ),
}


_EMPTY_SCHEMA = ClientSchema()


def _normalize_bucket(bucket: str) -> str:
    if not bucket:
        return ""
    b = bucket.strip()
    if b.startswith("s3://"):
        b = b[len("s3://") :]
    return b.rstrip("/")


def client_schema_for_bucket(bucket: str) -> ClientSchema:
    """Return the schema for `bucket`, or an empty schema if no entry exists.

    An empty schema means: trust the client CSV as-is (header present, no
    per-client renames). Global alias canonicalisation still runs on top.
    """
    return CLIENT_SCHEMAS.get(_normalize_bucket(bucket), _EMPTY_SCHEMA)
