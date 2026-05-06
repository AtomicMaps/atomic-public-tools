"""Load + clean + merge a client-supplied sidecar CSV.

Originally `tasks/generate_sidecar_csv/client_sidecar.py`.
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from atomic_tools.client_schemas import ClientSchema, client_schema_for_bucket
from atomic_tools.utils.utils import find_sidecar_row, load_sidecar_df

logger = logging.getLogger(__name__)


class SidecarMergeError(RuntimeError):
    """Raised when a client sidecar cannot be normalised or merged unambiguously."""


# ---- Cleaning passes -------------------------------------------------------


def _apply_positional_names(df: pd.DataFrame, schema: ClientSchema) -> pd.DataFrame:
    """Validate column count when a headerless schema is in effect.

    The actual `header=None, names=…` was applied at read time in
    `load_sidecar_df` (which received `headerless_columns` from the schema).
    Here we just sanity-check that the column count we got matches.
    """
    if not schema.headerless_columns:
        return df
    expected = len(schema.headerless_columns)
    actual = df.shape[1]
    if actual != expected:
        raise SidecarMergeError(
            f"Headerless client CSV column count mismatch: schema declares "
            f"{expected} columns ({list(schema.headerless_columns)}), "
            f"but CSV has {actual}."
        )
    return df


def _apply_client_renames(df: pd.DataFrame, schema: ClientSchema) -> pd.DataFrame:
    """Apply per-client column renames. Raises on rename-target collisions."""
    if not schema.column_renames:
        return df
    return _safe_rename(df, dict(schema.column_renames), context="per-client renames")


def build_global_alias_map(
    required_field_groups: dict[str, list[list[str]]],
) -> dict[str, str]:
    """Invert the required-field-groups dict into a flat {alias: canonical} map.

    First entry of each group is canonical; remaining entries are aliases that
    should be renamed to canonical when seen in a client sidecar.
    """
    alias_to_canonical: dict[str, str] = {}
    for groups in required_field_groups.values():
        for group in groups:
            if not group:
                continue
            canonical, *aliases = group
            for alias in aliases:
                if alias == canonical:
                    continue
                existing = alias_to_canonical.get(alias)
                if existing and existing != canonical:
                    logger.warning(
                        f"Alias collision: {alias!r} maps to both "
                        f"{existing!r} and {canonical!r}; using {canonical!r}."
                    )
                alias_to_canonical[alias] = canonical
    return alias_to_canonical


def _apply_global_aliases(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    """Rename alias columns to their canonical names.

    If both an alias and its canonical column are present, the canonical wins
    and the alias column is dropped with a warning rather than overwritten —
    silently picking would risk shipping wrong data.
    """
    if not alias_map:
        return df
    cols = set(df.columns)
    renames: dict[str, str] = {}
    drops: list[str] = []
    for alias, canonical in alias_map.items():
        if alias not in cols:
            continue
        if canonical in cols:
            logger.warning(
                f"Client sidecar has both {alias!r} (alias) and "
                f"{canonical!r} (canonical); dropping the alias column."
            )
            drops.append(alias)
        else:
            renames[alias] = canonical
    if drops:
        df = df.drop(columns=drops)
    if renames:
        df = _safe_rename(df, renames, context="global alias map")
    return df


def _safe_rename(df: pd.DataFrame, renames: dict[str, str], *, context: str) -> pd.DataFrame:
    """Apply a rename, raising if two source columns collide on the same target."""
    target_to_sources: dict[str, list[str]] = defaultdict(list)
    for src, dst in renames.items():
        if src in df.columns:
            target_to_sources[dst].append(src)
    collisions = {dst: srcs for dst, srcs in target_to_sources.items() if len(srcs) > 1}
    if collisions:
        raise SidecarMergeError(
            f"Rename collision in {context}: multiple source columns map to "
            f"the same target: {collisions}"
        )
    for dst, srcs in target_to_sources.items():
        if dst in df.columns and dst not in renames:
            raise SidecarMergeError(
                f"Rename collision in {context}: column {srcs[0]!r} would be "
                f"renamed to {dst!r}, which already exists."
            )
    return df.rename(columns=renames)


def _validate_filename_column(df: pd.DataFrame) -> None:
    """The first column is the filename/match column — validate it.

    Reuses `find_sidecar_row`'s contract that the first column is the filename
    column. Empty values are allowed (they simply won't match any image);
    duplicates are not.
    """
    if df.shape[1] == 0:
        raise SidecarMergeError("Client sidecar has no columns.")
    file_col = df.columns[0]
    series = df[file_col].fillna("").astype(str).str.strip()
    non_empty = series[series != ""]
    duplicates = non_empty[non_empty.duplicated()].unique().tolist()
    if duplicates:
        raise SidecarMergeError(
            f"Client sidecar's filename column {file_col!r} has duplicate values: {duplicates}"
        )


# ---- Top-level: load + clean ----------------------------------------------


def load_and_clean_client_sidecar(
    url: str,
    bucket: str,
    required_field_groups: dict[str, list[list[str]]],
) -> pd.DataFrame:
    """Load the client sidecar at `url` and run the cleaning pipeline.

    `bucket` is used to look up the per-client schema. `required_field_groups`
    is injected (rather than imported) to avoid a circular dependency with the
    sidecar generator section below.
    """
    schema = client_schema_for_bucket(bucket)
    logger.info(
        f"Loading client sidecar {url!r} for bucket {bucket!r} "
        f"(headerless={'yes' if schema.headerless_columns else 'no'}, "
        f"renames={len(schema.column_renames)})"
    )
    df = load_sidecar_df(url, headerless_columns=schema.headerless_columns or None)
    df = _apply_positional_names(df, schema)
    df = _apply_client_renames(df, schema)
    df = _apply_global_aliases(df, build_global_alias_map(required_field_groups))
    _validate_filename_column(df)
    logger.info(
        f"Client sidecar cleaned: {len(df)} row(s), {len(df.columns)} column(s); "
        f"columns={list(df.columns)}"
    )
    return df


# ---- Matching + merge ------------------------------------------------------


def _suffix_on_boundary_match(csv_id: str, image_basenames: list[str]) -> list[str]:
    """Return image basenames whose stem ends with `csv_id` on a separator boundary.

    `csv_id` matches a stem if the stem == csv_id, or the stem ends with
    `[-_]<csv_id>`. Anchoring on a separator means `0001` does NOT match
    `IMG_10001`.
    """
    csv_id = csv_id.strip()
    if not csv_id:
        return []
    pattern = re.compile(rf"(^|[-_]){re.escape(csv_id)}$")
    matches: list[str] = []
    for basename in image_basenames:
        stem = Path(basename).stem
        if pattern.search(stem):
            matches.append(basename)
    return matches


def _client_meta_from_row(row: pd.Series, file_col: str) -> dict[str, str]:
    """Convert a matched client-CSV row to a meta dict, dropping empties + filename col."""
    meta: dict[str, str] = {}
    for col, val in row.items():
        if col == file_col:
            continue
        s = str(val).strip() if val is not None else ""
        if not s or s.lower() == "nan":
            continue
        meta[col] = s
    return meta


def _build_suffix_match_index(
    client_df: pd.DataFrame,
    file_col: str,
    image_basenames: list[str],
) -> dict[str, pd.Series]:
    """Map image basename → matched CSV row via suffix-on-boundary.

    Walks each CSV id once, evaluating the regex against every image stem;
    raises `SidecarMergeError` if any CSV id matches more than one image.
    """
    if not image_basenames:
        return {}
    index: dict[str, pd.Series] = {}
    for csv_idx in range(len(client_df)):
        csv_id = str(client_df.iloc[csv_idx][file_col]).strip()
        if not csv_id:
            continue
        matches = _suffix_on_boundary_match(csv_id, image_basenames)
        if len(matches) > 1:
            raise SidecarMergeError(
                f"Ambiguous suffix match: client id {csv_id!r} matches "
                f"{len(matches)} files: {matches}"
            )
        if matches:
            index[matches[0]] = client_df.iloc[csv_idx]
    return index


def merge_client_metadata(
    file_metadata: list[tuple[str, dict[str, Any]]],
    client_df: pd.DataFrame,
) -> list[tuple[str, dict[str, Any]]]:
    """Merge cleaned client rows into per-file metadata. Client wins on conflict.

    Mutates the dicts inside `file_metadata` in place and returns the same list.
    Empty cells in the client row do NOT overwrite EXIF values.
    """
    file_col = client_df.columns[0]
    image_basenames = [name for name, _ in file_metadata]

    suffix_index = _build_suffix_match_index(client_df, file_col, image_basenames)

    primary_hits = 0
    suffix_hits = 0
    no_match = 0

    for basename, meta in file_metadata:
        matched, _default = find_sidecar_row(client_df, basename)
        if matched is None:
            matched = suffix_index.get(basename)
            if matched is None:
                no_match += 1
                continue
            suffix_hits += 1
        else:
            primary_hits += 1

        meta.update(_client_meta_from_row(matched, file_col))

    logger.info(
        f"Client sidecar merge: {primary_hits} primary match(es), "
        f"{suffix_hits} suffix-fallback match(es), {no_match} unmatched."
    )
    return file_metadata
