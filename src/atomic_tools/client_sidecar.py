"""Load + clean + merge a client-supplied sidecar CSV.

Originally `tasks/generate_sidecar_csv/client_sidecar.py`.
"""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from atomic_tools.utils.utils import find_sidecar_row, load_sidecar_df

logger = logging.getLogger(__name__)


# ---- Schema ----------------------------------------------------------------
#
# A client schema describes how to normalise a client-supplied sidecar CSV
# before merging:
#
#   * headerless_columns — positional column names applied when the client
#     CSV ships without a header row. Empty / omitted means "the CSV has a
#     header".
#   * column_renames — per-client column renames applied AFTER positional
#     naming and BEFORE the global alias canonicalisation.
#
# See ``schemas/example.json`` for an example.


@dataclass(frozen=True)
class ClientSchema:
    headerless_columns: tuple[str, ...] = ()
    column_renames: Mapping[str, str] = field(default_factory=dict)


_EMPTY_SCHEMA = ClientSchema()


def load_client_schema(path: Path | None) -> ClientSchema:
    """Load a client schema from a JSON file.

    Returns the empty schema (no headerless columns, no renames) when ``path``
    is ``None`` — i.e. when the caller hasn't supplied a schema.
    """
    if path is None:
        return _EMPTY_SCHEMA
    data = json.loads(path.read_text(encoding="utf-8"))
    return ClientSchema(
        headerless_columns=tuple(data.get("headerless_columns", ())),
        column_renames=dict(data.get("column_renames", {})),
    )


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
    field_groups: Iterable[list[str]],
) -> dict[str, str]:
    """Invert a sequence of ``[canonical, *aliases]`` groups into a flat
    ``{alias: canonical}`` map.
    """
    alias_to_canonical: dict[str, str] = {}
    for group in field_groups:
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


def _safe_rename(
    df: pd.DataFrame, renames: dict[str, str], *, context: str
) -> pd.DataFrame:
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
    schema_path: Path | None,
    required_field_groups: dict[str, list[list[str]]],
) -> pd.DataFrame:
    """Load the client sidecar at `url` and run the cleaning pipeline.

    `schema_path` points to a client-provided JSON schema (see
    ``schemas/example.json``); when ``None``, no headerless-column naming or
    per-client renames are applied. `required_field_groups` is injected
    (rather than imported) to avoid a circular dependency with the sidecar
    generator.
    """
    schema = load_client_schema(schema_path)
    logger.info(
        f"Loading client sidecar {url!r} "
        f"(schema={str(schema_path) if schema_path else 'none'}, "
        f"headerless={'yes' if schema.headerless_columns else 'no'}, "
        f"renames={len(schema.column_renames)})"
    )
    df = load_sidecar_df(url, headerless_columns=schema.headerless_columns or None)
    df = _apply_positional_names(df, schema)
    df = _apply_client_renames(df, schema)
    flat_groups = [g for groups in required_field_groups.values() for g in groups]
    df = _apply_global_aliases(df, build_global_alias_map(flat_groups))
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
        col_name = str(col)
        if col_name == file_col:
            continue
        s = str(val).strip() if val is not None else ""
        if not s or s.lower() == "nan":
            continue
        meta[col_name] = s
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
    """Merge cleaned client rows into per-file metadata. File metadata wins on conflict.

    Mutates the dicts inside `file_metadata` in place and returns the same list.
    Client values fill in only fields that are missing or empty on the file;
    when both sides have a value and they disagree, the file value is kept and
    a warning is logged. A per-file specific row (matched by filename or
    suffix-fallback) takes precedence over the client CSV's DEFAULT row.
    After the merge a summary of which columns were added is logged.
    """
    file_col = client_df.columns[0]
    image_basenames = [name for name, _ in file_metadata]
    total_files = len(file_metadata)

    suffix_index = _build_suffix_match_index(client_df, file_col, image_basenames)

    primary_hits = 0
    suffix_hits = 0
    no_match = 0

    additions: dict[str, dict[str, int]] = defaultdict(
        lambda: {"specific": 0, "default": 0}
    )

    for basename, meta in file_metadata:
        matched, default_row = find_sidecar_row(client_df, basename)
        used_suffix = False
        if matched is None:
            matched = suffix_index.get(basename)
            if matched is not None:
                used_suffix = True

        if matched is None and default_row is None:
            no_match += 1
            continue

        if matched is not None:
            if used_suffix:
                suffix_hits += 1
            else:
                primary_hits += 1

        specific_meta = (
            _client_meta_from_row(matched, file_col) if matched is not None else {}
        )
        default_meta = (
            _client_meta_from_row(default_row, file_col)
            if default_row is not None
            else {}
        )

        for key in set(specific_meta) | set(default_meta):
            if key in specific_meta:
                client_val = specific_meta[key]
                source = "specific"
            else:
                client_val = default_meta[key]
                source = "default"

            file_val = meta.get(key)
            file_str = str(file_val).strip() if file_val is not None else ""
            if not file_str:
                meta[key] = client_val
                additions[key][source] += 1
            elif file_str != client_val:
                logger.warning(
                    f"Client sidecar disagrees with file metadata for "
                    f"{basename!r} field {key!r}: file={file_str!r}, "
                    f"client={client_val!r} ({source}); keeping file value."
                )

    logger.info(
        f"Client sidecar merge: {primary_hits} primary match(es), "
        f"{suffix_hits} suffix-fallback match(es), {no_match} unmatched."
    )
    _log_addition_summary(additions, total_files)
    return file_metadata


def _log_addition_summary(
    additions: Mapping[str, Mapping[str, int]],
    total_files: int,
) -> None:
    """Log a per-column summary of metadata added by the client merge."""
    if not additions:
        logger.info("No metadata columns added from client sidecar.")
        return

    lines = [f"Added {len(additions)} columns of metadata:"]
    for key in sorted(additions):
        counts = additions[key]
        default_n = counts.get("default", 0)
        specific_n = counts.get("specific", 0)
        parts: list[str] = []
        if default_n:
            if default_n == total_files:
                parts.append("Default value added to all files")
            else:
                parts.append(f"Default value on {default_n}/{total_files} files")
        if specific_n:
            if specific_n == total_files:
                parts.append("File-specific value added to all files")
            else:
                parts.append(f"File-specific value on {specific_n}/{total_files} files")
        lines.append(f"[{key}] {', '.join(parts)}")
    logger.info("\n\t".join(lines))
