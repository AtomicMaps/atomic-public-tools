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
from urllib.parse import urlparse

import pandas as pd

from atomic_tools.utils.object_store import REMOTE_SCHEME_TO_STORE_TYPE, ObjectStore
from atomic_tools.utils.utils import (
    find_sidecar_row,
    get_object_keys,
    is_remote_uri,
    load_sidecar_df,
    read_text_uri,
)

logger = logging.getLogger(__name__)

_CSV_SUFFIX = ".csv"


# ---- Schema ----------------------------------------------------------------
#
# A client schema describes how to normalise a client-supplied sidecar CSV
# before merging:
#
#   * column_names — positional column names applied when the client
#     CSV ships without a header row. Empty / omitted means "the CSV has a
#     header".
#   * column_name_mapping — per-client column renames applied AFTER positional
#     naming and BEFORE the global alias canonicalisation.
#
# See ``schemas/example.json`` for an example.


@dataclass(frozen=True)
class ClientSchema:
    column_names: tuple[str, ...] = ()
    column_name_mapping: Mapping[str, str] = field(default_factory=dict)


_EMPTY_SCHEMA = ClientSchema()


def load_client_schema(path: str | Path | None) -> ClientSchema:
    """Load a client schema from a JSON file (local path or remote URI).

    `path` may be a local filesystem path or an object-store URI (``s3://…``,
    ``gs://…``, ``az://…``). Returns the empty schema (no positional column
    names, no renames) when ``path`` is ``None`` — i.e. when the caller hasn't
    supplied a schema.
    """
    if path is None:
        return _EMPTY_SCHEMA
    data = json.loads(read_text_uri(str(path)))
    return ClientSchema(
        column_names=tuple(data.get("column_names", ())),
        column_name_mapping=dict(data.get("column_name_mapping", {})),
    )


class SidecarMergeError(RuntimeError):
    """Raised when a client sidecar cannot be normalised or merged unambiguously."""


# ---- Cleaning passes -------------------------------------------------------


def _apply_positional_names(df: pd.DataFrame, schema: ClientSchema) -> pd.DataFrame:
    """Validate column count when column_names is in effect.

    The actual `header=None, names=…` was applied at read time in
    `load_sidecar_df` (which received `column_names` from the schema).
    Here we just sanity-check that the column count we got matches.
    """
    if not schema.column_names:
        return df
    expected = len(schema.column_names)
    actual = df.shape[1]
    if actual != expected:
        raise SidecarMergeError(
            f"Headerless client CSV column count mismatch: schema declares "
            f"{expected} columns ({list(schema.column_names)}), "
            f"but CSV has {actual}."
        )
    return df


def _apply_client_renames(df: pd.DataFrame, schema: ClientSchema) -> pd.DataFrame:
    """Apply per-client column renames. Raises on rename-target collisions."""
    if not schema.column_name_mapping:
        return df
    return _safe_rename(df, dict(schema.column_name_mapping), context="per-client renames")


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


def _is_sidecar_directory(url: str) -> bool:
    """Return True when `url` points at a directory of sidecars rather than a file.

    For object-store URIs we can't cheaply stat the target, so a URI is treated
    as a directory unless it ends in ``.csv``. Local paths are checked directly.
    """
    if is_remote_uri(url):
        return not url.rstrip("/").lower().endswith(_CSV_SUFFIX)
    return Path(url).expanduser().is_dir()


def _list_sidecar_csvs_below(url: str) -> list[str]:
    """Return loadable paths/URIs for every ``.csv`` in a subdirectory of `url`.

    CSVs sitting directly in `url` are skipped: that top level is where the
    generated sidecar is written, so it must never be picked up as an input.
    Only files nested at least one directory deeper are returned, sorted.
    """
    if is_remote_uri(url):
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        store_type = REMOTE_SCHEME_TO_STORE_TYPE[scheme]
        prefix = parsed.path.strip("/")
        store = ObjectStore(store_type).from_url(f"{scheme}://{parsed.netloc}")
        keys = get_object_keys(store, prefix, include=[_CSV_SUFFIX], exclude=[])
        results: list[str] = []
        for key in keys:
            rel = key[len(prefix) :].lstrip("/") if prefix else key
            if "/" not in rel:
                continue  # directly under the top level — skip it
            results.append(f"{scheme}://{parsed.netloc}/{key}")
        return sorted(results)

    root = Path(url).expanduser()
    results = []
    for path in root.rglob("*"):
        if path.suffix.lower() != _CSV_SUFFIX or not path.is_file():
            continue
        if path.parent == root:
            continue  # top-level CSV — that's where the output goes
        results.append(str(path))
    return sorted(results)


def _load_single_sidecar(url: str, schema: ClientSchema) -> pd.DataFrame:
    """Read one client sidecar CSV and apply positional naming/count validation."""
    df = load_sidecar_df(url, column_names=schema.column_names or None)
    return _apply_positional_names(df, schema)


def _dedupe_default_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple ``DEFAULT`` rows (one per merged sidecar) to the first.

    Merging several sidecars can produce several DEFAULT rows, which would
    otherwise trip the duplicate-filename check. Keep the first and drop the rest.
    """
    if df.shape[1] == 0:
        return df
    file_col = df.columns[0]
    is_default = df[file_col].fillna("").astype(str).str.strip() == "DEFAULT"
    if is_default.sum() <= 1:
        return df
    logger.warning(
        f"Found {int(is_default.sum())} DEFAULT rows across the merged sidecars; "
        f"keeping the first and dropping the rest."
    )
    default_positions = list(df.index[is_default])
    drop_idx = default_positions[1:]
    return df.drop(index=drop_idx).reset_index(drop=True)


def _load_and_merge_sidecar_dir(url: str, schema: ClientSchema) -> pd.DataFrame:
    """Load every sidecar CSV below `url`, verify a shared schema, and concatenate.

    The first CSV (sorted by path) defines the reference schema. Any sidecar
    whose columns differ from it is reported by name and aborts the run.
    """
    csv_paths = _list_sidecar_csvs_below(url)
    if not csv_paths:
        raise SidecarMergeError(
            f"No sidecar CSVs found in any subdirectory of {url!r}. Place each "
            f"client sidecar in a subfolder — the top-level folder is reserved "
            f"for the generated sidecar and is not scanned."
        )
    logger.info(f"Merging {len(csv_paths)} client sidecar CSV(s) from subdirectories of {url!r}.")

    frames: list[pd.DataFrame] = []
    reference_cols: list[str] | None = None
    reference_path: str | None = None
    for path in csv_paths:
        try:
            df = _load_single_sidecar(path, schema)
        except SidecarMergeError as e:
            raise SidecarMergeError(f"Client sidecar {path!r} is malformed: {e}") from e
        cols = list(df.columns)
        if reference_cols is None:
            reference_cols, reference_path = cols, path
        elif cols != reference_cols:
            raise SidecarMergeError(
                f"Client sidecar {path!r} has a different format than "
                f"{reference_path!r}: expected {len(reference_cols)} column(s) "
                f"{reference_cols}, but found {len(cols)} column(s) {cols}. All "
                f"sidecars under a directory must share the same schema."
            )
        logger.info(f"  loaded {path!r}: {len(df)} row(s).")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return _dedupe_default_rows(combined)


def load_and_clean_client_sidecar(
    url: str,
    schema_path: str | Path | None,
    required_field_groups: dict[str, list[list[str]]],
) -> pd.DataFrame:
    """Load the client sidecar(s) at `url` and run the cleaning pipeline.

    `url` may point at a single CSV (local path or remote URI) or at a
    directory. When it's a directory, every ``.csv`` in a *subdirectory* below
    it is merged into one frame (the top level itself is not scanned, since the
    generated sidecar is written there); all such CSVs must share the same
    schema or the run aborts naming the offending file.

    `schema_path` points to a client-provided JSON schema (local path or
    remote URI); when ``None``, no positional column naming or per-client
    renames are applied. `required_field_groups` is injected (rather than
    imported) to avoid a circular dependency with the sidecar generator.
    """
    schema = load_client_schema(schema_path)
    logger.info(
        f"Loading client sidecar {url!r} "
        f"(schema={str(schema_path) if schema_path else 'none'}, "
        f"headerless={'yes' if schema.column_names else 'no'}, "
        f"renames={len(schema.column_name_mapping)})"
    )
    if _is_sidecar_directory(url):
        df = _load_and_merge_sidecar_dir(url, schema)
    else:
        df = _load_single_sidecar(url, schema)
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
    file_labels = [name for name, _ in file_metadata]
    total_files = len(file_metadata)

    suffix_index = _build_suffix_match_index(client_df, file_col, file_labels)

    # Resolve all matches before merging so we can detect a client row that
    # ends up matched to multiple files (ambiguous and worth a warning).
    matched_rows: dict[str, tuple[pd.Series | None, bool]] = {}
    rows_to_files: dict[Any, list[str]] = defaultdict(list)
    default_row: pd.Series | None = None

    for label, _ in file_metadata:
        matched, default_row = find_sidecar_row(client_df, label)
        used_suffix = False
        if matched is None:
            matched = suffix_index.get(label)
            used_suffix = matched is not None
        matched_rows[label] = (matched, used_suffix)
        if matched is not None:
            rows_to_files[matched.name].append(label)

    for row_idx, files in rows_to_files.items():
        if len(files) > 1:
            client_id = str(client_df.iloc[row_idx][file_col]).strip()
            logger.warning(
                f"Client sidecar row {client_id!r} is ambiguous — it matches "
                f"{len(files)} files: {files}. Provide a more specific path "
                f"in the client sidecar to disambiguate."
            )

    primary_hits = 0
    suffix_hits = 0
    no_match = 0

    additions: dict[str, dict[str, int]] = defaultdict(lambda: {"specific": 0, "default": 0})

    for label, meta in file_metadata:
        matched, used_suffix = matched_rows[label]
        if matched is None and default_row is None:
            no_match += 1
            continue

        if matched is not None:
            if used_suffix:
                suffix_hits += 1
            else:
                primary_hits += 1

        specific_meta = _client_meta_from_row(matched, file_col) if matched is not None else {}
        default_meta = (
            _client_meta_from_row(default_row, file_col) if default_row is not None else {}
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
                    f"{label!r} field {key!r}: file={file_str!r}, "
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
