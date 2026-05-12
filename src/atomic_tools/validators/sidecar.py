"""Validate a sidecar CSV (client or generated)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from atomic_tools.client_sidecar import (
    SidecarMergeError,
    _apply_client_renames,
    _apply_global_aliases,
    build_global_alias_map,
    load_client_schema,
)
from atomic_tools.io.storage import from_directory
from atomic_tools.utils.utils import (
    DATA_TYPE_INFO,
    DataTypeEnum,
    _is_path_tail_suffix,
    _split_path_components,
    load_sidecar_df,
)
from atomic_tools.validators.report import LintReport
from atomic_tools.validators.required_fields import REQUIRED_SIDECAR_FIELD_GROUPS
from atomic_tools.validators.values import VALIDATORS

_MAX_LISTED_FILES = 20
_DEFAULT_ROW_NAME = "DEFAULT"


def lint_sidecar_file(
    sidecar_path: str,
    *,
    final: bool,
    data_type: DataTypeEnum,
    schema_path: str | Path | None,
    input_files_path: str | None,
) -> LintReport:
    report = LintReport()

    if final and schema_path is not None:
        report.add_warning(
            "--schema is ignored when --final is set (a final sidecar is already canonical).",
            fix_hint="Drop --schema, or run without --final to lint a client sidecar.",
        )
        schema_path = None

    if data_type not in REQUIRED_SIDECAR_FIELD_GROUPS:
        report.add_error(
            f"Unsupported data type for lint: {data_type}.",
            fix_hint=f"Allowed values: {sorted(REQUIRED_SIDECAR_FIELD_GROUPS.keys())}.",
        )
        return report

    schema = None
    if schema_path is not None:
        try:
            schema = load_client_schema(schema_path)
        except (OSError, ValueError) as e:
            report.add_error(
                f"Could not load schema {schema_path}: {e}",
                fix_hint="Run `am-tools lint schema <path>` to diagnose the schema file.",
            )
            return report

    column_names = schema.column_names if schema and schema.column_names else None

    try:
        df = load_sidecar_df(sidecar_path, column_names=column_names)
    except FileNotFoundError as e:
        report.add_error(f"Sidecar file does not exist: {e}")
        return report
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        report.add_error(
            f"Could not parse sidecar CSV: {e}",
            fix_hint="Check the file is a valid CSV and not empty.",
        )
        return report
    except (OSError, UnicodeDecodeError, ValueError) as e:
        report.add_error(f"Could not read sidecar: {e}")
        return report

    if schema is not None:
        try:
            df = _apply_client_renames(df, schema)
        except SidecarMergeError as e:
            report.add_error(
                f"Schema rename failed: {e}",
                fix_hint="Run `am-tools lint schema <path>` to diagnose the schema file.",
            )
            return report
        flat_groups = [g for groups in REQUIRED_SIDECAR_FIELD_GROUPS.values() for g in groups]
        df = _apply_global_aliases(df, build_global_alias_map(flat_groups))

    required_groups = REQUIRED_SIDECAR_FIELD_GROUPS[data_type]
    columns = list(df.columns)
    columns_set = set(columns)

    default_row_idx = _find_default_row_index(df)
    if final and default_row_idx is None:
        report.add_warning(
            "Generated sidecar has no DEFAULT row.",
            fix_hint="Final sidecars conventionally include a row where Filename == 'DEFAULT'.",
        )

    default_satisfied: dict[str, bool] = {}
    if final:
        _check_required_columns(required_groups, columns_set, report)
        default_satisfied = _check_required_values(
            df, required_groups, columns_set, default_row_idx, report
        )
    else:
        for group in required_groups:
            default_satisfied[group[0]] = False

    _check_value_formats(df, columns, default_row_idx, report)

    if input_files_path:
        _check_file_inventory(
            df=df,
            data_type=data_type,
            input_files_path=input_files_path,
            final=final,
            default_satisfied=default_satisfied,
            report=report,
        )

    if not report.findings:
        mode = "final" if final else "client"
        report.add_info(
            f"Sidecar OK ({mode} mode, datatype={data_type}, "
            f"{len(df)} row(s), {len(columns)} column(s))."
        )

    return report


def _find_default_row_index(df: pd.DataFrame) -> int | None:
    if df.shape[1] == 0 or len(df) == 0:
        return None
    stripped = df[df.columns[0]].astype(str).str.strip()
    matches = stripped[stripped == _DEFAULT_ROW_NAME]
    if matches.empty:
        return None
    return int(matches.index[0])


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    return bool(s) and s.lower() != "nan"


def _check_required_columns(
    required_groups: list[list[str]],
    columns_set: set[str],
    report: LintReport,
) -> None:
    for group in required_groups:
        if not any(field in columns_set for field in group):
            canonical = group[0]
            aliases = ", ".join(group)
            report.add_error(
                f"Required column for {canonical!r} is missing.",
                fix_hint=f"Add a column named {canonical!r} (accepted aliases: {aliases}).",
            )


def _check_required_values(
    df: pd.DataFrame,
    required_groups: list[list[str]],
    columns_set: set[str],
    default_row_idx: int | None,
    report: LintReport,
) -> dict[str, bool]:
    """Return {canonical: covered_by_default}."""
    file_col = df.columns[0] if df.shape[1] else None
    default_row = df.iloc[default_row_idx] if default_row_idx is not None else None
    non_default = df.drop(index=default_row_idx) if default_row_idx is not None else df
    default_satisfied: dict[str, bool] = {}

    for group in required_groups:
        canonical = group[0]
        present_fields = [f for f in group if f in columns_set]
        if not present_fields:
            default_satisfied[canonical] = False
            continue

        default_has = default_row is not None and any(
            _has_value(default_row[f]) for f in present_fields
        )
        default_satisfied[canonical] = default_has

        if default_has:
            continue

        missing_rows: list[tuple[int, str]] = []
        for idx, row in non_default.iterrows():
            if not any(_has_value(row[f]) for f in present_fields):
                fname = str(row[file_col]).strip() if file_col else ""
                missing_rows.append((int(idx), fname))

        if missing_rows:
            sample = missing_rows[:_MAX_LISTED_FILES]
            tail = (
                f" (+{len(missing_rows) - _MAX_LISTED_FILES} more)"
                if len(missing_rows) > _MAX_LISTED_FILES
                else ""
            )
            row_summary = ", ".join(f"row {idx} ({name!r})" for idx, name in sample) + tail
            report.add_error(
                f"{canonical!r} is missing on {len(missing_rows)} row(s) and "
                f"DEFAULT does not provide a value: {row_summary}",
                fix_hint=(
                    f"Either fill {canonical!r} on the listed rows or set a value in DEFAULT."
                ),
            )

    return default_satisfied


def _check_value_formats(
    df: pd.DataFrame,
    columns: list[str],
    default_row_idx: int | None,
    report: LintReport,
) -> None:
    file_col = columns[0] if columns else None
    file_col_series = df[file_col].astype(str).str.strip() if file_col else None
    for col in columns:
        validator = VALIDATORS.get(col)
        if validator is None:
            continue
        for idx, value in df[col].items():
            if value is None:
                continue
            value_str = str(value).strip()
            if not value_str or value_str.lower() == "nan":
                continue
            ok, err = validator(value_str)
            if ok:
                continue
            if idx == default_row_idx:
                label = _DEFAULT_ROW_NAME
            else:
                fname = file_col_series[idx] if file_col_series is not None else ""
                label = f"row {idx} ({fname!r})" if fname else f"row {idx}"
            report.add_error(
                f"{col}: {err}",
                location=f"{label}, col {col!r}",
            )

    _check_bounds_consistency(df, columns, report)


def _check_bounds_consistency(
    df: pd.DataFrame,
    columns: list[str],
    report: LintReport,
) -> None:
    axes = [
        ("bounds.minx", "bounds.maxx"),
        ("bounds.miny", "bounds.maxy"),
        ("bounds.minz", "bounds.maxz"),
    ]
    file_col = columns[0] if columns else None
    for min_col, max_col in axes:
        if min_col not in columns or max_col not in columns:
            continue
        for idx, row in df.iterrows():
            mn_raw, mx_raw = row[min_col], row[max_col]
            if not (_has_value(mn_raw) and _has_value(mx_raw)):
                continue
            try:
                mn = float(str(mn_raw).strip())
                mx = float(str(mx_raw).strip())
            except ValueError:
                continue
            if mn > mx:
                fname = str(row[file_col]).strip() if file_col else ""
                label = f"row {idx} ({fname!r})" if fname else f"row {idx}"
                report.add_warning(
                    f"{min_col} ({mn}) > {max_col} ({mx}).",
                    location=f"{label}",
                    fix_hint="Swap min/max for this axis.",
                )


def _check_file_inventory(
    *,
    df: pd.DataFrame,
    data_type: DataTypeEnum,
    input_files_path: str,
    final: bool,
    default_satisfied: dict[str, bool],
    report: LintReport,
) -> None:
    info = DATA_TYPE_INFO[data_type]
    include = list(info.get("include") or [])
    exclude = list(info.get("exclude") or [])

    try:
        backend = from_directory(input_files_path)
        keys = backend.list_keys(include=include, exclude=exclude)
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as e:
        report.add_error(
            f"Could not enumerate input files at {input_files_path}: {e}",
            fix_hint="Check the path exists and is readable.",
        )
        return

    if not keys:
        report.add_warning(
            f"No files matching {include} found in {input_files_path}.",
            fix_hint="Confirm the directory and datatype are correct.",
        )
        return

    file_col = df.columns[0]
    sidecar_filenames = df[file_col].astype(str).str.strip()
    non_default_mask = sidecar_filenames != _DEFAULT_ROW_NAME

    # Bucket by basename so each key only tail-checks rows sharing its
    # basename — tail-suffix requires equal last component.
    rows_by_basename: dict[str, list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
    for idx, name in sidecar_filenames[non_default_mask].items():
        parts = _split_path_components(name)
        if parts:
            rows_by_basename[parts[-1]].append((int(idx), parts))

    row_to_keys: dict[int, list[str]] = defaultdict(list)
    missing: list[str] = []
    for key in keys:
        key_parts = _split_path_components(key)
        if not key_parts:
            missing.append(key)
            continue
        matches = [
            row_idx
            for row_idx, parts in rows_by_basename.get(key_parts[-1], [])
            if _is_path_tail_suffix(key_parts, parts)
        ]
        if len(matches) > 1:
            report.add_warning(
                f"Input file {key!r} matches {len(matches)} sidecar rows: {matches}.",
                fix_hint="Use more specific paths in the sidecar to disambiguate.",
            )
            continue
        if not matches:
            missing.append(key)
            continue
        row_to_keys[matches[0]].append(key)

    for row_idx, files in row_to_keys.items():
        if len(files) > 1:
            sidecar_name = sidecar_filenames.loc[row_idx]
            report.add_warning(
                f"Sidecar row {row_idx} ({sidecar_name!r}) matches "
                f"{len(files)} input files: {files}.",
                fix_hint="Use more specific paths in the sidecar to disambiguate.",
            )

    if not missing:
        return
    missing.sort()
    sample = missing[:_MAX_LISTED_FILES]
    if len(missing) > _MAX_LISTED_FILES:
        tail = f" (+{len(missing) - _MAX_LISTED_FILES} more)"
    else:
        tail = ""
    listing = ", ".join(repr(n) for n in sample) + tail

    if final:
        all_default_covered = bool(default_satisfied) and all(default_satisfied.values())
        if all_default_covered:
            report.add_warning(
                f"{len(missing)} input file(s) have no sidecar row, but DEFAULT "
                f"covers every required field: {listing}",
                fix_hint=(
                    "This is OK — DEFAULT supplies the values. Add per-file rows "
                    "if some files need overrides."
                ),
            )
        else:
            report.add_error(
                f"{len(missing)} input file(s) have no sidecar row and DEFAULT "
                f"does not cover every required field: {listing}",
                fix_hint="Add a row for each missing file or set values in DEFAULT.",
            )
    else:
        report.add_warning(
            f"{len(missing)} input file(s) have no row in this client sidecar: {listing}",
            fix_hint=(
                "The generator will fill in any missing files automatically; this is informational."
            ),
        )
