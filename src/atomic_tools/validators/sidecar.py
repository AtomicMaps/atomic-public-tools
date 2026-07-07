"""Validate a sidecar CSV (client or generated)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from atomic_tools.client_sidecar import (
    SidecarMergeError,
    _apply_client_renames,
    _apply_global_aliases,
    build_global_alias_map,
    load_client_schema,
)
from atomic_tools.io.storage import from_directory
from atomic_tools.utils.aws_errors import S3AuthError
from atomic_tools.utils.utils import (
    DATA_TYPE_INFO,
    DataTypeEnum,
    _is_path_tail_suffix,
    _split_path_components,
    has_value,
    infer_data_type,
    load_sidecar_df,
)
from atomic_tools.validators.constants import DEFAULT_ROW_NAME, MAX_LISTED_FILES
from atomic_tools.validators.geography import analyze_spatial_distribution
from atomic_tools.validators.report import MISSING_MARKER, LintReport, MissingDataReport
from atomic_tools.validators.required_fields import (
    ALL_SIDECAR_FIELD_GROUPS,
    OPTIONAL_SIDECAR_FIELD_GROUPS,
    REQUIRED_SIDECAR_FIELD_GROUPS,
)
from atomic_tools.validators.values import VALIDATORS


def lint_sidecar_file(
    sidecar_path: str,
    *,
    final: bool,
    data_type: DataTypeEnum | None,
    schema_path: str | Path | None,
    input_files_path: str | None,
    ignore_missing_orientation: bool = False,
    coco_path: str | None = None,
    coco_not_on_disk_is_error: bool = True,
) -> LintReport:
    report = LintReport()

    if final and schema_path is not None:
        report.add_warning(
            "--schema is ignored when --final is set (a final sidecar is already canonical).",
            fix_hint="Drop --schema, or run without --final to lint a client sidecar.",
        )
        schema_path = None

    if data_type is not None and data_type not in ALL_SIDECAR_FIELD_GROUPS:
        report.add_error(
            f"Unsupported data type for lint: {data_type}.",
            fix_hint=f"Allowed values: {sorted(ALL_SIDECAR_FIELD_GROUPS.keys())}.",
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
        flat_groups = [g for groups in ALL_SIDECAR_FIELD_GROUPS.values() for g in groups]
        df = _apply_global_aliases(df, build_global_alias_map(flat_groups))

    columns = list(df.columns)
    columns_set = set(columns)

    default_row_idx = _find_default_row_index(df)
    if final and default_row_idx is None:
        report.add_warning(
            "Generated sidecar has no DEFAULT row.",
            fix_hint="Final sidecars conventionally include a row where Filename == 'DEFAULT'.",
        )

    # Per-row data type (from the DataType column, else filename inference),
    # honouring the optional `data_type` filter. DEFAULT and unclassifiable rows
    # map to None and are excluded from per-type checks.
    row_types = _row_types(df, data_type, default_row_idx, report)
    detected_types = list(dict.fromkeys(t for t in row_types.values() if t is not None))
    multi = len(detected_types) > 1

    # An explicit filter is always a "rule type" even with zero matching rows, so
    # the file-inventory check can still learn whether DEFAULT covers its
    # required fields (a DEFAULT-only sidecar is a valid pattern).
    rule_types = list(detected_types)
    if data_type is not None and data_type not in rule_types:
        rule_types.append(data_type)
    rules_by_type = _rules_by_type(rule_types, ignore_missing_orientation)

    # default_satisfied is keyed by (type_value, canonical).
    default_satisfied: dict[tuple[str, str], bool] = {}
    if final:
        for detected in rule_types:
            required_groups, optional_groups = rules_by_type[detected]
            has_rows = detected in detected_types
            label = detected.value if multi else None
            if has_rows:
                # Column-presence errors apply only to a type that has rows.
                _check_required_columns(required_groups, columns_set, report, type_label=label)
                _check_optional_columns(optional_groups, columns_set, report, type_label=label)
            type_rows = df.loc[
                [idx for idx, tt in row_types.items() if tt == detected]
            ]
            sat = _check_required_values(
                df, type_rows, required_groups, columns_set, default_row_idx, report,
                type_label=label,
            )
            for canonical, ok in sat.items():
                default_satisfied[(detected.value, canonical)] = ok
    else:
        for detected in rule_types:
            required_groups, _ = rules_by_type[detected]
            for group in required_groups:
                default_satisfied[(detected.value, group[0])] = False

    _check_value_formats(df, columns, default_row_idx, report)

    if input_files_path:
        _check_file_inventory(
            df=df,
            row_types=row_types,
            data_type_filter=data_type,
            rules_by_type=rules_by_type,
            input_files_path=input_files_path,
            final=final,
            default_satisfied=default_satisfied,
            report=report,
        )

    report.missing_data = _build_missing_data_report(
        df, row_types, rules_by_type, columns_set, default_row_idx
    )

    if coco_path:
        _apply_coco_impact(
            df=df,
            row_types=row_types,
            coco_path=coco_path,
            data_type_filter=data_type,
            rules_by_type=rules_by_type,
            ignore_missing_orientation=ignore_missing_orientation,
            columns_set=columns_set,
            default_row_idx=default_row_idx,
            report=report,
            not_on_disk_is_error=coco_not_on_disk_is_error,
        )

    analyze_spatial_distribution(df, report)

    if not report.findings:
        mode = "final" if final else "client"
        if detected_types:
            counts = Counter(t.value for t in row_types.values() if t is not None)
            detected_str = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        else:
            detected_str = "none"
        report.add_info(
            f"Sidecar OK ({mode} mode, detected: {detected_str}, "
            f"{len(df)} row(s), {len(columns)} column(s))."
        )

    return report


def _row_types(
    df: pd.DataFrame,
    data_type_filter: DataTypeEnum | None,
    default_row_idx: int | None,
    report: LintReport,
) -> dict[int, DataTypeEnum | None]:
    """Classify each sidecar row into a DataTypeEnum (or None).

    Resolution order per row:
      1. A non-empty ``DataType`` column value (an unrecognised value warns and
         maps to None).
      2. Filename inference via ``infer_data_type`` (filename-only; ambiguous
         images default to oriented_image, matching the backend's no-byte-access
         behaviour).
      3. DEFAULT and unclassifiable rows map to None (type-agnostic / skipped).

    When ``data_type_filter`` is set, rows of other types map to None (excluded
    from every per-type check) and an info records how many rows survived.
    """
    file_col = df.columns[0] if df.shape[1] else None
    has_col = "DataType" in df.columns
    result: dict[int, DataTypeEnum | None] = {}
    unknown_values: list[tuple[int, str, str]] = []
    uninferable: list[str] = []
    non_default_total = 0

    for idx, row in df.iterrows():
        idx = int(idx)
        fname = str(row[file_col]).strip() if file_col else ""
        if idx == default_row_idx or fname == DEFAULT_ROW_NAME:
            result[idx] = None
            continue
        non_default_total += 1

        detected: DataTypeEnum | None = None
        if has_col:
            raw = str(row["DataType"]).strip()
            if raw:
                try:
                    detected = DataTypeEnum(raw)
                except ValueError:
                    unknown_values.append((idx, fname, raw))
                    result[idx] = None
                    continue

        if detected is None:
            inferred = infer_data_type(fname)
            if inferred is not None:
                try:
                    detected = DataTypeEnum(inferred)
                except ValueError:
                    detected = None

        if detected is None:
            uninferable.append(fname)
            result[idx] = None
            continue

        if data_type_filter is not None and detected != data_type_filter:
            result[idx] = None
            continue

        result[idx] = detected

    for idx, fname, raw in unknown_values:
        report.add_warning(
            f"Row {idx} ({fname!r}) has unrecognized DataType {raw!r}; "
            "skipped for required-field checks.",
            fix_hint=f"Use one of: {[t.value for t in DataTypeEnum]}, or leave it blank.",
        )
    if uninferable:
        sample = uninferable[:MAX_LISTED_FILES]
        tail = (
            f" (+{len(uninferable) - MAX_LISTED_FILES} more)"
            if len(uninferable) > MAX_LISTED_FILES
            else ""
        )
        report.add_info(
            f"{len(uninferable)} row(s) could not be classified by filename and "
            f"were skipped for required-field checks: {sample}{tail}"
        )
    if data_type_filter is not None:
        kept = sum(1 for t in result.values() if t is not None)
        report.add_info(
            f"Filtered to {data_type_filter.value}: {kept} of {non_default_total} row(s)."
        )

    return result


def _rules_by_type(
    detected_types: list[DataTypeEnum],
    ignore_missing_orientation: bool,
) -> dict[DataTypeEnum, tuple[list[list[str]], list[list[str]]]]:
    """Return per-type ``(required_groups, optional_groups)``.

    Orientation (Pitch/Heading/Roll) is optional-by-default for oriented images,
    but unless the caller opts to ignore it we promote those groups to required
    so missing orientation is an error rather than a warning.
    """
    rules: dict[DataTypeEnum, tuple[list[list[str]], list[list[str]]]] = {}
    for detected in detected_types:
        required_groups = list(REQUIRED_SIDECAR_FIELD_GROUPS.get(detected, []))
        optional_groups = list(OPTIONAL_SIDECAR_FIELD_GROUPS.get(detected, []))
        if detected == DataTypeEnum.oriented_image and not ignore_missing_orientation:
            required_groups = [*required_groups, *optional_groups]
            optional_groups = []
        rules[detected] = (required_groups, optional_groups)
    return rules


def _find_default_row_index(df: pd.DataFrame) -> int | None:
    if df.shape[1] == 0 or len(df) == 0:
        return None
    stripped = df[df.columns[0]].astype(str).str.strip()
    matches = stripped[stripped == DEFAULT_ROW_NAME]
    if matches.empty:
        return None
    return int(matches.index[0])


def _check_required_columns(
    required_groups: list[list[str]],
    columns_set: set[str],
    report: LintReport,
    type_label: str | None = None,
) -> None:
    suffix = f" ({type_label})" if type_label else ""
    for group in required_groups:
        if not any(field in columns_set for field in group):
            canonical = group[0]
            aliases = ", ".join(group)
            report.add_error(
                f"Required column for {canonical!r}{suffix} is missing.",
                fix_hint=f"Add a column named {canonical!r} (accepted aliases: {aliases}).",
            )


def _check_optional_columns(
    optional_groups: list[list[str]],
    columns_set: set[str],
    report: LintReport,
    type_label: str | None = None,
) -> None:
    """Warn (don't error) when an optional column group is entirely absent.

    Present-but-blank values in an optional column are allowed and not flagged;
    any values that *are* present get format-checked by ``_check_value_formats``.
    """
    suffix = f" ({type_label})" if type_label else ""
    for group in optional_groups:
        if not any(field in columns_set for field in group):
            canonical = group[0]
            aliases = ", ".join(group)
            report.add_warning(
                f"Optional column for {canonical!r}{suffix} is not present.",
                fix_hint=(
                    f"Add {canonical!r} (accepted aliases: {aliases}) if available. "
                    "Images missing orientation (Pitch/Heading/Roll) will appear in "
                    "Lens without orientation but still process successfully."
                ),
            )


def _check_required_values(
    df: pd.DataFrame,
    non_default: pd.DataFrame,
    required_groups: list[list[str]],
    columns_set: set[str],
    default_row_idx: int | None,
    report: LintReport,
    type_label: str | None = None,
) -> dict[str, bool]:
    """Return {canonical: covered_by_default} for one type's row subset.

    ``non_default`` is the subset of rows to check (this type's rows, excluding
    DEFAULT); DEFAULT-row coverage is read from ``df``. ``type_label`` names the
    type in the error message when a scan detected more than one.
    """
    file_col = df.columns[0] if df.shape[1] else None
    default_row = df.iloc[default_row_idx] if default_row_idx is not None else None
    default_satisfied: dict[str, bool] = {}
    suffix = f" ({type_label})" if type_label else ""

    for group in required_groups:
        canonical = group[0]
        present_fields = [f for f in group if f in columns_set]
        if not present_fields:
            default_satisfied[canonical] = False
            continue

        default_has = default_row is not None and any(
            has_value(default_row[f]) for f in present_fields
        )
        default_satisfied[canonical] = default_has

        if default_has:
            continue

        missing_rows: list[tuple[int, str]] = []
        for idx, row in non_default.iterrows():
            if not any(has_value(row[f]) for f in present_fields):
                fname = str(row[file_col]).strip() if file_col else ""
                missing_rows.append((int(idx), fname))

        if missing_rows:
            sample = missing_rows[:MAX_LISTED_FILES]
            tail = (
                f" (+{len(missing_rows) - MAX_LISTED_FILES} more)"
                if len(missing_rows) > MAX_LISTED_FILES
                else ""
            )
            row_summary = ", ".join(f"row {idx} ({name!r})" for idx, name in sample) + tail
            report.add_error(
                f"{canonical!r}{suffix} is missing on {len(missing_rows)} row(s) and "
                f"DEFAULT does not provide a value: {row_summary}",
                fix_hint=(
                    f"Either fill {canonical!r} on the listed rows or set a value in DEFAULT."
                ),
            )

    return default_satisfied


def _build_missing_data_report(
    df: pd.DataFrame,
    row_types: dict[int, DataTypeEnum | None],
    rules_by_type: dict[DataTypeEnum, tuple[list[list[str]], list[list[str]]]],
    columns_set: set[str],
    default_row_idx: int | None,
) -> MissingDataReport | None:
    """Tabulate which required fields each typed row is missing.

    ``field_columns`` is the ordered union of required canonicals across all
    detected types. A cell can be "missing" only for the field groups that apply
    to that row's own type; non-applicable cells stay ``""``. A field group
    counts as present for a row when the row (or the DEFAULT row) supplies any
    field in the group — the same rule ``_check_required_values`` uses. Returns
    None when there are no typed rows, and an empty report when nothing is
    missing.
    """
    detected_types = list(dict.fromkeys(t for t in row_types.values() if t is not None))
    if not detected_types or df.shape[1] == 0:
        return None

    file_col = df.columns[0]
    default_row = df.iloc[default_row_idx] if default_row_idx is not None else None

    # Per-type required groups (with oriented promotion already applied) and the
    # ordered union of canonicals across every detected type.
    required_by_type = {t: rules_by_type[t][0] for t in detected_types}
    field_columns: list[str] = []
    for detected in detected_types:
        for group in required_by_type[detected]:
            if group[0] not in field_columns:
                field_columns.append(group[0])

    # Resolve present fields + DEFAULT coverage per (type, canonical) once.
    present_by_type: dict[DataTypeEnum, dict[str, list[str]]] = {}
    default_covers_by_type: dict[DataTypeEnum, dict[str, bool]] = {}
    for detected in detected_types:
        present = {
            group[0]: [f for f in group if f in columns_set]
            for group in required_by_type[detected]
        }
        present_by_type[detected] = present
        default_covers_by_type[detected] = {
            canonical: default_row is not None
            and any(has_value(default_row[f]) for f in fields)
            for canonical, fields in present.items()
        }

    rows: list[dict[str, str]] = []
    for idx, row in df.iterrows():
        idx = int(idx)
        detected = row_types.get(idx)
        if idx == default_row_idx or detected is None:
            continue
        present = present_by_type[detected]
        default_covers = default_covers_by_type[detected]
        record: dict[str, str] = {str(file_col): str(row[file_col]).strip()}
        any_missing = False
        for canonical in field_columns:
            if canonical not in present:
                # Not applicable to this row's type.
                record[canonical] = ""
                continue
            if default_covers[canonical]:
                missing = False
            else:
                missing = not any(has_value(row[f]) for f in present[canonical])
                any_missing = any_missing or missing
            record[canonical] = MISSING_MARKER if missing else ""
        if any_missing:
            rows.append(record)

    return MissingDataReport(
        filename_column=str(file_col),
        field_columns=field_columns,
        rows=rows,
    )


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
                label = DEFAULT_ROW_NAME
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
            if not (has_value(mn_raw) and has_value(mx_raw)):
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
    row_types: dict[int, DataTypeEnum | None],
    data_type_filter: DataTypeEnum | None,
    rules_by_type: dict[DataTypeEnum, tuple[list[list[str]], list[list[str]]]],
    input_files_path: str,
    final: bool,
    default_satisfied: dict[tuple[str, str], bool],
    report: LintReport,
) -> None:
    detected_types = list(dict.fromkeys(t for t in row_types.values() if t is not None))

    if data_type_filter is not None:
        # Explicit filter: keep the exact single-type listing.
        info = DATA_TYPE_INFO[data_type_filter]
        include = list(info.get("include") or [])
        exclude = list(info.get("exclude") or [])
    else:
        # Auto: list the union include of the detected row types; per-type
        # excludes are applied by infer_data_type when we classify each key.
        include = []
        for detected in detected_types:
            for ext in DATA_TYPE_INFO.get(detected, {}).get("include") or []:
                if ext not in include:
                    include.append(ext)
        exclude = []

    if not include:
        # No typed rows to inventory against (auto mode, nothing classified).
        return

    try:
        backend = from_directory(input_files_path)
        keys = backend.list_keys(include=include, exclude=exclude)
    except S3AuthError as e:
        report.add_error(
            f"Could not enumerate input files at {input_files_path}: {e}",
            fix_hint=e.help_text(),
        )
        return
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

    # Classify each listed key to a type; in auto mode drop keys whose type isn't
    # among the detected row types (or is unclassifiable).
    detected_values = {t.value for t in detected_types}
    key_type: dict[str, DataTypeEnum] = {}
    kept_keys: list[str] = []
    for key in keys:
        if data_type_filter is not None:
            key_type[key] = data_type_filter
            kept_keys.append(key)
            continue
        inferred = infer_data_type(key)
        if inferred is None or inferred not in detected_values:
            continue
        key_type[key] = DataTypeEnum(inferred)
        kept_keys.append(key)
    keys = kept_keys
    if not keys:
        return

    file_col = df.columns[0]
    sidecar_filenames = df[file_col].astype(str).str.strip()
    non_default_mask = sidecar_filenames != DEFAULT_ROW_NAME

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

    def _listing(items: list[str]) -> str:
        sample = items[:MAX_LISTED_FILES]
        tail = (
            f" (+{len(items) - MAX_LISTED_FILES} more)"
            if len(items) > MAX_LISTED_FILES
            else ""
        )
        return ", ".join(repr(n) for n in sample) + tail

    if final:
        # Whether DEFAULT covers every required field is per-type: a missing file
        # is OK only if its own type's required canonicals are all default-filled.
        # Iterate every rule type (includes an explicit filter with zero rows).
        covered_by_type: dict[str, bool] = {}
        for detected, (required_groups, _optional) in rules_by_type.items():
            canonicals = [group[0] for group in required_groups]
            covered_by_type[detected.value] = all(
                default_satisfied.get((detected.value, canonical), False)
                for canonical in canonicals
            )
        covered = [k for k in missing if covered_by_type.get(key_type[k].value, False)]
        uncovered = [k for k in missing if not covered_by_type.get(key_type[k].value, False)]

        if uncovered:
            report.add_error(
                f"{len(uncovered)} input file(s) have no sidecar row and DEFAULT "
                f"does not cover every required field: {_listing(uncovered)}",
                fix_hint="Add a row for each missing file or set values in DEFAULT.",
            )
        if covered:
            report.add_warning(
                f"{len(covered)} input file(s) have no sidecar row, but DEFAULT "
                f"covers every required field: {_listing(covered)}",
                fix_hint=(
                    "This is OK — DEFAULT supplies the values. Add per-file rows "
                    "if some files need overrides."
                ),
            )
    else:
        report.add_warning(
            f"{len(missing)} input file(s) have no row in this client sidecar: "
            f"{_listing(missing)}",
            fix_hint=(
                "The generator will fill in any missing files automatically; this is informational."
            ),
        )


def _image_group_union(
    image_types: list[DataTypeEnum],
    rules_by_type: dict[DataTypeEnum, tuple[list[list[str]], list[list[str]]]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Union the required/optional groups across the detected image types.

    Oriented promotion is already baked into ``rules_by_type``. A canonical that
    is required for any image type wins over an optional listing for another, so
    a field never counts as both required and optional.
    """
    required_union: list[list[str]] = []
    optional_union: list[list[str]] = []
    for detected in image_types:
        required_groups, optional_groups = rules_by_type[detected]
        for group in required_groups:
            if group not in required_union:
                required_union.append(group)
        for group in optional_groups:
            if group not in optional_union:
                optional_union.append(group)
    required_canonicals = {group[0] for group in required_union}
    optional_union = [g for g in optional_union if g[0] not in required_canonicals]
    return required_union, optional_union


def _apply_coco_impact(
    *,
    df: pd.DataFrame,
    row_types: dict[int, DataTypeEnum | None],
    coco_path: str,
    data_type_filter: DataTypeEnum | None,
    rules_by_type: dict[DataTypeEnum, tuple[list[list[str]], list[list[str]]]],
    ignore_missing_orientation: bool,
    columns_set: set[str],
    default_row_idx: int | None,
    report: LintReport,
    not_on_disk_is_error: bool = True,
) -> None:
    """Map COCO labels onto the sidecar's per-row verdicts and report impact."""
    from atomic_tools.validators import coco as coco_mod

    # An explicit non-image filter means COCO can't apply at all.
    if data_type_filter is not None and data_type_filter not in coco_mod.IMAGE_DATA_TYPES:
        report.add_warning(
            f"COCO file ignored: label impact only applies to image data types, "
            f"not {data_type_filter.value}.",
            fix_hint="Drop --coco for this datatype.",
        )
        return

    image_indices = [
        idx
        for idx, detected in row_types.items()
        if detected is not None and detected in coco_mod.IMAGE_DATA_TYPES
    ]
    if not image_indices:
        report.add_warning(
            "COCO file ignored: no image rows detected in the sidecar.",
            fix_hint="COCO label impact only applies to image data types.",
        )
        return

    image_types = list(dict.fromkeys(row_types[idx] for idx in image_indices))
    required_groups, optional_groups = _image_group_union(image_types, rules_by_type)

    try:
        resolved_path, coco_images = coco_mod.load_coco(coco_path)
    except coco_mod.CocoError as e:
        report.add_error(f"Could not use COCO file: {e}")
        return

    if not coco_images:
        report.add_warning(f"COCO file {resolved_path} has no images[] entries.")
        return

    # Restrict matching to the image rows (+ DEFAULT) and re-index so the mixed
    # label/positional access inside analyze_coco_impact stays consistent.
    keep = ([default_row_idx] if default_row_idx is not None else []) + image_indices
    image_df = df.loc[keep].reset_index(drop=True)
    image_default_idx = 0 if default_row_idx is not None else None

    impact = coco_mod.analyze_coco_impact(
        df=image_df,
        coco_path=resolved_path,
        coco_images=coco_images,
        required_groups=required_groups,
        optional_groups=optional_groups,
        columns_set=columns_set,
        default_row_idx=image_default_idx,
    )
    impact.not_on_disk_is_error = not_on_disk_is_error
    report.coco_impact = impact

    for line in impact.summary_lines():
        report.add_info(line)

    if report.missing_data is not None and impact.verdicts:
        coco_mod.augment_missing_data(report.missing_data, impact)

    csv_hint = (
        "Per-image detail (including degraded rows) is in the failed-rows "
        "CSV — use --report to save it."
    )

    def _format(verdicts: list) -> str:
        return "; ".join(
            f"{v.report_name or '?'} [{v.tier}, {v.labels} label(s)]" for v in verdicts
        )

    # not-on-disk means the COCO references images that were never extracted
    # into the sidecar (no matching file), so their labels can't be carried
    # through Flow. When building a sidecar this blocks (must be fixed before the
    # file ships); during `validate` it's downgraded to a warning so an
    # informational "is my data clean?" run reports the gap without failing.
    nod_flagged, nod_truncated = coco_mod.flagged_sample(
        impact, tiers={coco_mod.TIER_NOT_ON_DISK}
    )
    if nod_flagged:
        detail = _format(nod_flagged)
        if nod_truncated:
            detail += f" (+{nod_truncated} more)"
        add_finding = report.add_error if not_on_disk_is_error else report.add_warning
        add_finding(
            f"{impact.not_on_disk} COCO image(s) have no matching file in the input "
            f"directory, carrying {impact.labels_in_tier(coco_mod.TIER_NOT_ON_DISK)} "
            f"label(s): {detail}",
            fix_hint=(
                "These labels reference images that aren't in the scanned directory. "
                "Either add the missing images, or trim the COCO to the images present. "
                f"{csv_hint}"
            ),
        )

    # unusable-but-present images (a required field is missing, or zero width/
    # height) stay a warning — the image is here, the metadata just needs fixing.
    unusable_flagged, unusable_truncated = coco_mod.flagged_sample(
        impact, tiers={coco_mod.TIER_UNUSABLE}
    )
    if unusable_flagged:
        detail = _format(unusable_flagged)
        if unusable_truncated:
            detail += f" (+{unusable_truncated} more)"
        report.add_warning(
            f"{impact.unusable} image(s) unusable carry "
            f"{impact.labels_in_tier(coco_mod.TIER_UNUSABLE)} label(s): {detail}",
            fix_hint=csv_hint,
        )
