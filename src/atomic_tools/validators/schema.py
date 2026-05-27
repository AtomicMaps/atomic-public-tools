"""Validate a client schema JSON file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from atomic_tools.utils.utils import is_remote_uri, read_text_uri
from atomic_tools.validators.report import LintReport

ALLOWED_TOP_LEVEL_KEYS = {"column_names", "column_name_mapping"}


def lint_schema_file(path: str | Path) -> LintReport:
    report = LintReport()
    path_str = str(path)

    try:
        text = read_text_uri(path_str)
    except FileNotFoundError:
        report.add_error(
            f"Schema file does not exist: {path_str}",
            fix_hint="Check the path and try again.",
        )
        return report
    except IsADirectoryError:
        report.add_error(f"Schema path is not a file: {path_str}")
        return report
    except OSError as e:
        report.add_error(f"Could not read schema file: {e}")
        return report
    except Exception as e:
        if is_remote_uri(path_str):
            report.add_error(
                f"Could not fetch schema from {path_str}: {e}",
                fix_hint="Check the URI, your credentials, and network access.",
            )
            return report
        raise

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        report.add_error(
            f"Invalid JSON: {e.msg}",
            location=f"line {e.lineno}, col {e.colno}",
            fix_hint="Fix the syntax error at the indicated position. Comments are not supported.",
        )
        return report

    if not isinstance(data, dict):
        report.add_error(
            f"Schema must be a JSON object, got {type(data).__name__}.",
            fix_hint='Wrap the schema in {{}} like {{"column_names": [...]}}.',
        )
        return report

    has_column_names = "column_names" in data
    has_mapping = "column_name_mapping" in data
    if not has_column_names and not has_mapping:
        report.add_error(
            "Schema must contain at least one of 'column_names' or 'column_name_mapping'.",
            fix_hint='Add e.g. "column_names": ["Filename", "GPSLatitude", ...]',
        )

    for key in data:
        if key not in ALLOWED_TOP_LEVEL_KEYS:
            report.add_error(
                f"Unknown top-level key: {key!r}.",
                fix_hint=(
                    f"Allowed keys are {sorted(ALLOWED_TOP_LEVEL_KEYS)}. "
                    "Check for typos (e.g. 'columnNames' vs 'column_names')."
                ),
            )

    if has_column_names:
        _validate_column_names(data["column_names"], report)
    if has_mapping:
        _validate_column_name_mapping(data["column_name_mapping"], report)

    if has_column_names and has_mapping:
        _cross_check(data["column_names"], data["column_name_mapping"], report)

    if not report.has_errors():
        n_column_names = len(data.get("column_names", []) or []) if has_column_names else 0
        n_mappings = len(data.get("column_name_mapping", {}) or {}) if has_mapping else 0
        report.add_info(f"Schema OK: {n_column_names} column name(s), {n_mappings} mapping(s).")

    return report


def _validate_column_names(value: object, report: LintReport) -> None:
    if not isinstance(value, list):
        report.add_error(
            f"'column_names' must be a list, got {type(value).__name__}.",
            fix_hint='Use a JSON array, e.g. ["Filename", "GPSLatitude"].',
        )
        return

    if len(value) == 0:
        report.add_warning(
            "'column_names' is empty.",
            fix_hint="Either remove the key or fill in the column names in CSV order.",
        )
        return

    seen: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(value):
        if not isinstance(entry, str):
            report.add_error(
                f"'column_names[{i}]' must be a string, got {type(entry).__name__}.",
                location=f"index {i}",
            )
            continue
        if not entry.strip():
            report.add_error(
                f"'column_names[{i}]' is empty.",
                location=f"index {i}",
                fix_hint="Use the column name from your CSV.",
            )
            continue
        seen[entry].append(i)

    for name, indices in seen.items():
        if len(indices) > 1:
            report.add_warning(
                f"Duplicate column name {name!r} at indices {indices}.",
                fix_hint="Each column name should be unique.",
            )


def _validate_column_name_mapping(value: object, report: LintReport) -> None:
    if not isinstance(value, dict):
        report.add_error(
            f"'column_name_mapping' must be an object, got {type(value).__name__}.",
            fix_hint='Use a JSON object, e.g. {"old_name": "GPSLatitude"}.',
        )
        return

    if len(value) == 0:
        report.add_warning(
            "'column_name_mapping' is empty.",
            fix_hint="Either remove the key or add at least one mapping.",
        )
        return

    target_to_sources: dict[str, list[str]] = defaultdict(list)
    for src, dst in value.items():
        if not isinstance(src, str) or not src.strip():
            report.add_error(
                f"'column_name_mapping' has an invalid source key: {src!r}.",
                fix_hint="Source keys must be non-empty strings.",
            )
            continue
        if not isinstance(dst, str) or not dst.strip():
            report.add_error(
                f"'column_name_mapping[{src!r}]' must map to a non-empty string, got {dst!r}.",
                fix_hint="Use the canonical column name as the value.",
            )
            continue
        if src == dst:
            report.add_warning(
                f"Identity rename: {src!r} -> {dst!r} has no effect.",
                fix_hint="Remove this entry — source and target are the same.",
            )
        target_to_sources[dst].append(src)

    for dst, sources in target_to_sources.items():
        if len(sources) > 1:
            report.add_error(
                f"Multiple source columns rename to the same target {dst!r}: {sources}.",
                fix_hint=(
                    "A target name can only be produced from one source — "
                    "remove or fix the duplicates."
                ),
            )


def _cross_check(column_names: object, mapping: object, report: LintReport) -> None:
    if not isinstance(column_names, list) or not isinstance(mapping, dict):
        return
    column_names_set = {h for h in column_names if isinstance(h, str)}
    for src in mapping:
        if isinstance(src, str) and src not in column_names_set:
            report.add_warning(
                f"Rename source {src!r} is not in 'column_names'.",
                fix_hint=(
                    "Renames are applied to the column names, so this "
                    "rename would no-op. Either add the source to 'column_names' "
                    "or remove the rename."
                ),
            )
