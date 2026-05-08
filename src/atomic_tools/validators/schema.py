"""Validate a client schema JSON file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from atomic_tools.validators.report import LintReport

ALLOWED_TOP_LEVEL_KEYS = {"headerless_columns", "column_renames"}


def lint_schema_file(path: Path) -> LintReport:
    report = LintReport()
    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        report.add_error(
            f"Schema file does not exist: {path}",
            fix_hint="Check the path and try again.",
        )
        return report
    except IsADirectoryError:
        report.add_error(f"Schema path is not a file: {path}")
        return report
    except OSError as e:
        report.add_error(f"Could not read schema file: {e}")
        return report

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
            fix_hint='Wrap the schema in {{}} like {{"headerless_columns": [...]}}.',
        )
        return report

    has_headerless = "headerless_columns" in data
    has_renames = "column_renames" in data
    if not has_headerless and not has_renames:
        report.add_error(
            "Schema must contain at least one of 'headerless_columns' or 'column_renames'.",
            fix_hint='Add e.g. "headerless_columns": ["Filename", "GPSLatitude", ...]',
        )

    for key in data:
        if key not in ALLOWED_TOP_LEVEL_KEYS:
            report.add_error(
                f"Unknown top-level key: {key!r}.",
                fix_hint=(
                    f"Allowed keys are {sorted(ALLOWED_TOP_LEVEL_KEYS)}. "
                    "Check for typos (e.g. 'headerlessColumns' vs 'headerless_columns')."
                ),
            )

    if has_headerless:
        _validate_headerless_columns(data["headerless_columns"], report)
    if has_renames:
        _validate_column_renames(data["column_renames"], report)

    if has_headerless and has_renames:
        _cross_check(data["headerless_columns"], data["column_renames"], report)

    if not report.has_errors():
        n_headerless = len(data.get("headerless_columns", []) or []) if has_headerless else 0
        n_renames = len(data.get("column_renames", {}) or {}) if has_renames else 0
        report.add_info(
            f"Schema OK: {n_headerless} headerless column(s), {n_renames} rename(s)."
        )

    return report


def _validate_headerless_columns(value: object, report: LintReport) -> None:
    if not isinstance(value, list):
        report.add_error(
            f"'headerless_columns' must be a list, got {type(value).__name__}.",
            fix_hint='Use a JSON array, e.g. ["Filename", "GPSLatitude"].',
        )
        return

    if len(value) == 0:
        report.add_warning(
            "'headerless_columns' is empty.",
            fix_hint="Either remove the key or fill in the column names in CSV order.",
        )
        return

    seen: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(value):
        if not isinstance(entry, str):
            report.add_error(
                f"'headerless_columns[{i}]' must be a string, got {type(entry).__name__}.",
                location=f"index {i}",
            )
            continue
        if not entry.strip():
            report.add_error(
                f"'headerless_columns[{i}]' is empty.",
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


def _validate_column_renames(value: object, report: LintReport) -> None:
    if not isinstance(value, dict):
        report.add_error(
            f"'column_renames' must be an object, got {type(value).__name__}.",
            fix_hint='Use a JSON object, e.g. {"old_name": "GPSLatitude"}.',
        )
        return

    if len(value) == 0:
        report.add_warning(
            "'column_renames' is empty.",
            fix_hint="Either remove the key or add at least one rename.",
        )
        return

    target_to_sources: dict[str, list[str]] = defaultdict(list)
    for src, dst in value.items():
        if not isinstance(src, str) or not src.strip():
            report.add_error(
                f"'column_renames' has an invalid source key: {src!r}.",
                fix_hint="Source keys must be non-empty strings.",
            )
            continue
        if not isinstance(dst, str) or not dst.strip():
            report.add_error(
                f"'column_renames[{src!r}]' must map to a non-empty string, got {dst!r}.",
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


def _cross_check(headerless: object, renames: object, report: LintReport) -> None:
    if not isinstance(headerless, list) or not isinstance(renames, dict):
        return
    headerless_set = {h for h in headerless if isinstance(h, str)}
    for src in renames:
        if isinstance(src, str) and src not in headerless_set:
            report.add_warning(
                f"Rename source {src!r} is not in 'headerless_columns'.",
                fix_hint=(
                    "Renames are applied to the headerless column names, so this "
                    "rename would no-op. Either add the source to 'headerless_columns' "
                    "or remove the rename."
                ),
            )
