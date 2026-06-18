"""LintReport: accumulates findings and renders colored output."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Literal

import click

Level = Literal["error", "warning", "info"]

# Cell value written for a required field a row is missing. Present fields are
# left blank so gaps are easy to spot when the CSV is opened in a spreadsheet.
MISSING_MARKER = "MISSING"


@dataclass(frozen=True)
class LintFinding:
    level: Level
    message: str
    fix_hint: str | None = None
    location: str | None = None


@dataclass
class MissingDataReport:
    """Per-row matrix of which required fields a sidecar row is missing.

    Only rows missing at least one required field are kept. ``field_columns``
    are the canonical required-field names; each row maps every field column to
    ``MISSING_MARKER`` (absent) or ``""`` (present, on the row or via DEFAULT).
    """

    filename_column: str
    field_columns: list[str]
    rows: list[dict[str, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.rows

    def write_csv(self, path: str) -> None:
        header = [self.filename_column, *self.field_columns]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(self.rows)


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)
    # Populated by the sidecar linter when a datatype (and thus a set of
    # required fields) is known. None when there's nothing to tabulate.
    missing_data: MissingDataReport | None = None

    def add_error(
        self,
        message: str,
        *,
        fix_hint: str | None = None,
        location: str | None = None,
    ) -> None:
        self.findings.append(LintFinding("error", message, fix_hint, location))

    def add_warning(
        self,
        message: str,
        *,
        fix_hint: str | None = None,
        location: str | None = None,
    ) -> None:
        self.findings.append(LintFinding("warning", message, fix_hint, location))

    def add_info(self, message: str) -> None:
        self.findings.append(LintFinding("info", message))

    def errors(self) -> list[LintFinding]:
        return [f for f in self.findings if f.level == "error"]

    def warnings(self) -> list[LintFinding]:
        return [f for f in self.findings if f.level == "warning"]

    def infos(self) -> list[LintFinding]:
        return [f for f in self.findings if f.level == "info"]

    def has_errors(self) -> bool:
        return any(f.level == "error" for f in self.findings)

    def exit_code(self) -> int:
        return 1 if self.has_errors() else 0

    def render(self) -> str:
        lines: list[str] = []
        for finding in self.findings:
            if finding.level == "error":
                tag = click.style("ERROR", fg="bright_red", bold=True)
            elif finding.level == "warning":
                tag = click.style("WARN ", fg="yellow", bold=True)
            else:
                tag = click.style("INFO ", fg="cyan")
            loc = f" [{finding.location}]" if finding.location else ""
            lines.append(f"{tag}{loc} {finding.message}")
            if finding.fix_hint:
                lines.append(click.style(f"        fix: {finding.fix_hint}", fg="bright_black"))

        n_err = len(self.errors())
        n_warn = len(self.warnings())
        if n_err:
            footer = click.style(
                f"FAILED: {n_err} error(s), {n_warn} warning(s).",
                fg="bright_red",
                bold=True,
            )
        elif n_warn:
            footer = click.style(f"PASSED with {n_warn} warning(s).", fg="yellow", bold=True)
        else:
            footer = click.style("PASSED.", fg="green", bold=True)
        lines.append(footer)
        return "\n".join(lines)
