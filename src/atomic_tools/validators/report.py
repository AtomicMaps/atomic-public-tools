"""LintReport: accumulates findings and renders colored output."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import click

if TYPE_CHECKING:
    from atomic_tools.validators.coco import CocoImpact

Level = Literal["error", "warning", "info"]

# Extra columns appended to the failed-rows CSV when a COCO file is supplied.
COCO_STATUS_COLUMN = "coco_status"
COCO_LABELS_COLUMN = "coco_labels"

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
    # Set when a COCO file is supplied: each row then also carries a
    # ``coco_status`` (tier) and ``coco_labels`` (annotation count) cell, and
    # the CSV grows two matching columns.
    include_coco: bool = False

    def is_empty(self) -> bool:
        return not self.rows

    def write_csv(self, path: str) -> None:
        header = [self.filename_column, *self.field_columns]
        if self.include_coco:
            header += [COCO_STATUS_COLUMN, COCO_LABELS_COLUMN]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)
    # Populated by the sidecar linter when a datatype (and thus a set of
    # required fields) is known. None when there's nothing to tabulate.
    missing_data: MissingDataReport | None = None
    # Populated when a COCO file is supplied (image datatypes only).
    coco_impact: CocoImpact | None = None

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

        if self.coco_impact is not None:
            lines.extend(self._render_coco_breakdown())

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

    def _render_coco_breakdown(self) -> list[str]:
        """An aligned per-tier image/label table, shown just above the footer so
        the COCO matching outcome is the last thing the user sees.
        """
        impact = self.coco_impact
        assert impact is not None
        rows = impact.tier_counts()
        sev_color = {"ok": "green", "warn": "yellow", "error": "bright_red"}

        imgs = [r[1] for r in rows] + [impact.images_in_coco]
        labs = [r[2] for r in rows] + [impact.total_labels]
        name_w = max(len(r[0]) for r in rows)
        img_w = max(len(str(v)) for v in imgs)
        lab_w = max(len(str(v)) for v in labs)

        out = ["", click.style(f"COCO label matching ({impact.display_name}):", bold=True)]
        for tier, n_img, n_lab, sev in rows:
            # Dim tiers with nothing in them; colour the rest by severity.
            fg = sev_color[sev] if n_img else "bright_black"
            out.append(
                click.style(
                    f"  {tier:<{name_w}}  {n_img:>{img_w}} image(s)  {n_lab:>{lab_w}} label(s)",
                    fg=fg,
                )
            )
        out.append(
            click.style(
                f"  {'total':<{name_w}}  {impact.images_in_coco:>{img_w}} image(s)  "
                f"{impact.total_labels:>{lab_w}} label(s)",
                bold=True,
            )
        )
        return out
