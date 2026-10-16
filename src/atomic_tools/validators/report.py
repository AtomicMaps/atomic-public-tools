"""LintReport: accumulates findings and renders colored output."""

from __future__ import annotations

import csv
from collections.abc import Iterable
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
    # Terminal-only output (e.g. the geography histograms); omitted from the
    # API response.
    cli_only: bool = False
    # The files this finding is about — sidecar Filename labels, or input keys
    # for files with no sidecar row; empty for a batch-level finding. Feeds the
    # API's per-file pass/fail lists.
    files: tuple[str, ...] = ()


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
    # Filename → data type value for every typed row, i.e. every row the
    # per-type checks ran on (rows added from COCO alone have none).
    row_types: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.rows

    def mark(self, name: str, field_name: str, data_type: str) -> None:
        """Record ``name`` as missing ``field_name`` (a check outside the
        required-field groups, e.g. a CRS)."""
        if field_name not in self.field_columns:
            self.field_columns.append(field_name)
        row = next((r for r in self.rows if r.get(self.filename_column) == name), None)
        if row is None:
            row = {self.filename_column: name}
            self.rows.append(row)
        row[field_name] = MISSING_MARKER
        self.row_types.setdefault(name, data_type)

    def write_csv(self, path: str) -> None:
        header = [self.filename_column, *self.field_columns]
        if self.include_coco:
            header += [COCO_STATUS_COLUMN, COCO_LABELS_COLUMN]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def to_dict(self, keys_by_name: dict[str, str] | None = None) -> dict:
        """``{data_type: [{"file", "missing", ...}]}`` naming only missing fields.

        ``keys_by_name`` maps a Filename to the path to report (default: as is).
        Rows with no data type (COCO images with no sidecar row) go under
        ``"unknown"``. Rows COCO added with nothing missing are left out unless
        their images are unusable or not on disk; the rest (e.g. ``degraded``)
        stay in ``details.coco`` and the CSV.
        """
        from atomic_tools.validators.coco import TIER_NOT_ON_DISK, TIER_UNUSABLE

        keys_by_name = keys_by_name or {}
        grouped: dict[str, list[dict]] = {}
        for row in self.rows:
            name = row.get(self.filename_column, "")
            missing = [c for c in self.field_columns if row.get(c) == MISSING_MARKER]
            status = row.get(COCO_STATUS_COLUMN, "")
            if not missing and status not in (TIER_UNUSABLE, TIER_NOT_ON_DISK):
                continue
            entry: dict = {"file": keys_by_name.get(name, name), "missing": missing}
            if self.include_coco:
                entry["coco_status"] = status
                entry["coco_labels"] = int(row.get(COCO_LABELS_COLUMN) or 0)
            grouped.setdefault(self.row_types.get(name, "unknown"), []).append(entry)
        return grouped


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)
    # Populated by the sidecar linter when a datatype (and thus a set of
    # required fields) is known. None when there's nothing to tabulate.
    missing_data: MissingDataReport | None = None
    # Same, for optional fields (orientation under --ignore-missing-orientation):
    # gaps that warn rather than fail.
    optional_missing: MissingDataReport | None = None
    # Populated when a COCO file is supplied (image datatypes only).
    coco_impact: CocoImpact | None = None

    def add_error(
        self,
        message: str,
        *,
        fix_hint: str | None = None,
        location: str | None = None,
        files: Iterable[str] = (),
    ) -> None:
        self.findings.append(
            LintFinding("error", message, fix_hint, location, files=tuple(files))
        )

    def add_warning(
        self,
        message: str,
        *,
        fix_hint: str | None = None,
        location: str | None = None,
        files: Iterable[str] = (),
    ) -> None:
        self.findings.append(
            LintFinding("warning", message, fix_hint, location, files=tuple(files))
        )

    def add_info(self, message: str, *, cli_only: bool = False) -> None:
        self.findings.append(LintFinding("info", message, cli_only=cli_only))

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

    def api_findings(self) -> list[LintFinding]:
        """The findings the API reports: all but the ``cli_only`` ones."""
        return [f for f in self.findings if not f.cli_only]

    def findings_as_dicts(self) -> list[dict]:
        """:meth:`api_findings` as dicts, without the internal fields."""
        return [
            {"level": f.level, "message": f.message, "fix_hint": f.fix_hint,
             "location": f.location}
            for f in self.api_findings()
        ]

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
