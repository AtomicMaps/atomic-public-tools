"""Interactively build a client schema (column-name mapping) for a sidecar CSV.

``am-tools build-schema`` reads a client-supplied sidecar CSV (local or remote),
shows per-column statistics and samples, and walks the user through mapping each
column to a canonical name. It writes a ``schema.json`` in the exact format
consumed by ``sidecar generate --client-schema`` (see ``client_sidecar`` and
``validators/schema.py``):

  * CSV has a header row  -> ``{"column_name_mapping": {<original>: <canonical>}}``
  * CSV is headerless     -> ``{"column_names": [<name>, ...]}`` (positional)
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Annotated

import click
import questionary
import typer

from atomic_tools.io.storage import from_directory
from atomic_tools.utils.utils import (
    DataTypeEnum,
    has_value,
    is_remote_uri,
    load_sidecar_df,
    uri_stem,
)
from atomic_tools.validators.required_fields import (
    ALL_SIDECAR_FIELD_GROUPS,
    ALL_SIDECAR_FIELD_GROUPS_FULL,
    OPTIONAL_SIDECAR_FIELD_GROUPS,
    REFERENCED_FULL_SIDECAR_FIELD_GROUPS,
    REQUIRED_SIDECAR_FIELD_GROUPS,
)
from atomic_tools.validators.schema import lint_schema_file

logger = logging.getLogger(__name__)

# The filename/match column is the implicit first column of every sidecar; it
# isn't part of the required/optional field groups, so offer it explicitly.
_FILENAME = "Filename"

_PREVIEW_ROWS = 5
_PREVIEW_COLS = 5
_SAMPLE_VALUES = 5

# Sentinel choices appended after the canonical names in each per-column prompt.
_CUSTOM = "✎ Type a custom name…"
_LEAVE = "\x00leave"  # sentinel value distinct from any canonical name


def _leave_label(name: str) -> str:
    return f"↩ Leave as it is (keep {name!r})"


# ---- Field tiers (color + ranking) -----------------------------------------
#
# Each canonical falls in one tier, which drives both its dropdown color and how
# high it sorts. Required fields rank first and show green; optional and curated
# referenced fields show yellow; only the comprehensive ``--full`` tier shows
# orange (and always sorts last).
_TIER_REQUIRED = "required"
_TIER_OPTIONAL = "optional"
_TIER_REFERENCED = "referenced"
_TIER_FULL = "full"

_TIER_WEIGHT = {
    _TIER_REQUIRED: 0,
    _TIER_OPTIONAL: 1,
    _TIER_REFERENCED: 2,
    _TIER_FULL: 3,
}

# prompt_toolkit inline styles for questionary Choice titles.
_TIER_STYLE = {
    _TIER_REQUIRED: "fg:#3dd13d",  # green
    _TIER_OPTIONAL: "fg:#d1c93d",  # yellow
    _TIER_REFERENCED: "fg:#d1c93d",  # yellow (non-required, offered by default)
    _TIER_FULL: "fg:#ff9933",  # orange (--full comprehensive only)
}

# click colors for the printed legend (orange has no click name -> rgb tuple).
_GREEN = "green"
_YELLOW = "yellow"
_ORANGE = (255, 153, 51)


def field_tier(canonical: str, data_type: DataTypeEnum) -> str:
    """Which tier `canonical` belongs to for `data_type`.

    ``Filename`` is the implicit match column and counts as required. Curated
    referenced fields are the ``referenced`` tier; the comprehensive fields only
    offered under ``--full`` are the ``full`` tier. Anything else is treated as
    referenced.
    """
    if canonical == _FILENAME:
        return _TIER_REQUIRED
    for tier, groups in (
        (_TIER_REQUIRED, REQUIRED_SIDECAR_FIELD_GROUPS),
        (_TIER_OPTIONAL, OPTIONAL_SIDECAR_FIELD_GROUPS),
        (_TIER_FULL, REFERENCED_FULL_SIDECAR_FIELD_GROUPS),
    ):
        if any(g and g[0] == canonical for g in groups.get(data_type, [])):
            return tier
    return _TIER_REFERENCED


# ---- Pure helpers (unit-tested) --------------------------------------------


def canonical_candidates(
    data_type: DataTypeEnum, *, full: bool = False
) -> dict[str, list[str]]:
    """Map each canonical name to all the names that should resolve to it.

    The first entry of every field group is the canonical name written into the
    final sidecar; the remaining entries are accepted aliases. ``Filename`` has
    no group, so it maps to just itself. When `full` is set, the comprehensive
    referenced tier (calibration internals, etc.) is included as well.
    """
    groups = (ALL_SIDECAR_FIELD_GROUPS_FULL if full else ALL_SIDECAR_FIELD_GROUPS).get(
        data_type, []
    )
    candidates: dict[str, list[str]] = {_FILENAME: [_FILENAME]}
    for group in groups:
        if not group:
            continue
        candidates[group[0]] = list(group)
    return candidates


def canonical_names_for(data_type: DataTypeEnum, *, full: bool = False) -> list[str]:
    """Ordered canonical names offered for `data_type` (``Filename`` first)."""
    return list(canonical_candidates(data_type, full=full).keys())


def canonical_label(canonical: str, data_type: DataTypeEnum, *, full: bool = False) -> str:
    """Dropdown label for a canonical name, listing its aliases in parentheses.

    e.g. ``"Heading (Yaw, GimbalYawDegree, …)"``. Names with no aliases (e.g.
    ``GPSLatitude``, ``Filename``) render as just the canonical name.
    """
    aliases = canonical_candidates(data_type, full=full).get(canonical, [canonical])[1:]
    if aliases:
        return f"{canonical} ({', '.join(aliases)})"
    return canonical


def _normalize(name: str) -> str:
    """Lowercase and strip non-alphanumerics for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Shortest shared run (in normalized chars) that still counts as a fuzzy match.
# Three keeps "Lat"->"GPSLatitude" and "Alt"->"GPSAltitude" while rejecting
# incidental one/two-character overlaps.
_MIN_OVERLAP = 3


def _longest_common_substring(a: str, b: str) -> int:
    """Length of the longest run of characters common to `a` and `b`."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                best = max(best, curr[j])
        prev = curr
    return best


# A partial overlap only counts as a *confident* guess (one worth floating to
# the top, above the required fields) when the shared run covers most of the
# shorter name. This keeps real matches ("Lat"->"GPSLatitude", "Cam Make"->
# "Make") while demoting incidental short overlaps ("Cam Make"'s "cam" against
# "CameraPitch"/"CameraSource") down into their normal tier position.
_GUESS_COVERAGE = 0.6


def _match_strength(header: str, names: list[str]) -> tuple[int, bool]:
    """``(score, is_guess)`` for `header` against any name in `names`.

    `score` (higher is better) ranks candidates: an exact normalized match beats
    a partial overlap beats no match (0); it grows with the longest shared run so
    the most specific candidate wins. `is_guess` is True only for confident
    matches — an exact/alias hit or a partial overlap covering most of the
    shorter name — and is what promotes a candidate into the guesses section.
    """
    h = _normalize(header)
    if not h:
        return 0, False
    best = 0
    is_guess = False
    for name in names:
        n = _normalize(name)
        if not n:
            continue
        if n == h:
            return 10_000, True
        overlap = _longest_common_substring(h, n)
        if overlap >= _MIN_OVERLAP:
            best = max(best, 100 + overlap)
            if overlap >= _GUESS_COVERAGE * min(len(h), len(n)):
                is_guess = True
    return best, is_guess


def rank_canonicals(header: str, data_type: DataTypeEnum, *, full: bool = False) -> list[str]:
    """Canonical names ordered for `header`: guesses, then required, then optional.

    Confident matches/guesses lead (best score first), followed by the remaining
    names grouped by tier — required, then optional, then referenced, then the
    comprehensive ``--full`` fields. Within a group the natural (field-group)
    order is preserved (the sort is stable).
    """
    candidates = canonical_candidates(data_type, full=full)
    ordered: list[tuple[int, int, str]] = []
    for canonical, names in candidates.items():
        score, is_guess = _match_strength(header, names)
        if is_guess:
            ordered.append((0, -score, canonical))  # guesses section
        else:
            ordered.append((1 + _TIER_WEIGHT[field_tier(canonical, data_type)], 0, canonical))
    ordered.sort(key=lambda t: (t[0], t[1]))
    return [canonical for _, _, canonical in ordered]


def exact_or_alias_match(
    header: str, data_type: DataTypeEnum, *, full: bool = False
) -> str | None:
    """Return the canonical name `header` exactly matches (itself or an alias)."""
    h = _normalize(header)
    if not h:
        return None
    for canonical, names in canonical_candidates(data_type, full=full).items():
        if any(_normalize(name) == h for name in names):
            return canonical
    return None


def column_stats(values: list[str]) -> dict[str, object]:
    """Summary statistics for one column's string values."""
    non_blank = [v.strip() for v in values if has_value(v)]
    unique = list(dict.fromkeys(non_blank))  # insertion-ordered dedup
    return {
        "total": len(values),
        "non_blank": len(non_blank),
        "blank": len(values) - len(non_blank),
        "unique": len(unique),
        "samples": unique[:_SAMPLE_VALUES],
    }


def build_schema(decisions: list[tuple[str, str]], *, has_header: bool) -> dict:
    """Build the schema dict from ``(original_name, target_name)`` decisions.

    Headered CSVs produce a sparse ``column_name_mapping`` (only columns whose
    target differs from the original); headerless CSVs produce a positional
    ``column_names`` list covering every column.
    """
    if has_header:
        mapping = {orig: target for orig, target in decisions if target != orig}
        return {"column_name_mapping": mapping}
    return {"column_names": [target for _, target in decisions]}


# ---- Prompts ---------------------------------------------------------------


def _ask_csv_path() -> str:
    return (
        questionary.text(
            "Client sidecar CSV to build a schema for:",
            instruction="(Local path or object-store URI like s3://bucket/prefix/file.csv)",
            validate=lambda v: bool(v.strip()) or "Required.",
        )
        .unsafe_ask()
        .strip()
    )


def _ask_data_type() -> DataTypeEnum:
    choice = questionary.select(
        "Data type of the input data (scopes the canonical column names):",
        choices=[v.value for v in DataTypeEnum if v in REQUIRED_SIDECAR_FIELD_GROUPS],
    ).unsafe_ask()
    return DataTypeEnum(choice)


def _ask_has_header() -> bool:
    return questionary.confirm(
        "Does this CSV have a header row?",
        default=True,
    ).unsafe_ask()


_SAVE_SAME = "Same folder as the sidecar CSV"
_SAVE_LOCAL = "A local path"
_SAVE_BOTH = "Both"


def _ask_save_target() -> str:
    return questionary.select(
        "Where should the schema be saved?",
        choices=[_SAVE_SAME, _SAVE_LOCAL, _SAVE_BOTH],
        default=_SAVE_SAME,
    ).unsafe_ask()


def _default_schema_filename(csv_path: str) -> str:
    """Suggested schema filename derived from the CSV name (``<stem>_schema.json``)."""
    return f"{uri_stem(csv_path) or 'schema'}_schema.json"


def _ask_filename(csv_path: str) -> str:
    default = _default_schema_filename(csv_path)
    return (
        questionary.text(
            "Schema filename:",
            default=default,
            instruction="(Press Enter to keep the default)",
        )
        .unsafe_ask()
        .strip()
        or default
    )


def _ask_target_for_column(
    original: str,
    ranked: list[str],
    data_type: DataTypeEnum,
    default: str | None,
    *,
    full: bool = False,
) -> str:
    """Prompt for the canonical name a column maps to; return the target name.

    Canonical names are shown with their aliases in parentheses and colored by
    tier (required green, optional yellow, referenced orange). Each choice's
    value is the bare canonical name, so the selection needs no resolving.
    """
    choices = [
        questionary.Choice(
            title=[
                (_TIER_STYLE[field_tier(c, data_type)], canonical_label(c, data_type, full=full))
            ],
            value=c,
        )
        for c in ranked
    ]
    choices.append(questionary.Choice(title=_CUSTOM, value=_CUSTOM))
    choices.append(questionary.Choice(title=_leave_label(original), value=_LEAVE))
    selection = questionary.select(
        f"Map column {original!r} to:",
        choices=choices,
        default=default if default in ranked else _LEAVE,
    ).unsafe_ask()
    if selection == _CUSTOM:
        return (
            questionary.text(
                "Custom column name:",
                validate=lambda v: bool(v.strip()) or "Required.",
            )
            .unsafe_ask()
            .strip()
        )
    if selection == _LEAVE:
        return original
    return selection


# ---- Rendering -------------------------------------------------------------


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an aligned text table, truncated to the preview window."""
    headers = headers[:_PREVIEW_COLS]
    rows = [row[:_PREVIEW_COLS] for row in rows[:_PREVIEW_ROWS]]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    out = [sep.join(click.style(h.ljust(widths[i]), bold=True) for i, h in enumerate(headers))]
    for row in rows:
        out.append(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def _print_tier_legend(*, full: bool) -> None:
    """Print a one-line key for the colored canonical-name choices."""
    parts = [
        ("● required", _GREEN),
        ("● optional / referenced", _YELLOW),
    ]
    if full:
        parts.append(("● comprehensive (--full)", _ORANGE))
    click.secho("\nColumn-name colors:", fg="cyan", bold=True, err=True)
    legend = "   ".join(click.style(text, fg=color) for text, color in parts)
    click.echo("  " + legend, err=True)


def _print_column_summary(original: str, stats: dict[str, object]) -> None:
    click.secho(f"\nColumn: {original!r}", fg="cyan", bold=True, err=True)
    click.secho(
        f"  {stats['non_blank']}/{stats['total']} non-blank, "
        f"{stats['blank']} blank, {stats['unique']} unique.",
        fg="bright_black",
        err=True,
    )
    samples = stats["samples"]
    if samples:
        rendered = ", ".join(repr(s) for s in samples)  # type: ignore[union-attr]
        click.secho(f"  samples: {rendered}", fg="bright_black", err=True)


# ---- Save ------------------------------------------------------------------


def _parent_location(csv_path: str) -> str:
    """The directory containing `csv_path` (remote URI or local path)."""
    if is_remote_uri(csv_path):
        return csv_path.rstrip("/").rsplit("/", 1)[0]
    return str(Path(csv_path).expanduser().resolve().parent)


def _save_schema(schema: dict, csv_path: str, target: str, filename: str) -> list[str]:
    """Write the schema to the requested destination(s); return written paths."""
    serialized = json.dumps(schema, indent=2) + "\n"
    written: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / filename
        source.write_text(serialized, encoding="utf-8")

        if target in (_SAVE_SAME, _SAVE_BOTH):
            backend = from_directory(_parent_location(csv_path))
            written.append(backend.write_output(filename, source))

        if target in (_SAVE_LOCAL, _SAVE_BOTH):
            default_local = str(Path.cwd() / filename)
            local_path = (
                questionary.text(
                    "Local path to save the schema:",
                    default=default_local,
                    instruction="(Press Enter to keep the default)",
                )
                .unsafe_ask()
                .strip()
                or default_local
            )
            dest = Path(local_path).expanduser()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(serialized, encoding="utf-8")
            written.append(str(dest))
    return written


# ---- Command ---------------------------------------------------------------


def build_schema_command(
    ctx: typer.Context,
    csv_path: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Client sidecar CSV. Either a local path or an object-store URI "
                "(s3://bucket/prefix/file.csv). Prompted for if omitted."
            ),
        ),
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            "-f",
            help=(
                "Also offer comprehensive/advanced referenced EXIF fields "
                "(e.g. calibration internals, quaternion orientation)."
            ),
        ),
    ] = False,
) -> None:
    """Interactively build a client schema for a sidecar CSV."""
    if csv_path is None:
        csv_path = _ask_csv_path()

    data_type = _ask_data_type()

    click.secho(f"\nReading {csv_path!r}…", fg="cyan", err=True)
    df = load_sidecar_df(csv_path)
    width = df.shape[1]
    if width == 0:
        raise typer.BadParameter(f"CSV at {csv_path!r} has no columns.")

    # Show the first rows as they sit on disk so the user can judge whether the
    # top row is a header. `load_sidecar_df` consumed row 0 as the header, so
    # reconstruct the raw view by putting the column names back as the first row.
    # Positional labels (col 1, col 2, …) head the table so the raw first row is
    # shown exactly once, as data.
    raw_first_row = [str(c) for c in df.columns]
    data_rows = [[str(v) for v in row] for row in df.head(_PREVIEW_ROWS - 1).values.tolist()]
    position_headers = [f"col {i + 1}" for i in range(width)]
    click.secho("\nPreview (first rows × columns):", fg="cyan", bold=True, err=True)
    typer.echo(_render_table(position_headers, [raw_first_row, *data_rows]))
    if width > _PREVIEW_COLS:
        click.secho(f"  …and {width - _PREVIEW_COLS} more column(s).", fg="bright_black", err=True)

    has_header = _ask_has_header()
    if not has_header:
        placeholder_names = [f"column_{i + 1}" for i in range(width)]
        df = load_sidecar_df(csv_path, column_names=placeholder_names)

    _print_tier_legend(full=full)

    decisions: list[tuple[str, str]] = []
    for col in df.columns:
        original = str(col)
        stats = column_stats([str(v) for v in df[col].tolist()])
        _print_column_summary(original, stats)

        matched = exact_or_alias_match(original, data_type, full=full) if has_header else None
        if matched is not None:
            click.secho(
                f"  matches canonical name {matched!r}.", fg="green", err=True
            )
        ranked = (
            rank_canonicals(original, data_type, full=full)
            if has_header
            else canonical_names_for(data_type, full=full)
        )
        target = _ask_target_for_column(
            original, ranked, data_type, default=matched, full=full
        )
        decisions.append((original, target))

    schema = build_schema(decisions, has_header=has_header)
    if has_header and not schema["column_name_mapping"]:
        click.secho(
            "\nNo columns were renamed — the schema will contain an empty mapping.",
            fg="yellow",
            err=True,
        )

    click.secho("\nGenerated schema:", fg="cyan", bold=True, err=True)
    typer.echo(json.dumps(schema, indent=2))

    save_target = _ask_save_target()
    filename = _ask_filename(csv_path)
    written = _save_schema(schema, csv_path, save_target, filename)

    for path in written:
        click.secho(f"\nSchema written to: {path}", fg="green", bold=True, err=True)
        report = lint_schema_file(path)
        typer.echo(report.render())

    # Offer to keep going: the schema we just wrote, the client sidecar it was
    # built from, and the data type are exactly the inputs `sidecar generate`
    # needs, so carry them over and ask only the remaining wizard questions.
    # Prefer a remote schema path for the follow-on so the printed replay
    # command matches what you'd run against object storage.
    schema_for_sidecar = next(
        (p for p in written if is_remote_uri(p)),
        written[0] if written else None,
    )
    if schema_for_sidecar is not None:
        from atomic_tools.cli import Verbosity
        from atomic_tools.commands.sidecar import offer_sidecar_after_schema

        verbosity: Verbosity = ctx.ensure_object(Verbosity)
        offer_sidecar_after_schema(
            data_type=data_type,
            client_sidecar=csv_path,
            client_schema=schema_for_sidecar,
            verbosity=verbosity.choice,
            verbosity_provided=verbosity.verbose or verbosity.silent,
        )
