"""``am-tools update`` — pull the latest code and reinstall.

The tool is installed as an editable checkout (``pip install -e ".[dev]"``), so
"updating" means ``git pull`` in that clone followed by a reinstall to pick up
any dependency changes. This command finds the clone from the installed package
location and runs both steps, streaming their output.

If the install isn't an editable git checkout (e.g. installed from a wheel),
there's nothing to ``git pull``, so the command explains that instead.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

import atomic_tools
from atomic_tools import vendor_sync

logger = logging.getLogger(__name__)

_REPO_URL = "https://github.com/AtomicMaps/atomic-public-tools"


def find_repo_root() -> Path | None:
    """Locate the editable git checkout this package is installed from.

    Walks up from the installed package file looking for a directory that holds
    both a ``.git`` and a ``pyproject.toml`` — i.e. the repo clone. Returns
    ``None`` when the package isn't running from such a checkout (a wheel/site
    install), where there's nothing to pull.
    """
    start = Path(atomic_tools.__file__).resolve()
    for parent in start.parents:
        if (parent / ".git").exists() and (parent / "pyproject.toml").exists():
            return parent
    return None


def _run(cmd: list[str], cwd: Path) -> None:
    """Run `cmd` in `cwd`, streaming output. Raise ``typer.Exit`` on failure."""
    printable = " ".join(cmd)
    typer.secho(f"\n$ {printable}", fg=typer.colors.CYAN, bold=True, err=True)
    try:
        result = subprocess.run(cmd, cwd=cwd)  # noqa: S603 — args are fixed, not user input
    except FileNotFoundError as exc:
        typer.secho(
            f"Could not run {cmd[0]!r}: {exc}. Is it installed and on your PATH?",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from exc
    if result.returncode != 0:
        typer.secho(
            f"Command failed (exit {result.returncode}): {printable}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(result.returncode)


def _refresh_vendored(repo: Path) -> None:
    """Best-effort refresh of the vendored data-engineering files.

    On a machine that can reach the canonical source (sibling checkout or a
    GitHub token) this regenerates the vendored files to the latest canonical
    version. It is purely advisory: any failure or unreachable source is
    reported in a single line and never aborts the update (clients have no
    access to the private repo).
    """
    try:
        result = vendor_sync.refresh(repo)
    except Exception as exc:  # noqa: BLE001 — never fail the update over this
        logger.debug("Vendored refresh errored: %s", exc)
        typer.secho(
            "Skipping vendored refresh — regeneration failed (harmless).",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return

    if not result.available:
        typer.secho(
            "Skipping vendored refresh — data-engineering source not reachable.",
            fg=typer.colors.BLUE,
            err=True,
        )
        return

    if result.any_changed:
        changed = ", ".join(name for name, did in result.changed.items() if did)
        typer.secho(
            f"Refreshed vendored data-engineering files ({changed}); "
            "review and commit the change.",
            fg=typer.colors.GREEN,
            err=True,
        )
    else:
        typer.secho(
            "Vendored data-engineering files already up to date.",
            fg=typer.colors.BLUE,
            err=True,
        )


_DEFAULT_BRANCH = "main"


def update_command(
    dev: Annotated[
        bool,
        typer.Option(
            "--dev/--no-dev",
            help="Reinstall with dev extras (pytest, ruff, mypy). On by default.",
        ),
    ] = True,
    branch: Annotated[
        str,
        typer.Option(
            "--branch",
            "-b",
            help="Branch to update to.",
        ),
    ] = _DEFAULT_BRANCH,
) -> None:
    """Update am-tools: switch to the latest branch, pull it, then reinstall."""
    repo = find_repo_root()
    if repo is None:
        typer.secho(
            "am-tools isn't running from an editable git checkout, so there's "
            "nothing to pull.\n"
            f"To update, clone or pull {_REPO_URL} and reinstall from there:\n"
            '    git pull && pip install -e ".[dev]"',
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(1)

    typer.secho(
        f"Updating am-tools in {repo} (branch {branch})",
        fg=typer.colors.CYAN,
        bold=True,
        err=True,
    )

    # Switch to the target branch and pull it explicitly. Naming the remote and
    # branch avoids "no tracking information" failures on clones whose current
    # branch has no upstream, and keeps every client on the same branch.
    _run(["git", "checkout", branch], cwd=repo)
    _run(["git", "pull", "origin", branch], cwd=repo)

    _refresh_vendored(repo)

    target = ".[dev]" if dev else "."
    _run([sys.executable, "-m", "pip", "install", "-e", target], cwd=repo)

    typer.secho(
        "\nam-tools is up to date. Run `am-tools --version` to confirm.",
        fg=typer.colors.GREEN,
        bold=True,
        err=True,
    )
