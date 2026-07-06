"""Daily "are you up to date?" check against the GitHub repo.

The first time ``am-tools`` runs on any given calendar day, this fetches the
version declared in ``pyproject.toml`` on the repo's default branch and compares
it to the locally installed ``__version__``. If the local copy is behind, it
prints a one-line upgrade nudge; otherwise it stays silent.

Everything here is best-effort and fail-safe: no network error, parse failure,
or unwritable cache can break (or even slow down noticeably) a normal run. The
check runs at most once per day thanks to a small stamp file, and can be turned
off entirely with ``AM_TOOLS_NO_VERSION_CHECK=1``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import sys
import urllib.request
from pathlib import Path

from atomic_tools import __version__

logger = logging.getLogger(__name__)

# Raw pyproject.toml on the default branch — the canonical source of the
# repo's current version.
_REPO_SLUG = "AtomicMaps/atomic-public-tools"
_REMOTE_PYPROJECT_URL = (
    f"https://raw.githubusercontent.com/{_REPO_SLUG}/main/pyproject.toml"
)
_REPO_URL = f"https://github.com/{_REPO_SLUG}"

_DISABLE_ENV = "AM_TOOLS_NO_VERSION_CHECK"
_FETCH_TIMEOUT_SECONDS = 3.0

_VERSION_RE = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def _cache_path() -> Path:
    """Location of the once-per-day stamp file (honours ``XDG_CACHE_HOME``)."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "atomic-tools" / "version_check.json"


def _today() -> str:
    return _dt.date.today().isoformat()


def _parse_version(text: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple of ints.

    Uses ``packaging`` when available for full PEP 440 semantics, falling back
    to a lenient dotted-integer parse (``"1.2.3" -> (1, 2, 3)``). Non-numeric
    or trailing components (e.g. ``rc1``) are dropped in the fallback.
    """
    try:
        from packaging.version import Version

        v = Version(text)
        return v.release
    except Exception:
        parts: list[int] = []
        for chunk in text.strip().split("."):
            m = re.match(r"\d+", chunk)
            if not m:
                break
            parts.append(int(m.group()))
        return tuple(parts)


def _is_outdated(local: str, remote: str) -> bool:
    """True when ``local`` is strictly behind ``remote``."""
    lv, rv = _parse_version(local), _parse_version(remote)
    if not lv or not rv:
        return False
    return lv < rv


def _read_stamp(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_stamp(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception as exc:  # unwritable cache must never break a run
        logger.debug("Could not write version-check stamp: %s", exc)


def _fetch_remote_version() -> str | None:
    """Fetch and parse the version from the repo's ``pyproject.toml``."""
    req = urllib.request.Request(
        _REMOTE_PYPROJECT_URL,
        headers={"User-Agent": f"am-tools/{__version__}"},
    )
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", errors="replace")
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def _emit_outdated_warning(local: str, remote: str) -> None:
    """Print an upgrade nudge. Uses typer styling when available."""
    message = (
        f"A newer version of am-tools is available: {remote} "
        f"(you have {local}).\n"
        f"Update with:  git -C <your clone> pull && pip install -e \".[dev]\"\n"
        f"Repo: {_REPO_URL}"
    )
    try:
        import typer

        typer.secho(message, fg=typer.colors.YELLOW, err=True)
    except Exception:
        print(message, file=sys.stderr)


def check_for_update(*, force: bool = False) -> None:
    """Run the daily version check. Best-effort; never raises.

    On the first invocation of a given calendar day, fetches the repo's declared
    version and warns if the local install is behind. Subsequent runs the same
    day are no-ops. Pass ``force=True`` to bypass the once-per-day gate (used by
    an explicit ``--check-version`` and by tests).
    """
    try:
        if os.environ.get(_DISABLE_ENV):
            return

        stamp_path = _cache_path()
        today = _today()

        if not force:
            stamp = _read_stamp(stamp_path)
            if stamp.get("last_check") == today:
                return

        remote = _fetch_remote_version()
        if remote is None:
            # Couldn't determine the remote version — don't record the check so
            # the next run retries rather than staying silent for the rest of
            # the day.
            logger.debug("Version check: could not read remote version.")
            return

        _write_stamp(stamp_path, {"last_check": today, "latest_known": remote})

        if _is_outdated(__version__, remote):
            _emit_outdated_warning(__version__, remote)
    except Exception as exc:  # absolutely never break the CLI over this
        logger.debug("Version check skipped due to error: %s", exc)
