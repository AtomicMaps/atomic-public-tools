"""Discovery, regeneration, and refresh of the vendored data-engineering files.

amtools carries verbatim copies of a handful of definitions from the private
``data-engineering`` repo (the data-type registry + classifier as
``data_type_registry.py``, and the canonical source-field registry as
``field_registry.json``) under :mod:`atomic_tools.vendored`. This module is the
single code path that:

* **discovers** the canonical source (env override → sibling checkout → GitHub
  API), returning ``None`` when it is unreachable — the normal case on a client
  machine with no access to the private repo;
* **regenerates** the vendored module contents byte-exactly from that source;
* **refreshes** the on-disk vendored files (used by ``am-tools update`` on dev
  machines).

The same discovery/regeneration logic backs ``tests/test_vendored_drift.py``,
so the drift test compares against exactly what a refresh would write.

No absolute paths are ever hardcoded: the sibling checkout is resolved relative
to this repo's root (see :func:`atomic_tools.commands.update.find_repo_root`),
with an ``AM_TOOLS_DATA_ENGINEERING_REPO`` env override for non-standard layouts.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Canonical source coordinates -------------------------------------------

CANONICAL_REPO_SLUG = "AtomicMaps/data-engineering"
CANONICAL_REF = "main"
UTILS_PATH = "atomicmapspy/atomicmapspy/utils/utils.py"
FIELD_REGISTRY_PATH = "atomicmapspy/atomicmapspy/schemas/field_registry.json"

# Top-level definitions vendored verbatim from the canonical utils.py, in the
# order they should appear in the generated data_type_registry.py.
VENDORED_NAMES: tuple[str, ...] = (
    "DATA_TYPE_INFO",
    "DataTypeEnum",
    "ImageDataTypeEnum",
    "get_valid_subtypes",
    "_INFER_DATA_TYPE_ORDER",
    "_AMBIGUOUS_IMAGE_EXTENSIONS",
    "_SPHERICAL_ASPECT_RATIO_MIN",
    "_SPHERICAL_ASPECT_RATIO_MAX",
    "_xmp_indicates_spherical",
    "_aspect_ratio_indicates_spherical",
    "_user_comment_indicates_spherical",
    "infer_data_type",
)

# Import header prepended to the generated data_type_registry.py. These are the
# only names the vendored definitions reference beyond builtins (verified: no
# atomicmapspy/obstore/numpy usage inside the selected nodes).
_REGISTRY_IMPORTS = (
    "import json\n"
    "import re\n"
    "from enum import Enum\n"
    "from typing import Dict, List, Optional\n"
)

# Env override: an absolute or relative path to a data-engineering checkout.
_REPO_ENV = "AM_TOOLS_DATA_ENGINEERING_REPO"
# Tokens for the (private) GitHub contents API.
_TOKEN_ENVS = ("AM_TOOLS_DRIFT_TOKEN", "GITHUB_TOKEN")

_GIT_TIMEOUT_SECONDS = 10
_HTTP_TIMEOUT_SECONDS = 5.0


def _banner(source_path: str, *, selected: bool) -> str:
    """Return the ``#``-comment banner prepended to a vendored file.

    ``#`` comments are invisible to :func:`ast.parse`, so the banner never
    disturbs the AST-based drift comparison. ``selected`` marks the registry
    file (only some definitions copied) versus a whole-file copy.
    """
    scope = " (selected definitions)" if selected else ""
    return (
        f"# VENDORED from data-engineering@{CANONICAL_REF}\n"
        f"#   {source_path}{scope}\n"
        "# Do not edit by hand. Re-vendor with `am-tools update` (dev machines) —\n"
        "# drift is detected by tests/test_vendored_drift.py.\n"
    )


# --- Discovery --------------------------------------------------------------


def _sibling_repo_root() -> Path | None:
    """Locate a ``data-engineering`` checkout as a sibling of this repo's root.

    Never a hardcoded absolute path: the repo root is found from the installed
    package location, and the sibling is ``root.parent / "data-engineering"``.
    """
    # Imported lazily to avoid a hard import cycle (commands import utils which
    # imports the vendored package) and to keep this module importable in tests
    # that stub the CLI.
    from atomic_tools.commands.update import find_repo_root

    root = find_repo_root()
    if root is None:
        return None
    candidate = root.parent / "data-engineering"
    if candidate.is_dir() and (candidate / ".git").exists():
        return candidate
    return None


def _read_from_checkout(repo: Path, path: str) -> str | None:
    """Read ``path`` from a git checkout at the canonical ref.

    Tries ``git show {ref}:path`` first (the committed ref), then
    ``origin/{ref}``, and finally the plain working-tree file. Returns ``None``
    if every attempt fails.
    """
    for ref in (CANONICAL_REF, f"origin/{CANONICAL_REF}"):
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), "show", f"{ref}:{path}"],
                capture_output=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("git show %s:%s failed: %s", ref, path, exc)
            continue
        if result.returncode == 0 and result.stdout:
            return result.stdout

    working_tree = repo / path
    try:
        if working_tree.is_file():
            return working_tree.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("reading working-tree %s failed: %s", working_tree, exc)
    return None


def _fetch_from_github(path: str) -> str | None:
    """Fetch ``path`` from the private repo via the GitHub contents API.

    Requires a token (the repo is private); returns ``None`` when no token is
    configured or the request fails. Best-effort, fail-safe.
    """
    token = next((os.environ[e] for e in _TOKEN_ENVS if os.environ.get(e)), None)
    if not token:
        return None
    url = (
        f"https://api.github.com/repos/{CANONICAL_REPO_SLUG}/contents/{path}"
        f"?ref={CANONICAL_REF}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.raw",
            "Authorization": f"Bearer {token}",
            "User-Agent": "am-tools-vendor-sync",
        },
    )
    try:
        with urllib.request.urlopen(
            req, timeout=_HTTP_TIMEOUT_SECONDS
        ) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        logger.debug("GitHub fetch of %s failed: %s", path, exc)
        return None


def fetch_canonical_source(path: str) -> str | None:
    """Fetch the canonical source text for ``path``; first hit wins.

    Order: ``AM_TOOLS_DATA_ENGINEERING_REPO`` override → sibling checkout →
    GitHub contents API. Every failure falls through; total failure returns
    ``None`` (expected for client installs with no data-engineering access).
    """
    override = os.environ.get(_REPO_ENV)
    if override:
        repo = Path(override).expanduser()
        text = _read_from_checkout(repo, path)
        if text is not None:
            return text

    sibling = _sibling_repo_root()
    if sibling is not None:
        text = _read_from_checkout(sibling, path)
        if text is not None:
            return text

    return _fetch_from_github(path)


# --- Regeneration -----------------------------------------------------------


def extract_named_nodes(source: str, names: tuple[str, ...] | list[str]) -> dict:
    """Return ``{name: ast_node}`` for the requested top-level definitions.

    Matches ``FunctionDef``/``ClassDef`` by ``.name``, ``Assign`` by
    ``targets[0].id``, and ``AnnAssign`` by ``target.id``. Names not found are
    simply absent from the result.
    """
    wanted = set(names)
    found: dict = {}
    for node in ast.parse(source).body:
        name: str | None = None
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            name = node.name
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in wanted:
            found[name] = node
    return found


def render_data_type_registry(upstream_utils_source: str) -> str:
    """Render the vendored ``data_type_registry.py`` from canonical utils.py.

    Extracts the vendored nodes and emits their source segments in
    :data:`VENDORED_NAMES` order under the fixed banner + import header. Produces
    byte-exact expected content for the on-disk vendored module.

    Raises ``ValueError`` if a vendored name is missing upstream (an upstream
    rename must surface as an error, not silently drop a definition).
    """
    nodes = extract_named_nodes(upstream_utils_source, VENDORED_NAMES)
    missing = [n for n in VENDORED_NAMES if n not in nodes]
    if missing:
        raise ValueError(
            f"Canonical utils.py is missing vendored definitions: {missing}"
        )
    segments = [
        ast.get_source_segment(upstream_utils_source, nodes[n]) for n in VENDORED_NAMES
    ]
    body = "\n\n\n".join(seg for seg in segments if seg)
    banner = _banner(UTILS_PATH, selected=True)
    return f"{banner}\n{_REGISTRY_IMPORTS}\n\n{body}\n"


def render_field_registry(upstream_source: str) -> str:
    """Return the vendored ``field_registry.json`` content: the upstream JSON verbatim.

    JSON can't carry a ``#`` banner, so unlike the Python copies there is no
    header to prepend — the file is copied byte-for-byte (its own ``$comment``
    already says "External repos: copy this file verbatim"). Drift is caught by
    a structural (parsed-JSON) comparison in ``test_vendored_drift.py``.
    """
    return upstream_source


# --- Refresh (used by ``am-tools update``) ----------------------------------


@dataclass
class RefreshResult:
    """Outcome of a :func:`refresh` run.

    ``available`` is ``False`` when the canonical source could not be reached
    (the normal client case). ``changed`` maps each vendored filename to whether
    its content was rewritten.
    """

    available: bool
    changed: dict[str, bool]

    @property
    def any_changed(self) -> bool:
        return any(self.changed.values())


def _vendored_dir(repo_root: Path) -> Path:
    return repo_root / "src" / "atomic_tools" / "vendored"


def _write_if_changed(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` only when it differs. Returns True if written."""
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        existing = None
    if existing == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def refresh(repo_root: Path) -> RefreshResult:
    """Regenerate the vendored files from the canonical source, best-effort.

    Fetches both canonical sources; if either is unreachable, returns
    ``RefreshResult(available=False, ...)`` without touching disk. Otherwise
    renders and writes both vendored files, reporting which changed.
    """
    # Short-circuit: if the first fetch can't reach the source, neither can the
    # second. Worth skipping — a client with an unrelated GITHUB_TOKEN set does
    # reach the network here, and this halves the 404 round-trip they wait on.
    utils_source = fetch_canonical_source(UTILS_PATH)
    if utils_source is None:
        return RefreshResult(available=False, changed={})
    field_registry_source = fetch_canonical_source(FIELD_REGISTRY_PATH)
    if field_registry_source is None:
        return RefreshResult(available=False, changed={})

    vendored = _vendored_dir(repo_root)
    changed = {
        "data_type_registry.py": _write_if_changed(
            vendored / "data_type_registry.py",
            render_data_type_registry(utils_source),
        ),
        "field_registry.json": _write_if_changed(
            vendored / "field_registry.json",
            render_field_registry(field_registry_source),
        ),
    }
    return RefreshResult(available=True, changed=changed)
