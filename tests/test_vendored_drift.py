"""Drift check: the vendored data-engineering files must match the canonical source.

Clients have **no access** to the private ``data-engineering`` repo. When the
canonical source is unreachable (no sibling checkout, no env override, no token)
this test emits a warning and *skips* — it must never fail for that reason. Only
genuine content drift between the vendored copy and the canonical source fails.

Comparison is AST-based (``ast.dump``), so it is insensitive to whitespace and
line numbers but sensitive to docstrings and actual code — exactly the drift we
care about.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

from atomic_tools import vendor_sync

_SKIP_MESSAGE = (
    "vendored-drift check skipped: canonical data-engineering source unreachable "
    "(expected for client installs)"
)
_FIX_HINT = (
    "run `am-tools update` on a machine with data-engineering access, then commit "
    "the refreshed vendored files"
)

_VENDORED_DIR = Path(vendor_sync.__file__).resolve().parent / "vendored"


def _require_source(path: str) -> str:
    """Fetch a canonical source or skip (never fail) when it's unreachable."""
    source = vendor_sync.fetch_canonical_source(path)
    if source is None:
        warnings.warn(_SKIP_MESSAGE, stacklevel=2)
        pytest.skip(_SKIP_MESSAGE)
    return source


def test_data_type_registry_matches_canonical():
    upstream = _require_source(vendor_sync.UTILS_PATH)
    vendored_source = (_VENDORED_DIR / "data_type_registry.py").read_text(encoding="utf-8")

    upstream_nodes = vendor_sync.extract_named_nodes(upstream, vendor_sync.VENDORED_NAMES)
    vendored_nodes = vendor_sync.extract_named_nodes(vendored_source, vendor_sync.VENDORED_NAMES)

    for name in vendor_sync.VENDORED_NAMES:
        assert name in upstream_nodes, (
            f"vendored symbol {name!r} no longer exists in the canonical utils.py "
            f"(upstream rename is drift); {_FIX_HINT}"
        )
        assert name in vendored_nodes, (
            f"vendored symbol {name!r} is missing from data_type_registry.py; {_FIX_HINT}"
        )
        assert ast.dump(vendored_nodes[name]) == ast.dump(upstream_nodes[name]), (
            f"vendored {name!r} has drifted from the canonical data-engineering "
            f"source; {_FIX_HINT}"
        )


def test_field_names_matches_canonical():
    upstream = _require_source(vendor_sync.FIELD_NAMES_PATH)
    vendored_source = (_VENDORED_DIR / "field_names.py").read_text(encoding="utf-8")

    assert ast.dump(ast.parse(vendored_source)) == ast.dump(ast.parse(upstream)), (
        f"vendored field_names.py has drifted from the canonical source; {_FIX_HINT}"
    )


def test_skip_path_works_offline(monkeypatch):
    """When the source is unreachable, _require_source raises pytest.skip."""
    monkeypatch.setattr(vendor_sync, "fetch_canonical_source", lambda path: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(pytest.skip.Exception):
            _require_source(vendor_sync.UTILS_PATH)
