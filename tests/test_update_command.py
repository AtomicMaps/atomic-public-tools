import logging
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atomic_tools import vendor_sync
from atomic_tools.cli import app
from atomic_tools.commands import update

runner = CliRunner()


class _Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "find_repo_root", lambda: tmp_path)
    # Default: vendored refresh is unreachable (the client case) so it never
    # touches disk or the network during these tests. Individual tests override.
    monkeypatch.setattr(
        update.vendor_sync,
        "refresh",
        lambda repo: vendor_sync.RefreshResult(available=False, changed={}),
    )
    return tmp_path


def test_find_repo_root_detects_this_checkout():
    # The test suite runs from the editable checkout, so it should be found and
    # contain the expected marker files.
    root = update.find_repo_root()
    assert root is not None
    assert (root / ".git").exists()
    assert (root / "pyproject.toml").exists()


def test_update_checks_out_pulls_then_installs(fake_repo, monkeypatch):
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, cwd):
        calls.append((cmd, cwd))
        return _Result(0)

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert [c[0] for c in calls] == [
        ["git", "checkout", "main"],
        ["git", "pull", "origin", "main"],
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
    ]
    assert all(c[1] == fake_repo for c in calls)


def test_update_branch_option_targets_that_branch(fake_repo, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        update.subprocess, "run", lambda cmd, cwd: calls.append(cmd) or _Result(0)
    )

    result = runner.invoke(app, ["update", "--branch", "staging"])

    assert result.exit_code == 0
    assert calls[0] == ["git", "checkout", "staging"]
    assert calls[1] == ["git", "pull", "origin", "staging"]


def test_update_no_dev_installs_without_extras(fake_repo, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        update.subprocess, "run", lambda cmd, cwd: calls.append(cmd) or _Result(0)
    )

    result = runner.invoke(app, ["update", "--no-dev"])

    assert result.exit_code == 0
    assert calls[-1] == [sys.executable, "-m", "pip", "install", "-e", "."]


def test_update_without_checkout_explains(monkeypatch):
    monkeypatch.setattr(update, "find_repo_root", lambda: None)
    ran = []
    monkeypatch.setattr(
        update.subprocess, "run", lambda *a, **k: ran.append(1) or _Result(0)
    )

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert not ran  # never shells out when there's no clone
    assert "nothing to pull" in result.output


def test_update_stops_when_git_fails(fake_repo, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, cwd):
        calls.append(cmd)
        # Fail the pull; the preceding checkout succeeds.
        return _Result(1 if cmd[:2] == ["git", "pull"] else 0)

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    # checkout + pull ran; pip install never runs after the failed pull.
    assert calls == [["git", "checkout", "main"], ["git", "pull", "origin", "main"]]


def test_update_reports_missing_git(fake_repo, monkeypatch):
    def fake_run(cmd, cwd):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "PATH" in result.output


def test_update_refresh_unreachable_is_silent_for_clients(fake_repo, monkeypatch, caplog):
    """A client has no access to data-engineering, so the skip is a non-event:
    the update succeeds and says nothing about it at default verbosity."""
    monkeypatch.setattr(
        update.subprocess, "run", lambda cmd, cwd: _Result(0)
    )

    with caplog.at_level(logging.INFO, logger="atomic_tools.commands.update"):
        result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "vendored" not in result.output.lower()
    # …but the reason is there under --verbose.
    assert any("not reachable" in rec.getMessage() for rec in caplog.records)


def test_update_refresh_error_is_silent_for_clients(fake_repo, monkeypatch, caplog):
    """Same for a regeneration failure — a client can't act on it, and drift on
    a dev machine is caught by tests/test_vendored_drift.py, not by this line."""
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, cwd: _Result(0))

    def boom(repo):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(update.vendor_sync, "refresh", boom)

    with caplog.at_level(logging.INFO, logger="atomic_tools.commands.update"):
        result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "vendored" not in result.output.lower()
    assert any("kaboom" in rec.getMessage() for rec in caplog.records)


def test_update_refresh_up_to_date_is_silent(fake_repo, monkeypatch):
    """Nothing changed is also a non-event — no console line."""
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, cwd: _Result(0))
    monkeypatch.setattr(
        update.vendor_sync,
        "refresh",
        lambda repo: vendor_sync.RefreshResult(
            available=True, changed={"data_type_registry.py": False}
        ),
    )

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "vendored" not in result.output.lower()


def test_update_refresh_changed_prompts_commit(fake_repo, monkeypatch):
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, cwd: _Result(0))
    monkeypatch.setattr(
        update.vendor_sync,
        "refresh",
        lambda repo: vendor_sync.RefreshResult(
            available=True, changed={"data_type_registry.py": True, "field_registry.json": False}
        ),
    )

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "Refreshed vendored data-engineering files" in result.output
    assert "data_type_registry.py" in result.output


def test_update_listed_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "update" in result.stdout
