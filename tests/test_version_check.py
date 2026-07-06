import json

import pytest

from atomic_tools import version_check


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    """Point the stamp file at a temp dir and enable the check."""
    monkeypatch.delenv("AM_TOOLS_NO_VERSION_CHECK", raising=False)
    path = tmp_path / "cache" / "atomic-tools" / "version_check.json"
    monkeypatch.setattr(version_check, "_cache_path", lambda: path)
    return path


def test_parse_version_basic():
    assert version_check._parse_version("1.2.3") == (1, 2, 3)
    assert version_check._parse_version("0.1.0") == (0, 1, 0)


def test_is_outdated():
    assert version_check._is_outdated("0.1.0", "0.2.0") is True
    assert version_check._is_outdated("0.2.0", "0.1.0") is False
    assert version_check._is_outdated("1.0.0", "1.0.0") is False
    # Unparseable versions are treated as "can't tell" -> not outdated.
    assert version_check._is_outdated("", "1.0.0") is False


def test_warns_when_outdated(cache_file, monkeypatch, capsys):
    monkeypatch.setattr(version_check, "__version__", "0.1.0")
    monkeypatch.setattr(version_check, "_fetch_remote_version", lambda: "0.9.0")

    version_check.check_for_update()

    err = capsys.readouterr().err
    assert "newer version" in err
    assert "0.9.0" in err
    # Stamp written so the next run today is a no-op.
    assert json.loads(cache_file.read_text())["last_check"] == version_check._today()


def test_silent_when_up_to_date(cache_file, monkeypatch, capsys):
    monkeypatch.setattr(version_check, "__version__", "1.0.0")
    monkeypatch.setattr(version_check, "_fetch_remote_version", lambda: "1.0.0")

    version_check.check_for_update()

    assert capsys.readouterr().err == ""
    assert cache_file.exists()


def test_skips_second_run_same_day(cache_file, monkeypatch):
    calls = []

    def fake_fetch():
        calls.append(1)
        return "0.9.0"

    monkeypatch.setattr(version_check, "__version__", "0.1.0")
    monkeypatch.setattr(version_check, "_fetch_remote_version", fake_fetch)

    version_check.check_for_update()
    version_check.check_for_update()

    assert len(calls) == 1  # second run short-circuited on the stamp


def test_force_bypasses_daily_gate(cache_file, monkeypatch):
    calls = []
    monkeypatch.setattr(version_check, "__version__", "0.1.0")
    monkeypatch.setattr(
        version_check, "_fetch_remote_version", lambda: calls.append(1) or "0.9.0"
    )

    version_check.check_for_update()
    version_check.check_for_update(force=True)

    assert len(calls) == 2


def test_disabled_by_env(cache_file, monkeypatch):
    monkeypatch.setenv("AM_TOOLS_NO_VERSION_CHECK", "1")
    monkeypatch.setattr(
        version_check,
        "_fetch_remote_version",
        lambda: pytest.fail("should not fetch when disabled"),
    )

    version_check.check_for_update()

    assert not cache_file.exists()


def test_fetch_failure_does_not_stamp(cache_file, monkeypatch):
    def boom():
        raise OSError("network down")

    monkeypatch.setattr(version_check, "_fetch_remote_version", boom)

    # Must not raise, and must not record a check (so the next run retries).
    version_check.check_for_update()

    assert not cache_file.exists()


def test_remote_none_does_not_stamp(cache_file, monkeypatch):
    monkeypatch.setattr(version_check, "_fetch_remote_version", lambda: None)

    version_check.check_for_update()

    assert not cache_file.exists()
