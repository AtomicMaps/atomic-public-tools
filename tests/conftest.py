import pytest


@pytest.fixture(autouse=True)
def _disable_version_check(monkeypatch):
    """Keep the daily update check from making network calls during tests.

    The root CLI callback runs `check_for_update()` on every invocation; setting
    this env var makes it a no-op. Tests that exercise the check itself clear the
    var and mock the fetch.
    """
    monkeypatch.setenv("AM_TOOLS_NO_VERSION_CHECK", "1")
