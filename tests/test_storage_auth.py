"""Verify that storage-layer auth failures surface as S3AuthError subclasses."""

from __future__ import annotations

import pytest
from obstore.exceptions import PermissionDeniedError, UnauthenticatedError

from atomic_tools.io import storage
from atomic_tools.io.storage import ObjectStoreBackend, from_directory
from atomic_tools.utils import object_store as object_store_module
from atomic_tools.utils.aws_errors import (
    S3CredentialsMissingError,
    S3PermissionDeniedError,
)


class _FakeSession:
    """Minimal stand-in for ``boto3.Session`` used by init_session()."""

    profile_name = "fake-profile"

    def __init__(self, *, has_credentials: bool) -> None:
        self._has_credentials = has_credentials

    def get_credentials(self):
        return object() if self._has_credentials else None


def test_from_directory_raises_credentials_missing(monkeypatch):
    monkeypatch.setattr(
        object_store_module.boto3,
        "Session",
        lambda *a, **kw: _FakeSession(has_credentials=False),
    )
    with pytest.raises(S3CredentialsMissingError) as ei:
        from_directory("s3://example-bucket/some/prefix")
    assert ei.value.profile == "fake-profile"
    assert "fake-profile" in ei.value.help_text()


def _make_backend_without_store(monkeypatch) -> ObjectStoreBackend:
    """Build an ObjectStoreBackend without touching the obstore S3 client."""
    monkeypatch.setattr(
        ObjectStoreBackend, "__init__", lambda self, **kwargs: None
    )
    backend = ObjectStoreBackend()
    backend._scheme = "s3"
    backend._bucket = "denied-bucket"
    backend._prefix = "deep/path"
    backend._store = object()
    return backend


@pytest.mark.parametrize(
    "obstore_exc_cls", [PermissionDeniedError, UnauthenticatedError]
)
def test_list_keys_translates_obstore_auth_failure(monkeypatch, obstore_exc_cls):
    backend = _make_backend_without_store(monkeypatch)

    def fake_get_object_keys(**kwargs):
        # Mirror the wrap-as-RuntimeError pattern in utils.utils
        inner = obstore_exc_cls("denied")
        outer = RuntimeError("Failed to list keys with prefix deep/path: denied")
        raise outer from inner

    monkeypatch.setattr(storage, "get_object_keys", fake_get_object_keys)

    with pytest.raises(S3PermissionDeniedError) as ei:
        backend.list_keys(include=[".jpg"], exclude=[])

    err = ei.value
    assert err.bucket == "denied-bucket"
    assert err.prefix == "deep/path"
    assert err.operation == "list"
    assert "s3:ListBucket" in err.help_text()


def test_open_local_translates_obstore_auth_failure(monkeypatch):
    backend = _make_backend_without_store(monkeypatch)

    def fake_download(store, key, tmp_dir):
        inner = PermissionDeniedError("nope")
        outer = RuntimeError("Storage error: nope")
        raise outer from inner

    monkeypatch.setattr(storage, "download", fake_download)

    with (
        pytest.raises(S3PermissionDeniedError) as ei,
        backend.open_local("foo.jpg"),
    ):
        pass
    assert ei.value.operation == "download"
    assert "s3:GetObject" in ei.value.help_text()


def test_write_output_translates_obstore_auth_failure(monkeypatch, tmp_path):
    backend = _make_backend_without_store(monkeypatch)

    def fake_upload(store, *, key, source):
        inner = PermissionDeniedError("nope")
        outer = RuntimeError("Failed to upload x to y: nope")
        raise outer from inner

    monkeypatch.setattr(storage, "upload", fake_upload)

    src = tmp_path / "sidecar.csv"
    src.write_text("Filename\n")
    with pytest.raises(S3PermissionDeniedError) as ei:
        backend.write_output("sidecar.csv", src)
    assert ei.value.operation == "upload"
    assert "s3:PutObject" in ei.value.help_text()


def test_list_keys_passes_through_non_auth_runtime_error(monkeypatch):
    backend = _make_backend_without_store(monkeypatch)

    def fake_get_object_keys(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(storage, "get_object_keys", fake_get_object_keys)

    with pytest.raises(RuntimeError, match="disk full"):
        backend.list_keys(include=[], exclude=[])
