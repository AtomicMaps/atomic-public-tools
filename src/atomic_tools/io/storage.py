"""Storage backend abstraction for sidecar generation.

A `StorageBackend` hides whether the source data lives in an object store
(S3/GCS/Azure) or on the local filesystem. The sidecar generator uses the
backend to list candidate files, materialise each one as a local path so
exiftool/pdal can run on it, and write the resulting CSV back alongside
the inputs.

`from_directory(directory)` is the entry point: it inspects the URI scheme
and returns the appropriate backend, raising a clear error if a local path
doesn't exist.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from obstore.exceptions import PermissionDeniedError, UnauthenticatedError

from atomic_tools.utils.aws_errors import (
    S3PermissionDeniedError,
    current_profile,
)
from atomic_tools.utils.object_store import (
    REMOTE_SCHEME_TO_STORE_TYPE,
    ObjectStore,
)
from atomic_tools.utils.utils import (
    download,
    filter_keys,
    get_object_keys,
    upload,
)

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """A storage backend the sidecar pipeline can read from and write to."""

    @property
    @abstractmethod
    def display_root(self) -> str:
        """Human-readable root for log messages, e.g. ``s3://bucket/prefix`` or ``/abs/path``."""

    @abstractmethod
    def list_keys(self, include: Sequence[str], exclude: Sequence[str]) -> list[str]:
        """List candidate files under the root, filtered by suffix rules."""

    @abstractmethod
    @contextmanager
    def open_local(self, key: str) -> Iterator[str]:
        """Yield a local filesystem path for `key`. Local backends yield in place;
        remote backends download to a temp file and clean up on exit."""

    @abstractmethod
    def write_output(self, filename: str, source: Path) -> str:
        """Write `source` to the root as `filename`. Returns a display path."""


class LocalBackend(StorageBackend):
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def display_root(self) -> str:
        return str(self._root)

    def list_keys(self, include: Sequence[str], exclude: Sequence[str]) -> list[str]:
        # Stat only files whose suffix could plausibly match — avoids walking
        # large trees of unrelated files just to discard them.
        inc_suffixes = tuple(e.lower() for e in include)
        key_sizes: dict[str, int] = {}
        for path in self._root.rglob("*"):
            if not path.name.lower().endswith(inc_suffixes):
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(self._root).as_posix()
            try:
                key_sizes[rel] = path.stat().st_size
            except OSError:
                key_sizes[rel] = 0
        return filter_keys(key_sizes, include, exclude)

    @contextmanager
    def open_local(self, key: str) -> Iterator[str]:
        yield str(self._root / key)

    def write_output(self, filename: str, source: Path) -> str:
        target = self._root / filename
        shutil.copy2(source, target)
        return str(target)


class ObjectStoreBackend(StorageBackend):
    def __init__(self, scheme: str, bucket: str, prefix: str) -> None:
        store_type = REMOTE_SCHEME_TO_STORE_TYPE[scheme]
        self._scheme = scheme
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        factory = ObjectStore(store_type)
        self._store = (
            factory.init_session(bucket=bucket)
            if store_type == "s3"
            else factory.from_env(bucket=bucket)
        )

    @property
    def display_root(self) -> str:
        return f"{self._scheme}://{self._bucket}/{self._prefix}"

    def _reraise_if_auth(self, error: BaseException, operation: str) -> None:
        """Translate an obstore auth failure into ``S3PermissionDeniedError``.

        `utils.py` wraps obstore errors as ``RuntimeError(...) from <obstore exc>``,
        so we walk both the error itself and ``__cause__``.
        """
        candidates: list[BaseException] = [error]
        if error.__cause__ is not None:
            candidates.append(error.__cause__)
        for candidate in candidates:
            if isinstance(candidate, (PermissionDeniedError, UnauthenticatedError)):
                raise S3PermissionDeniedError(
                    profile=current_profile(),
                    bucket=self._bucket,
                    prefix=self._prefix,
                    operation=operation,
                ) from error

    def list_keys(self, include: Sequence[str], exclude: Sequence[str]) -> list[str]:
        try:
            return get_object_keys(
                store=self._store,
                directory=self._prefix,
                include=list(include),
                exclude=list(exclude),
            )
        except RuntimeError as e:
            self._reraise_if_auth(e, operation="list")
            raise

    @contextmanager
    def open_local(self, key: str) -> Iterator[str]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                local_path = download(self._store, key, tmp_dir)
            except RuntimeError as e:
                self._reraise_if_auth(e, operation="download")
                raise
            yield local_path

    def write_output(self, filename: str, source: Path) -> str:
        output_key = f"{self._prefix}/{filename}" if self._prefix else filename
        try:
            upload(self._store, key=output_key, source=str(source))
        except RuntimeError as e:
            self._reraise_if_auth(e, operation="upload")
            raise
        return f"{self._scheme}://{self._bucket}/{output_key}"


def from_directory(directory: str) -> StorageBackend:
    """Return a `StorageBackend` for `directory`.

    `directory` may be an object-store URI (``s3://bucket/prefix``,
    ``gs://...``, ``az://...``) or a local filesystem path. Local paths
    must already exist and be directories; this raises otherwise.
    """
    if not directory or not directory.strip():
        raise ValueError("--directory cannot be empty.")

    raw = directory.strip().rstrip("/")
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower() if parsed.scheme else ""
    store_type = REMOTE_SCHEME_TO_STORE_TYPE.get(scheme) if scheme else None
    if store_type:
        if not parsed.netloc:
            raise ValueError(f"Object-store URI missing bucket: {directory!r}")
        prefix = parsed.path.lstrip("/")
        return ObjectStoreBackend(scheme=scheme, bucket=parsed.netloc, prefix=prefix)

    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"--directory is not a directory: {path}")
    return LocalBackend(path)
