"""Storage backend abstraction for sidecar generation.

A `StorageBackend` hides whether the source data lives in an object store
(S3/GCS/Azure) or on the local filesystem. The sidecar generator uses the
backend to list candidate files, materialise each one as a local path so
exiftool/pdal can run on it, and write the resulting CSV back alongside
the inputs.

`from_path(path)` is the entry point: it inspects the URI scheme and returns
the appropriate backend, raising a clear error if a local path doesn't exist.
``path`` may be a directory/prefix or a single file/object; a single-file
backend lists just that file and writes output beside it.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import obstore
from obstore.exceptions import NotFoundError, PermissionDeniedError, UnauthenticatedError

if TYPE_CHECKING:
    import pandas as pd

from atomic_tools.errors import InvalidInputError
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
    get_keys_and_metadata,
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

    @abstractmethod
    def read_output(self, filename: str) -> tuple[pd.DataFrame, str | None]:
        """Read an existing output CSV (all cells as strings) plus its
        :meth:`output_version`, for read-modify-write updates."""

    @abstractmethod
    def output_version(self, filename: str) -> str | None:
        """An opaque token that changes whenever the output file changes
        (None if it doesn't exist). Compare before writing a merged file."""

    @abstractmethod
    def output_exists(self, filename: str) -> bool:
        """Whether :meth:`write_output` would replace an existing file."""

    @property
    @abstractmethod
    def output_location(self) -> str:
        """Display form of the folder :meth:`write_output` writes into (for a
        single-file target, the file's folder rather than the file)."""

    @property
    def single_file(self) -> str | None:
        """The one key this backend targets when it was given a file, else None.

        For object stores this is only known after ``list_keys`` (a URI can't
        be told apart from a prefix without asking the store).
        """
        return None


class LocalBackend(StorageBackend):
    def __init__(self, root: Path, only: str | None = None) -> None:
        # `only` restricts the backend to one file (a name relative to `root`);
        # `root` is then the file's parent, so outputs land beside it.
        self._root = root
        self._only = only

    @property
    def display_root(self) -> str:
        return str(self._root / self._only) if self._only else str(self._root)

    @property
    def single_file(self) -> str | None:
        return self._only

    def list_keys(self, include: Sequence[str], exclude: Sequence[str]) -> list[str]:
        if self._only is not None:
            try:
                size = (self._root / self._only).stat().st_size
            except OSError:
                size = 0
            return filter_keys({self._only: size}, include, exclude)
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

    def output_exists(self, filename: str) -> bool:
        return (self._root / filename).exists()

    def output_version(self, filename: str) -> str | None:
        try:
            st = (self._root / filename).stat()
        except FileNotFoundError:
            return None
        return f"{st.st_mtime_ns}:{st.st_size}"

    def read_output(self, filename: str) -> tuple[pd.DataFrame, str | None]:
        from atomic_tools.utils.utils import load_sidecar_df

        version = self.output_version(filename)
        return load_sidecar_df(str(self._root / filename)), version

    @property
    def output_location(self) -> str:
        return str(self._root)


class ObjectStoreBackend(StorageBackend):
    def __init__(self, scheme: str, bucket: str, prefix: str) -> None:
        store_type = REMOTE_SCHEME_TO_STORE_TYPE[scheme]
        self._scheme = scheme
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        # Set by list_keys when the prefix turns out to name a single object.
        self._single_key: str | None = None
        self._single_size = 0
        # {key: size} of everything under "<prefix>/", listed once per backend.
        self._listing: dict[str, int] | None = None
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

    @property
    def single_file(self) -> str | None:
        return self._single_key

    def _raw_listing(self) -> dict[str, int]:
        """``{key: size}`` for everything under ``<prefix>/``, listed once."""
        if self._listing is None:
            if not self._prefix.strip():
                raise InvalidInputError("Directory path cannot be empty")
            try:
                info = get_keys_and_metadata(self._store, self._prefix)
            except RuntimeError as e:
                self._reraise_if_auth(e, operation="list")
                raise
            self._listing = {k: int(v["size"]) for k, v in info.items()}
        return self._listing

    def list_keys(self, include: Sequence[str], exclude: Sequence[str]) -> list[str]:
        if self._single_key is None:
            listing = self._raw_listing()
            if listing:
                # Something lives under "<prefix>/", so the URI names a folder —
                # even one with a dot in its name — matching files or not.
                return filter_keys(listing, include, exclude)
            # Nothing under it as a folder (listing treats the prefix as
            # "<prefix>/"): either one object at exactly this key, or nothing.
            self._single_size = self._head_size(self._prefix)
            self._single_key = self._prefix
        return filter_keys({self._single_key: self._single_size}, include, exclude)

    def _head_size(self, key: str) -> int:
        """Size of the object at ``key``.

        Only called when nothing is listed under ``key/``, so a 404 means the
        path doesn't exist at all (a bad path, like a missing local file), and
        a denied HEAD is a real permission problem on that object.
        """
        try:
            meta = obstore.head(self._store, key)
        except (FileNotFoundError, NotFoundError):
            raise FileNotFoundError(
                f"No file or folder exists at {self.display_root}"
            ) from None
        except (PermissionDeniedError, UnauthenticatedError) as e:
            self._reraise_if_auth(e, operation="download")
            raise
        return int(meta["size"])

    @contextmanager
    def open_local(self, key: str) -> Iterator[str]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                local_path = download(self._store, key, tmp_dir)
            except RuntimeError as e:
                self._reraise_if_auth(e, operation="download")
                raise
            yield local_path

    def _output_folder(self) -> str:
        # For a single-object target, outputs go beside the object, not under it.
        return self._prefix.rpartition("/")[0] if self._single_key else self._prefix

    def _output_key(self, filename: str) -> str:
        folder = self._output_folder()
        return f"{folder}/{filename}" if folder else filename

    @property
    def output_location(self) -> str:
        return f"{self._scheme}://{self._bucket}/{self._output_folder()}"

    def output_exists(self, filename: str) -> bool:
        try:
            obstore.head(self._store, self._output_key(filename))
        except (FileNotFoundError, NotFoundError):
            return False
        except (PermissionDeniedError, UnauthenticatedError) as e:
            self._reraise_if_auth(e, operation="download")
            raise
        return True

    @staticmethod
    def _version_of(meta: dict) -> str:
        return str(meta.get("e_tag") or f"{meta.get('last_modified')}:{meta.get('size')}")

    def output_version(self, filename: str) -> str | None:
        try:
            meta = obstore.head(self._store, self._output_key(filename))
        except (FileNotFoundError, NotFoundError):
            return None
        except (PermissionDeniedError, UnauthenticatedError) as e:
            self._reraise_if_auth(e, operation="download")
            raise
        return self._version_of(meta)

    def read_output(self, filename: str) -> tuple[pd.DataFrame, str | None]:
        import io

        import pandas

        try:
            result = obstore.get(self._store, self._output_key(filename))
            data = bytes(result.bytes())
        except (PermissionDeniedError, UnauthenticatedError) as e:
            self._reraise_if_auth(e, operation="download")
            raise
        df = pandas.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8")
        return df, self._version_of(result.meta)

    def write_output(self, filename: str, source: Path) -> str:
        output_key = self._output_key(filename)
        try:
            upload(self._store, key=output_key, source=str(source))
        except RuntimeError as e:
            self._reraise_if_auth(e, operation="upload")
            raise
        return f"{self._scheme}://{self._bucket}/{output_key}"


def _join_s3(bucket: str, prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"


def console_url_to_s3_uri(directory: str) -> str | None:
    """Translate an AWS S3 *console* or HTTPS URL to an ``s3://bucket/prefix`` URI.

    Returns ``None`` for anything that isn't a recognisable AWS S3 web URL
    (already-``s3://`` URIs, local paths, other https links). Handles the three
    shapes users paste from a browser:

      * console:        ``https://<region>.console.aws.amazon.com/s3/buckets/<bucket>?prefix=<prefix>``
      * virtual-hosted: ``https://<bucket>.s3[.<region>].amazonaws.com/<key>``
      * path-style:     ``https://s3[.<region>].amazonaws.com/<bucket>/<key>``
    """
    parsed = urlparse(directory.strip())
    if parsed.scheme not in ("http", "https"):
        return None
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]

    # S3 web console — bucket is a path segment, prefix is a query parameter.
    if host.endswith("console.aws.amazon.com"):
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 3 and parts[0] == "s3" and parts[1] == "buckets":
            prefix = parse_qs(parsed.query).get("prefix", [""])[0]
            return _join_s3(parts[2], prefix)
        return None

    if host.endswith(".amazonaws.com"):
        labels = host.split(".")
        # path-style: s3.amazonaws.com/<bucket>/<key> (or s3.<region>.amazonaws.com/...)
        if labels[0] == "s3":
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                bucket, *key = parts
                return _join_s3(bucket, "/".join(key))
            return None
        # virtual-hosted: <bucket>.s3[.<region>].amazonaws.com/<key>
        if "s3" in labels:
            bucket = ".".join(labels[: labels.index("s3")])
            if bucket:
                return _join_s3(bucket, parsed.path.lstrip("/"))
    return None


def normalize_directory(directory: str) -> str:
    """Return ``directory`` as an ``s3://`` URI when it's an AWS console/HTTPS URL.

    Warns when a translation happens; a no-op for ``s3://`` URIs and local paths.
    """
    s3_uri = console_url_to_s3_uri(directory)
    if s3_uri is not None:
        logger.warning(
            "Interpreting the AWS console URL as %s "
            "(paste the s3:// URI directly to skip this step).",
            s3_uri,
        )
        return s3_uri
    return directory


def from_path(path: str) -> StorageBackend:
    """Return a `StorageBackend` for `path`.

    `path` may be an object-store URI (``s3://bucket/prefix`` or
    ``s3://bucket/key.jpg``, ``gs://...``, ``az://...``) or a local filesystem
    path to a directory or a single file. Local paths must already exist.
    """
    if not path or not path.strip():
        raise InvalidInputError("--path cannot be empty.")

    directory = normalize_directory(path)
    raw = directory.strip().rstrip("/")
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower() if parsed.scheme else ""
    store_type = REMOTE_SCHEME_TO_STORE_TYPE.get(scheme) if scheme else None
    if store_type:
        if not parsed.netloc:
            raise InvalidInputError(f"Object-store URI missing bucket: {directory!r}")
        prefix = parsed.path.lstrip("/")
        return ObjectStoreBackend(scheme=scheme, bucket=parsed.netloc, prefix=prefix)

    local = Path(raw).expanduser().resolve()
    if not local.exists():
        raise FileNotFoundError(f"Local path does not exist: {local}")
    if local.is_file():
        return LocalBackend(local.parent, only=local.name)
    return LocalBackend(local)
