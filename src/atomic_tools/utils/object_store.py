"""Cloud object-store wrapper, originally `atomicmapspy/object_store.py`."""

import logging
import time
from typing import Any, TypeAlias, cast

import boto3
from botocore.session import (  # noqa: F401  # used by ObjectStore.from_session typing
    Session as BotocoreSession,
)
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import AzureStore, GCSStore, S3Store

logger = logging.getLogger(__name__)

# Instances returned by boto/obstore bindings; accepted by obstore.head/put/sign/etc.
ObstoreBackend: TypeAlias = S3Store | GCSStore | AzureStore
ObstoreBackendClass: TypeAlias = type[S3Store] | type[GCSStore] | type[AzureStore]

REMOTE_SCHEME_TO_STORE_TYPE: dict[str, str] = {
    "s3": "s3",
    "s3a": "s3",
    "gs": "gcs",
    "az": "azure",
    "adl": "azure",
    "abfs": "azure",
    "abfss": "azure",
    "azure": "azure",
}


class ObjectStore:
    """Wrapper for creating cloud storage instances dynamically.

    Only `init_session` and `from_url` are exercised by this script, but the
    full surface from atomicmapspy is preserved here so the class can be lifted
    cleanly into a utils module in the destination repo.
    """

    def __init__(
        self,
        store_type: str,
        config: dict[str, str] | None = None,
        client_options: dict[str, str] | None = None,
        retry_config: dict[str, Any] | None = None,
    ) -> None:
        self.store_type = self.validate_store_type(store_type)
        self.config = config
        self.client_options = client_options
        self.retry_config = retry_config
        self.store: ObstoreBackend | None = None
        self.init_time: float | None = None

    def _record_backend_init(self) -> None:
        self.init_time = time.time()

    def seconds_since_backend_init(self) -> float:
        if self.init_time is None:
            return float("inf")
        return time.time() - self.init_time

    def _initialize_store(
        self, init_method: str, *args, **kwargs
    ) -> S3Store | GCSStore | AzureStore:
        store_classes: dict[str, ObstoreBackendClass] = {
            "s3": S3Store,
            "gcs": GCSStore,
            "azure": AzureStore,
        }

        store_class = store_classes[self.store_type]
        init_method = self.validate_init_method(init_method)

        if self.store_type == "azure" and init_method != "from_url":
            kwargs["container_name"] = kwargs.pop("bucket")

        common_kwargs = {
            "config": self.config,
            "client_options": self.client_options,
            "retry_config": self.retry_config,
        }

        if init_method == "from_url":
            return store_class.from_url(*args, **kwargs, **common_kwargs)
        elif init_method == "from_session":
            if self.store_type != "s3":
                raise ValueError(
                    f"Invalid store_type: {self.store_type}. "
                    "'from_session' is only supported for 's3'."
                )
            session = kwargs.pop("session")
            credential_provider = Boto3CredentialProvider(session=session)
            return S3Store(
                **kwargs,
                credential_provider=credential_provider,
                **common_kwargs,
            )
        else:
            return store_class(*args, **kwargs, **common_kwargs)

    def validate_store_type(self, store_type: str) -> str:
        store_type = store_type.lower()
        supported_types = ["s3", "gcs", "azure"]
        if store_type not in supported_types:
            raise ValueError(
                f"Unsupported store_type: {store_type}. "
                f"Supported store_types are: {', '.join(supported_types)}"
            )
        return store_type

    def validate_init_method(self, init_method: str):
        if not self.store_type:
            raise ValueError("No store_type found")

        init_method = init_method.lower()
        supported_init_methods = [
            "from_env",
            "from_url",
            "from_session",
            "init_session",
            "download",
        ]

        if (
            init_method == "from_session" or init_method == "init_session"
        ) and self.store_type != "s3":
            raise ValueError(
                "Unsupported init_method. 's3' is the only store_type "
                "supported for 'from_session' and 'init_session'."
            )
        elif init_method not in supported_init_methods:
            raise ValueError(
                f"Unsupported init_method: {init_method}. "
                f"Supported init_methods are: {', '.join(supported_init_methods)}"
            )
        return init_method

    def from_env(self, bucket: str) -> S3Store | GCSStore | AzureStore:
        store = self._initialize_store("from_env", bucket=bucket)
        self.store = store
        self._record_backend_init()
        return store

    def from_url(self, url: str) -> S3Store | GCSStore | AzureStore:
        store = self._initialize_store("from_url", url=url)
        self.store = store
        self._record_backend_init()
        return store

    def from_session(self, session, bucket: str) -> S3Store:
        if not self.store_type or self.store_type != "s3":
            raise ValueError(
                f"Invalid store_type: {self.store_type}. "
                "'store_type' must be 's3' for 'from_session'."
            )
        store = self._initialize_store("from_session", session=session, bucket=bucket)
        self.store = store
        self._record_backend_init()
        return cast(S3Store, store)

    def init_session(self, bucket: str) -> S3Store:
        if not self.store_type or self.store_type != "s3":
            raise ValueError(
                f"Invalid store_type: {self.store_type}. "
                "'store_type' must be 's3' for 'init_session()'."
            )

        if not self.config:
            self.config = {}
            session = boto3.Session()
        else:
            self.config = {key.lower(): value for key, value in self.config.items()}
            session = boto3.Session(
                aws_access_key_id=self.config.get("aws_access_key_id"),
                aws_secret_access_key=self.config.get("aws_secret_access_key"),
                aws_session_token=self.config.get("aws_session_token"),
                region_name=self.config.get("region_name"),
                profile_name=self.config.get("profile_name"),
            )

        store = self._initialize_store("from_session", session=session, bucket=bucket)
        self.store = store
        self._record_backend_init()
        return cast(S3Store, store)
