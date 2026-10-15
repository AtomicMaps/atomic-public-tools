"""atomic_tools — format and validate metadata for Atomic Flow intake.

Library exports load lazily so ``am-tools --version`` stays fast.
"""

import logging
from typing import TYPE_CHECKING, Any

__version__ = "0.3.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

_LAZY = {
    "validate_path": "atomic_tools.api",
    "validate_upload": "atomic_tools.api",
    "check_dependencies": "atomic_tools.api",
    "UploadedFile": "atomic_tools.api",
    "ValidationResponse": "atomic_tools.models",
    "DependencyReport": "atomic_tools.models",
}

__all__ = ["__version__", *_LAZY]

if TYPE_CHECKING:
    from atomic_tools.api import UploadedFile as UploadedFile
    from atomic_tools.api import check_dependencies as check_dependencies
    from atomic_tools.api import validate_path as validate_path
    from atomic_tools.api import validate_upload as validate_upload
    from atomic_tools.models import DependencyReport as DependencyReport
    from atomic_tools.models import ValidationResponse as ValidationResponse


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'atomic_tools' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
