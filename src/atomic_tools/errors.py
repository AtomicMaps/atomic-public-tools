"""Exception types shared across commands and the library API."""

from __future__ import annotations


class InvalidInputError(ValueError):
    """Something the caller supplied is wrong: a path, a flag value, an upload.

    The library API reports these as ``invalid_input`` (exit 2 in
    ``validate --api-mode``). Any other ``ValueError`` escaping validation is a
    bug and is reported as ``internal_error``. Subclassing ValueError keeps
    existing ``except ValueError`` handlers working.
    """
