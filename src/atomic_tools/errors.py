"""Exception types shared across commands and the library API."""

from __future__ import annotations


class InvalidInputError(ValueError):
    """Bad caller input, reported as ``invalid_input``.

    Any other ``ValueError`` is a bug and is reported as ``internal_error``.
    """
