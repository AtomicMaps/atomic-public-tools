__version__ = "0.2.0"


def validate_directory(*args, **kwargs):  # noqa: ANN002, ANN003
    """Validate a directory and return a JSON-ready dict.

    Lazy wrapper around :func:`atomic_tools.api.validate_directory` so that
    ``import atomic_tools`` (and ``am-tools --version``) stays cheap.
    """
    from atomic_tools.api import validate_directory as _impl

    return _impl(*args, **kwargs)


__all__ = ["__version__", "validate_directory"]
