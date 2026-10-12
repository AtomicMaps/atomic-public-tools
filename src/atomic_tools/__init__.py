__version__ = "0.2.0"


def validate_path(*args, **kwargs):  # noqa: ANN002, ANN003
    """Validate a directory or single file and return a JSON-ready dict.

    Lazy wrapper around :func:`atomic_tools.api.validate_path` so that
    ``import atomic_tools`` (and ``am-tools --version``) stays cheap.
    """
    from atomic_tools.api import validate_path as _impl

    return _impl(*args, **kwargs)


__all__ = ["__version__", "validate_path"]
