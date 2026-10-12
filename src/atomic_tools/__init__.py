"""atomic_tools — format and validate metadata for Atomic Flow intake."""

import logging

__version__ = "0.2.0"

# Library etiquette: never emit log records unless the host app configures
# logging (the CLI does; a web server decides for itself).
logging.getLogger(__name__).addHandler(logging.NullHandler())
