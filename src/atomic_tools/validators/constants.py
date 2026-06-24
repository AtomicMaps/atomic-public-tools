"""Constants shared across the sidecar validators."""

from __future__ import annotations

# Cap on how many files/rows a finding lists before collapsing the rest into
# a "(+N more)" tail.
MAX_LISTED_FILES = 20

# Filename sentinel for the row whose values apply to every file in the batch.
DEFAULT_ROW_NAME = "DEFAULT"
