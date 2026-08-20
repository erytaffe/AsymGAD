"""Repository path configuration.

All data access is rooted at ``DATA_ROOT`` and all experiment outputs are
written under ``OUTPUT_ROOT``.  Both can be overridden with environment
variables so that the repository itself never stores large data artifacts:

    ASYGAD_DATA_ROOT    (default: <repo>/data)
    ASYGAD_OUTPUT_ROOT  (default: <repo>/output)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = Path(os.environ.get("ASYGAD_DATA_ROOT", REPO_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("ASYGAD_OUTPUT_ROOT", REPO_ROOT / "output"))
