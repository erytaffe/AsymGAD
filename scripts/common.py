"""Shared helpers for the AsymGAD experiment scripts.

The scripts are plain entry points that run from the repository root.
They add ``src/`` to ``sys.path`` and apply the data/output roots from
the selected JSON config (or the ``ASYGAD_DATA_ROOT`` /
``ASYGAD_OUTPUT_ROOT`` environment variables) before importing the
``asymgad`` package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_data_roots(cfg: dict) -> None:
    """Set ASYGAD_DATA_ROOT / ASYGAD_OUTPUT_ROOT from the config."""
    data_root = cfg.get("data_root")
    if data_root:
        p = Path(data_root)
        if not p.is_absolute():
            p = REPO_ROOT / p
        os.environ.setdefault("ASYGAD_DATA_ROOT", str(p))
    output_root = cfg.get("output_root")
    if output_root:
        p = Path(output_root)
        if not p.is_absolute():
            p = REPO_ROOT / p
        os.environ.setdefault("ASYGAD_OUTPUT_ROOT", str(p))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "lanl.json"),
                        help="Path to the JSON experiment configuration.")
    parser.add_argument("--out-dir", default=None,
                        help="Override the output directory for this run.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many windows (0 = all).")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seed list; overrides config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the number of training epochs.")
    return parser


def parse_seeds(value: str | None, cfg_defaults) -> list[int]:
    if value:
        return [int(s) for s in value.split(",") if s.strip()]
    return list(cfg_defaults)


def windows_path(cfg: dict) -> Path:
    """Resolve the windows.json path from the config or the default."""
    rel = cfg.get("windows_json", "windows.json")
    p = Path(rel)
    if p.is_absolute():
        return p
    output_root = Path(os.environ.get("ASYGAD_OUTPUT_ROOT", REPO_ROOT / "output"))
    if p.parts[0] == "output":
        return REPO_ROOT / p
    return output_root / p
