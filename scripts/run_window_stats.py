"""Summarize the adaptive window construction statistics.

Prints the window-level statistics reported in the paper (duration,
event count, directed edges, unique sources/destinations) and the
per-pivot window coverage, and writes a Markdown report.

Usage:
    python scripts/run_window_stats.py [--config configs/lanl.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import add_common_args, apply_data_roots, load_config, windows_path

args_parser = argparse.ArgumentParser(description=__doc__)
add_common_args(args_parser)
args = args_parser.parse_args()

cfg = load_config(args.config)
apply_data_roots(cfg)

from asymgad.paths import OUTPUT_ROOT


def _qs(arr):
    a = sorted(arr)
    n = len(a)
    return a[0], a[n // 4], a[n // 2], a[3 * n // 4], a[-1]


def main():
    wpath = windows_path(cfg)
    with open(wpath, "r", encoding="utf-8") as f:
        wdata = json.load(f)
    windows = wdata["windows"]
    attack_windows = [w for w in windows if w.get("has_attack", False)]

    lines = []
    lines.append("# Adaptive Window Statistics")
    lines.append("")
    lines.append(f"- Total windows: {len(windows)}")
    lines.append(f"- Attack windows: {len(attack_windows)} "
                 f"({100 * len(attack_windows) / max(len(windows), 1):.1f}%)")
    lines.append("")
    lines.append("| Metric | Min | Q1 | Median | Q3 | Max | Mean |")
    lines.append("|--------|-----|----|--------|----|-----|------|")
    for label, key, fmt in [
        ("Elapsed Duration (h)", "dur_h", ".1f"),
        ("Raw Event Count", "M", ",.0f"),
        ("Directed Edges", "E", ",.0f"),
        ("Unique Sources", "V_src", ",.0f"),
        ("Unique Destinations", "V_dst", ",.0f"),
        ("Total Active Nodes", "V_total", ",.0f"),
    ]:
        vals = [w.get(key, 0) for w in windows]
        mn, q1, md, q3, mx = _qs(vals)
        mu = np.mean(vals)
        lines.append(f"| {label} | {mn:{fmt}} | {q1:{fmt}} | {md:{fmt}} | "
                     f"{q3:{fmt}} | {mx:{fmt}} | {mu:{fmt}} |")

    pivots = ["C17693", "C19932", "C22409", "C18025"]
    lines.append("")
    lines.append("## Per-Pivot Coverage")
    lines.append("")
    for p in pivots:
        wis = [w["idx"] for w in windows if p in w.get("attack_pivots", [])]
        lines.append(f"- {p}: {len(wis)} windows - {wis}")

    report = "\n".join(lines)
    print(report)
    out_path = OUTPUT_ROOT / "window_stats.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
