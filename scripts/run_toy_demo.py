"""Self-contained toy demo (no external data required).

Generates the toy stream, runs adaptive window construction, builds the
observation graph of the attack window, trains AsymGAD, and reports the
pivot ranking.  Useful for verifying the installation.

Usage:
    python scripts/run_toy_demo.py [--epochs 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REPO_ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data-root", default=str(REPO_ROOT / "data_toy"))
args = parser.parse_args()

import os
import sys

os.environ["ASYGAD_DATA_ROOT"] = args.data_root
sys.path.insert(0, str(REPO_ROOT / "src"))

from asymgad.data.toy import make_toy_data, toy_thresholds
from asymgad.window import build_adaptive_windows_streaming, _annotate_attacks
from asymgad.graph_data import build_window_graph
from asymgad import train_asymgad
from asymgad.metrics import average_precision, node_ranks


def main():
    data_root = Path(args.data_root)
    make_toy_data(data_root=data_root, seed=0)
    print(f"Toy data written to {data_root}")

    th = toy_thresholds()
    windows, total_events = build_adaptive_windows_streaming(
        categories=("AUTH",),
        M_min=th["M_min"], E_min=th["E_min"],
        V_min_src=th["V_min_src"], V_min_dst=th["V_min_dst"],
        verbose=False,
    )
    _annotate_attacks(windows, verbose=False)
    print(f"Windows: {len(windows)} (events={total_events})")

    out_dir = REPO_ROOT / "output_toy"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "windows.json", "w", encoding="utf-8") as f:
        json.dump({
            "windows": [
                {
                    "idx": w.idx, "ts_start": w.ts_start, "ts_end": w.ts_end,
                    "dur_h": round(w.duration_sec / 3600, 2),
                    "M": w.M, "E": w.E, "V_src": w.V_src, "V_dst": w.V_dst,
                    "V_total": w.V_total, "has_attack": w.has_attack,
                    "attack_pivots": w.attack_pivots,
                    "attack_event_count": w.attack_event_count,
                }
                for w in windows
            ],
            "thresholds": th,
        }, f, indent=2)

    aw = next(w for w in windows if w.has_attack)
    print(f"Attack window {aw.idx}: pivots={aw.attack_pivots}")
    g = build_window_graph(aw, J=4, verbose=True)
    r = train_asymgad(g, epochs=args.epochs, seed=42, verbose=True)
    ap = average_precision(r["scores"], g.labels)
    ranks = node_ranks(r["scores"], g.pivot_ids, g.pivot_names)
    print(f"\nToy result: N={g.N} E={g.E} AP={ap:.4f} pivot ranks={ranks}")
    print("Demo complete.")


if __name__ == "__main__":
    main()
