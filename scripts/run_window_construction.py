"""Label-Independent Adaptive Window Construction.

Scans the cleaned AUTH stream chronologically and closes a window when
all four evidence-sufficiency thresholds are met:

    M >= M_min, E >= E_min, V_src >= V_min_src, V_dst >= V_min_dst

The paper uses M_min=2,000,000, E_min=50,000, V_src_min=5,000,
V_dst_min=3,000, yielding 526 windows on the full LANL stream (54 with a
recorded attack).  Attack annotations come from the redteam records and
are used only for evaluation.

Usage:
    python scripts/run_window_construction.py [--config configs/lanl.json]
                                              [--fixed-event]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import REPO_ROOT, add_common_args, apply_data_roots, load_config

args_parser = argparse.ArgumentParser(description=__doc__)
add_common_args(args_parser)
args_parser.add_argument("--fixed-event", action="store_true",
                         help="Build fixed-event windows (every 2,000,000 events) "
                              "for the windowing ablation instead of adaptive windows.")
args_parser.add_argument("--events-per-window", type=int, default=2_000_000)
args = args_parser.parse_args()

cfg = load_config(args.config)
apply_data_roots(cfg)

from asymgad.paths import DATA_ROOT, OUTPUT_ROOT
from asymgad.window import WindowInfo, _annotate_attacks


def _adaptive_windows(thresholds: dict, max_files: int | None = None):
    from asymgad.window import build_adaptive_windows_streaming

    t0 = time.time()
    windows, total_events = build_adaptive_windows_streaming(
        categories=("AUTH",),
        M_min=thresholds["M_min"],
        E_min=thresholds["E_min"],
        V_min_src=thresholds["V_min_src"],
        V_min_dst=thresholds["V_min_dst"],
        max_files=max_files,
        verbose=True,
    )
    _annotate_attacks(windows, verbose=True)
    print(f"Scanned {total_events:,} events in {time.time() - t0:.0f}s -> "
          f"{len(windows)} windows")
    return windows


def _fixed_event_windows(events_per_window: int = 2_000_000):
    """Build contiguous fixed-event windows (windowing ablation)."""
    import pyarrow.parquet as pq

    cat_dir = DATA_ROOT / "cleaned" / "auth"
    files = sorted(f for f in os.listdir(str(cat_dir)) if f.endswith(".parquet"))
    windows: list[WindowInfo] = []
    window_start = None
    global_before = 0
    tail_start = None
    t0 = time.time()

    for fi, fp in enumerate(files, 1):
        table = pq.read_table(str(cat_dir / fp))
        ts = table.column("timestamp").to_numpy().astype(np.int64)
        n = int(ts.size)
        if n == 0:
            continue
        if window_start is None:
            window_start = int(ts[0])
        global_after = global_before + n
        first_multiple = ((global_before // events_per_window) + 1) * events_per_window
        for boundary in range(first_multiple, global_after + 1, events_per_window):
            idx = boundary - global_before - 1
            if idx < 0:
                continue
            window_end = int(ts[idx])
            windows.append(WindowInfo(
                idx=len(windows),
                ts_start=int(window_start),
                ts_end=window_end,
                duration_sec=float(max(0, window_end - int(window_start))),
                M=events_per_window, E=0, V_src=0, V_dst=0, V_total=0,
            ))
            if idx + 1 < n:
                window_start = int(ts[idx + 1])
            else:
                window_start = None
        tail_start = window_start
        global_before = global_after
        if fi % 25 == 0 or fi == len(files):
            print(f"  [{fi}/{len(files)}] events={global_before:,} "
                  f"windows={len(windows)}", flush=True)

    # Merge the residual tail into the previous window.
    if tail_start is not None and windows:
        windows[-1].ts_end = int(ts[-1])
        windows[-1].duration_sec = float(max(0, windows[-1].ts_end - windows[-1].ts_start))
        windows[-1].is_merged_tail = True

    _annotate_attacks(windows, verbose=True)
    print(f"Fixed-event windows: {len(windows)} in {time.time() - t0:.0f}s")
    return windows


def _dump(windows, thresholds: dict, total_events: int, kind: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "windows": [
            {
                "idx": w.idx, "ts_start": w.ts_start, "ts_end": w.ts_end,
                "dur_h": round(w.duration_sec / 3600.0, 2),
                "M": w.M, "E": w.E, "V_src": w.V_src, "V_dst": w.V_dst,
                "V_total": w.V_total, "has_attack": w.has_attack,
                "attack_pivots": w.attack_pivots,
                "attack_event_count": w.attack_event_count,
            }
            for w in windows
        ],
        "kind": kind,
        "thresholds": thresholds,
        "n_windows": len(windows),
        "total_events": total_events,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_path}")


def main():
    thresholds = cfg.get("thresholds", {})
    if args.fixed_event:
        windows = _fixed_event_windows(args.events_per_window)
        out_path = OUTPUT_ROOT / "fixed_event_2M" / "windows.json"
        _dump(windows, {"events_per_window": args.events_per_window},
              sum(w.M for w in windows), "fixed_event", out_path)
    else:
        windows = _adaptive_windows(thresholds)
        out_path = OUTPUT_ROOT / "windows.json"
        _dump(windows, thresholds, sum(w.M for w in windows), "adaptive", out_path)


if __name__ == "__main__":
    main()
