"""Main benchmark: AsymGAD on the attack-containing windows.

Reproduces the paper's end-to-end ranking benchmark.  Each positive
window is turned into one observation graph and trained independently
(no parameter transfer across windows), with three random seeds per
window; the reported metric is the mean over seeds of the per-window
Average Precision.

Usage:
    python scripts/run_benchmark.py [--config configs/lanl.json]
                                    [--limit 4] [--epochs 100]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    REPO_ROOT,
    add_common_args,
    apply_data_roots,
    load_config,
    parse_seeds,
    windows_path,
)

args_parser = argparse.ArgumentParser(description=__doc__)
add_common_args(args_parser)
args = args_parser.parse_args()

cfg = load_config(args.config)
apply_data_roots(cfg)

from asymgad import WindowInfo, build_window_graph, train_asymgad
from asymgad.metrics import average_precision, node_ranks, summarize_window_aps
from asymgad.paths import OUTPUT_ROOT


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def cleanup():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    training = cfg.get("training", {})
    seeds = parse_seeds(args.seeds, cfg.get("seeds", [42, 123, 456]))
    epochs = args.epochs or int(training.get("epochs", 100))
    wpath = windows_path(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    summary_path = out_dir / "summary.json"

    wdata = load_json(wpath)
    attack_windows = [w for w in wdata["windows"] if w["has_attack"]]
    print(f"Attack windows: {len(attack_windows)}")
    print(f"Seeds: {seeds}  Epochs: {epochs}  Output: {out_dir}")
    print(f"Config: {json.dumps(training, indent=2)}")

    results = load_json(results_path, []) or []
    done = {r["window"] for r in results}
    remaining = [w for w in attack_windows if w["idx"] not in done]
    if args.limit and args.limit > 0:
        remaining = remaining[: args.limit]
    print(f"Remaining: {len(remaining)} windows")

    t_total = time.time()
    n_ok = 0
    n_fail = 0

    for i, w in enumerate(remaining, 1):
        wid = w["idx"]
        wi = WindowInfo(
            idx=wid, ts_start=w["ts_start"], ts_end=w["ts_end"],
            duration_sec=w["dur_h"] * 3600, M=w["M"], E=w["E"],
            V_src=w["V_src"], V_dst=w["V_dst"], V_total=w["V_total"],
            has_attack=True, attack_pivots=w["attack_pivots"],
            attack_event_count=w["attack_event_count"],
        )
        print(f"\n[{i}/{len(remaining)}] Window {wid}: "
              f"pivots={wi.attack_pivots}  E={wi.E:,}", flush=True)
        g = None
        try:
            g = build_window_graph(wi, J=4, verbose=False)
            if g is None or g.N == 0:
                print("  SKIP: empty graph", flush=True)
                continue

            seed_aps = []
            seed_ranks = {}
            train_times = []
            for seed in seeds:
                r = train_asymgad(
                    g, epochs=epochs, seed=seed, verbose=False,
                    use_pe=bool(training.get("use_pe", False)),
                    use_ief=bool(training.get("use_ief", True)),
                    lambda_deg=float(training.get("lambda_deg", 0.5)),
                    lambda_pr=float(training.get("lambda_pr", 0.3)),
                    fusion_alpha=float(training.get("fusion_alpha", 0.60)),
                    fusion_beta=float(training.get("fusion_beta", 0.0)),
                    fusion_delta=float(training.get("fusion_delta", 0.40)),
                )
                apv = average_precision(r["scores"], g.labels)
                seed_aps.append(apv)
                train_times.append(r["train_s"])
                for pid, pname in zip(g.pivot_ids, g.pivot_names):
                    rank = int(np.sum(r["scores"] > r["scores"][pid]) + 1)
                    seed_ranks.setdefault(pname, []).append(rank)
                print(f"    seed={seed}: AP={apv:.6f}  t={r['train_s']:.1f}s", flush=True)

            if not seed_aps:
                n_fail += 1
                continue

            per_pivot = {
                pname: {
                    "best_rank": int(np.min(ranks_p)),
                    "ranks": ranks_p,
                }
                for pname, ranks_p in seed_ranks.items()
            }
            entry = {
                "window": wid,
                "N": g.N,
                "E": g.E,
                "pivots": wi.attack_pivots,
                "ap_mean": round(float(np.mean(seed_aps)), 6),
                "ap_std": round(float(np.std(seed_aps)), 6),
                "ap_best": round(float(np.max(seed_aps)), 6),
                "ap_worst": round(float(np.min(seed_aps)), 6),
                "ap_per_seed": [round(v, 6) for v in seed_aps],
                "n_seeds": len(seed_aps),
                "per_pivot": per_pivot,
                "t_train_mean": round(float(np.mean(train_times)), 1),
            }
            results.append(entry)
            done.add(wid)
            n_ok += 1
            print(f"  => mean AP={entry['ap_mean']:.6f}", flush=True)
            save_json(results, results_path)
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}", flush=True)
            traceback.print_exc()
            n_fail += 1
        finally:
            del g
            cleanup()

    results.sort(key=lambda r: r["window"])
    aps = [r["ap_mean"] for r in results]
    summary = {
        "method": "AsymGAD",
        "n_windows": len(results),
        "n_seeds": len(seeds),
        "epochs": epochs,
        "config": training,
        "stats": summarize_window_aps(aps) if aps else {},
        "elapsed_s": round(time.time() - t_total, 1),
    }
    save_json(summary, summary_path)
    print(f"\nSaved: {results_path}")
    print(f"Summary: {summary['stats']}")
    print("Done.")


if __name__ == "__main__":
    main()
