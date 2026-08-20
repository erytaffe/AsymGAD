"""Ranking stability on the attack-unlabeled windows.

Applies the paper's configuration (single seed 42) to the windows that
contain no recorded attack (472 windows on the full LANL stream) and
computes the stability statistics reported in the paper: top-1 host
frequency, top-50 overlap between adjacent windows, persistent
top-ranked hosts, and the ranks of the four known pivots during quiet
periods.

Usage:
    python scripts/run_nonattack.py [--config configs/lanl.json]
                                    [--limit 4] [--epochs 100]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common import add_common_args, apply_data_roots, load_config, windows_path

args_parser = argparse.ArgumentParser(description=__doc__)
add_common_args(args_parser)
args_parser.add_argument("--top-k-save", type=int, default=200)
args = args_parser.parse_args()

cfg = load_config(args.config)
apply_data_roots(cfg)

from asymgad import WindowInfo, build_window_graph, train_asymgad
from asymgad.paths import OUTPUT_ROOT


PIVOTS = ["C17693", "C19932", "C22409", "C18025"]
SEED = 42


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    training = cfg.get("training", {})
    epochs = args.epochs or int(training.get("epochs", 100))
    wpath = windows_path(cfg)
    wdata = load_json(wpath)
    nonattack = [w for w in wdata["windows"] if not w.get("has_attack", False)]
    if args.limit and args.limit > 0:
        nonattack = nonattack[: args.limit]
    print(f"Non-attack windows: {len(nonattack)}  Seed: {SEED}  Epochs: {epochs}")

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT / "nonattack"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    results = []
    t_total = time.time()
    for i, w in enumerate(nonattack, 1):
        wid = w["idx"]
        wi = WindowInfo(
            idx=wid, ts_start=w["ts_start"], ts_end=w["ts_end"],
            duration_sec=w["dur_h"] * 3600, M=w["M"], E=w["E"],
            V_src=w["V_src"], V_dst=w["V_dst"], V_total=w["V_total"],
            has_attack=False,
        )
        try:
            g = build_window_graph(wi, J=4, verbose=False)
            if g is None or g.N == 0:
                continue
            r = train_asymgad(
                g, epochs=epochs, seed=SEED, verbose=False,
                use_pe=bool(training.get("use_pe", False)),
                use_ief=bool(training.get("use_ief", True)),
                lambda_deg=float(training.get("lambda_deg", 0.5)),
                lambda_pr=float(training.get("lambda_pr", 0.3)),
                fusion_alpha=float(training.get("fusion_alpha", 0.60)),
                fusion_beta=float(training.get("fusion_beta", 0.0)),
                fusion_delta=float(training.get("fusion_delta", 0.40)),
            )
            scores = r["scores"]
            cand_mask = g.out_deg > 0
            cand_idx = np.where(cand_mask)[0]
            order = cand_idx[np.argsort(-scores[cand_idx])]

            top = []
            for rank, nid in enumerate(order[: args.top_k_save], start=1):
                top.append({
                    "rank": rank,
                    "host": g.nodes[nid],
                    "score": round(float(scores[nid]), 6),
                    "alpha": round(float(r["score_alpha"][nid]), 6),
                    "delta": round(float(r["score_delta"][nid]), 6),
                    "out_deg": int(g.out_deg[nid]),
                    "in_deg": int(g.in_deg[nid]),
                })

            pivot_ranks = {}
            for pid, pname in zip(g.pivot_ids, g.pivot_names):
                pivot_ranks[pname] = int(np.sum(scores > scores[pid]) + 1)

            results.append({
                "window": wid,
                "n_candidates": int(len(cand_idx)),
                "top": top,
                "pivot_ranks": pivot_ranks,
                "top1_host": top[0]["host"] if top else None,
            })
            if i % 20 == 0 or i == len(nonattack):
                print(f"  [{i}/{len(nonattack)}] done "
                      f"({time.time() - t_total:.0f}s)", flush=True)
                save_json(results, results_path)
        except Exception as exc:
            import traceback
            print(f"  Window {wid} FAILED: {exc}", flush=True)
            traceback.print_exc()
        finally:
            del g
            gc.collect()

    save_json(results, results_path)

    # ------------------------------------------------------------------
    # Stability statistics
    # ------------------------------------------------------------------
    top1_counter = Counter(r["top1_host"] for r in results if r["top1_host"])
    n_windows = len(results)

    overlaps = []
    for prev, cur in zip(results[:-1], results[1:]):
        a = {t["host"] for t in prev["top"][:50]}
        b = {t["host"] for t in cur["top"][:50]}
        overlaps.append(jaccard(a, b))

    top50_freq = Counter()
    for r in results:
        for t in r["top"][:50]:
            top50_freq[t["host"]] += 1
    persistent = {h for h, c in top50_freq.items() if c >= 25}

    pivot_ranks_all = {p: [] for p in PIVOTS}
    for r in results:
        for p in PIVOTS:
            if p in r["pivot_ranks"]:
                pivot_ranks_all[p].append(r["pivot_ranks"][p])

    summary = {
        "n_windows": n_windows,
        "top1_host_counts": dict(top1_counter.most_common(10)),
        "n_top1_hosts": len(top1_counter),
        "top50_jaccard_mean": round(float(np.mean(overlaps)), 4) if overlaps else None,
        "top50_jaccard_n_pairs": len(overlaps),
        "n_persistent_top50_hosts": len(persistent),
        "pivot_median_ranks": {
            p: int(np.median(v)) if v else None
            for p, v in pivot_ranks_all.items()
        },
        "pivot_min_ranks": {
            p: int(np.min(v)) if v else None
            for p, v in pivot_ranks_all.items()
        },
        "elapsed_s": round(time.time() - t_total, 1),
    }
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
