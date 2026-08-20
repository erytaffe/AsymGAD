"""Ablation studies from the paper (single seed 42).

Reproduces four ablation tables on the 54 positive windows:

  1. Signal composition (Asymmetry / Residual / Rarity combinations)
  2. Self-supervised auxiliary objectives (reconstruction, degree,
     centrality, unified)
  3. Structural residual definitions (degree / PageRank / max)
  4. Windowing ablation (adaptive vs fixed-event), when the fixed-event
     windows.json is provided with ``--fixed-windows``

Usage:
    python scripts/run_ablation.py [--config configs/lanl.json]
                                   [--limit 4] [--epochs 100]
                                   [--fixed-windows output/fixed_event_2M/windows.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from common import add_common_args, apply_data_roots, load_config, windows_path

args_parser = argparse.ArgumentParser(description=__doc__)
add_common_args(args_parser)
args_parser.add_argument("--fixed-windows", default=None,
                         help="Fixed-event windows.json for the windowing ablation.")
args = args_parser.parse_args()

cfg = load_config(args.config)
apply_data_roots(cfg)

from asymgad import WindowInfo, build_window_graph, train_asymgad
from asymgad.metrics import average_precision
from asymgad.paths import OUTPUT_ROOT


SEED = 42

# ----------------------------------------------------------------------
# Ablation configurations (paper Tables 5-8)
# ----------------------------------------------------------------------

SIGNAL_ABLATION = {
    "A+R":           {"fusion_alpha": 0.60, "fusion_beta": 0.00, "fusion_delta": 0.40, "delta_mode": "both"},
    "R":             {"fusion_alpha": 0.00, "fusion_beta": 0.00, "fusion_delta": 1.00, "delta_mode": "both"},
    "A":             {"fusion_alpha": 1.00, "fusion_beta": 0.00, "fusion_delta": 0.00, "delta_mode": "both"},
    "A+Rarity+R":    {"fusion_alpha": 0.45, "fusion_beta": 0.30, "fusion_delta": 0.25, "delta_mode": "both"},
    "Rarity+R":      {"fusion_alpha": 0.00, "fusion_beta": 0.55, "fusion_delta": 0.45, "delta_mode": "both"},
    "A+Rarity":      {"fusion_alpha": 0.60, "fusion_beta": 0.40, "fusion_delta": 0.00, "delta_mode": "both"},
    "Rarity":        {"fusion_alpha": 0.00, "fusion_beta": 1.00, "fusion_delta": 0.00, "delta_mode": "both"},
}

OBJECTIVE_ABLATION = {
    "rec":       {"lambda_deg": 0.0, "lambda_pr": 0.0},
    "rec+deg":   {"lambda_deg": 0.5, "lambda_pr": 0.0},
    "rec+pr":    {"lambda_deg": 0.0, "lambda_pr": 0.3},
    "unified":   {"lambda_deg": 0.5, "lambda_pr": 0.3},
}

RESIDUAL_ABLATION = {
    "R_deg":  {"delta_mode": "deg"},
    "R_pr":   {"delta_mode": "pr"},
    "R_both": {"delta_mode": "both"},
}


def train_and_fuse(g, training: dict, fusion: dict, epochs: int) -> np.ndarray:
    """Train once and recompute the fused score for an ablation config."""
    r = train_asymgad(
        g, epochs=epochs, seed=SEED, verbose=False,
        use_pe=bool(training.get("use_pe", False)),
        use_ief=bool(training.get("use_ief", True)),
        lambda_deg=float(fusion.get("lambda_deg", training.get("lambda_deg", 0.5))),
        lambda_pr=float(fusion.get("lambda_pr", training.get("lambda_pr", 0.3))),
        fusion_alpha=0.0, fusion_beta=0.0, fusion_delta=0.0,
    )
    delta_mode = fusion.get("delta_mode", "both")
    if delta_mode == "deg":
        s_delta = r["score_delta_deg"]
    elif delta_mode == "pr":
        s_delta = r["score_delta_pr"]
    else:
        s_delta = r["score_delta"]

    wa = float(fusion.get("fusion_alpha", 0.0))
    wb = float(fusion.get("fusion_beta", 0.0))
    wd = float(fusion.get("fusion_delta", 0.0))
    total = wa + wb + wd
    if total > 0:
        wa /= total
        wb /= total
        wd /= total
    return (wa * r["score_alpha"] + wb * r["score_beta"] + wd * s_delta).astype(np.float64)


def load_windows(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        wdata = json.load(f)
    return wdata["windows"]


def run_ablations(configs: dict, windows, training: dict, epochs: int,
                  out_path: Path) -> dict:
    results = {}
    for name, fusion in configs.items():
        aps = []
        for w in windows:
            wi = WindowInfo(
                idx=w["idx"], ts_start=w["ts_start"], ts_end=w["ts_end"],
                duration_sec=w["dur_h"] * 3600, M=w["M"], E=w["E"],
                V_src=w["V_src"], V_dst=w["V_dst"], V_total=w["V_total"],
                has_attack=bool(w.get("has_attack", False)),
                attack_pivots=list(w.get("attack_pivots", [])),
                attack_event_count=int(w.get("attack_event_count", 0)),
            )
            g = build_window_graph(wi, J=4, verbose=False)
            if g is None or g.N == 0:
                continue
            scores = train_and_fuse(g, training, fusion, epochs)
            aps.append(average_precision(scores, g.labels))
            del g
            gc.collect()
        mean_ap = float(np.mean(aps)) if aps else float("nan")
        results[name] = {"mean_ap": round(mean_ap, 4), "n_windows": len(aps),
                         "config": fusion}
        print(f"  {name:<12s}: mAP={mean_ap:.4f}  (n={len(aps)})", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    training = cfg.get("training", {})
    epochs = args.epochs or int(training.get("epochs", 100))
    wpath = windows_path(cfg)
    windows = load_windows(wpath)
    attack_windows = [w for w in windows if w["has_attack"]]
    if args.limit and args.limit > 0:
        attack_windows = attack_windows[: args.limit]
    print(f"Attack windows: {len(attack_windows)}  Seed: {SEED}  Epochs: {epochs}")

    out_dir = OUTPUT_ROOT / "ablation"
    print("[1/3] Signal ablation")
    run_ablations(SIGNAL_ABLATION, attack_windows, training, epochs,
                  out_dir / "signal.json")
    print("[2/3] Objective ablation")
    obj_configs = {
        name: {"lambda_deg": fusion["lambda_deg"], "lambda_pr": fusion["lambda_pr"],
               "fusion_alpha": 0.45, "fusion_beta": 0.30, "fusion_delta": 0.25,
               "delta_mode": "both"}
        for name, fusion in OBJECTIVE_ABLATION.items()
    }
    run_ablations(obj_configs, attack_windows, training, epochs,
                  out_dir / "objective.json")
    print("[3/3] Residual ablation")
    res_configs = {
        name: {"fusion_alpha": 0.0, "fusion_beta": 0.0, "fusion_delta": 1.0,
               "delta_mode": fusion["delta_mode"]}
        for name, fusion in RESIDUAL_ABLATION.items()
    }
    run_ablations(res_configs, attack_windows, training, epochs,
                  out_dir / "residual.json")

    if args.fixed_windows:
        print("[4/4] Windowing ablation (adaptive vs fixed-event)")
        fixed_windows = load_windows(args.fixed_windows)
        fixed_attack = [w for w in fixed_windows if w["has_attack"]]
        if args.limit and args.limit > 0:
            fixed_attack = fixed_attack[: args.limit]
        run_ablations({"A+R": SIGNAL_ABLATION["A+R"]}, fixed_attack, training,
                      epochs, out_dir / "windowing_fixed.json")
    print("Ablation complete.")


if __name__ == "__main__":
    main()
