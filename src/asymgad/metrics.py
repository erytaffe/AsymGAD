"""Evaluation metrics used in the paper.

AP is defined only when the candidate set contains a positive instance,
so all ranking evaluation is performed on the attack-containing windows.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision over a ranked list (higher score = higher rank)."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = int(sorted_labels.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(sorted_labels).astype(np.float64)
    precision = tp / np.arange(1, len(sorted_labels) + 1)
    return float(sum(precision[i] for i in range(len(sorted_labels))
                     if sorted_labels[i]) / n_pos)


def node_ranks(
    scores: np.ndarray,
    pivot_ids: Sequence[int],
    pivot_names: Sequence[str],
) -> Dict[str, int]:
    """Return {pivot_name: rank} (1-based, descending score)."""
    order = np.argsort(-scores)
    rank_map = {int(n): int(r) + 1 for r, n in enumerate(order)}
    return {
        pname: rank_map.get(int(pid), len(scores))
        for pid, pname in zip(pivot_ids, pivot_names)
    }


def hit_at(
    scores: np.ndarray,
    labels: np.ndarray,
    ks: Sequence[int] = (10, 50, 100),
) -> Dict[int, bool]:
    """Whether at least one positive appears in the top-K for each K."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    out = {}
    for k in ks:
        out[int(k)] = bool(sorted_labels[:k].sum() > 0)
    return out


def summarize_window_aps(aps: Sequence[float]) -> Dict[str, float]:
    """Aggregate per-window AP values into the paper's summary statistics."""
    arr = np.asarray(aps, dtype=np.float64)
    return {
        "n_windows": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "best": float(arr.max()),
        "worst": float(arr.min()),
        "n_strong": int((arr >= 0.01).sum()),
        "n_medium": int(((arr >= 0.001) & (arr < 0.01)).sum()),
        "n_weak": int((arr < 0.001).sum()),
    }


def paired_difference_stats(
    method_aps: np.ndarray,
    baseline_aps: np.ndarray,
    n_bootstrap: int = 20_000,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """Percentile-bootstrap CI for the mean paired difference.

    Mirrors the paper's statistical protocol: a nonparametric percentile
    bootstrap with 20,000 window resamples gives the 95% confidence
    interval for the mean difference
    Delta_k = AP_{k, method} - AP_{k, baseline}.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    diff = np.asarray(method_aps, dtype=np.float64) - np.asarray(baseline_aps, dtype=np.float64)
    n = len(diff)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
