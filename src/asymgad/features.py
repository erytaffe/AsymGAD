"""Time-augmented features for AsymGAD.

- Peak ACP (per-slice maximum of d_out/(d_in+1))
- Edge time targets: occupancy vector + micro-count per slice
- Category mask: skip categories with insufficient window samples
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .micro_slice import MicroSliceResult


# ---
# Peak ACP
# ---

def compute_peak_acp(
    msr: MicroSliceResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-node peak ACP and full-window ACP.

    For each node, computes ACP_j = d_out_j / (d_in_j + 1) in each slice j,
    then takes the maximum across slices.

    Parameters
    ----------
    msr : MicroSliceResult
        Micro-slice data for one window.

    Returns
    -------
    acp_full  : (N,) float64 - full-window ACP
    acp_peak  : (N,) float64 - peak ACP across slices
    """
    N = msr.N
    J = msr.J

 # Full-window ACP from aggregate degrees
    out_full = np.zeros(N, dtype=np.float64)
    in_full = np.zeros(N, dtype=np.float64)
    for s in msr.slices:
        out_full += s.out_deg.astype(np.float64)
        in_full += s.in_deg.astype(np.float64)
    acp_full = out_full / (in_full + 1.0)

 # Per-slice ACP -> peak
    acp_per_slice = np.zeros((J, N), dtype=np.float64)
    for j, s in enumerate(msr.slices):
        out_j = s.out_deg.astype(np.float64)
        in_j = s.in_deg.astype(np.float64)
        acp_per_slice[j] = out_j / (in_j + 1.0)

    acp_peak = acp_per_slice.max(axis=0)

    return acp_full, acp_peak


# ---
# Edge time targets
# ---

def build_edge_time_targets(
    msr: MicroSliceResult,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """Build per-edge occupancy vector and micro-count vector.

    For each unique edge (u,v) in the window, computes:
      o_uv[j] = 1 if edge appears in slice j, else 0   (J,)
      q_uv[j] = log(1 + count_in_slice_j)                (J,)

    Parameters
    ----------
    msr : MicroSliceResult

    Returns
    -------
    dict mapping (u,v) -> (o_vec, q_vec) where each is a float32 array of length J.
    """
    J = msr.J
    result: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

 # Collect all unique pairs across all slices
    all_pairs: set = set()
    for ec in msr.edge_counts_per_slice:
        all_pairs.update(ec.keys())

    for uv in all_pairs:
        o_vec = np.zeros(J, dtype=np.float32)
        q_vec = np.zeros(J, dtype=np.float32)
        for j in range(J):
            count = msr.edge_counts_per_slice[j].get(uv, 0)
            o_vec[j] = 1.0 if count > 0 else 0.0
            q_vec[j] = np.log1p(count)
        result[uv] = (o_vec, q_vec)

    return result


# ---
# Category mask
# ---

def compute_category_mask(
    ef: np.ndarray,
    n_min_cat: int = 5,
) -> np.ndarray:
    """Compute a binary mask for edge-feature categories.

    A category is active if at least n_min_cat positive samples exist
    in the current window.  Categories below this threshold have their
    loss contribution zeroed to avoid unstable gradients.

    Parameters
    ----------
    ef : (E, F) float32 - binary edge-feature matrix for the window.
    n_min_cat : int - minimum positive count for a category to be active.

    Returns
    -------
    mask : (F,) float32 - 1.0 for active categories, 0.0 otherwise.
    """
    F = ef.shape[1]
    counts = ef.sum(axis=0)
    mask = (counts >= n_min_cat).astype(np.float32)
    return mask


def count_active_categories(
    ef: np.ndarray,
    n_min_cat: int = 5,
) -> Tuple[int, int, float]:
    """Return (n_active, n_total, active_ratio) for diagnostics."""
    F = ef.shape[1]
    counts = ef.sum(axis=0)
    n_active = int((counts >= n_min_cat).sum())
    return n_active, F, n_active / max(F, 1)


# ---
# Structural score (time-enhanced)
# ---

def structural_scores(
    acp_full: np.ndarray,
    acp_peak: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-enhanced structural anomaly scores.

    S_alpha(v) = max(P_full(v), P_peak(v))

    where P_* is the empirical percentile within the current window.

    Parameters
    ----------
    acp_full : (N,) - full-window ACP per node.
    acp_peak : (N,) - peak per-slice ACP per node.

    Returns
    -------
    p_full : (N,) - percentile of full-window ACP in [0, 1].
    p_peak : (N,) - percentile of peak ACP in [0, 1].
    s_alpha : (N,) - max(p_full, p_peak).
    """
    N = len(acp_full)

    def _percentile(arr: np.ndarray) -> np.ndarray:
        """Empirical percentile (fraction of values <= each value)."""
        order = np.argsort(arr)
        ranks = np.empty(N, dtype=np.float64)
        for i, idx in enumerate(order):
            ranks[idx] = (i + 1) / N
        return ranks

    p_full = _percentile(acp_full)
    p_peak = _percentile(acp_peak)
    s_alpha = np.maximum(p_full, p_peak)

    return p_full, p_peak, s_alpha
