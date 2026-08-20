"""Scoring heads used by AsymGAD.

The final algorithm combines two complementary branches:

  * Structural Asymmetry  A(v): window-local percentile of the raw
    out-degree / (in-degree + 1) fanout ratio (non-learned).
  * Structural Residual   R(v): max of the degree residual percentile
    and the directed-centrality (PageRank) residual percentile, computed
    from a relation-conditioned encoder that predicts node roles
    (learned).

An internal rarity head (inverse-frequency edge rarity, Top-K pooled) is
kept only for the ablation study in the paper; it is not part of the
final ranking score.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Non-learned heads
# ----------------------------------------------------------------------


def compute_acp_scores(out_deg: np.ndarray, in_deg: np.ndarray) -> np.ndarray:
    """Structural asymmetry percentile in [0, 1].

    ACP(v) = out_deg(v) / (in_deg(v) + 1), rank-transformed within the
    current window.  Higher out-degree relative to in-degree yields a
    higher score (scanner-type pivots).
    """
    acp = out_deg / (in_deg + 1.0)
    order = np.argsort(acp)
    pct = np.zeros(len(acp), dtype=np.float64)
    for i, idx in enumerate(order):
        pct[idx] = (i + 1) / len(acp)
    return pct


def compute_ief_rarity_scores(
    ef: np.ndarray,
    src: np.ndarray,
    N: int,
    K: int = 1,
) -> np.ndarray:
    """Inverse-frequency edge rarity, Top-K pooled per source, percentile.

    ef: (E, F) edge features (binary protocol/port dimensions; the last
    log-count column is excluded).

    Used only in the paper's ablation study; the final score does not
    include this head (fusion weight 0).
    """
    n_binary = max(1, ef.shape[1] - 1)
    ef_bin = ef[:, :n_binary]
    feat_freq = ef_bin.mean(axis=0)
    feat_freq = np.maximum(feat_freq, 1e-10)
    edge_rarity = (ef_bin * -np.log(feat_freq)).sum(axis=1)

    buckets = defaultdict(list)
    for i in range(len(src)):
        buckets[int(src[i])].append(float(edge_rarity[i]))
    node_scores = np.zeros(N, dtype=np.float64)
    for v, vals in buckets.items():
        topk = sorted(vals, reverse=True)[:min(K, len(vals))]
        node_scores[v] = np.mean(topk) if topk else 0.0

    order = np.argsort(node_scores)
    pct = np.zeros(N, dtype=np.float64)
    for i, idx in enumerate(order):
        pct[idx] = (i + 1) / N
    return pct


# ----------------------------------------------------------------------
# Learned structural predictor
# ----------------------------------------------------------------------


class StructuralPredictor(nn.Module):
    """Predicts node structural properties from GNN embeddings.

    Input:  z_v in R^embed_dim (64 from the EnhancedEncoder).
    Output: predicted log(1 + d_in), log(1 + d_out), and PageRank.

    The prediction error is the learned anomaly signal: nodes whose
    observed role differs from the role predicted for comparable
    machines receive a high structural residual score.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.deg_head = nn.Linear(hidden, 2)
        self.pr_head = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> tuple:
        """Returns (deg_pred, pr_pred).

        deg_pred: (N, 2) - [log(1 + d_in), log(1 + d_out)]
        pr_pred:  (N,)   - predicted PageRank
        """
        h = self.shared(z)
        deg_pred = self.deg_head(h)
        pr_pred = self.pr_head(h).squeeze(-1)
        return deg_pred, pr_pred


def _percentile(arr: np.ndarray) -> np.ndarray:
    """Rank-transform an array to (1..N)/N percentiles in [0, 1]."""
    order = np.argsort(arr)
    N = len(arr)
    ranks = np.empty(N, dtype=np.float64)
    for i, idx in enumerate(order):
        ranks[idx] = (i + 1) / N
    return ranks


def compute_surprise_scores(
    z_np: np.ndarray,
    out_deg: np.ndarray,
    in_deg: np.ndarray,
    pr_target: np.ndarray,
    predictor: StructuralPredictor,
    device: torch.device,
) -> Dict:
    """Compute structural surprise (residual) scores for all nodes.

    Returns a dict with:
      delta_deg : (N,) degree residual percentile [0, 1]
      delta_pr  : (N,) PageRank residual percentile [0, 1]
      delta     : (N,) max(delta_deg, delta_pr) [0, 1]
      deg_pred  : (N, 2) raw degree predictions
      pr_pred   : (N,) raw PageRank predictions
    """
    predictor.eval()

    with torch.no_grad():
        z_t = torch.tensor(z_np, dtype=torch.float32, device=device)
        deg_pred_t, pr_pred_t = predictor(z_t)
        deg_pred = deg_pred_t.cpu().numpy()
        pr_pred = pr_pred_t.cpu().numpy()

    d_in_log = np.log1p(in_deg)
    d_out_log = np.log1p(out_deg)

    deg_error = np.sqrt(
        (deg_pred[:, 0] - d_in_log) ** 2 +
        (deg_pred[:, 1] - d_out_log) ** 2
    )
    delta_deg = _percentile(deg_error.astype(np.float64))

    pr_error = np.abs(pr_pred - pr_target)
    delta_pr = _percentile(pr_error.astype(np.float64))

    delta = np.maximum(delta_deg, delta_pr)

    return {
        "delta_deg": delta_deg,
        "delta_pr": delta_pr,
        "delta": delta,
        "deg_pred": deg_pred,
        "pr_pred": pr_pred,
    }


def compute_deg_targets(out_deg: np.ndarray, in_deg: np.ndarray) -> np.ndarray:
    """Build normalized degree regression targets: (N, 2)."""
    d_in_log = np.log1p(in_deg).astype(np.float32)
    d_out_log = np.log1p(out_deg).astype(np.float32)
    targets = np.column_stack([d_in_log, d_out_log])

    mean = targets.mean(axis=0, keepdims=True)
    std = targets.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    targets = (targets - mean) / std

    return targets


def compute_pr_targets_processed(pr_target: np.ndarray) -> np.ndarray:
    """Normalize PageRank targets for stable training."""
    pr = pr_target.astype(np.float32)
    pr = (pr - pr.mean()) / max(pr.std(), 1e-8)
    return pr
