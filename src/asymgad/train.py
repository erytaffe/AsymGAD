"""AsymGAD: Structural Residual Prediction.

The GNN predicts node structural properties from graph context; anomaly
scores are positive residuals (actual property - predicted property).

Architecture:
  α(v) = ACP percentile                 (non-learned, structural asymmetry)
  β(v) = IEF rarity Top-1 percentile    (non-learned, semantic rarity, K=1)
  δ(v) = max(δ_deg, δ_pr) percentile   (learned, structural surprise)
  S(v) = w_α-α + w_β-β + w_δ-δ        (weighted fusion)

Training (multi-task):
  L_total = L_rec(IEF-BCE) + λ_deg-L_deg(MSE) + λ_pr-L_pr(MSE)
  where L_deg predicts log-degrees, L_pr predicts the directed PageRank
  target computed by power iteration on the window graph.

No Laplacian PE by default (ablation: +77% without PE).
"""
from __future__ import annotations
import time, gc, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict

from .encoder import EnhancedEncoder, compute_lapeig_fast
from .heads import (
    StructuralPredictor,
    compute_acp_scores,
    compute_ief_rarity_scores,
    compute_surprise_scores,
    compute_deg_targets,
    compute_pr_targets_processed,
)
from .graph_data import WindowGraphData


def _percentile(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    N = len(arr)
    ranks = np.empty(N, dtype=np.float64)
    for i, idx in enumerate(order):
        ranks[idx] = (i + 1) / N
    return ranks


def train_asymgad(
    g: WindowGraphData,
    epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 1e-5,
    gamma: float = 2.0,
    use_pe: bool = False,
    use_ief: bool = True,
    lambda_deg: float = 0.5,
    lambda_pr: float = 0.3,
    fusion_alpha: float = 0.45,
    fusion_beta: float = 0.30,
    fusion_delta: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Train AsymGAD on one observation window.

    Three signals:
      α: raw ACP percentile (non-learned, structural asymmetry)
      β: IEF edge rarity Top-1 (non-learned, semantic rarity, K=1)
      δ: structural surprise (learned - GNN predicts degrees & PageRank)

    Fusion: weighted sum with configurable weights.
    Training: L_rec (IEF-BCE) + λ_deg-L_deg (MSE) + λ_pr-L_pr (MSE)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = g.N
    X_arr = g.X
    src_arr = g.src
    dst_arr = g.dst
    ef_arr = g.ef

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 # -- Edge tensors ---
    ei_dir_np = np.vstack([src_arr.astype(np.int64), dst_arr.astype(np.int64)])
    ei_dir_t = torch.tensor(ei_dir_np, dtype=torch.long, device=device)

 # -- Laplacian PE (optional, OFF by default) ---
    if use_pe:
        pe_np = compute_lapeig_fast(ei_dir_t, N, pe_dim=8)
        pe_dim_actual = 8
    else:
        pe_np = np.zeros((N, 0), dtype=np.float32)
        pe_dim_actual = 0

    node_dim = 8
    X_aug = np.concatenate([X_arr, pe_np], axis=1)
    xt = torch.tensor(X_aug, dtype=torch.float32, device=device)

 # -- Edge features ---
    eft = torch.tensor(ef_arr, dtype=torch.float32, device=device)
    F_dim = ef_arr.shape[1]
    n_binary = max(1, F_dim - 1)

    counts = ef_arr[:, :n_binary].sum(axis=0)
    counts = np.maximum(counts, 1)
    if use_ief:
        freq = counts / max(g.E, 1)
        ief_w = (1.0 / np.sqrt(freq)).astype(np.float32)
        ief_w /= ief_w.mean()
    else:
        ief_w = np.ones(n_binary, dtype=np.float32)
    fwt = torch.tensor(ief_w, dtype=torch.float32, device=device)
    cm = np.ones(n_binary, dtype=np.float32)
    cm[counts < 5] = 0.0
    cmt = torch.tensor(cm, dtype=torch.float32, device=device)

 # -- In-degree ---
    in_deg_t = torch.zeros(N, device=device)
    in_deg_t.index_add_(0, ei_dir_t[1], torch.ones(len(ei_dir_t[1]), device=device))

 # -- Node degrees (numpy) ---
    out_d = np.bincount(src_arr, minlength=N).astype(np.float64)
    in_d = np.bincount(dst_arr, minlength=N).astype(np.float64)

 # -- Non-learned head scores ---
    S_alpha = compute_acp_scores(out_d, in_d)                      # (N,) [0,1]
    S_beta = compute_ief_rarity_scores(ef_arr, src_arr, N, K=1)    # (N,) [0,1] K=1

 # -- Structural targets (for training) ---
    deg_targets_np = compute_deg_targets(out_d, in_d)           # (N, 2) normalized
    pr_targets_np = compute_pr_targets_processed(g.pr_target)   # (N,)  normalized
    deg_targets_t = torch.tensor(deg_targets_np, dtype=torch.float32, device=device)
    pr_targets_t = torch.tensor(pr_targets_np, dtype=torch.float32, device=device)

 # -- Model ---
    encoder = EnhancedEncoder(
        node_dim=node_dim, edge_dim=F_dim, pe_dim=pe_dim_actual,
        hidden_dims=[32, 64, 64], n_gate_heads=3,
        gamma=gamma, dropout=0.3, dropedge=0.1,
    ).to(device)

 # Edge decoder: [z_src z_dst] -> edge features
    edge_decoder = nn.Sequential(
        nn.Linear(128, 64),  # za(32)*2 + zb(32)*2 = 128
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, n_binary),
    ).to(device)

 # Structural predictor: z_v -> [log(1+d_in), log(1+d_out)], pr
    struct_pred = StructuralPredictor(embed_dim=64, hidden=32).to(device)

    params = (
        list(encoder.parameters()) +
        list(edge_decoder.parameters()) +
        list(struct_pred.parameters())
    )
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

 # -- Training ---
    t0 = time.time()
    src_t = ei_dir_t[0]
    dst_t = ei_dir_t[1]

    for ep in range(1, epochs + 1):
        encoder.train()
        edge_decoder.train()
        struct_pred.train()
        opt.zero_grad()

 # GNN forward
        z = encoder(xt, ei_dir_t, eft, N, in_deg_t, training=True)

 # -- Edge reconstruction loss --
        edge_h = torch.cat([z[src_t], z[dst_t]], dim=-1)  # (E, 128)
        e_hat = torch.sigmoid(edge_decoder(edge_h))
        bce = F.binary_cross_entropy(e_hat, eft[:, :n_binary], reduction="none")
        L_rec = (bce * fwt.unsqueeze(0) * cmt.unsqueeze(0)).sum(dim=1).mean()

 # -- Structural prediction losses --
        deg_pred, pr_pred = struct_pred(z)
        L_deg = F.mse_loss(deg_pred, deg_targets_t)
        L_pr = F.mse_loss(pr_pred, pr_targets_t)

 # -- Total loss --
        L_total = L_rec + lambda_deg * L_deg + lambda_pr * L_pr

        L_total.backward()
        opt.step()

        if verbose and ep % 50 == 0:
            print(f"    Epoch {ep:3d}/{epochs}  rec={L_rec.item():.4f}  "
                  f"deg={L_deg.item():.4f}  pr={L_pr.item():.4f}", flush=True)

    train_time = time.time() - t0

 # -- Inference ---
    encoder.eval()
    struct_pred.eval()

    with torch.no_grad():
        z = encoder(xt, ei_dir_t, eft, N, in_deg_t, training=False)
        z_np = z.cpu().numpy()

 # Compute structural surprise δ
    surprise = compute_surprise_scores(
        z_np, out_d, in_d, g.pr_target,
        struct_pred, device,
    )
    S_delta = surprise["delta"]           # (N,) [0,1]
    S_delta_deg = surprise["delta_deg"]   # (N,) [0,1]
    S_delta_pr = surprise["delta_pr"]     # (N,) [0,1]

 # -- Three-signal weighted fusion ---
    scores = (fusion_alpha * S_alpha +
              fusion_beta * S_beta +
              fusion_delta * S_delta)

    if verbose:
        print(f"    Training: {train_time:.1f}s  Params: {n_params:,}", flush=True)
        print(f"    S_alpha: [{S_alpha.min():.3f}, {S_alpha.max():.3f}]  "
              f"S_beta: [{S_beta.min():.3f}, {S_beta.max():.3f}]  "
              f"S_delta: [{S_delta.min():.3f}, {S_delta.max():.3f}]", flush=True)

    return {
        "scores": scores.astype(np.float64),
        "score_alpha": S_alpha,
        "score_beta": S_beta,
        "score_delta": S_delta,
        "score_delta_deg": S_delta_deg,
        "score_delta_pr": S_delta_pr,
        "deg_pred": surprise["deg_pred"],
        "pr_pred": surprise["pr_pred"],
        "z_np": z_np.astype(np.float32),  # (N, 64) GNN embeddings
        "train_s": round(train_time, 2),
        "n_params": n_params,
        "seed": seed,
    }
