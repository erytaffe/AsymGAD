"""Enhanced 3-layer directed GNN encoder for AsymGAD.

Features:
  - 3 layers (32->64->64) with LayerNorm + residual
  - Multi-head edge-conditioned gate (3 heads)
  - Laplacian Positional Encoding (8-dim)
  - DropEdge regularization during training
  - Hub Penalty γ=2.0 (retained)
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, scipy.sparse as sp


def compute_lapeig_fast(ei_directed: torch.Tensor, N: int, pe_dim: int = 8) -> np.ndarray:
    """Compute Laplacian PE for directed graph (symmetrized)."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    rev = ei_directed.flip(0)
    ei = torch.cat([ei_directed, rev], dim=1)
    sl = torch.arange(N, device=ei.device).repeat(2, 1)
    ei = torch.cat([ei, sl], dim=1)
    ei = torch.unique(ei, dim=1)

    src = ei[0].cpu().numpy().astype(np.int32)
    dst = ei[1].cpu().numpy().astype(np.int32)
    data = np.ones(len(src), dtype=np.float32)
    A = sp.coo_matrix((data, (src, dst)), shape=(N, N)).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, deg**(-0.5), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L_sym = sp.eye(N, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt

    k = min(pe_dim + 1, N - 2)
    evals, evecs = sp.linalg.eigsh(L_sym, k=k, which='SM', tol=1e-3, maxiter=300)
    idx = np.argsort(evals)
    evecs = evecs[:, idx]
    pe = evecs[:, 1:pe_dim + 1].astype(np.float32)
    pe = (pe - pe.mean(axis=0, keepdims=True)) / (pe.std(axis=0, keepdims=True).clip(min=1e-10))
    return pe


class MultiHeadEdgeGate(nn.Module):
    """Multi-head edge-conditioned gate: 3 independent tanh MLPs -> concatenated."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 32, n_heads: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            ) for _ in range(n_heads)
        ])

    def forward(self, x_src, x_dst, edge_feat):
        """Returns (E, n_heads) - per-head gate values."""
        edge_repr = torch.cat([x_src, x_dst, edge_feat], dim=-1)
        return torch.cat([mlp(edge_repr) for mlp in self.mlps], dim=-1)


class EnhancedEncoder(nn.Module):
    """3-layer directed GNN with multi-head gates, LayerNorm, residual, DropEdge."""

    def __init__(self, node_dim: int = 8, edge_dim: int = 37, pe_dim: int = 8,
                 hidden_dims: list = [32, 64, 64], n_gate_heads: int = 3,
                 gamma: float = 2.0, dropout: float = 0.3, dropedge: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.dropedge = dropedge
        self.n_layers = len(hidden_dims)
        in_dim = node_dim + pe_dim  # structural + Laplacian PE

        self.gates = nn.ModuleList()
        self.W_self = nn.ModuleList()
        self.W_neigh = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()  # for dim change

        prev_dim = in_dim
        for i, hd in enumerate(hidden_dims):
            self.gates.append(MultiHeadEdgeGate(prev_dim, edge_dim, hd, n_gate_heads))
            self.W_self.append(nn.Linear(prev_dim, hd, bias=False))
            self.W_neigh.append(nn.Linear(prev_dim, hd, bias=False))
            self.norms.append(nn.LayerNorm(hd))
 # Residual projection if dims change
            if prev_dim != hd:
                self.residuals.append(nn.Linear(prev_dim, hd, bias=False))
            else:
                self.residuals.append(nn.Identity())
            prev_dim = hd

        self.drop = nn.Dropout(dropout)
        self.embed_dim = hidden_dims[-1]

    def hub_penalty(self, edge_index, in_degree):
        _, dst = edge_index
        return torch.exp(-self.gamma * torch.log1p(in_degree[dst].float()))

    @staticmethod
    def _aggregate_multihead(feat, src, dst, weights, num_nodes, n_heads):
        """Aggregate with (E, n_heads) weights -> (N, hidden*n_heads) then mean across heads."""
        E = len(src)
 # weights: (E, n_heads)
        feat_expanded = feat[src].unsqueeze(1)  # (E, 1, D)
        wf = feat_expanded * weights.unsqueeze(-1)  # (E, n_heads, D)
        out = torch.zeros(num_nodes, n_heads, feat.size(1), device=feat.device)
        cnt = torch.zeros(num_nodes, n_heads, device=feat.device)
        for h in range(n_heads):
            out[:, h, :].index_add_(0, dst, wf[:, h, :])
            cnt[:, h].index_add_(0, dst, weights[:, h].abs())
 # Mean across heads
        out = out.sum(dim=1) / cnt.sum(dim=1).clamp(min=1).unsqueeze(1)
        return out

    def forward(self, x, edge_index, edge_feat, num_nodes, in_degree, training=True):
        src, dst = edge_index
        alpha = self.hub_penalty(edge_index, in_degree)  # (E,)

 # DropEdge: randomly drop edges during training
        if training and self.dropedge > 0:
            mask = torch.rand(len(src), device=edge_index.device) > self.dropedge
            src = src[mask]; dst = dst[mask]
            edge_feat_curr = edge_feat[mask]
            alpha = alpha[mask]
        else:
            edge_feat_curr = edge_feat

        for i in range(self.n_layers):
 # Multi-head gate
            beta = self.gates[i](x[src], x[dst], edge_feat_curr)  # (E', n_heads)
            weights = alpha.unsqueeze(1) * beta  # (E', n_heads)

 # Aggregate with multi-head weights
            h_neigh = self._aggregate_multihead(x, src, dst, weights, num_nodes, len(self.gates[i].mlps))

 # Transform
            h_new = self.norms[i](self.W_self[i](x) + self.W_neigh[i](h_neigh))
            h_new = F.relu(self.drop(h_new))

 # Residual
            x = self.residuals[i](x) + h_new

        return x
