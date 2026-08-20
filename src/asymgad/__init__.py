"""AsymGAD: Label-Free Lateral-Movement Pivot Ranking.

AsymGAD is a fully label-free framework that turns a raw enterprise
authentication stream into a ranked list of pivot candidates.  It
combines Label-Independent Adaptive Window Construction, a
relation-conditioned directed GNN encoder, self-supervised structural
expectation learning, and a dual-branch structural ranking score.

Reference:
    "AsymGAD: Label-Free Lateral-Movement Pivot Ranking via Asymmetric
     Graph Anomaly Detection"
"""

from .window import (
    WindowInfo,
    build_adaptive_windows_streaming,
    stream_events,
)
from .graph_data import WindowGraphData, build_window_graph, directed_pagerank
from .encoder import EnhancedEncoder, MultiHeadEdgeGate
from .heads import (
    StructuralPredictor,
    compute_acp_scores,
    compute_ief_rarity_scores,
    compute_surprise_scores,
    compute_deg_targets,
    compute_pr_targets_processed,
)
from .train import train_asymgad
from .metrics import (
    average_precision,
    node_ranks,
    hit_at,
)

__all__ = [
    "WindowInfo",
    "build_adaptive_windows_streaming",
    "stream_events",
    "WindowGraphData",
    "build_window_graph",
    "directed_pagerank",
    "EnhancedEncoder",
    "MultiHeadEdgeGate",
    "StructuralPredictor",
    "compute_acp_scores",
    "compute_ief_rarity_scores",
    "compute_surprise_scores",
    "compute_deg_targets",
    "compute_pr_targets_processed",
    "train_asymgad",
    "average_precision",
    "node_ranks",
    "hit_at",
]
