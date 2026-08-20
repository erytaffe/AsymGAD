"""Window graph data with micro-slice time targets.

Extends the basic graph construction with:
  - Peak ACP per node (max across J micro-slices)
  - Edge occupancy vector o_uv[0:J] - binary, is edge present in slice j?
  - Edge micro-count vector q_uv[0:J] - log(1 + count) in slice j
  - Category mask: which edge-feature dims have enough samples
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from .window import WindowInfo
from .micro_slice import MicroSliceResult, build_micro_slices
from .paths import DATA_ROOT


def directed_pagerank(src: np.ndarray, dst: np.ndarray, n: int,
                      alpha: float = 0.85, tol: float = 1e-6,
                      max_iter: int = 200):
    """Power-iteration directed PageRank on the window graph.

    Matches the paper's current ``p(v)`` auxiliary target: the stationary
    distribution of the directed edge multigraph (unique ordered pairs),
    computed by power iteration with dangling-node redistribution.
    """
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)
    out_deg = np.bincount(src, minlength=n).astype(np.float64)
    dangling = out_deg == 0.0
    weight = 1.0 / np.maximum(out_deg[src], 1.0)
    r = np.full(n, 1.0 / n)
    iterations = 0
    for iteration in range(1, max_iter + 1):
        contrib = alpha * weight * r[src]
        r_new = np.bincount(dst, weights=contrib, minlength=n)
        r_new += ((1.0 - alpha) + alpha * float(r[dangling].sum())) / n
        delta = float(np.abs(r_new - r).sum())
        r = r_new
        iterations = iteration
        if delta < tol:
            break
    return r, iterations


@dataclass
class WindowGraphData:
    """Label-free directed graph for one adaptive window.

    Extends the original GraphData with time-augmented targets.
    """

 # -- Identity ---
    window_idx: int
    ts_start: int
    ts_end: int

 # -- Topology ---
    N: int = 0
    E: int = 0
    src: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    dst: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    nodes: List[str] = field(default_factory=list)
    node_to_id: Dict[str, int] = field(default_factory=dict)

 # -- Node features ---
    X: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

 # -- Edge features ---
    ef: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    feat_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # IEF
    cat_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))       # category mask

 # -- Distillation targets ---
    pr_target: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    acp_full: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    acp_peak: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

 # -- Micro-slice time targets (J-dim per edge) ---
    J: int = 4
    edge_occ: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))   # (E, J) occupancy
    edge_mcount: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32)) # (E, J) micro-count

 # -- Evaluation labels ---
    pivot_ids: List[int] = field(default_factory=list)
    pivot_names: List[str] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

 # -- Cached ---
    _out_deg: Optional[np.ndarray] = field(default=None, repr=False)
    _in_deg: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def F(self) -> int:
        return self.ef.shape[1]

    @property
    def out_deg(self) -> np.ndarray:
        if self._out_deg is None:
            self._out_deg = np.bincount(self.src, minlength=self.N).astype(np.float64)
        return self._out_deg

    @property
    def in_deg(self) -> np.ndarray:
        if self._in_deg is None:
            self._in_deg = np.bincount(self.dst, minlength=self.N).astype(np.float64)
        return self._in_deg

    @property
    def n_pivots(self) -> int:
        return len(self.pivot_ids)


# ---
# Rich event loader (with protocol/status)
# ---

def _load_window_events_rich(
    ts_start: int,
    ts_end: int,
) -> List[dict]:
    """Load auth events within [ts_start, ts_end) WITH protocol details.

    Uses the global file index from micro_slice for fast file selection.
    """
    import json as _json
    from .micro_slice import _file_time_cache, _build_file_index
    from pathlib import Path as _Path

    global _file_time_cache
    if _file_time_cache is None:
        _file_time_cache = _build_file_index(("AUTH",))

    cat_dir = DATA_ROOT / "cleaned" / "auth"
    if not cat_dir.exists():
        return []

    events: List[dict] = []
    for fp, f_start, f_end in _file_time_cache.get("AUTH", []):
        if f_end < ts_start or f_start >= ts_end:
            continue

        t = pq.read_table(str(cat_dir / fp))
        ts_col = t.column("timestamp")
        src_col = t.column("source_entity")
        dst_col = t.column("destination_entity")
        detail_col = t.column("event_detail")

        for i in range(t.num_rows):
            ts = int(ts_col[i].as_py())
            if ts < ts_start or ts >= ts_end:
                continue
            try:
                detail = _json.loads(detail_col[i].as_py())
            except Exception:
                detail = {}
            events.append({
                "ts": ts,
                "src": src_col[i].as_py(),
                "dst": dst_col[i].as_py(),
                "status": detail.get("status", "?"),
                "protocol": detail.get("protocol", "?"),
                "action": detail.get("action_category", "?"),
            })

    return events


# ---
# Port vocabulary (loaded once)
# ---

_port_vocab_cache: Optional[Dict] = None


def _load_port_vocab() -> Dict:
    """Load the 30-port vocabulary from the pre-built feature_names.json."""
    global _port_vocab_cache
    if _port_vocab_cache is not None:
        return _port_vocab_cache

    vocab_path = DATA_ROOT / "graph" / "feature_names.json"
    if vocab_path.exists():
        import json as _json
        with open(vocab_path) as f:
            _port_vocab_cache = _json.load(f)
        return _port_vocab_cache

 # Fallback: build from top-30 known ports
    _port_vocab_cache = {
        "auth_features": ["is_auth", "status_success", "proto_NTLM",
                          "proto_Kerberos", "proto_Negotiate", "proto_unknown"],
        "port_features": [f"port_{p}" for p in [
            "445", "389", "80", "88", "139", "135", "443", "22", "1433", "2049",
            "111", "161", "7002", "6002", "137", "3306", "1241", "8080", "8081", "1094",
            "427", "53", "2989", "2432", "1300", "3493", "2000", "1434", "1109", "2010"]],
    }
    return _port_vocab_cache


# ---
# Flow event loader (port features per edge)
# ---

def _load_window_flows(
    ts_start: int,
    ts_end: int,
) -> Dict[Tuple[str, str], set]:
    """Load flow events within [ts_start, ts_end) - extract service ports per edge.

    Returns dict mapping (src_node, dst_node) -> set of port strings.
    Only non-N (service) ports are included.
    """
    import json as _json

    flows_dir = DATA_ROOT / "cleaned" / "flows"
    if not flows_dir.exists():
        return {}

 # Build flow file index (cached once)
    global _flow_file_cache
    if _flow_file_cache is None:
        import os as _os
        entries = []
        for fp in sorted([f for f in _os.listdir(str(flows_dir)) if f.endswith(".parquet")]):
            pf = pq.ParquetFile(str(flows_dir / fp))
            md = pf.metadata
            f_min, f_max = float("inf"), float("-inf")
            for rg_idx in range(md.num_row_groups):
                rg = md.row_group(rg_idx)
                for ci in range(rg.num_columns):
                    if rg.column(ci).path_in_schema == "timestamp":
                        stats = rg.column(ci).statistics
                        if stats and stats.has_min_max:
                            f_min = min(f_min, int(stats.min))
                            f_max = max(f_max, int(stats.max))
                        break
            if f_min == float("inf"):
                t = pf.read()
                f_min = int(t.column("timestamp")[0].as_py())
                f_max = int(t.column("timestamp")[t.num_rows - 1].as_py())
            entries.append((fp, int(f_min), int(f_max)))
        _flow_file_cache = entries

 # Collect flow edges with service ports
    edge_ports: Dict[Tuple[str, str], set] = {}
    for fp, f_start, f_end in _flow_file_cache:
        if f_end < ts_start or f_start >= ts_end:
            continue

        t = pq.read_table(str(flows_dir / fp))
        ts_col = t.column("timestamp")
        src_col = t.column("source_entity")
        dst_col = t.column("destination_entity")
        detail_col = t.column("event_detail")

        for i in range(t.num_rows):
            ts = int(ts_col[i].as_py())
            if ts < ts_start or ts >= ts_end:
                continue
            src = src_col[i].as_py()
            dst = dst_col[i].as_py()
            try:
                detail = _json.loads(detail_col[i].as_py())
            except Exception:
                continue
 # Extract service ports (numeric, non-ephemeral)
            for port_key in ("src_port", "dst_port"):
                p_str = str(detail.get(port_key, ""))
                if p_str and not p_str.startswith("N"):
                    try:
                        int(p_str)  # validate numeric
                        edge_ports.setdefault((src, dst), set()).add(p_str)
                    except ValueError:
                        pass

    return edge_ports


_flow_file_cache: Optional[List] = None


# ---
# Main builder
# ---

def build_window_graph(
    window: WindowInfo,
    J: int = 4,
    n_min_cat: int = 5,
    verbose: bool = True,
) -> Optional[WindowGraphData]:
    """Build the observation graph from one adaptive window.

    Steps:
    1. Build J micro-slices from the window's events
    2. Construct the directed graph (unique pairs)
    3. Compute peak ACP, edge time targets, category masks
    4. Package into WindowGraphData

    Parameters
    ----------
    window : WindowInfo
    J : int - micro-slices per window
    n_min_cat : int - min positive samples for a category to be active

    Returns
    -------
    WindowGraphData or None if the window has no events.
    """
    if verbose:
        print(f"  Building graph for window {window.idx} "
              f"(ts={window.ts_start}-{window.ts_end}, J={J})", flush=True)

 # -- Build micro-slices ---
    msr = build_micro_slices(window, J=J, categories=("AUTH",), verbose=False)
    if msr is None or msr.N == 0:
        if verbose:
            print(f"    No events in window {window.idx}")
        return None

    N = msr.N
    node_list = sorted(msr.node_to_id.keys(), key=lambda n: msr.node_to_id[n])
    n2i = msr.node_to_id

 # -- Load event details for feature building ---
 # Read auth events with protocol/status info for this window
    rich_events = _load_window_events_rich(window.ts_start, window.ts_end + 1)

 # -- Load port vocabulary + flow events ---
    vocab = _load_port_vocab()
    AUTH_FEATURES = vocab["auth_features"]              # 6 auth features
    PORT_FEATURES = vocab["port_features"]               # 30 port features
    F_auth = len(AUTH_FEATURES)
    F_port = len(PORT_FEATURES)
    F_dim = F_auth + F_port                              # 36 binary features

    port_to_idx = {p.replace("port_", ""): i for i, p in enumerate(PORT_FEATURES)}
    flow_port_map = _load_window_flows(window.ts_start, window.ts_end + 1)

 # -- Build edge list with protocol + port features ---
    all_pairs: Dict[Tuple[int, int], Dict] = {}
    for j in range(J):
        for (u, v), count in msr.edge_counts_per_slice[j].items():
            if (u, v) not in all_pairs:
                all_pairs[(u, v)] = {
                    "count": 0,
                    "slices": np.zeros(J, dtype=np.float32),
                    "mcount": np.zeros(J, dtype=np.float32),
                    "auth_vec": np.zeros(F_auth, dtype=np.float32),
                    "port_vec": np.zeros(F_port, dtype=np.float32),
                }
            all_pairs[(u, v)]["count"] += count
            all_pairs[(u, v)]["slices"][j] = 1.0
            all_pairs[(u, v)]["mcount"][j] = np.log1p(count)

 # Aggregate protocol features from rich auth events
    for evt in rich_events:
        src = evt["src"]
        dst = evt["dst"]
        u = n2i.get(src)
        v = n2i.get(dst)
        if u is None or v is None:
            continue
        pk = (u, v)
        if pk in all_pairs:
            all_pairs[pk]["auth_vec"][0] = 1.0  # is_auth
            if evt.get("status") == "Success":
                all_pairs[pk]["auth_vec"][1] = 1.0
            proto_key = f"proto_{evt.get('protocol', '?')}"
            if proto_key in AUTH_FEATURES:
                all_pairs[pk]["auth_vec"][AUTH_FEATURES.index(proto_key)] = 1.0
            else:
                all_pairs[pk]["auth_vec"][5] = 1.0  # proto_unknown

 # Aggregate port features from flow events
    for (src_name, dst_name), ports in flow_port_map.items():
        u = n2i.get(src_name)
        v = n2i.get(dst_name)
        if u is None or v is None:
            continue
        pk = (u, v)
        if pk in all_pairs:
            for p_str in ports:
                idx = port_to_idx.get(p_str)
                if idx is not None:
                    all_pairs[pk]["port_vec"][idx] = 1.0

 # Build final arrays
    E_pairs = len(all_pairs)
    src_arr = np.zeros(E_pairs, dtype=np.int32)
    dst_arr = np.zeros(E_pairs, dtype=np.int32)
    ef_arr = np.zeros((E_pairs, F_dim + 1), dtype=np.float32)  # 36 binary + log_count
    edge_occ_arr = np.zeros((E_pairs, J), dtype=np.float32)
    edge_mcount_arr = np.zeros((E_pairs, J), dtype=np.float32)

    for i, ((u, v), data) in enumerate(sorted(all_pairs.items())):
        src_arr[i] = u
        dst_arr[i] = v
        ef_arr[i, :F_auth] = data["auth_vec"]
        ef_arr[i, F_auth:F_dim] = data["port_vec"]
        ef_arr[i, F_dim] = np.log1p(data["count"])  # log intensity
        edge_occ_arr[i] = data["slices"]
        edge_mcount_arr[i] = data["mcount"]

 # -- Node features ---
    out_deg = np.bincount(src_arr, minlength=N).astype(np.float64)
    in_deg = np.bincount(dst_arr, minlength=N).astype(np.float64)

    X = np.zeros((N, 8), dtype=np.float32)
    X[:, 0] = np.log1p(in_deg)
    X[:, 1] = np.log1p(out_deg)
    X[:, 3] = np.log1p(out_deg / (in_deg + 1.0))
    pr_proxy = in_deg / max(in_deg.max(), 1)
    acp_proxy = out_deg / (in_deg + 1.0)
    X[:, 4] = np.log1p(pr_proxy * 1e7) / 16.0
    X[:, 5] = np.log1p(acp_proxy) / 12.0
    X[:, 6] = np.log1p(in_deg) / max(np.log1p(in_deg).max(), 1.0)
    X[:, 7] = np.log1p(out_deg) / max(np.log1p(out_deg).max(), 1.0)
    mu, sigma = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma

 # -- Distillation targets ---
    # Current paper version: true directed PageRank (power iteration), not
    # the in-degree proxy.  Input features (X) are unchanged.
    pr_scores, _pr_iterations = directed_pagerank(src_arr, dst_arr, N)
    pr_target = np.log1p(1.0 / (pr_scores * 1e7 + 1e-10)).astype(np.float32)
    acp_full_arr = (out_deg / (in_deg + 1.0)).astype(np.float32)

 # Peak ACP from micro-slices
    from .features import compute_peak_acp
    _, acp_peak_arr = compute_peak_acp(msr)
    acp_peak_arr = acp_peak_arr.astype(np.float32)

 # -- IEF weights + category mask ---
 # F_dim = 36 (binary features only); ef_arr has 37 cols (36 binary + log_count)
    F_total = ef_arr.shape[1]           # 37 = F_dim + 1 (log_count)
    counts = ef_arr[:, :F_dim].sum(axis=0)  # binary features only
    counts = np.maximum(counts, 1)
    freq = counts / E_pairs
 # Build weights: 36 IEF weights for binary features + 1.0 for log_count
    ief_w = (1.0 / np.sqrt(freq)).astype(np.float32)
    ief_w /= ief_w.mean()
    feat_weights = np.ones(F_total, dtype=np.float32)
    feat_weights[:F_dim] = ief_w       # IEF for binary features
    feat_weights[F_dim] = 0.5           # reduced weight for log_count

    cat_mask_arr = np.ones(F_total, dtype=np.float32)
    cat_mask_arr[:F_dim] = (ef_arr[:, :F_dim].sum(axis=0) >= n_min_cat).astype(np.float32)

 # -- Evaluation labels ---
    pivot_ids = [n2i[p] for p in window.attack_pivots if p in n2i]
    labels = np.zeros(N, dtype=np.int32)
    for pid in pivot_ids:
        labels[pid] = 1

    g = WindowGraphData(
        window_idx=window.idx,
        ts_start=window.ts_start,
        ts_end=window.ts_end,
        N=N, E=E_pairs,
        src=src_arr, dst=dst_arr,
        nodes=node_list, node_to_id=n2i,
        X=X, ef=ef_arr,
        feat_weights=feat_weights, cat_mask=cat_mask_arr,
        pr_target=pr_target,
        acp_full=acp_full_arr, acp_peak=acp_peak_arr,
        J=J, edge_occ=edge_occ_arr, edge_mcount=edge_mcount_arr,
        pivot_ids=pivot_ids, pivot_names=window.attack_pivots,
        labels=labels,
    )

    if verbose:
        n_occ_spike = float((edge_occ_arr.mean(axis=1) <= 0.25).mean()) * 100
        n_occ_dense = float((edge_occ_arr.mean(axis=1) >= 1.0).mean()) * 100
        active_cats = int(cat_mask_arr[:F_dim].sum())
        print(f"    N={N:,}  E={E_pairs:,}  F[bin]={F_dim}  F[total]={F_total}  "
              f"acp_full>0={(acp_full_arr>0).sum()}/{N}  "
              f"acp_peak>0={(acp_peak_arr>0).sum()}/{N}", flush=True)
        print(f"    Edge occ spike(<=0.25): {n_occ_spike:.1f}%  "
              f"dense(1.0): {n_occ_dense:.1f}%  "
              f"active_cats: {active_cats}/{F_dim}", flush=True)

    return g
