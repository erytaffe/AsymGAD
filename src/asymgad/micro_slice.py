"""Micro-slice construction within adaptive windows.

Each information-sufficient window is divided into J equal-duration
micro-slices.  This preserves coarse-grained temporal structure inside
the window without introducing cross-window state.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from .window import WindowInfo

from .paths import DATA_ROOT as ROOT


@dataclass
class MicroSlice:
    """Per-slice statistics for one window."""

    idx: int                          # 0-based slice index within window
    ts_start: int                     # inclusive start timestamp
    ts_end: int                       # exclusive end timestamp

 # Graph statistics
    n_events: int                     # raw event count
    n_pairs: int                      # unique (src,dst) pairs
    n_sources: int                    # unique source nodes
    n_dests: int                      # unique destination nodes

 # Per-node degree arrays (indexed by node_id in window)
    out_deg: np.ndarray               # (N,) int32
    in_deg: np.ndarray                # (N,) int32


@dataclass
class MicroSliceResult:
    """Micro-slice output for one window."""

    window_idx: int
    J: int
    N: int                            # total unique nodes in window
    E: int                            # total unique pairs in window
    slices: List[MicroSlice]

 # Window-level node mapping (node_name -> int)
    node_to_id: Dict[str, int]

 # Per-slice edge counts: edges_index[slice_j][(u,v)] = count
    edge_counts_per_slice: List[Dict[Tuple[int, int], int]]


# ---
# Main API
# ---

def build_micro_slices(
    window: WindowInfo,
    J: int = 4,
    categories: Tuple[str, ...] = ("AUTH",),
    verbose: bool = True,
) -> Optional[MicroSliceResult]:
    """Divide one adaptive window into J equal-duration micro-slices.

    Parameters
    ----------
    window : WindowInfo
        The adaptive window to split.
    J : int
        Number of micro-slices (default 4).
    categories : tuple
        Event categories to include.

    Returns
    -------
    MicroSliceResult, or None if the window has no events.
    """
    duration = window.ts_end - window.ts_start
    if duration <= 0:
        if verbose:
            print(f"  Window {window.idx}: zero duration, skipping")
        return None

    slice_width = duration / J

 # -- Load events within this window ---
    events = _load_window_events(window.ts_start, window.ts_end + 1, categories)
    if not events:
        if verbose:
            print(f"  Window {window.idx}: no events found")
        return None

 # -- Build node mapping ---
    all_nodes = set()
    for e in events:
        all_nodes.add(e["src"])
        all_nodes.add(e["dst"])
    node_list = sorted(all_nodes)
    node_to_id = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

 # -- Assign events to slices ---
    slices_data: List[Dict] = []
    edge_counts_per_slice: List[Dict[Tuple[int, int], int]] = []

    for j in range(J):
        slice_start = window.ts_start + int(j * slice_width)
        slice_end = window.ts_start + int((j + 1) * slice_width)

 # Events in [slice_start, slice_end)
        slice_events = [e for e in events
                        if slice_start <= e["ts"] < slice_end]

        out_deg = np.zeros(N, dtype=np.int32)
        in_deg = np.zeros(N, dtype=np.int32)
        edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        for e in slice_events:
            u = node_to_id[e["src"]]
            v = node_to_id[e["dst"]]
            out_deg[u] += 1
            in_deg[v] += 1
            edge_counts[(u, v)] += 1

        slices_data.append({
            "j": j,
            "n_events": len(slice_events),
            "n_pairs": len(edge_counts),
            "out_deg": out_deg,
            "in_deg": in_deg,
        })
        edge_counts_per_slice.append(dict(edge_counts))

 # Build MicroSlice objects
    micro_slices = []
    for j in range(J):
        sd = slices_data[j]
        n_src = int((sd["out_deg"] > 0).sum())
        n_dst = int((sd["in_deg"] > 0).sum())
        micro_slices.append(MicroSlice(
            idx=j,
            ts_start=window.ts_start + int(j * slice_width),
            ts_end=window.ts_start + int((j + 1) * slice_width),
            n_events=sd["n_events"],
            n_pairs=sd["n_pairs"],
            n_sources=n_src,
            n_dests=n_dst,
            out_deg=sd["out_deg"],
            in_deg=sd["in_deg"],
        ))

 # Window-level E is the union of per-slice pairs
    all_pairs = set()
    for ec in edge_counts_per_slice:
        all_pairs.update(ec.keys())

    result = MicroSliceResult(
        window_idx=window.idx,
        J=J,
        N=N,
        E=len(all_pairs),
        slices=micro_slices,
        node_to_id=node_to_id,
        edge_counts_per_slice=edge_counts_per_slice,
    )

    if verbose:
        _print_slice_summary(window, result)

    return result


# ---
# Event loading (window-specific)
# ---

# -- File time-range cache (built once) ---
_file_time_cache: Optional[Dict[str, List[Tuple[str, int, int]]]] = None


def _build_file_index(categories: Tuple[str, ...] = ("AUTH",)) -> Dict[str, List[Tuple[str, int, int]]]:
    """Build index: category -> [(filename, ts_min, ts_max), ...].

    Iterates all row groups of each parquet file to get the true min/max
    timestamp.  This is O(num_row_groups) per file (typically ~10 groups),
    much faster than reading the full table.
    """
    cat_dirs_map = {"AUTH": "auth", "FLOW": "flows", "DNS": "dns", "PROC": "proc"}
    index: Dict[str, List[Tuple[str, int, int]]] = {}

    for cat in categories:
        cat_dir = ROOT / "cleaned" / cat_dirs_map.get(cat, cat.lower())
        if not cat_dir.exists():
            index[cat] = []
            continue

        import os
        parquet_files = sorted(
            [f for f in os.listdir(str(cat_dir)) if f.endswith(".parquet")]
        )

        entries = []
        for fp in parquet_files:
            pf = pq.ParquetFile(str(cat_dir / fp))
            md = pf.metadata
            f_min = float("inf")
            f_max = float("-inf")

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
 # Fallback: read the full table (shouldn't happen)
                t = pf.read()
                f_min = int(t.column("timestamp")[0].as_py())
                f_max = int(t.column("timestamp")[t.num_rows - 1].as_py())

            entries.append((fp, int(f_min), int(f_max)))

        index[cat] = entries

    return index


def _load_window_events(
    ts_start: int,
    ts_end: int,
    categories: Tuple[str, ...] = ("AUTH",),
) -> List[dict]:
    """Load events within [ts_start, ts_end) from parquet files.

    Uses a pre-built file index to only read files overlapping the window.
    """
    global _file_time_cache
    if _file_time_cache is None:
        _file_time_cache = _build_file_index(categories)

    events: List[dict] = []
    cat_dirs_map = {"AUTH": "auth", "FLOW": "flows", "DNS": "dns", "PROC": "proc"}

    for cat in categories:
        cat_dir = ROOT / "cleaned" / cat_dirs_map.get(cat, cat.lower())
        if not cat_dir.exists():
            continue

        for fp, f_start, f_end in _file_time_cache.get(cat, []):
 # Quick skip
            if f_end < ts_start or f_start >= ts_end:
                continue

            t = pq.read_table(str(cat_dir / fp))
            ts_col = t.column("timestamp")
            src_col = t.column("source_entity")
            dst_col = t.column("destination_entity")

            for i in range(t.num_rows):
                ts = int(ts_col[i].as_py())
                if ts < ts_start or ts >= ts_end:
                    continue
                events.append({
                    "ts": ts,
                    "src": src_col[i].as_py(),
                    "dst": dst_col[i].as_py(),
                    "category": cat,
                })

    return events


# ---
# Reporting
# ---

def _print_slice_summary(window: WindowInfo, result: MicroSliceResult) -> None:
    """Print per-slice statistics for one window."""
    print(f"  Window {window.idx}: J={result.J}, N={result.N}, E={result.E}")
    print(f"    {'Slice':<8s} {'Events':>10s} {'Pairs':>8s} "
          f"{'Src':>6s} {'Dst':>6s} {'Events/s':>10s}")
    print(f"    {'-'*50}")
    for s in result.slices:
        dur = max(s.ts_end - s.ts_start, 1)
        rate = s.n_events / dur
        print(f"    [{s.idx}]     {s.n_events:>10,} {s.n_pairs:>8,} "
              f"{s.n_sources:>6,} {s.n_dests:>6,} {rate:>10.1f}")
