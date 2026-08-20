"""Label-free adaptive window construction.

Builds information-sufficient time windows by scanning the event stream
chronologically and closing each window when all four label-free
thresholds are met:

    M(W) >= M_min          raw event count
    E(W) >= E_min          unique directed (src,dst) pairs
    V_s(W) >= V_min_src    unique source nodes
    V_d(W) >= V_min_dst    unique destination nodes

No attack labels, timestamps, or identities are consulted during
window-boundary decisions.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

# Data root (override with the ASYGAD_DATA_ROOT environment variable).
from .paths import DATA_ROOT as ROOT


# ---
# Data structures
# ---

@dataclass
class WindowInfo:
    """Metadata for one information-sufficient window."""
    idx: int                          # zero-based window index
    ts_start: int                     # first event timestamp (inclusive)
    ts_end: int                       # last event timestamp (inclusive)
    duration_sec: float               # ts_end - ts_start
    M: int                            # raw event count
    E: int                            # unique directed (src,dst) pairs
    V_src: int                        # unique source nodes
    V_dst: int                        # unique destination nodes
    V_total: int                      # unique nodes (union)

 # Evaluation-only (populated after window construction)
    has_attack: bool = False
    attack_pivots: List[str] = field(default_factory=list)
    attack_event_count: int = 0

 # For the last window (may be merged with trailing data)
    is_merged_tail: bool = False


@dataclass
class WindowBuildResult:
    """Complete output of adaptive window construction."""
    windows: List[WindowInfo]
    M_min: int
    E_min: int
    V_min_src: int
    V_min_dst: int
    total_events: int
    total_unique_pairs: int
    total_unique_nodes: int

    @property
    def n_windows(self) -> int:
        return len(self.windows)


# ---
# Event stream reader
# ---

def _read_events_stream(
    categories: Tuple[str, ...] = ("AUTH",),
) -> List[dict]:
    """Read all cleaned events, sorted by timestamp.

    Processes category directories (AUTH, FLOW) and returns a single
    chronologically-sorted list.  For datasets that fit in memory (~10  events
    per day x 60 days ~ 6x10  records for auth), this is practical.

    For larger streams, replace with a file-by-file streaming accumulator.
    """
    cat_dirs_map = {"AUTH": "auth", "FLOW": "flows", "DNS": "dns", "PROC": "proc"}

    events: List[dict] = []
    for cat in categories:
        cat_dir = ROOT / "cleaned" / cat_dirs_map.get(cat, cat.lower())
        if not cat_dir.exists():
            print(f"  [WARN] directory not found: {cat_dir}")
            continue

        parquet_files = sorted(
            [f for f in os.listdir(str(cat_dir)) if f.endswith(".parquet")]
        )

        n_cat = 0
        for fp in parquet_files:
            t = pq.read_table(str(cat_dir / fp))
            ts_col = t.column("timestamp")
            src_col = t.column("source_entity")
            dst_col = t.column("destination_entity")

            for i in range(t.num_rows):
                events.append({
                    "ts": int(ts_col[i].as_py()),
                    "src": src_col[i].as_py(),
                    "dst": dst_col[i].as_py(),
                    "category": cat,
                })
                n_cat += 1

        print(f"  {cat}: {n_cat:,} events from {len(parquet_files)} files")

 # Sort by timestamp
    events.sort(key=lambda e: e["ts"])
    return events


# ---
# Streaming event generator (file-by-file, memory-efficient)
# ---

def stream_events(
    categories: Tuple[str, ...] = ("AUTH",),
    max_files: Optional[int] = None,
    chunk_size: int = 100_000,
    verbose: bool = True,
):
    """Generate events in approximate chronological order.

    Processes parquet files one at a time, reading rows in small
    chunks to avoid materialising 5M-row Python lists.  Files are
    already time-partitioned; within each file events are consumed
    in storage order (approximately chronological).

    Yields
    ------
    dict with keys: ts, src, dst, category
    """
    cat_dirs_map = {"AUTH": "auth", "FLOW": "flows", "DNS": "dns", "PROC": "proc"}

    for cat in categories:
        cat_dir = ROOT / "cleaned" / cat_dirs_map.get(cat, cat.lower())
        if not cat_dir.exists():
            if verbose:
                print(f"  [WARN] directory not found: {cat_dir}")
            continue

        parquet_files = sorted(
            [f for f in os.listdir(str(cat_dir)) if f.endswith(".parquet")]
        )
        if max_files is not None:
            parquet_files = parquet_files[:max_files]

        n_cat = 0
        for fp in parquet_files:
            t = pq.read_table(str(cat_dir / fp))
            n = t.num_rows
            ts_col = t.column("timestamp")
            src_col = t.column("source_entity")
            dst_col = t.column("destination_entity")

 # Iterate in chunks to balance Python overhead vs memory
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                for i in range(start, end):
                    ts = int(ts_col[i].as_py())
                    src = src_col[i].as_py()
                    dst = dst_col[i].as_py()
                    yield {"ts": ts, "src": src, "dst": dst, "category": cat}
                    n_cat += 1

        if verbose:
            print(f"  {cat}: {n_cat:,} events from {len(parquet_files)} files")


# ---
# Adaptive window builder (in-memory version - for small datasets)
# ---

def build_adaptive_windows(
    categories: Tuple[str, ...] = ("AUTH",),
    M_min: int = 20_000,
    E_min: int = 2_000,
    V_min_src: int = 500,
    V_min_dst: int = 500,
    merge_tail: bool = True,
    tail_ratio: float = 0.5,
    verbose: bool = True,
) -> WindowBuildResult:
    """Build information-sufficient windows by scanning the event stream.

    The algorithm walks events chronologically and closes the current
    window as soon as ALL FOUR thresholds are met.  "First to satisfy all"
    is the key property - it prevents the window from accumulating
    superfluous data once information adequacy is reached.

    Parameters
    ----------
    categories : tuple
        Event categories to include (default: AUTH only).
    M_min : int
        Minimum raw event count per window.
    E_min : int
        Minimum unique directed (src,dst) pairs per window.
    V_min_src : int
        Minimum unique source nodes per window.
    V_min_dst : int
        Minimum unique destination nodes per window.
    merge_tail : bool
        If True, merge the trailing window that fails thresholds into
        the previous window (and mark it).
    tail_ratio : float
        If the trailing window's event count is < tail_ratio * M_min,
        merge it.  Otherwise it becomes its own window anyway.
    verbose : bool
        Print progress.

    Returns
    -------
    WindowBuildResult with all window metadata.
    """
    if verbose:
        print("=" * 60)
        print("Adaptive window construction")
        print(f"  M_min={M_min:,}  E_min={E_min:,}  "
              f"V_src_min={V_min_src:,}  V_dst_min={V_min_dst:,}")
        print("=" * 60)

 # -- Load events ---
    if verbose:
        print("\n[1/3] Loading events ...")
    events = _read_events_stream(categories)
    M_total = len(events)
    if verbose:
        print(f"  Total: {M_total:,} events")

    if M_total == 0:
        return WindowBuildResult(
            windows=[], M_min=M_min, E_min=E_min,
            V_min_src=V_min_src, V_min_dst=V_min_dst,
            total_events=0, total_unique_pairs=0, total_unique_nodes=0,
        )

 # -- Scan and accumulate ---
    if verbose:
        print("\n[2/3] Scanning event stream ...")

    windows: List[WindowInfo] = []

 # Current window accumulators
    cur_M = 0
    cur_pairs: Dict[Tuple[str, str], int] = {}   # (src,dst) -> count
    cur_sources: set = set()
    cur_dests: set = set()
    cur_nodes: set = set()
    cur_ts_start: Optional[int] = None
    cur_ts_end: Optional[int] = None

    def _flush_window(idx: int) -> WindowInfo:
        """Package the current accumulator state into a WindowInfo."""
        return WindowInfo(
            idx=idx,
            ts_start=cur_ts_start or 0,
            ts_end=cur_ts_end or 0,
            duration_sec=float((cur_ts_end or 0) - (cur_ts_start or 0)),
            M=cur_M,
            E=len(cur_pairs),
            V_src=len(cur_sources),
            V_dst=len(cur_dests),
            V_total=len(cur_nodes),
        )

    last_report = 0
    for evt in events:
        ts = evt["ts"]
        src = evt["src"]
        dst = evt["dst"]

 # Update accumulators
        cur_M += 1
        pair_key = (src, dst)
        cur_pairs[pair_key] = cur_pairs.get(pair_key, 0) + 1
        cur_sources.add(src)
        cur_dests.add(dst)
        cur_nodes.add(src)
        cur_nodes.add(dst)
        if cur_ts_start is None:
            cur_ts_start = ts
        cur_ts_end = ts

 # Check thresholds - flush when ALL are met
        if (cur_M >= M_min
                and len(cur_pairs) >= E_min
                and len(cur_sources) >= V_min_src
                and len(cur_dests) >= V_min_dst):
            windows.append(_flush_window(len(windows)))

 # Reset accumulators
            cur_M = 0
            cur_pairs.clear()
            cur_sources.clear()
            cur_dests.clear()
            cur_nodes.clear()
            cur_ts_start = None
            cur_ts_end = None

 # Progress
            if verbose and len(windows) % 10 == 0 and len(windows) > last_report:
                w = windows[-1]
                print(f"  Window {w.idx:3d}: M={w.M:>8,}  E={w.E:>6,}  "
                      f"V_s={w.V_src:>5,}  V_d={w.V_dst:>5,}  "
                      f"dur={w.duration_sec/3600:.1f}h")
                last_report = len(windows)

 # Handle trailing data
    if cur_M > 0:
        if (merge_tail
                and len(windows) > 0
                and cur_M < tail_ratio * M_min):
 # Merge into previous window
            prev = windows[-1]
            prev.M += cur_M
 # Update E/V estimates (these are approximate for the merged
 # window since we discarded the pair accumulators)
            prev.ts_end = cur_ts_end
            prev.duration_sec = float(prev.ts_end - prev.ts_start)
            prev.is_merged_tail = True
            if verbose:
                print(f"  Tail merged: +{cur_M:,} events -> window {prev.idx}")
        else:
            windows.append(_flush_window(len(windows)))
            if verbose:
                print(f"  Final window {windows[-1].idx}: M={cur_M:,} (tail)")

 # -- Annotate with attack presence (evaluation-only) ---
    if verbose:
        print("\n[3/3] Annotating attack presence (evaluation-only) ...")

    _annotate_attacks(windows, verbose=verbose)

 # -- Compute global aggregates ---
    total_pairs = sum(w.E for w in windows)  # approximate (pairs may repeat)
    total_nodes_union = len(set().union(*[
        set()  # We don't hold node sets across windows - estimate as max
    ]))
    if verbose:
        _print_summary(windows, M_total)

    return WindowBuildResult(
        windows=windows,
        M_min=M_min, E_min=E_min,
        V_min_src=V_min_src, V_min_dst=V_min_dst,
        total_events=M_total,
        total_unique_pairs=total_pairs,
        total_unique_nodes=total_nodes_union,
    )


# ---
# Adaptive window builder (streaming - for large datasets)
# ---

def build_adaptive_windows_streaming(
    categories: Tuple[str, ...] = ("AUTH",),
    M_min: int = 20_000,
    E_min: int = 2_000,
    V_min_src: int = 500,
    V_min_dst: int = 500,
    merge_tail: bool = True,
    tail_ratio: float = 0.5,
    max_files: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[WindowInfo], int]:
    """Build adaptive windows using streaming (file-by-file) event reader.

    Same semantics as build_adaptive_windows() but processes events
    incrementally - suitable for 1B+ event datasets.

    Returns
    -------
    (windows, total_events)
    """
    if verbose:
        print("=" * 60)
        print("Adaptive Window Construction (Streaming)")
        print(f"  M_min={M_min:,}  E_min={E_min:,}  "
              f"V_src_min={V_min_src:,}  V_dst_min={V_min_dst:,}")
        if max_files:
            print(f"    TEST MODE: max_files={max_files}")
        print("=" * 60)

    windows: List[WindowInfo] = []
    total_events = 0

    cur_M = 0
    cur_pairs: dict = {}
    cur_sources: set = set()
    cur_dests: set = set()
    cur_nodes: set = set()
    cur_ts_start: Optional[int] = None
    cur_ts_end: Optional[int] = None

    def _flush(idx: int) -> WindowInfo:
        return WindowInfo(
            idx=idx,
            ts_start=cur_ts_start or 0,
            ts_end=cur_ts_end or 0,
            duration_sec=float((cur_ts_end or 0) - (cur_ts_start or 0)),
            M=cur_M,
            E=len(cur_pairs),
            V_src=len(cur_sources),
            V_dst=len(cur_dests),
            V_total=len(cur_nodes),
        )

    import time as _time
    t0 = _time.time()
    last_report = 0

    if verbose:
        print("\n[1/2] Streaming events and building windows ...")

    for evt in stream_events(categories, max_files=max_files, verbose=verbose):
        ts = evt["ts"]
        src = evt["src"]
        dst = evt["dst"]
        total_events += 1

        cur_M += 1
        pk = (src, dst)
        cur_pairs[pk] = cur_pairs.get(pk, 0) + 1
        cur_sources.add(src)
        cur_dests.add(dst)
        cur_nodes.add(src)
        cur_nodes.add(dst)
        if cur_ts_start is None:
            cur_ts_start = ts
        cur_ts_end = ts

 # Flush when all four thresholds are met
        if (cur_M >= M_min
                and len(cur_pairs) >= E_min
                and len(cur_sources) >= V_min_src
                and len(cur_dests) >= V_min_dst):
            windows.append(_flush(len(windows)))
            cur_M = 0
            cur_pairs.clear()
            cur_sources.clear()
            cur_dests.clear()
            cur_nodes.clear()
            cur_ts_start = None
            cur_ts_end = None

            if verbose and len(windows) % 20 == 0 and len(windows) > last_report:
                w = windows[-1]
                elapsed = _time.time() - t0
                print(f"  Win {w.idx:4d}: M={w.M:>8,}  E={w.E:>6,}  "
                      f"V_s={w.V_src:>5,}  V_d={w.V_dst:>5,}  "
                      f"dur={w.duration_sec/3600:.1f}h  "
                      f"[{total_events/1e6:6.1f}M events, {elapsed:.0f}s]")
                last_report = len(windows)

 # Handle trailing data
    if cur_M > 0:
        if merge_tail and len(windows) > 0 and cur_M < tail_ratio * M_min:
            prev = windows[-1]
            prev.M += cur_M
            prev.E += len(cur_pairs)
            prev.ts_end = cur_ts_end
            prev.duration_sec = float(prev.ts_end - prev.ts_start)
            prev.is_merged_tail = True
            if verbose:
                print(f"  Tail merged: +{cur_M:,} events -> window {prev.idx}")
        else:
            windows.append(_flush(len(windows)))

    elapsed = _time.time() - t0
    if verbose:
        print(f"\n  Scanned {total_events:,} events in {elapsed:.0f}s "
              f"({total_events/elapsed/1e6:.1f}M events/s)")
        print(f"  Produced {len(windows)} windows")

 # -- Annotate attacks ---
    if verbose:
        print("\n[2/2] Annotating attack presence ...")
    _annotate_attacks(windows, verbose=verbose)

    if verbose:
        _print_summary(windows, total_events)

    return windows, total_events


# ---
# Attack annotation (evaluation-only - NEVER used for window boundaries)
# ---

def _annotate_attacks(
    windows: List[WindowInfo],
    verbose: bool = True,
) -> None:
    """Annotate each window with attack presence (post-hoc, evaluation only)."""
    rt_dir = ROOT / "cleaned" / "redteam"
    if not rt_dir.exists():
        if verbose:
            print("  [WARN] No redteam directory found")
        return

 # Collect all redteam events: (ts, pivot_name)
    attack_events: List[Tuple[int, str]] = []
    for fp in sorted(os.listdir(str(rt_dir))):
        if not fp.endswith(".parquet"):
            continue
        t = pq.read_table(str(rt_dir / fp))
        for i in range(t.num_rows):
            ts = int(t.column("timestamp")[i].as_py())
            detail = json.loads(t.column("event_detail")[i].as_py())
            pivot = detail.get("pivot_machine", "unknown")
            attack_events.append((ts, pivot))

    if not attack_events:
        if verbose:
            print("  No attack events found")
        return

    attack_events.sort(key=lambda x: x[0])

 # Binary-search each attack event into the right window
    window_starts = np.array([w.ts_start for w in windows], dtype=np.int64)
    window_ends = np.array([w.ts_end for w in windows], dtype=np.int64)

    n_attack_windows = 0
    for ats, pivot in attack_events:
 # Find first window whose end >= ats
        idx = int(np.searchsorted(window_ends, ats, side="left"))
        if idx < len(windows) and windows[idx].ts_start <= ats <= windows[idx].ts_end:
            windows[idx].has_attack = True
            if pivot not in windows[idx].attack_pivots:
                windows[idx].attack_pivots.append(pivot)
            windows[idx].attack_event_count += 1
            n_attack_windows += 1

    if verbose:
        n_has = sum(1 for w in windows if w.has_attack)
        print(f"  Attack events: {len(attack_events)} across {n_has}/{len(windows)} windows")


# ---
# Reporting
# ---

def _print_summary(windows: List[WindowInfo], total_events: int) -> None:
    """Print window construction summary statistics."""
    n = len(windows)
    if n == 0:
        print("\n  No windows constructed.")
        return

    Ms = [w.M for w in windows]
    Es = [w.E for w in windows]
    Vs = [w.V_src for w in windows]
    Vd = [w.V_dst for w in windows]
    Vt = [w.V_total for w in windows]
    durs = [w.duration_sec / 3600.0 for w in windows]  # hours

    def _qs(arr):
        a = sorted(arr)
        return a[0], a[len(a)//4], a[len(a)//2], a[3*len(a)//4], a[-1]

    print(f"\n{'='*60}")
    print(f"  Window Construction Summary")
    print(f"{'='*60}")
    print(f"  Total windows:     {n}")
    print(f"  Attack windows:    {sum(1 for w in windows if w.has_attack)}")
    print(f"  Merged-tail:       {sum(1 for w in windows if w.is_merged_tail)}")
    print()
    print(f"  {'Metric':<20s} {'Min':>8s} {'Q1':>8s} {'Median':>8s} {'Q3':>8s} {'Max':>8s} {'Mean':>8s}")
    print(f"  {'-'*68}")
    for label, arr in [
        ("Duration (h)", durs),
        ("Events (M)", Ms),
        ("Unique pairs (E)", Es),
        ("Src nodes (V_s)", Vs),
        ("Dst nodes (V_d)", Vd),
        ("Total nodes (V)", Vt),
    ]:
        mn, q1, md, q3, mx = _qs(arr)
        mu = np.mean(arr)
        if isinstance(arr[0], float):
            print(f"  {label:<20s} {mn:>8.2f} {q1:>8.2f} {md:>8.2f} {q3:>8.2f} {mx:>8.2f} {mu:>8.2f}")
        else:
            print(f"  {label:<20s} {mn:>8,} {q1:>8,} {md:>8,} {q3:>8,} {mx:>8,} {mu:>8,.0f}")


def window_stats_table(windows: List[WindowInfo]) -> str:
    """Build a markdown table of per-window statistics."""
    lines = [
        "| Win | t_start | t_end | Dur(h) | M | E | V_src | V_dst | V | Attack? | Pivots |",
        "|-----|---------|-------|--------|---|---|-------|-------|---|---------|--------|",
    ]
    for w in windows:
        piv_str = ",".join(w.attack_pivots) if w.attack_pivots else ""
        att_str = "Y" if w.has_attack else ""
        lines.append(
            f"| {w.idx} | {w.ts_start} | {w.ts_end} | {w.duration_sec/3600:.1f} | "
            f"{w.M:,} | {w.E:,} | {w.V_src:,} | {w.V_dst:,} | {w.V_total:,} | "
            f"{att_str} | {piv_str} |"
        )
    return "\n".join(lines)
