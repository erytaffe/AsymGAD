"""Build the destination-port vocabulary from cleaned flow events.

The paper's edge features use 6 auth/protocol indicators and the top-30
service ports (36 binary dimensions) plus a log-count dimension.  The
vocabulary is derived from the LANL flow records and stored as
``data/graph/feature_names.json``, which :func:`asymgad.graph_data.build_window_graph`
reads when constructing observation graphs.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from ..paths import DATA_ROOT


AUTH_FEATURES = [
    "is_auth",
    "status_success",
    "proto_NTLM",
    "proto_Kerberos",
    "proto_Negotiate",
    "proto_unknown",
]


def extract_service_ports(src_port, dst_port):
    """Return the numeric (non-N) ports among (src_port, dst_port)."""
    ports = []
    for p in (src_port, dst_port):
        p_str = str(p).strip()
        if p_str and not p_str.startswith("N"):
            try:
                ports.append(int(p_str))
            except ValueError:
                pass
    return ports


def build_port_vocab(
    flow_dir: str | Path | None = None,
    top_k: int = 30,
    out_path: str | Path | None = None,
) -> dict:
    """Scan cleaned flow parquet files and write the top-K port vocabulary."""
    flow_dir = Path(flow_dir) if flow_dir else DATA_ROOT / "cleaned" / "flows"
    out_path = Path(out_path) if out_path else DATA_ROOT / "graph" / "feature_names.json"

    if not flow_dir.exists():
        raise FileNotFoundError(f"No flow directory at {flow_dir}")

    files = sorted(f for f in os.listdir(str(flow_dir)) if f.endswith(".parquet"))
    port_counter: Counter = Counter()
    total_events = 0
    t0 = time.time()

    for fi, fp in enumerate(files):
        t = pq.read_table(str(flow_dir / fp))
        detail_col = t.column("event_detail")
        n = t.num_rows
        total_events += n
        for i in range(n):
            try:
                detail = json.loads(detail_col[i].as_py())
            except Exception:
                continue
            for sp in extract_service_ports(
                detail.get("src_port", ""), detail.get("dst_port", "")
            ):
                port_counter[str(sp)] += 1
        if (fi + 1) % 5 == 0:
            print(f"  File {fi + 1:3d}/{len(files)}: "
                  f"{total_events / 1e6:.1f}M events, {time.time() - t0:.0f}s", flush=True)

    top_ports = [p for p, _ in port_counter.most_common(top_k)]
    port_features = [f"port_{p}" for p in top_ports]
    vocab = {
        "top_k_ports": len(top_ports),
        "auth_features": AUTH_FEATURES,
        "port_features": port_features,
        "feature_names": AUTH_FEATURES + port_features,
        "port_index": {p: i for i, p in enumerate(top_ports)},
        "total_flow_events": total_events,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved {len(AUTH_FEATURES) + len(port_features)} features to {out_path}")
    return vocab


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-dir", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--out-path", type=str, default=None)
    args = parser.parse_args()
    build_port_vocab(args.flow_dir, args.top_k, args.out_path)


if __name__ == "__main__":
    main()
