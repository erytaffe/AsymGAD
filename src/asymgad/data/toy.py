"""Generate a small, fully self-contained toy dataset for smoke tests.

The toy dataset mimics the LANL-derived schema used by the full
pipeline: cleaned parquet files for ``auth``, ``flows``, and ``redteam``
plus the port vocabulary.  No external data is required.

The stream contains three observation windows of roughly equal size;
the middle window contains an injected pivot (``C5``) that exhibits
scanner-like fanout, so the end-to-end test can verify that ranking
recovers the injected pivot without labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..paths import DATA_ROOT


SCHEMA = pa.schema([
    ("timestamp", pa.string()),
    ("source_entity", pa.string()),
    ("destination_entity", pa.string()),
    ("event_category", pa.string()),
    ("event_type", pa.string()),
    ("event_detail", pa.string()),
])

AUTH_FEATURES = [
    "is_auth", "status_success", "proto_NTLM", "proto_Kerberos",
    "proto_Negotiate", "proto_unknown",
]

PORT_FEATURES = [
    "port_445", "port_389", "port_80", "port_88", "port_139", "port_135",
    "port_443", "port_22", "port_1433", "port_2049", "port_111", "port_161",
    "port_7002", "port_6002", "port_137", "port_3306", "port_1241",
    "port_8080", "port_8081", "port_1094", "port_427", "port_53",
    "port_2989", "port_2432", "port_1300", "port_3493", "port_2000",
    "port_1434", "port_1109", "port_2010",
]


def _write_parquet(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "timestamp": [r["timestamp"] for r in rows],
            "source_entity": [r["source_entity"] for r in rows],
            "destination_entity": [r["destination_entity"] for r in rows],
            "event_category": [r["event_category"] for r in rows],
            "event_type": [r["event_type"] for r in rows],
            "event_detail": [r["event_detail"] for r in rows],
        },
        schema=SCHEMA,
    )
    pq.write_table(table, path)


def make_toy_data(data_root: str | Path | None = None, seed: int = 0) -> Path:
    """Write the toy parquet files and return the data root directory."""
    root = Path(data_root) if data_root else DATA_ROOT
    rng = np.random.default_rng(seed)

    auth_rows: list[dict] = []
    flow_rows: list[dict] = []

    window_ts = [(1, 100), (101, 200), (201, 300)]
    window_srcs = [["C1", "C2", "C3", "C4", "C6", "C7", "C8", "C9", "C10",
                    "C11", "C12", "C13", "C14", "C15", "C16", "C17"],
                   ["C1", "C2", "C3", "C4", "C6", "C7", "C8", "C9", "C10",
                    "C11", "C12", "C13", "C14", "C15", "C16", "C17"],
                   ["C1", "C2", "C3", "C4", "C6", "C7", "C8", "C9", "C10",
                    "C11", "C12", "C13", "C14", "C15", "C16", "C17"]]
    window_dsts = [["C101", "C102", "C103", "C104", "C105", "C106", "C107",
                    "C108", "C109", "C110", "C111", "C112", "C113", "C114",
                    "C115", "C116", "C117", "C118", "C119", "C120"]] * 3

    protocols = ["NTLM", "Kerberos", "Negotiate"]
    statuses = ["Success", "Success", "Failure"]

    ts = 1
    for wi, (ts_lo, ts_hi) in enumerate(window_ts):
        srcs = window_srcs[wi]
        dsts = window_dsts[wi]
        n_normal = 100
        for _ in range(n_normal):
            src = str(rng.choice(srcs))
            dst = str(rng.choice(dsts))
            if src == dst:
                dst = "C200"
            proto = str(rng.choice(protocols))
            status = str(rng.choice(statuses))
            detail = {
                "action_category": "LogOn",
                "status": status,
                "subtype": "Network",
                "protocol": proto,
                "account": f"U{rng.integers(1, 50)}",
                "account_domain": "DOM1",
                "source_machine_raw": src,
            }
            auth_rows.append({
                "timestamp": str(ts),
                "source_entity": src,
                "destination_entity": dst,
                "event_category": "AUTH",
                "event_type": f"LogOn_{status}_Network_{proto}",
                "event_detail": json.dumps(detail),
            })
            # one flow record for roughly half of the auth events
            if rng.random() < 0.5:
                port = str(rng.choice(["445", "389", "88", "80", "443", "139"]))
                flow_rows.append({
                    "timestamp": str(ts),
                    "source_entity": src,
                    "destination_entity": dst,
                    "event_category": "FLOW",
                    "event_type": f"NetFlow_6_{port}>N1",
                    "event_detail": json.dumps({"protocol": "6", "src_port": port, "dst_port": "N1"}),
                })
            ts += 1

        if wi == 1:  # injected pivot in the middle window (ts 140..159)
            for j in range(20):
                dst = f"C3{j:02d}" if j < 10 else f"C4{j - 10:02d}"
                detail = {
                    "action_category": "LogOn",
                    "status": "Success",
                    "subtype": "Network",
                    "protocol": "Kerberos",
                    "account": "U7",
                    "account_domain": "DOM1",
                    "source_machine_raw": "C5",
                }
                auth_rows.append({
                    "timestamp": str(140 + j),
                    "source_entity": "C5",
                    "destination_entity": dst,
                    "event_category": "AUTH",
                    "event_type": "LogOn_Success_Network_Kerberos",
                    "event_detail": json.dumps(detail),
                })
                flow_rows.append({
                    "timestamp": str(140 + j),
                    "source_entity": "C5",
                    "destination_entity": dst,
                    "event_category": "FLOW",
                    "event_type": "NetFlow_6_445>N1",
                    "event_detail": json.dumps({"protocol": "6", "src_port": "445", "dst_port": "N1"}),
                })

    # redteam labels (evaluation-only): C5 is active around ts=150
    redteam_rows = []
    for k in range(5):
        redteam_rows.append({
            "timestamp": str(150 + k),
            "source_entity": "C5",
            "destination_entity": f"C30{k}",
            "event_category": "ALERT",
            "event_type": "RedTeam_via_C5",
            "event_detail": json.dumps({"pivot_machine": "C5"}),
        })

    auth_rows.sort(key=lambda r: int(r["timestamp"]))
    flow_rows.sort(key=lambda r: int(r["timestamp"]))

    _write_parquet(auth_rows, root / "cleaned" / "auth" / "auth_part001.parquet")
    _write_parquet(flow_rows, root / "cleaned" / "flows" / "flows_part001.parquet")
    _write_parquet(redteam_rows, root / "cleaned" / "redteam" / "redteam_part001.parquet")

    vocab = {
        "top_k_ports": len(PORT_FEATURES),
        "auth_features": AUTH_FEATURES,
        "port_features": PORT_FEATURES,
        "feature_names": AUTH_FEATURES + PORT_FEATURES,
        "port_index": {p.replace("port_", ""): i for i, p in enumerate(PORT_FEATURES)},
    }
    (root / "graph").mkdir(parents=True, exist_ok=True)
    with open(root / "graph" / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    return root


def toy_thresholds() -> dict:
    """Thresholds that yield three windows on the toy stream."""
    return {
        "M_min": 90,
        "E_min": 40,
        "V_min_src": 15,
        "V_min_dst": 15,
    }
