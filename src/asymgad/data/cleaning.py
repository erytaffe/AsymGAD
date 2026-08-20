"""Streaming data-cleaning pipeline for the five quads datasets.

Each dataset is read in batches via PyArrow, transformed into the unified
six-column event schema, and written out as Parquet.  The unified schema
is designed to feed directly into graph-construction tooling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from tqdm import tqdm

from .normalizer import normalize_node, extract_account_and_source
from .parsers import (
    parse_auth_action,
    parse_flows_action,
    parse_proc_action,
    parse_redteam_action,
    is_noise_flow,
)

# -- Shared constants ---
INPUT_COLUMNS = ["timestamp", "src_node", "dst_node", "action_state", "source_file"]

OUTPUT_SCHEMA = pa.schema([
    ("timestamp",       pa.string()),
    ("source_entity",   pa.string()),
    ("destination_entity", pa.string()),
    ("event_category",  pa.string()),
    ("event_type",      pa.string()),
    ("event_detail",    pa.string()),
])

# CSV read options shared across all datasets
_READ_OPTS = pv.ReadOptions(use_threads=True, block_size=10_000_000)
_PARSE_OPTS = pv.ParseOptions(delimiter=",")


def _make_csv_reader(path: Path) -> pv.CSVStreamReader:
    return pv.open_csv(
        str(path),
        read_options=_READ_OPTS,
        parse_options=_PARSE_OPTS,
        convert_options=pv.ConvertOptions(
            column_types={c: pa.string() for c in INPUT_COLUMNS}
        ),
    )


def _write_batch(rows: list[dict], writer: pq.ParquetWriter | None, out_path: Path):
    """Persist a list of row dicts to a Parquet file, creating the writer on first call."""
    if not rows:
        return writer
    arrays = [
        pa.array([r["timestamp"] for r in rows], type=pa.string()),
        pa.array([r["source_entity"] for r in rows], type=pa.string()),
        pa.array([r["destination_entity"] for r in rows], type=pa.string()),
        pa.array([r["event_category"] for r in rows], type=pa.string()),
        pa.array([r["event_type"] for r in rows], type=pa.string()),
        pa.array([r["event_detail"] for r in rows], type=pa.string()),
    ]
    table = pa.table(pa.RecordBatch.from_arrays(arrays, schema=OUTPUT_SCHEMA))
    if writer is None:
        writer = pq.ParquetWriter(str(out_path), OUTPUT_SCHEMA)
    writer.write_table(table)
    return writer


# ---
# Per-dataset cleaning functions
# ---

def clean_auth(
    files: list[Path],
    output_dir: Path,
    chunk_size: int = 500_000,
) -> dict:
    """Clean the **auth** dataset.

    * Splits ``src_node`` into account + source machine.
    * Parses ``action_state`` into action_category / status / subtype / protocol.
    * Normalises all node names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_in": 0, "total_out": 0}

    for fpath in tqdm(files, desc="Cleaning auth", unit="file"):
        reader = _make_csv_reader(fpath)
        out_path = output_dir / f"{fpath.stem}.parquet"
        writer = None

        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            n = batch.num_rows
            stats["total_in"] += n
            rows: list[dict] = []

 # Extract columns once per batch
            timestamps = batch.column("timestamp").to_pylist()
            src_nodes = batch.column("src_node").to_pylist()
            dst_nodes = batch.column("dst_node").to_pylist()
            action_states = batch.column("action_state").to_pylist()

            for i in range(n):
                account, domain, src_machine = extract_account_and_source(src_nodes[i])
                detail = parse_auth_action(action_states[i])
                detail["account"] = account
                detail["account_domain"] = domain
                detail["source_machine_raw"] = src_machine

                rows.append({
                    "timestamp": timestamps[i],
                    "source_entity": src_machine,
                    "destination_entity": normalize_node(dst_nodes[i]),
                    "event_category": "AUTH",
                    "event_type": action_states[i],
                    "event_detail": json.dumps(detail, ensure_ascii=False),
                })

            writer = _write_batch(rows, writer, out_path)
            stats["total_out"] += len(rows)
            del batch, rows

        if writer:
            writer.close()

    return stats


def clean_flows(
    files: list[Path],
    output_dir: Path,
    chunk_size: int = 500_000,
    filter_noise: bool = True,
) -> dict:
    """Clean the **flows** dataset.

    * Parses ``action_state`` into protocol / src_port / dst_port.
    * Optionally removes UDP broadcast noise (NetBIOS 137/138, NTP 123).
    * Normalises all node names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_in": 0, "total_out": 0, "noise_filtered": 0}

    for fpath in tqdm(files, desc="Cleaning flows", unit="file"):
        reader = _make_csv_reader(fpath)
        out_path = output_dir / f"{fpath.stem}.parquet"
        writer = None

        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            n = batch.num_rows
            stats["total_in"] += n
            rows: list[dict] = []

            timestamps = batch.column("timestamp").to_pylist()
            src_nodes = batch.column("src_node").to_pylist()
            dst_nodes = batch.column("dst_node").to_pylist()
            action_states = batch.column("action_state").to_pylist()

            for i in range(n):
                detail = parse_flows_action(action_states[i])
                if filter_noise and is_noise_flow(
                    detail.get("protocol", ""), detail.get("dst_port", "")
                ):
                    stats["noise_filtered"] += 1
                    continue

                rows.append({
                    "timestamp": timestamps[i],
                    "source_entity": normalize_node(src_nodes[i]),
                    "destination_entity": normalize_node(dst_nodes[i]),
                    "event_category": "FLOW",
                    "event_type": action_states[i],
                    "event_detail": json.dumps(detail, ensure_ascii=False),
                })

            writer = _write_batch(rows, writer, out_path)
            stats["total_out"] += len(rows)
            del batch, rows

        if writer:
            writer.close()

    return stats


def clean_dns(
    files: list[Path],
    output_dir: Path,
    chunk_size: int = 500_000,
) -> dict:
    """Clean the **dns** dataset.

    * No composite-field parsing needed.
    * Normalises all node names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_in": 0, "total_out": 0}

    for fpath in tqdm(files, desc="Cleaning dns", unit="file"):
        reader = _make_csv_reader(fpath)
        out_path = output_dir / f"{fpath.stem}.parquet"
        writer = None

        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            n = batch.num_rows
            stats["total_in"] += n
            rows: list[dict] = []

            timestamps = batch.column("timestamp").to_pylist()
            src_nodes = batch.column("src_node").to_pylist()
            dst_nodes = batch.column("dst_node").to_pylist()
            action_states = batch.column("action_state").to_pylist()

            for i in range(n):
                rows.append({
                    "timestamp": timestamps[i],
                    "source_entity": normalize_node(src_nodes[i]),
                    "destination_entity": normalize_node(dst_nodes[i]),
                    "event_category": "DNS",
                    "event_type": action_states[i],
                    "event_detail": json.dumps({"query": "DNS_Query"}, ensure_ascii=False),
                })

            writer = _write_batch(rows, writer, out_path)
            stats["total_out"] += len(rows)
            del batch, rows

        if writer:
            writer.close()

    return stats


def clean_proc(
    files: list[Path],
    output_dir: Path,
    chunk_size: int = 500_000,
    include_self_loops: bool = False,
) -> dict:
    """Clean the **proc** dataset.

    * Strips ``$@DOM1`` suffix from ``src_node``.
    * Self-loops (C5170 -> C5170) are dropped by default.
    * Normalises all node names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_in": 0, "total_out": 0, "self_loops_dropped": 0}

    for fpath in tqdm(files, desc="Cleaning proc", unit="file"):
        reader = _make_csv_reader(fpath)
        out_path = output_dir / f"{fpath.stem}.parquet"
        writer = None

        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            n = batch.num_rows
            stats["total_in"] += n
            rows: list[dict] = []

            timestamps = batch.column("timestamp").to_pylist()
            src_nodes = batch.column("src_node").to_pylist()
            dst_nodes = batch.column("dst_node").to_pylist()
            action_states = batch.column("action_state").to_pylist()

            for i in range(n):
                src_norm = normalize_node(src_nodes[i])
                dst_norm = normalize_node(dst_nodes[i])

                if not include_self_loops and src_norm == dst_norm:
                    stats["self_loops_dropped"] += 1
                    continue

                rows.append({
                    "timestamp": timestamps[i],
                    "source_entity": src_norm,
                    "destination_entity": dst_norm,
                    "event_category": "PROC",
                    "event_type": action_states[i],
                    "event_detail": json.dumps(
                        parse_proc_action(action_states[i]), ensure_ascii=False
                    ),
                })

            writer = _write_batch(rows, writer, out_path)
            stats["total_out"] += len(rows)
            del batch, rows

        if writer:
            writer.close()

    return stats


def clean_redteam(
    files: list[Path],
    output_dir: Path,
    chunk_size: int = 500_000,
) -> dict:
    """Clean the **redteam** dataset.

    * Extracts the pivot / jump-host from ``action_state``.
    * Uses the extracted pivot machine as *source_entity*.
    * Normalises all node names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total_in": 0, "total_out": 0}

    for fpath in tqdm(files, desc="Cleaning redteam", unit="file"):
        reader = _make_csv_reader(fpath)
        out_path = output_dir / f"{fpath.stem}.parquet"
        writer = None

        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            n = batch.num_rows
            stats["total_in"] += n
            rows: list[dict] = []

            timestamps = batch.column("timestamp").to_pylist()
            src_nodes = batch.column("src_node").to_pylist()
            dst_nodes = batch.column("dst_node").to_pylist()
            action_states = batch.column("action_state").to_pylist()

            for i in range(n):
                detail = parse_redteam_action(action_states[i])
                pivot = detail.get("pivot_machine", normalize_node(src_nodes[i]))

                rows.append({
                    "timestamp": timestamps[i],
                    "source_entity": normalize_node(pivot),
                    "destination_entity": normalize_node(dst_nodes[i]),
                    "event_category": "ALERT",
                    "event_type": action_states[i],
                    "event_detail": json.dumps(detail, ensure_ascii=False),
                })

            writer = _write_batch(rows, writer, out_path)
            stats["total_out"] += len(rows)
            del batch, rows

        if writer:
            writer.close()

    return stats


# ---
# Master pipeline
# ---

def run_pipeline(
    data_dir: str | Path = "data/quads",
    output_dir: str | Path = "data/cleaned",
    include_dns: bool = False,
    include_proc: bool = False,
    chunk_size: int = 500_000,
) -> dict:
    """Run the full cleaning pipeline across all configured datasets.

    Parameters
    ----------
    data_dir : Path
        Directory containing the raw ``*_partNNN.csv`` files.
    output_dir : Path
        Directory where per-dataset Parquet directories are written.
    include_dns : bool
        If False (default), skip the DNS dataset.
    include_proc : bool
        If False (default), skip the Proc dataset (self-loops).
    chunk_size : int
        Rows per PyArrow batch.

    Returns
    -------
    dict
        Mapping dataset name -> stats dict.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}
    total_start = time.time()

 # Helper: find and sort part files
    def _find(prefix: str) -> list[Path]:
        import re
        files = sorted(
            data_dir.glob(f"{prefix}_part*.csv"),
            key=lambda p: int(re.search(r"_part(\d+)", p.name).group(1)),
        )
        return files

 # -- Auth --
    auth_files = _find("auth")
    if auth_files:
        print(f"\n{'='*60}\nCleaning [auth]: {len(auth_files)} files\n{'='*60}")
        t0 = time.time()
        all_stats["auth"] = clean_auth(auth_files, output_dir / "auth", chunk_size)
        print(f"  Done in {time.time()-t0:.0f}s: {all_stats['auth']}")

 # -- Flows --
    flows_files = _find("flows")
    if flows_files:
        print(f"\n{'='*60}\nCleaning [flows]: {len(flows_files)} files\n{'='*60}")
        t0 = time.time()
        all_stats["flows"] = clean_flows(flows_files, output_dir / "flows", chunk_size)
        print(f"  Done in {time.time()-t0:.0f}s: {all_stats['flows']}")

 # -- DNS (optional) --
    dns_files = _find("dns")
    if dns_files and include_dns:
        print(f"\n{'='*60}\nCleaning [dns]: {len(dns_files)} files\n{'='*60}")
        t0 = time.time()
        all_stats["dns"] = clean_dns(dns_files, output_dir / "dns", chunk_size)
        print(f"  Done in {time.time()-t0:.0f}s: {all_stats['dns']}")
    elif dns_files:
        print(f"\n[Skipping dns: {len(dns_files)} files (include_dns=False)]")

 # -- Proc (optional) --
    proc_files = _find("proc")
    if proc_files and include_proc:
        print(f"\n{'='*60}\nCleaning [proc]: {len(proc_files)} files\n{'='*60}")
        t0 = time.time()
        all_stats["proc"] = clean_proc(proc_files, output_dir / "proc", chunk_size)
        print(f"  Done in {time.time()-t0:.0f}s: {all_stats['proc']}")
    elif proc_files:
        print(f"\n[Skipping proc: {len(proc_files)} files (include_proc=False)]")

 # -- Redteam (always included - tiny) --
    redteam_files = _find("redteam")
    if redteam_files:
        print(f"\n{'='*60}\nCleaning [redteam]: {len(redteam_files)} files\n{'='*60}")
        t0 = time.time()
        all_stats["redteam"] = clean_redteam(redteam_files, output_dir / "redteam", chunk_size)
        print(f"  Done in {time.time()-t0:.0f}s: {all_stats['redteam']}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {total_elapsed/60:.1f} min")
    _print_summary(all_stats)
    return all_stats


def _print_summary(all_stats: dict) -> None:
    print("\n--- Cleaning Summary ---")
    total_in = 0
    total_out = 0
    for name, s in all_stats.items():
        rows_in = s.get("total_in", 0)
        rows_out = s.get("total_out", 0)
        pct = (rows_out / rows_in * 100) if rows_in else 0
        extra = ""
        if "noise_filtered" in s and s["noise_filtered"]:
            extra += f"  noise_filtered: {s['noise_filtered']:,}"
        if "self_loops_dropped" in s and s["self_loops_dropped"]:
            extra += f"  self_loops_dropped: {s['self_loops_dropped']:,}"
        print(f"  {name}: {rows_in:,} -> {rows_out:,} ({pct:.1f}%){extra}")
        total_in += rows_in
        total_out += rows_out
    print(f"  TOTAL: {total_in:,} -> {total_out:,} ({(total_out/total_in*100) if total_in else 0:.1f}%)")
