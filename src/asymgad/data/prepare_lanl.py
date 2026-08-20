"""Convert the raw LANL CMCSE text files into graph-ready CSV parts.

The LANL "Comprehensive, Multi-Source Cyber-Security Events" (CMCSE)
dataset provides one raw text file per event category.  This module
converts the auth, flows, and redteam files into the 5-column CSV format
consumed by :mod:`asymgad.data.cleaning`:

    timestamp,src_node,dst_node,action_state,source_file

Raw field mappings (LANL CMCSE):

auth.txt
    time,user@domain,user@domain,src_comp,dst_comp,
    auth_type,logon_type,auth_orientation,success
    -> src_node = "user@domain|src_comp"
    -> action_state = "{auth_orientation}_{success}_{logon_type}_{auth_type}"

flows.txt
    time,?,src_comp,src_port,dst_comp,dst_port,protocol,?,?
    -> action_state = "NetFlow_{protocol}_{src_port}>{dst_port}"

redteam.txt
    time,user@domain,pivot,target
    -> src_node = user@domain, dst_node = target
    -> action_state = "RedTeam_via_{pivot}"

The conversion streams line by line so the multi-GB LANL files can be
processed with bounded memory, and splits each category into parts of
``rows_per_part`` rows named ``<category>_partNNN.csv``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List, Tuple


_PART_RE = re.compile(r"_part(\d+)")


def _parts_paths(data_dir: Path, prefix: str) -> List[Path]:
    return sorted(
        data_dir.glob(f"{prefix}_part*.csv"),
        key=lambda p: int(_PART_RE.search(p.name).group(1)),
    )


def _writer_for(out_dir: Path, prefix: str, part_no: int, rows_per_part: int):
    """Return a CSV writer and the current output path."""
    out_path = out_dir / f"{prefix}_part{part_no:03d}.csv"
    fh = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(["timestamp", "src_node", "dst_node", "action_state", "source_file"])
    return writer, fh, out_path


def _stream_rows(raw_path: Path, converter, category: str, rows_per_part: int,
                 out_dir: Path) -> Tuple[int, int]:
    """Stream a raw file through ``converter`` into numbered CSV parts."""
    n_total = 0
    n_parts = 0
    writer = None
    fh = None
    out_path = None
    with open(raw_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            if not line.strip():
                continue
            row = converter(line, category)
            if row is None:
                continue
            if writer is None or (n_total % rows_per_part == 0 and n_total > 0):
                if fh is not None:
                    fh.close()
                n_parts += 1
                writer, fh, out_path = _writer_for(out_dir, category, n_parts, rows_per_part)
            writer.writerow(row)
            n_total += 1
    if fh is not None:
        fh.close()
    return n_total, n_parts


def _convert_auth(line: str, _category: str):
    fields = line.strip().split(",")
    if len(fields) < 9:
        return None
    ts, user, _dup, src_comp, dst_comp, auth_type, logon_type, orientation, success = fields[:9]
    action_state = f"{orientation}_{success}_{logon_type}_{auth_type}"
    return [ts, f"{user}|{src_comp}", dst_comp, action_state, "auth"]


def _convert_flows(line: str, _category: str):
    fields = line.strip().split(",")
    if len(fields) < 7:
        return None
    ts, _flag, src_comp, src_port, dst_comp, dst_port, protocol = fields[:7]
    action_state = f"NetFlow_{protocol}_{src_port}>{dst_port}"
    return [ts, src_comp, dst_comp, action_state, "flows"]


def _convert_redteam(line: str, _category: str):
    fields = line.strip().split(",")
    if len(fields) < 4:
        return None
    ts, user, pivot, target = fields[:4]
    action_state = f"RedTeam_via_{pivot}"
    return [ts, user, target, action_state, "redteam"]


CONVERTERS = {
    "auth": _convert_auth,
    "flows": _convert_flows,
    "redteam": _convert_redteam,
}


def prepare_lanl(
    raw_dir: str | Path,
    out_dir: str | Path,
    categories: Iterable[str] = ("auth", "flows", "redteam"),
    rows_per_part: int = 5_000_000,
) -> dict:
    """Convert raw LANL CMCSE files under ``raw_dir`` to CSV parts.

    Expected raw files (LANL CMCSE):
        raw_dir/auth.txt, raw_dir/flows.txt, raw_dir/redteam.txt
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for cat in categories:
        converter = CONVERTERS[cat]
        raw_path = raw_dir / f"{cat}.txt"
        if not raw_path.exists():
            stats[cat] = {"error": f"{raw_path.name} not found"}
            continue
        n_total, n_parts = _stream_rows(
            raw_path, converter, cat, rows_per_part, out_dir
        )
        stats[cat] = {"rows": n_total, "parts": n_parts}
    return stats


def main():
    import argparse
    from ..paths import DATA_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Directory containing the raw LANL CMCSE .txt files.")
    parser.add_argument("--out-dir", type=str, default=str(DATA_ROOT / "quads"),
                        help="Directory for the generated CSV parts.")
    parser.add_argument("--rows-per-part", type=int, default=5_000_000)
    args = parser.parse_args()
    stats = prepare_lanl(args.raw_dir, args.out_dir, rows_per_part=args.rows_per_part)
    for cat, s in stats.items():
        print(f"{cat}: {s}")


if __name__ == "__main__":
    main()
