"""Unit test for the CSV -> Parquet cleaning pipeline."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _run():
    tmp = Path(tempfile.mkdtemp(prefix="asymgad_clean_"))
    quads = tmp / "quads"
    cleaned = tmp / "cleaned"
    quads.mkdir()

    rows = [
        ["1", "ANONYMOUS LOGON@C586|C1250", "C586", "LogOn_Success_Network_NTLM", "auth"],
        ["2", "C101$@DOM1|C988", "C988", "LogOn_Success_Network_Kerberos", "auth"],
        ["3", "U66@DOM1|C586", "C1619", "LogOn_Failure_Network_NTLM", "auth"],
    ]
    with open(quads / "auth_part001.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "src_node", "dst_node", "action_state", "source_file"])
        w.writerows(rows)

    from asymgad.data.cleaning import run_pipeline
    import pyarrow.parquet as pq

    stats = run_pipeline(data_dir=quads, output_dir=cleaned,
                         include_dns=False, include_proc=False, chunk_size=500_000)
    assert stats["auth"]["total_in"] == 3, stats
    out = cleaned / "auth" / "auth_part001.parquet"
    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("source_entity").to_pylist() == ["C1250", "C988", "C586"]
    print("CLEANING TEST PASSED")


if __name__ == "__main__":
    _run()
