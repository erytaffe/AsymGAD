"""End-to-end LANL preprocessing: raw text -> CSV parts -> Parquet.

Steps:
  1. Convert the raw LANL CMCSE ``auth.txt`` / ``flows.txt`` /
     ``redteam.txt`` files into 5-column CSV parts.
  2. Clean each category into the unified 6-column Parquet schema.
  3. Build the destination-port vocabulary used for edge features.

Usage:
    python scripts/run_preprocess.py --raw-dir <lanl_raw_dir>
                                     [--config configs/lanl.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import REPO_ROOT, add_common_args, apply_data_roots, load_config

apply_data_roots(load_config(str(REPO_ROOT / "configs" / "lanl.json")))

from asymgad.data.cleaning import run_pipeline
from asymgad.data.prepare_lanl import prepare_lanl
from asymgad.data.port_vocab import build_port_vocab
from asymgad.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Directory containing LANL CMCSE auth.txt / flows.txt / redteam.txt.")
    parser.add_argument("--quads-dir", type=str, default=None,
                        help="Directory for the intermediate CSV parts (default: <data>/quads).")
    parser.add_argument("--cleaned-dir", type=str, default=None,
                        help="Directory for the cleaned Parquet files (default: <data>/cleaned).")
    parser.add_argument("--rows-per-part", type=int, default=5_000_000)
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip the raw-text to CSV step (CSV parts already exist).")
    parser.add_argument("--skip-clean", action="store_true",
                        help="Skip the CSV to Parquet cleaning step.")
    parser.add_argument("--skip-vocab", action="store_true",
                        help="Skip the port-vocabulary step.")
    args = parser.parse_args()

    quads_dir = Path(args.quads_dir) if args.quads_dir else DATA_ROOT / "quads"
    cleaned_dir = Path(args.cleaned_dir) if args.cleaned_dir else DATA_ROOT / "cleaned"

    if not args.skip_prepare:
        print(f"[1/3] Raw LANL text -> CSV parts ({args.raw_dir})")
        stats = prepare_lanl(args.raw_dir, quads_dir, rows_per_part=args.rows_per_part)
        for cat, s in stats.items():
            print(f"  {cat}: {s}")
    else:
        print("[1/3] Skipping raw-text conversion.")

    if not args.skip_clean:
        print(f"[2/3] CSV parts -> Parquet ({quads_dir})")
        stats = run_pipeline(data_dir=quads_dir, output_dir=cleaned_dir,
                             include_dns=False, include_proc=False,
                             chunk_size=500_000)
        print(f"  Cleaning stats: {stats}")
    else:
        print("[2/3] Skipping cleaning.")

    if not args.skip_vocab:
        print("[3/3] Building port vocabulary")
        build_port_vocab(cleaned_dir / "flows")
    else:
        print("[3/3] Skipping port vocabulary.")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
