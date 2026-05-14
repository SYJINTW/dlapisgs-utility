#!/usr/bin/env python3
"""Concatenate per-cell metrics/summary.csv files into one combined CSV.

Usage:
  python experiments/concat_summaries.py \
      --glob '<sweep_root>/*/metrics/summary.csv' \
      --out  <sweep_root>/summary_all.csv

Header is taken from the first matching file. Files with a different column
order are realigned to that header; missing columns are written as empty.
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True,
                    help="Glob pattern matching summary.csv files (quote it).")
    ap.add_argument("--out",  required=True, type=Path,
                    help="Destination combined CSV.")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in _glob.glob(args.glob, recursive=True))
    if not paths:
        print(f"[concat_summaries] no matches for: {args.glob}")
        return

    header: list[str] | None = None
    rows: list[dict] = []
    for p in paths:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)

    if not rows or not header:
        print(f"[concat_summaries] no rows across {len(paths)} files")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in header})
    print(f"[concat_summaries] wrote {len(rows)} rows from {len(paths)} files -> {args.out}")


if __name__ == "__main__":
    main()
