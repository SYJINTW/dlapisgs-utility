#!/usr/bin/env python3
"""Concatenate per-cell metrics/summary.csv files into one combined CSV.

Usage (single source, original form):
  python experiments/concat_summaries.py \
      --glob '<sweep_root>/*/metrics/summary.csv' \
      --out  <sweep_root>/summary_all.csv

Usage (multiple heterogeneous sources, each tagged with a column the source
CSVs don't already have -- e.g. combining a `full`-scope sweep and a
`visible`-scope sweep from two different output dirs into one CSV with a
`scope` column for downstream --group-by):
  python experiments/concat_summaries.py \
      --source '<full_root>/*/metrics/summary.csv:scope=full' \
      --source '<visible_root>/*/metrics/summary.csv:scope=visible' \
      --out combined.csv

Header is taken from the first matching file (plus any --source tag columns).
Files with a different column order are realigned to that header; missing
columns are written as empty.
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
from pathlib import Path


def _load_rows(pattern: str) -> tuple[list[str], list[dict]]:
    paths = sorted(Path(p) for p in _glob.glob(pattern, recursive=True))
    header: list[str] | None = None
    rows: list[dict] = []
    for p in paths:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = list(reader.fieldnames or [])
            rows.extend(reader)
    return header or [], rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob",
                    help="Glob pattern matching summary.csv files (quote it). "
                         "Single-source form; mutually exclusive with --source.")
    ap.add_argument("--source", action="append", default=None, metavar="GLOB:COL=VALUE",
                    help="Repeatable. Each source's rows get COL=VALUE added before "
                         "concatenation -- use when combining sweeps whose distinguishing "
                         "condition (e.g. gs-weight-scope) isn't already a CSV column.")
    ap.add_argument("--out",  required=True, type=Path,
                    help="Destination combined CSV.")
    args = ap.parse_args()

    if bool(args.glob) == bool(args.source):
        raise SystemExit("[concat_summaries] pass exactly one of --glob or --source")

    header: list[str] | None = None
    rows: list[dict] = []

    if args.glob:
        header, rows = _load_rows(args.glob)
        if not rows:
            print(f"[concat_summaries] no matches for: {args.glob}")
            return
    else:
        tag_cols: list[str] = []
        for spec in args.source:
            pattern, _, tag = spec.partition(":")
            col, _, val = tag.partition("=")
            if not tag or not col:
                raise SystemExit(f"[concat_summaries] bad --source spec (want GLOB:COL=VALUE): {spec!r}")
            src_header, src_rows = _load_rows(pattern)
            if not src_rows:
                print(f"[concat_summaries] no matches for: {pattern}")
                continue
            for row in src_rows:
                row[col] = val
            if col not in tag_cols:
                tag_cols.append(col)
            if header is None:
                header = src_header
            rows.extend(src_rows)
        if header is not None:
            header = header + [c for c in tag_cols if c not in header]
        if not rows or not header:
            print("[concat_summaries] no rows across any --source")
            return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in header})
    print(f"[concat_summaries] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
