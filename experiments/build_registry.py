"""Crawl output/ for params.yaml files and emit experiment_registry.csv.

Usage:
    conda run -n gsquic python experiments/build_registry.py \
        --root output/ --out output/experiment_registry.csv
"""
import argparse
import csv
import json
from pathlib import Path

import yaml

COLUMNS = [
    "output_root", "timestamp", "hostname", "scene",
    "weight_mode", "packing_mode", "schemes",
    "budget_pcts", "grid_shape", "num_lod", "img_w", "img_h", "script",
]


def _str(v):
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return "" if v is None else str(v)


def load_params(path: Path) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f)
    run = d.get("run", {})
    args = d.get("args", {})
    return {
        "output_root": str(path.parent),
        "timestamp": _str(run.get("timestamp")),
        "hostname": _str(run.get("hostname")),
        "scene": _str(args.get("scene")),
        "weight_mode": _str(args.get("weight_mode")),
        "packing_mode": _str(args.get("packing_mode")),
        "schemes": _str(args.get("schemes")),
        "budget_pcts": _str(args.get("budget_pct")),
        "grid_shape": _str(args.get("grid_shape")),
        "num_lod": _str(args.get("num_lod")),
        "img_w": _str(args.get("img_w")),
        "img_h": _str(args.get("img_h")),
        "script": _str(run.get("script")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/", help="Root dir to crawl")
    ap.add_argument("--out", default="output/experiment_registry.csv")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in sorted(root.rglob("params.yaml")):
        try:
            rows.append(load_params(p))
        except Exception as e:
            print(f"[skip] {p}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
