"""Print a scene x field table from per-scene diagnostic JSONs (e.g. diag_marginal_corr.json).

Replaces the ad-hoc python3 -c loop retyped ~4x in one session to pull
signals.<name>.median across all 10 scenes.

Usage:
    python3 experiments/print_diag_table.py \\
        "output/0703/diag_marginal_corr/*/diag_marginal_corr.json" \\
        signals.baseline_key.median signals.v_lod_w_key.median signals.N_k_raw.median
"""
import argparse
import glob
import json
from pathlib import Path


def _get(d, dotted_path):
    for key in dotted_path.split("."):
        d = d[key]
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_glob", help="glob matching one JSON per scene")
    parser.add_argument("fields", nargs="+", help="dotted paths, e.g. signals.W_k_raw.median")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.json_glob))
    if not paths:
        raise SystemExit(f"no files matched: {args.json_glob}")

    for p in paths:
        scene = Path(p).parent.name
        d = json.loads(Path(p).read_text())
        row = []
        for f in args.fields:
            v = _get(d, f)
            row.append(f"{v:+.3f}" if isinstance(v, float) else str(v))
        print(scene.ljust(10), *[f"{f.split('.')[-2] if '.' in f else f}={v}"
                                  for f, v in zip(args.fields, row)])


if __name__ == "__main__":
    main()
