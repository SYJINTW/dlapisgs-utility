#!/usr/bin/env python3
"""Aggregate per-stage timings from a sweep directory into a readable CSV.

Reads:
  <output_root>/timings.json           (per-camera selection stages, from test_utility.py)
  <output_root>/render_timings.json    (per-camera render+metrics stages, from render_metrics.py)

Writes:
  <output_root>/metrics/timings_summary.csv

Columns: source, stage, n, mean_sec, std_sec, ci95_sec, p50_sec, p95_sec, total_sec.

Also emits an "e2e" pseudo-stage per source: sum of per-camera stage times grouped
by (camera, scheme, budget_mb) when those labels exist — useful for "wall time per
selection" type questions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _ci95(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    return 1.96 * sd / math.sqrt(n)


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]


def _stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean_sec": float("nan"), "std_sec": float("nan"),
                "ci95_sec": float("nan"), "p50_sec": float("nan"),
                "p95_sec": float("nan"), "total_sec": 0.0}
    mean = sum(vals) / n
    if n >= 2:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = float("nan")
    return {
        "n": n,
        "mean_sec": mean,
        "std_sec": std,
        "ci95_sec": _ci95(vals),
        "p50_sec": _percentile(vals, 0.5),
        "p95_sec": _percentile(vals, 0.95),
        "total_sec": sum(vals),
    }


def _aggregate_one(path: Path, source: str) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    by_stage: dict[str, list[float]] = defaultdict(list)
    # per-camera totals (only for rows that have a camera label)
    per_camera_total: dict[int, float] = defaultdict(float)
    for r in rows:
        stage = r["stage"]
        t = float(r["t_sec"])
        by_stage[stage].append(t)
        cam = r.get("camera")
        if cam is not None:
            per_camera_total[int(cam)] += t

    out = []
    for stage, vals in sorted(by_stage.items()):
        out.append({"source": source, "stage": stage, **_stats(vals)})
    if per_camera_total:
        out.append({"source": source, "stage": "_per_camera_e2e",
                    **_stats(list(per_camera_total.values()))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Override CSV destination. Default: <output_root>/metrics/timings_summary.csv")
    args = ap.parse_args()

    out_csv = args.out_csv or (args.output_root / "metrics" / "timings_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += _aggregate_one(args.output_root / "timings.json",        source="select")
    rows += _aggregate_one(args.output_root / "render_timings.json", source="render")

    if not rows:
        print(f"[aggregate_timings] no timings.json or render_timings.json under {args.output_root}")
        return

    fieldnames = ["source", "stage", "n", "mean_sec", "std_sec", "ci95_sec",
                  "p50_sec", "p95_sec", "total_sec"]
    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"[aggregate_timings] wrote {len(rows)} rows -> {out_csv}")

    # Also print a compact human view
    print()
    print(f"{'source':<8} {'stage':<22} {'n':>4} {'mean(s)':>10} {'±ci95':>9} {'p95':>8} {'total':>10}")
    print("-" * 75)
    for r in rows:
        print(f"{r['source']:<8} {r['stage']:<22} {r['n']:>4} "
              f"{r['mean_sec']:>10.3f} {r['ci95_sec']:>9.3f} "
              f"{r['p95_sec']:>8.2f} {r['total_sec']:>10.1f}")


if __name__ == "__main__":
    main()
