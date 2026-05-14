#!/usr/bin/env python3
"""Aggregate per-stage timings from a sweep cell directory into readable CSVs.

Reads:
  <output_root>/timings.json           (per-stage selection rows, from test_utility.py)
  <output_root>/render_timings.json    (per-stage render+metrics rows, from render_metrics.py)

Writes:
  <output_root>/metrics/timings_stages.csv     — per-stage mean/std/CI/p50/p95/total
  <output_root>/metrics/timings_oneshot.csv    — per (scheme, budget_mb) "oneshot" wall time:
                                                  selection_steady_per_camera
                                                  + select_at_budget
                                                  + render_steady_per_(cam,sch,budg)
                                                 Useful for ETA + streaming real-time check.
  <output_root>/metrics/timings_setup.csv      — one-time setup costs (tiling, gt_render_all, ...)

Streaming feasibility lens: the "select" portion of oneshot is the part that has to
run on the server per viewpoint-change. Render time is client-side and listed alongside
for context.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


# Stage classification — kept here so future stage additions are easy to slot in.
ONE_TIME_SELECT  = {"ply_load", "tiling", "camera_load", "gs_attrs", "tile_npz"}
# Real per-camera scoring/selection compute (streaming-latency-relevant).
PER_CAM_COMPUTE  = {"visibility", "gaussian_weights", "tile_weights",
                    "utility", "greedy"}
# Per-camera disk IO from the experiment runner (NOT relevant to real-time streaming).
PER_CAM_IO       = {"vis_npz", "ply_writes_drained"}
PER_CAM_BUDGET   = {"select_at_budget"}            # tagged with (camera, scheme, budget_mb)
PER_CAM_RENDER   = {"render_selected", "metrics", "png_write"}  # tagged with (camera, scheme, budget_mb)
ONE_TIME_RENDER  = {"gt_ply_load", "gt_render_all"}              # camera_load on render side: also one-time-ish


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
        ci95 = 1.96 * std / math.sqrt(n)
    else:
        std = ci95 = float("nan")
    s = sorted(vals)
    p50 = s[len(s) // 2]
    p95 = s[max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))]
    return {"n": n, "mean_sec": mean, "std_sec": std, "ci95_sec": ci95,
            "p50_sec": p50, "p95_sec": p95, "total_sec": sum(vals)}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--metrics-dir", type=Path, default=None,
                    help="Override CSV destination. Default: <output_root>/metrics/")
    args = ap.parse_args()

    metrics_dir = args.metrics_dir or (args.output_root / "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    select_rows = _load(args.output_root / "timings.json")
    render_rows = _load(args.output_root / "render_timings.json")

    if not select_rows and not render_rows:
        print(f"[aggregate_timings] no timings under {args.output_root}")
        return

    # ---------------------------------------------------------------------- #
    # 1. Per-stage summary  ------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    stage_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in select_rows:
        if "t_sec" in r:
            stage_vals[("select", r["stage"])].append(float(r["t_sec"]))
    for r in render_rows:
        if "t_sec" in r:
            stage_vals[("render", r["stage"])].append(float(r["t_sec"]))

    stages_rows = []
    for (source, stage), vals in sorted(stage_vals.items()):
        stages_rows.append({"source": source, "stage": stage, **_stats(vals)})

    stages_csv = metrics_dir / "timings_stages.csv"
    with stages_csv.open("w", newline="") as fp:
        fields = ["source", "stage", "n", "mean_sec", "std_sec", "ci95_sec",
                  "p50_sec", "p95_sec", "total_sec"]
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in stages_rows:
            w.writerow({k: r.get(k) for k in fields})

    # ---------------------------------------------------------------------- #
    # 2. One-time setup costs (no camera/budget tag)  ---------------------- #
    # ---------------------------------------------------------------------- #
    setup_rows = []
    for r in select_rows + render_rows:
        if r["stage"] in (ONE_TIME_SELECT | ONE_TIME_RENDER) and "t_sec" in r:
            setup_rows.append({"stage": r["stage"], "t_sec": float(r["t_sec"])})
    setup_csv = metrics_dir / "timings_setup.csv"
    if setup_rows:
        with setup_csv.open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=["stage", "t_sec"])
            w.writeheader()
            w.writerows(setup_rows)

    # ---------------------------------------------------------------------- #
    # 3. Per (camera, scheme, budget) oneshot  ------------------------------ #
    # ---------------------------------------------------------------------- #
    # Bucket per-camera selection stages (independent of budget/scheme): once per camera
    per_cam_compute: dict[int, float] = defaultdict(float)
    per_cam_io:      dict[int, float] = defaultdict(float)
    for r in select_rows:
        if "t_sec" not in r:
            continue
        cam = r.get("camera")
        if cam is None:
            continue
        if r["stage"] in PER_CAM_COMPUTE:
            per_cam_compute[int(cam)] += float(r["t_sec"])
        elif r["stage"] in PER_CAM_IO:
            per_cam_io[int(cam)] += float(r["t_sec"])

    # Bucket per-(cam, scheme, budget) stages
    per_cell_select: dict[tuple, float] = defaultdict(float)
    for r in select_rows:
        if r["stage"] in PER_CAM_BUDGET:
            key = (int(r["camera"]), r["scheme"], float(r["budget_mb"]))
            per_cell_select[key] += float(r["t_sec"])

    per_cell_render: dict[tuple, float] = defaultdict(float)
    for r in render_rows:
        if r["stage"] in PER_CAM_RENDER:
            key = (int(r["camera"]), r["scheme"], float(r["budget_mb"]))
            per_cell_render[key] += float(r["t_sec"])

    # Aggregate by (scheme, budget_mb) across cameras
    by_sb_compute:  dict[tuple, list[float]] = defaultdict(list)
    by_sb_io:       dict[tuple, list[float]] = defaultdict(list)
    by_sb_atb:      dict[tuple, list[float]] = defaultdict(list)
    by_sb_render:   dict[tuple, list[float]] = defaultdict(list)
    by_sb_stream:   dict[tuple, list[float]] = defaultdict(list)   # streaming-feasibility: compute + render
    by_sb_oneshot:  dict[tuple, list[float]] = defaultdict(list)   # experiment-runner wall: compute + io + render

    all_cells = set(per_cell_select.keys()) | set(per_cell_render.keys())
    for (cam, scheme, budg) in all_cells:
        sb = (scheme, budg)
        compute = per_cam_compute.get(cam, 0.0)
        io      = per_cam_io.get(cam, 0.0)
        atb     = per_cell_select.get((cam, scheme, budg), 0.0)
        rnd     = per_cell_render.get((cam, scheme, budg), 0.0)
        by_sb_compute[sb].append(compute + atb)
        by_sb_io[sb].append(io)
        by_sb_atb[sb].append(atb)
        by_sb_render[sb].append(rnd)
        by_sb_stream[sb].append(compute + atb + rnd)
        by_sb_oneshot[sb].append(compute + atb + io + rnd)

    oneshot_csv = metrics_dir / "timings_oneshot.csv"
    fields = ["scheme", "budget_mb", "n_cams",
              "compute_mean", "compute_p95",
              "io_mean",      "io_p95",
              "render_mean",  "render_p95",
              "stream_latency_mean", "stream_latency_p95",
              "oneshot_mean", "oneshot_std", "oneshot_ci95",
              "oneshot_p50",  "oneshot_p95"]
    oneshot_summary_rows: list[dict] = []
    with oneshot_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for sb in sorted(by_sb_oneshot.keys()):
            scheme, budg = sb
            s_c  = _stats(by_sb_compute[sb])
            s_io = _stats(by_sb_io[sb])
            s_rd = _stats(by_sb_render[sb])
            s_st = _stats(by_sb_stream[sb])
            s_os = _stats(by_sb_oneshot[sb])
            row = {
                "scheme": scheme, "budget_mb": budg,
                "n_cams": s_os["n"],
                "compute_mean": s_c["mean_sec"],  "compute_p95": s_c["p95_sec"],
                "io_mean":      s_io["mean_sec"], "io_p95":      s_io["p95_sec"],
                "render_mean":  s_rd["mean_sec"], "render_p95":  s_rd["p95_sec"],
                "stream_latency_mean": s_st["mean_sec"], "stream_latency_p95": s_st["p95_sec"],
                "oneshot_mean": s_os["mean_sec"], "oneshot_std":  s_os["std_sec"],
                "oneshot_ci95": s_os["ci95_sec"], "oneshot_p50":  s_os["p50_sec"],
                "oneshot_p95":  s_os["p95_sec"],
            }
            oneshot_summary_rows.append(row)
            w.writerow(row)

    # ---------------------------------------------------------------------- #
    # 4. Human-readable summary  ------------------------------------------- #
    # ---------------------------------------------------------------------- #
    print(f"[aggregate_timings] {args.output_root}")
    print(f"  -> {stages_csv}")
    print(f"  -> {setup_csv}")
    print(f"  -> {oneshot_csv}")

    if setup_rows:
        total_setup = sum(r["t_sec"] for r in setup_rows)
        parts = ", ".join(f"{r['stage']}={r['t_sec']:.1f}s" for r in setup_rows)
        print(f"\nOne-time setup: {total_setup:.1f}s  ({parts})")

    if oneshot_summary_rows:
        total_oneshot = sum(r["oneshot_mean"] * r["n_cams"] for r in oneshot_summary_rows)
        print(f"\nPer (scheme, budget) per-camera timings:")
        print(f"{'scheme':<14} {'budget':>9} {'n':>3} "
              f"{'compute':>8} {'io':>8} {'render':>8} "
              f"{'stream':>8} {'oneshot':>9} {'p95':>7}")
        print("-" * 80)
        for r in oneshot_summary_rows:
            print(f"{r['scheme']:<14} {r['budget_mb']:>9.1f} {r['n_cams']:>3} "
                  f"{r['compute_mean']:>8.3f} {r['io_mean']:>8.2f} {r['render_mean']:>8.3f} "
                  f"{r['stream_latency_mean']:>8.3f} {r['oneshot_mean']:>9.2f} "
                  f"{r['oneshot_p95']:>7.2f}")
        print(f"\nCell-wall total ≈ {total_oneshot:.1f}s  (+ setup ≈ {total_setup if setup_rows else 0:.1f}s)")
        print("\nLegend: 'compute' = scoring + selection (streaming-relevant CPU/GPU work)")
        print("        'io'      = experiment-runner disk writes (NOT in real streaming)")
        print("        'render'  = client-side rasterization of selected subset")
        print("        'stream'  = compute + render  (streaming-feasibility lower bound)")
        print("        'oneshot' = compute + io + render  (predicts wall time of this sweep)")


if __name__ == "__main__":
    main()
