#!/usr/bin/env python3
"""Grounded ETA for a scene x grid x packing sweep, from real reference timings.json files
-- not a guess, not a single-scene extrapolation.

Real incident (2026-07-15): estimated a sweep's cost from chair's timings.json alone and got
it 4x wrong for real scenes (bicycle/garden/stump cost ~4x more per combo than synthetic).
This script sums *every* reference timings.json passed in, so scene-cost variance is real
data, not assumed-uniform.

Usage:
  python experiments/sweep_eta.py \
      --timings-glob "output/0704/quality_sweep/*/grid8_tile_partial/timings.json" \
      --grid-multiplier 5 --n-gpus 2

Assumes grid-flat cost (same combo costs ~the same at any grid size) unless you pass
reference timings.json at more than one grid and let --timings-glob cover them directly
(then omit --grid-multiplier, it defaults to 1 and the glob IS the full plan).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def _scene_from_path(p: str) -> str:
    # output/.../<scene>/grid{N}_<packing>/timings.json
    return Path(p).parts[-3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timings-glob", required=True, help="glob matching reference timings.json files")
    ap.add_argument("--grid-multiplier", type=int, default=1,
                     help="multiply each reference file's cost by this many grids (grid-flat assumption)")
    ap.add_argument("--n-gpus", type=int, default=2)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.timings_glob))
    if not paths:
        raise SystemExit(f"no files matched: {args.timings_glob}")

    per_scene: dict[str, float] = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        total = sum(e.get("t_sec", 0.0) for e in d)
        per_scene[_scene_from_path(p)] = per_scene.get(_scene_from_path(p), 0.0) + total

    for scene, cost in sorted(per_scene.items(), key=lambda kv: -kv[1]):
        scaled = cost * args.grid_multiplier
        print(f"  {scene:12s} {cost:8.1f}s/combo-set  x{args.grid_multiplier} grids = {scaled/3600:.2f}h")

    total = sum(per_scene.values()) * args.grid_multiplier
    print(f"\nTotal: {total/3600:.2f}h  |  ideal /{args.n_gpus} GPUs = {total/args.n_gpus/3600:.2f}h")

    # greedy balanced split: largest-first bin packing across n_gpus
    bins = [[] for _ in range(args.n_gpus)]
    bin_costs = [0.0] * args.n_gpus
    for scene, cost in sorted(per_scene.items(), key=lambda kv: -kv[1]):
        scaled = cost * args.grid_multiplier
        i = bin_costs.index(min(bin_costs))
        bins[i].append(scene)
        bin_costs[i] += scaled

    print("\nSuggested balanced split:")
    for i, (scenes, cost) in enumerate(zip(bins, bin_costs)):
        print(f"  GPU slot {i}: {' '.join(scenes)}  ~= {cost/3600:.2f}h")


if __name__ == "__main__":
    main()
