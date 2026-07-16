#!/usr/bin/env python3
"""Measure bz2 compression ratio on raw Gaussian attribute bytes for a handful of
representative tile-sized subsets (bicycle @ 8^3 tiling). Measurement only -- not
coupled into the streaming_sim.py hot loop (see .claude/PLAN.md fast-cadence entry
for why: folding a compression ratio into the effective per-tick byte budget is a
separate, unanswered design decision).

Usage:
  conda run -n gaussian_splatting python experiments/measure_gs_compression.py
"""
from __future__ import annotations

import bz2
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
WORKSPACE = HERE.parent

sys.path.insert(0, str(HERE))
import selection_core as sc  # noqa: E402

RENDERER_ROOT = WORKSPACE / "LapisGS-object-based-renderer"
sys.path.insert(0, str(RENDERER_ROOT))
import os  # noqa: E402
_DUMMY = WORKSPACE / "exp-dataset" / "chair" / "predictions" / "color" / "test" / "r_0.png"
os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))
from gaussian_renderer_lapisgs import GaussianModel  # noqa: E402

PLY = WORKSPACE / "exp-dataset/bicycle/point_cloud.ply"
TILING_CACHE = HERE / "output/oracle_tiling_cache/bicycle_8x8x8.npz"
ATTRS = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]


def raw_bytes_for(gs) -> bytes:
    chunks = [getattr(gs, a).detach().cpu().numpy().tobytes() for a in ATTRS]
    return b"".join(chunks)


def main():
    gs = GaussianModel(3)
    gs.load_ply(str(PLY))

    tc = np.load(str(TILING_CACHE), allow_pickle=True)
    off = tc["index_offsets"]
    flat = tc["flat_indices"]
    n_gs_per_tile = np.diff(off)

    tile_smallest = int(np.argmin(n_gs_per_tile))
    tile_median = int(np.argsort(n_gs_per_tile)[len(n_gs_per_tile) // 2])
    tile_largest = int(np.argmax(n_gs_per_tile))

    samples = {}
    for name, t in [("smallest_tile", tile_smallest), ("median_tile", tile_median),
                     ("largest_tile", tile_largest)]:
        idx = flat[off[t]:off[t + 1]]
        samples[f"{name} ({len(idx)} GS)"] = idx

    n_total = len(gs.get_xyz)
    rng = np.random.default_rng(0)
    for n in (75_000, 150_000, 300_000):
        samples[f"cadence_window_{n}gs"] = rng.choice(n_total, size=n, replace=False)

    print(f"{'sample':<28} {'raw MB':>10} {'compressed MB':>14} {'ratio':>8}")
    for name, idx in samples.items():
        sub = sc.subset_gaussians(gs, np.asarray(idx))
        raw = raw_bytes_for(sub)
        compressed = bz2.compress(raw, compresslevel=9)
        ratio = len(raw) / len(compressed)
        print(f"{name:<28} {len(raw)/1e6:>10.2f} {len(compressed)/1e6:>14.2f} {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
