#!/usr/bin/env python3
"""Build .tiling_cache.npz files at grid sizes 2/4/16 for all 10 scenes.

Standalone build of the same cache format test_utility_inmem.py builds on a
cache miss (test_utility_inmem.py:667-680) -- runs it directly instead of
through the full select+render pipeline, since we only need the tiling, not
a render.

Run: conda run -n gaussian_splatting python experiments/0702/build_tiling_caches.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKSPACE = REPO.parent

sys.path.insert(0, str(WORKSPACE / "GGSP"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))

import tiling as ggsp_tiling  # noqa: E402
import io_3dgs  # noqa: E402
from oracle_dq import ply_fingerprint  # noqa: E402
from test_utility_inmem import _build_tile_arrays  # noqa: E402

SCENES = ["chair", "drums", "ficus", "hotdog", "materials", "mic", "ship",
          "bicycle", "garden", "stump"]
GRID_SIZES = [2, 4, 16, 32]
OUT_ROOT = REPO / "output" / "0702" / "tiling_grid_sweep"


def build_one(scene: str, grid_n: int, sel_gs) -> None:
    tile_aabbs, tile_indices, _scene_min, _scene_max = ggsp_tiling.tiling_uniform_layered_gs(
        [sel_gs], grid_shape=(grid_n, grid_n, grid_n)
    )
    min_corners, max_corners, index_offsets, flat_indices, _keys = _build_tile_arrays(
        tile_aabbs, tile_indices, layer_idx=0
    )
    n_gs, xyz_sha1 = ply_fingerprint(
        sel_gs.data["x"]["data"], sel_gs.data["y"]["data"], sel_gs.data["z"]["data"]
    )
    out_path = OUT_ROOT / scene / f"grid{grid_n}" / ".tiling_cache.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path),
              min_corners=min_corners, max_corners=max_corners,
              index_offsets=index_offsets, flat_indices=flat_indices,
              grid_shape=np.array([grid_n, grid_n, grid_n], dtype=np.int32),
              n_gs=np.int64(n_gs), xyz_sha1=xyz_sha1)
    print(f"  grid{grid_n}: {len(index_offsets) - 1} tiles -> {out_path}", flush=True)


def main() -> None:
    for scene in SCENES:
        ply_path = WORKSPACE / "exp-dataset" / scene / "point_cloud.ply"
        print(f"=== {scene} ===", flush=True)
        t0 = time.time()
        sel_gs = io_3dgs.GaussianModelV2(str(ply_path))
        print(f"  loaded {len(sel_gs.data['x']['data'])} gaussians "
              f"({time.time() - t0:.1f}s)", flush=True)
        for grid_n in GRID_SIZES:
            t1 = time.time()
            build_one(scene, grid_n, sel_gs)
            print(f"    ({time.time() - t1:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
