#!/usr/bin/env python3
"""Per-camera visible-tile-fraction diagnostic, no rendering/LOO/scoring.

For each scene x grid size, loads a tiling cache and the 150-cam eval trace,
runs batched_check_tiles_visible per camera, and records visible_frac. Cheap
gate to decide whether a finer/coarser grid is worth an oracle_online timing
rerun (see .claude/handover.md / plan for 2026-07-02).

Run: conda run -n gaussian_splatting python experiments/0702/diag_visible_frac.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKSPACE = REPO.parent

sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))

import visibility_AABB_pytorch  # noqa: E402

SCENES = ["chair", "drums", "ficus", "hotdog", "materials", "mic", "ship",
          "bicycle", "garden", "stump"]
GRID_SIZES = [2, 4, 8, 16, 32]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = REPO / "output" / "0702" / "visible_frac_grid_sweep"

EXISTING_8_CACHE = REPO / "output" / "0605" / "exp1_gs_weights" / "{scene}" / ".tiling_cache.npz"
NEW_CACHE = REPO / "output" / "0702" / "tiling_grid_sweep" / "{scene}" / "grid{grid}" / ".tiling_cache.npz"


def cache_path(scene: str, grid_n: int) -> Path:
    if grid_n == 8:
        return Path(str(EXISTING_8_CACHE).format(scene=scene))
    return Path(str(NEW_CACHE).format(scene=scene, grid=grid_n))


def main() -> None:
    print(f"device={DEVICE}", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for scene in SCENES:
        trace_path = WORKSPACE / "exp-dataset" / scene / "sparse_views_eval.json"
        print(f"=== {scene} ===", flush=True)
        cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
            str(trace_path), 1600, 1600)
        sel_cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
        print(f"  cameras={len(sel_cameras)}", flush=True)

        rows = []
        for grid_n in GRID_SIZES:
            cpath = cache_path(scene, grid_n)
            if not cpath.exists():
                print(f"  grid{grid_n}: MISSING cache {cpath}, skipping", flush=True)
                continue
            tc = np.load(str(cpath))
            min_corners_t = torch.tensor(tc["min_corners"], dtype=torch.float32, device=DEVICE)
            max_corners_t = torch.tensor(tc["max_corners"], dtype=torch.float32, device=DEVICE)
            n_tiles = min_corners_t.shape[0]

            t0 = time.time()
            for cam_idx, cam in enumerate(sel_cameras):
                vis = visibility_AABB_pytorch.batched_check_tiles_visible(
                    min_corners_t, max_corners_t, cam, device=DEVICE)
                n_visible = int(vis.sum().item())
                rows.append({
                    "scene": scene,
                    "grid_size": grid_n,
                    "camera_index": cam_idx,
                    "n_tiles": n_tiles,
                    "n_visible": n_visible,
                    "visible_frac": n_visible / n_tiles if n_tiles else 0.0,
                })
            mean_frac = sum(r["visible_frac"] for r in rows if r["grid_size"] == grid_n) / len(sel_cameras)
            print(f"  grid{grid_n}: n_tiles={n_tiles} mean_visible_frac={mean_frac:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)

        out_csv = OUT_ROOT / scene / "visible_frac.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scene", "grid_size", "camera_index", "n_tiles", "n_visible", "visible_frac"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
