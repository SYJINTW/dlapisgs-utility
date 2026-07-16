#!/usr/bin/env python3
"""Pure Gaussian-downsampling degradation measurement: drop the bottom-N% of Gaussians by
the view-independent `volume` score (utility_calculation.compute_gaussian_weights_v2),
render the pruned scene, and diff against the full-scene GT render. No budget/selection/
streaming logic involved -- this measures what downsampling itself costs, decoupled from
any packing decision, before committing to a scene-size cut for the streaming pipeline
(Workstream A of the scene-size-reduction plan, 2026-07-15).

Two modes:
  per_tile:  within each tile independently, keep the top keep_frac by volume score.
  per_scene: rank all Gaussians globally, keep the top keep_frac scene-wide (tiles with
             uniformly low-volume Gaussians can be wiped entirely -- the hypothesis this
             mode is meant to test).

Run in the gaussian_splatting conda environment (matches test_utility_inmem.py).

Usage:
  python experiments/measure_downsample_degradation.py \
      --ply exp-dataset/bicycle/point_cloud.ply \
      --tiling-cache output/oracle_tiling_cache/bicycle_8x8x8.npz \
      --camera-trace exp-dataset/bicycle/sparse_views_eval.json \
      --scene bicycle --mode per_tile --keep-fracs 0.5 0.33 0.1 \
      --output-root output/0715/downsample_degradation/bicycle/per_tile
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
UTIL_DIR = HERE.parent

sys.path.insert(0, str(UTIL_DIR))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))

import io_3dgs  # noqa: E402
import utility_calculation as uc  # noqa: E402
import selection_core as sc  # noqa: E402

RENDERER_ROOT = WORKSPACE / "LapisGS-object-based-renderer"
if str(RENDERER_ROOT) not in sys.path:
    sys.path.insert(0, str(RENDERER_ROOT))

from gaussian_renderer_lapisgs import GaussianModel  # noqa: E402  # type: ignore
from streaming_utils.camera_loader import load_camera_from_streaming_config  # noqa: E402  # type: ignore


def volume_score(sel_gs) -> torch.Tensor:
    """View-independent per-Gaussian importance: sigmoid(opacity) * volume."""
    return uc.compute_gaussian_weights_v2(
        "volume",
        opacity=sel_gs.data["opacity"]["data"],
        scale_0=sel_gs.data["scale_0"]["data"],
        scale_1=sel_gs.data["scale_1"]["data"],
        scale_2=sel_gs.data["scale_2"]["data"],
    )


def keep_indices_per_scene(score: torch.Tensor, keep_frac: float) -> np.ndarray:
    n_keep = max(1, int(round(len(score) * keep_frac)))
    _, order = torch.sort(score, descending=True, stable=True)
    idx = order[:n_keep].cpu().numpy()
    return np.sort(idx).astype(np.int64)


def keep_indices_per_tile(score: torch.Tensor, index_offsets: np.ndarray,
                           flat_indices: np.ndarray, keep_frac: float) -> np.ndarray:
    kept = []
    n_tiles = len(index_offsets) - 1
    for t in range(n_tiles):
        tile_idx = flat_indices[index_offsets[t]:index_offsets[t + 1]]
        if len(tile_idx) == 0:
            continue
        n_keep = max(1, int(round(len(tile_idx) * keep_frac)))
        tile_score = score[tile_idx]
        _, order = torch.sort(tile_score, descending=True, stable=True)
        kept.append(tile_idx[order[:n_keep].cpu().numpy()])
    if not kept:
        return np.empty((0,), dtype=np.int64)
    return np.sort(np.unique(np.concatenate(kept))).astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--tiling-cache", required=True)
    p.add_argument("--camera-trace", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--mode", choices=["per_tile", "per_scene"], required=True)
    p.add_argument("--keep-fracs", nargs="+", type=float, default=[0.5, 0.33, 0.1])
    p.add_argument("--output-root", required=True)
    p.add_argument("--img-w", type=int, default=1600)
    p.add_argument("--img-h", type=int, default=1600)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--white-bg", action="store_true")
    args = p.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    logger.add(str(out_root / "log.txt"), level="INFO")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sel_gs = io_3dgs.GaussianModelV2(args.ply)
    score = volume_score(sel_gs).to(device)

    tc = np.load(args.tiling_cache, allow_pickle=True)
    index_offsets = np.asarray(tc["index_offsets"], dtype=np.int64)
    flat_indices = np.asarray(tc["flat_indices"], dtype=np.int64)

    full_gs = GaussianModel(args.sh_degree)
    full_gs.load_ply(args.ply)
    n_gs_total = len(full_gs.get_xyz)

    frames = sc.load_trace(args.camera_trace)
    rend_cameras = [
        load_camera_from_streaming_config(f, width=args.img_w, height=args.img_h)
        for f in frames
    ]
    for cam in rend_cameras:
        cam.original_image = None

    logger.info("Pre-rendering GT ({} cameras) ...", len(rend_cameras))
    gt_renders = []
    with torch.no_grad():
        for cam in tqdm(rend_cameras, desc="gt_render", leave=False):
            gt_renders.append(sc.render_gs(full_gs, cam, args.white_bg).cpu())

    rows = []
    for keep_frac in args.keep_fracs:
        if args.mode == "per_scene":
            keep_idx = keep_indices_per_scene(score, keep_frac)
        else:
            keep_idx = keep_indices_per_tile(score, index_offsets, flat_indices, keep_frac)

        pruned_gs = sc.subset_gaussians(full_gs, keep_idx)
        n_kept = len(keep_idx)
        logger.info("keep_frac={} mode={} n_kept={}/{} ({:.1%})",
                    keep_frac, args.mode, n_kept, n_gs_total, n_kept / n_gs_total)

        with torch.no_grad():
            for cam_idx, cam in enumerate(tqdm(rend_cameras, desc=f"render_kf{keep_frac}", leave=False)):
                rendered = sc.render_gs(pruned_gs, cam, args.white_bg)
                metrics = sc.compute_metrics(rendered, gt_renders[cam_idx].to(device), skip_lpips=False)
                rows.append({
                    "scene": args.scene, "mode": args.mode, "keep_frac": keep_frac,
                    "camera_index": cam_idx, "n_gs_kept": n_kept, "n_gs_total": n_gs_total,
                    **metrics,
                })
        del pruned_gs
        torch.cuda.empty_cache()

    out_csv = out_root / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.success("wrote {} rows to {}", len(rows), out_csv)


if __name__ == "__main__":
    main()
