#!/usr/bin/env python3
"""Per-method selection timing experiment (Exp5).

Measures wall time of each selection method's full pipeline, run
independently per camera. No shared-stage amortization across methods.
At each budget level, records n_gs selected (O(1) slice, untimed).

Methods:
  progressive_screen_area, progressive_volume, progressive_vol_d2
  vd_lod        — tile_partial, v/d scoring only (no gaussian_weights)
  heuristic     — tile_partial, (v/d)·W_k scoring (screen_area gaussian_weights)
  ml            — tile_partial, RF/LGBM/XGB predict; no gaussian_weights for scoring
  oracle_online — tile_partial, LOO computed live via opacity-zeroing per tile

Output per method: {output_root}/{scene}/{method}/
  params.yaml, timings.json, summary.csv

Run in gaussian_splatting conda env:
  CUDA_VISIBLE_DEVICES=2 conda run -n gaussian_splatting \\
    python experiments/0628/time_selection.py \\
    --ply <scene.ply> --tiling-cache <cache.npz> --camera-trace <eval.json> \\
    --ml-model-dir output/0606/ml_models/<scene>/AC --ml-model-type rf \\
    --methods heuristic ml oracle_online --scene <scene> \\
    --output-root output/0628/selection_timing
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
for _p in (
    WORKSPACE / "Frustum-for-3DGS",
    WORKSPACE / "GGSP",
    WORKSPACE / "GS-Interface",
    WORKSPACE / "dlapisgs-utility",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import visibility_AABB_pytorch  # noqa: E402
import utility_calculation as uc  # noqa: E402
import io_3dgs  # noqa: E402
from ml import predict as ml_predict, features as ml_features  # noqa: E402

RENDERER_ROOT = WORKSPACE / "LapisGS-object-based-renderer"
sys.path.insert(0, str(RENDERER_ROOT))
_DUMMY = WORKSPACE / "exp-dataset" / "chair" / "predictions" / "color" / "test" / "r_0.png"
os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))

from gaussian_renderer_lapisgs import GaussianModel, render as gs_render  # noqa: E402  # type: ignore
from streaming_utils.camera_loader import load_camera_from_streaming_config  # noqa: E402  # type: ignore

VALID_METHODS = [
    "progressive_screen_area",
    "progressive_volume",
    "progressive_vol_d2",
    "vd_lod",
    "heuristic",
    "ml",
    "oracle_online",
]

_PROG_WEIGHT_MODE = {
    "progressive_screen_area": "screen_area",
    "progressive_volume":      "volume",
    "progressive_vol_d2":      "volume_over_d2",
}

# Ordered list of all possible stage columns for summary.csv
ALL_STAGE_COLS = [
    "visibility", "gaussian_weights", "tile_weights", "utility",
    "ml_group_a", "ml_predict", "full_render", "tile_renders", "greedy",
]

SUMMARY_COLS = (
    ["scene", "budget_pct", "method", "camera_index",
     "selected_gaussians", "used_bytes", "bytes_per_gaussian",
     "total_s"]
    + [f"{s}_s" for s in ALL_STAGE_COLS]
)


class _FakePipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


PIPELINE = _FakePipe()


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def _timed(name: str, store: dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_per_gaussian(sel_gs) -> int:
    return int(np.sum([np.dtype(v["val_dtype"]).itemsize for v in sel_gs.data.values()]))


def _compute_camera_weights(sel_cam, opacity, scale_0, scale_1, scale_2,
                             rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
                             weight_mode: str, img_w: int, img_h: int):
    cam_center = sel_cam.camera_center.to(device)
    kw = dict(opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
              xyz=gs_xyz_t, cam_center=cam_center)
    if weight_mode == "screen_area":
        kw.update(rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
                  world_view=sel_cam.world_view_transform,
                  proj=sel_cam.projection_matrix,
                  img_w=img_w, img_h=img_h,
                  fov_x=getattr(sel_cam, "FoVx", None),
                  fov_y=getattr(sel_cam, "FoVy", None))
    return uc.compute_gaussian_weights_v2(weight_mode, **kw).to(device)


def _greedy_order_progressive(visibility, tile_index_offsets, tile_flat_indices,
                               w_gi, bytes_per_gaussian: int, max_budget_bytes: int):
    max_count = max_budget_bytes // bytes_per_gaussian
    device = w_gi.device
    vis = visibility.to(device=device, dtype=torch.bool)
    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    per_gs_vis = torch.repeat_interleave(vis, sizes)
    if per_gs_vis.numel() == 0:
        return np.empty(0, dtype=np.int64)
    visible_gs   = tile_flat_indices[per_gs_vis]
    invisible_gs = tile_flat_indices[~per_gs_vis]
    visible_sorted   = visible_gs[torch.argsort(w_gi[visible_gs], descending=True)]
    invisible_sorted = invisible_gs[torch.argsort(w_gi[invisible_gs], descending=True)]
    if visible_sorted.numel() >= max_count:
        return visible_sorted[:max_count].cpu().numpy().astype(np.int64)
    remaining = max_count - int(visible_sorted.numel())
    return torch.cat([visible_sorted, invisible_sorted[:remaining]]).cpu().numpy().astype(np.int64)


def _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices,
                  w_gi, bytes_per_gaussian: int, max_budget_bytes: int):
    max_count = max_budget_bytes // bytes_per_gaussian
    n_tiles = int(tile_index_offsets.numel()) - 1
    device = w_gi.device

    # Assign priority rank to every tile (one scatter, no Python loop over tiles)
    tile_prio = torch.zeros(n_tiles, dtype=torch.long, device=device)
    tile_order_t = torch.as_tensor(order_pairs[:, 0], dtype=torch.long, device=device)
    tile_prio[tile_order_t] = torch.arange(len(order_pairs), dtype=torch.long, device=device)

    # Expand tile priority to per-GS in flat_indices order
    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    tile_ranks_flat = torch.repeat_interleave(tile_prio, sizes)  # (n_total_gs,)

    w_flat = w_gi[tile_flat_indices].cpu().numpy()
    order = np.lexsort((-w_flat, tile_ranks_flat.cpu().numpy()))
    return tile_flat_indices.cpu().numpy()[order][:max_count].astype(np.int64)


def _sort_tiles(raw_scores, n_gs_per_tile, bytes_per_gaussian: int, greedy_key: str):
    if greedy_key == "marginal":
        byte_k = np.where(n_gs_per_tile > 0, n_gs_per_tile * bytes_per_gaussian, 1.0)
        scores = raw_scores / byte_k
    else:
        scores = raw_scores
    order = np.argsort(-scores, kind="stable").astype(np.int64)
    return np.stack([order, np.zeros(len(order), dtype=np.int64)], axis=1)


def _budget_slices(all_ordered, budget_pcts, total_scene_bytes, bytes_per_gaussian):
    """Compute n_gs selected at each budget pct. Untimed — O(1) per budget."""
    slices = []
    for pct in budget_pcts:
        n_budget = int(total_scene_bytes * pct / 100.0) // bytes_per_gaussian
        n_selected = int(min(len(all_ordered), n_budget))
        slices.append({
            "budget_pct":        pct,
            "selected_gaussians": n_selected,
            "used_bytes":         n_selected * bytes_per_gaussian,
        })
    return slices


def _render_gs(gaussians: GaussianModel, camera, white_bg: bool) -> torch.Tensor:
    bg = [1, 1, 1] if white_bg else [0, 0, 0]
    bg_color = torch.tensor(bg, dtype=torch.float32, device="cuda").view(3, 1, 1)
    bg_color = bg_color.expand(3, camera.image_height, camera.image_width)
    bg_depth = torch.zeros(1, camera.image_height, camera.image_width, device="cuda")
    gs_res = torch.ones(len(gaussians.get_xyz), device="cuda")
    with torch.no_grad():
        result = gs_render(camera, gaussians, PIPELINE, bg_color, bg_depth, gs_res=gs_res)
    return result["render"].clamp(0.0, 1.0)


def _load_trace(trace_path: Path) -> list:
    with open(trace_path) as f:
        data = json.load(f)
    angle_x = data["camera_angle_x"]
    for frame in data["frames"]:
        frame["camera_angle_x"] = angle_x
    return data["frames"]


# ---------------------------------------------------------------------------
# Per-method timing
# ---------------------------------------------------------------------------

def _time_method(
    method: str,
    camera_indices: list[int],
    sel_cameras,
    rend_cameras,
    sel_gs,
    rend_gs_full: GaussianModel | None,
    tile_index_offsets,    # torch.long
    tile_flat_indices,     # torch.long
    min_corners_t,
    max_corners_t,
    tile_centers,
    opacity, scale_0, scale_1, scale_2,
    rot_0, rot_1, rot_2, rot_3, gs_xyz_t,
    index_offsets: np.ndarray,  # numpy int64
    flat_indices: np.ndarray,
    n_gs_per_tile: np.ndarray,
    bytes_per_gaussian: int,
    device: str,
    args,
    budget_pcts: list[int],
    ml_model=None,
    ml_static_feats=None,
    ml_feature_names=None,
) -> list[dict]:
    n_total_gs = int(len(sel_gs.data["x"]["data"]))
    max_budget_bytes = n_total_gs * bytes_per_gaussian  # identity budget
    total_scene_bytes = max_budget_bytes

    cam_results = []
    for ci in camera_indices:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stages: dict = {}
        cam = sel_cameras[ci]
        all_ordered: np.ndarray | None = None

        # ── PROGRESSIVE variants ─────────────────────────────────────────────
        if method in _PROG_WEIGHT_MODE:
            wm = _PROG_WEIGHT_MODE[method]

            with _timed("visibility", stages):
                vis   = visibility_AABB_pytorch.batched_check_tiles_visible(
                    min_corners_t, max_corners_t, cam, device=device)
                dists = uc.calculate_distances(tile_centers, cam.camera_center.to(device))

            with _timed("gaussian_weights", stages):
                w_gi = _compute_camera_weights(
                    cam, opacity, scale_0, scale_1, scale_2,
                    rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
                    weight_mode=wm, img_w=args.img_w, img_h=args.img_h)

            with _timed("greedy", stages):
                all_ordered = _greedy_order_progressive(
                    vis, tile_index_offsets, tile_flat_indices,
                    w_gi, bytes_per_gaussian, max_budget_bytes)

        # ── VD_LOD (tile_partial, v/d only — no gaussian_weights) ────────────
        elif method == "vd_lod":
            with _timed("visibility", stages):
                vis   = visibility_AABB_pytorch.batched_check_tiles_visible(
                    min_corners_t, max_corners_t, cam, device=device)
                dists = uc.calculate_distances(tile_centers, cam.camera_center.to(device))

            with _timed("utility", stages):
                raw = uc.calculate_utility_param(
                    vis, dists,
                    num_of_level=1, weight_sum_tensor=None, complexity_tensor=None,
                    include_lod=True, include_w=False, include_c=False,
                )
                if hasattr(raw, "cpu"):
                    raw = raw.cpu().numpy()
                order_pairs = _sort_tiles(raw, n_gs_per_tile, bytes_per_gaussian, "marginal")

            with _timed("greedy", stages):
                w_ones = torch.ones(n_total_gs, device=device)
                all_ordered = _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices,
                                            w_ones, bytes_per_gaussian, max_budget_bytes)

        # ── HEURISTIC (tile_partial, vd_lod_w, screen_area W_k) ─────────────
        elif method == "heuristic":
            with _timed("visibility", stages):
                vis   = visibility_AABB_pytorch.batched_check_tiles_visible(
                    min_corners_t, max_corners_t, cam, device=device)
                dists = uc.calculate_distances(tile_centers, cam.camera_center.to(device))

            with _timed("gaussian_weights", stages):
                w_gi = _compute_camera_weights(
                    cam, opacity, scale_0, scale_1, scale_2,
                    rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
                    weight_mode="screen_area", img_w=args.img_w, img_h=args.img_h)

            with _timed("tile_weights", stages):
                W_k, _ = uc.compute_tile_weights_and_counts(
                    tile_index_offsets, tile_flat_indices, w_gi,
                    w_norm="sum", c_norm="sum", w_mode="sum")

            with _timed("utility", stages):
                raw = uc.calculate_utility_param(
                    vis, dists,
                    num_of_level=1, weight_sum_tensor=W_k, complexity_tensor=None,
                    include_lod=True, include_w=True, include_c=False,
                )
                if hasattr(raw, "cpu"):
                    raw = raw.cpu().numpy()
                order_pairs = _sort_tiles(raw, n_gs_per_tile, bytes_per_gaussian, "marginal")

            with _timed("greedy", stages):
                all_ordered = _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices,
                                            w_gi, bytes_per_gaussian, max_budget_bytes)

        # ── ML (tile_partial, group_a + model.predict — no gaussian_weights) ─
        elif method == "ml":
            with _timed("visibility", stages):
                vis   = visibility_AABB_pytorch.batched_check_tiles_visible(
                    min_corners_t, max_corners_t, cam, device=device)
                dists = uc.calculate_distances(tile_centers, cam.camera_center.to(device))
            np_vis   = vis.float().cpu().numpy()
            np_dists = dists.cpu().numpy()

            with _timed("ml_group_a", stages):
                cam_w2v       = cam.world_view_transform.cpu().numpy()
                cam_center_np = cam.camera_center.cpu().numpy()
                cam_fwd       = cam_w2v[:3, 2]
                fov_x = float(getattr(cam, "FoVx", math.pi / 2))
                fov_y = float(getattr(cam, "FoVy", math.pi / 2))
                tile_centers_np = tile_centers.cpu().numpy()
                group_a = ml_features.build_group_a(
                    cam_center_np, cam_fwd, fov_x, fov_y,
                    tile_centers_np, n_gs_per_tile.astype(np.float32),
                    np_dists, np_vis,
                )

            with _timed("ml_predict", stages):
                raw = np.exp(ml_predict.predict_utility(
                    model_dir=args.ml_model_dir,
                    model_type=args.ml_model_type,
                    static_features=ml_static_feats,
                    group_a=group_a,
                    group_b=None,
                    feature_names=ml_feature_names,
                    model=ml_model,
                ))
                order_pairs = _sort_tiles(raw, n_gs_per_tile, bytes_per_gaussian, "marginal")

            with _timed("greedy", stages):
                w_ones = torch.ones(n_total_gs, device=device)
                all_ordered = _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices,
                                            w_ones, bytes_per_gaussian, max_budget_bytes)

        # ── ORACLE ONLINE (tile_partial, LOO via opacity-zero per tile) ───────
        elif method == "oracle_online":
            rend_cam = rend_cameras[ci]
            w_gi = None  # intra-tile ordering uses w_ones in greedy

            with _timed("full_render", stages):
                R_full = _render_gs(rend_gs_full, rend_cam, white_bg=False).detach()

            n_tiles = len(index_offsets) - 1
            scores = np.full(n_tiles, -np.inf, dtype=np.float64)
            tile_render_t = 0.0
            n_scored = 0

            for tile_idx in range(n_tiles):
                start = int(index_offsets[tile_idx])
                end   = int(index_offsets[tile_idx + 1])
                if end == start:
                    continue
                gs_idx = tile_flat_indices[start:end]

                # Use .data to bypass leaf-variable grad check
                saved_op = rend_gs_full._opacity.data[gs_idx].clone()
                rend_gs_full._opacity.data[gs_idx] = -100.0

                t0 = time.perf_counter()
                R_k = _render_gs(rend_gs_full, rend_cam, white_bg=False)
                scores[tile_idx] = float(((R_full - R_k) ** 2).mean().item())
                tile_render_t += time.perf_counter() - t0

                rend_gs_full._opacity.data[gs_idx] = saved_op
                n_scored += 1

            stages["tile_renders"] = tile_render_t
            stages["n_tiles_scored"] = n_scored

            with _timed("greedy", stages):
                order_pairs = _sort_tiles(scores, n_gs_per_tile, bytes_per_gaussian, "marginal")
                w_ones = torch.ones(n_total_gs, device=device)
                all_ordered = _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices,
                                            w_ones, bytes_per_gaussian, max_budget_bytes)

        else:
            raise ValueError(f"Unknown method: {method}")

        total_s = sum(v for k, v in stages.items() if isinstance(v, float) and k != "n_tiles_scored")
        budgets = _budget_slices(all_ordered, budget_pcts, total_scene_bytes, bytes_per_gaussian)

        print(f"  [{method}] cam {ci:03d}  total={total_s*1000:.1f} ms  "
              + "  ".join(f"{k}={v*1000:.1f}ms" for k, v in stages.items()
                          if isinstance(v, float)),
              flush=True)

        cam_results.append({
            "camera":  ci,
            "stages":  stages,
            "total_s": total_s,
            "budgets": budgets,
        })

    return cam_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--tiling-cache", required=True)
    parser.add_argument("--camera-trace", required=True)
    parser.add_argument("--methods", nargs="+", required=True, choices=VALID_METHODS)
    parser.add_argument("--ml-model-dir", default=None)
    parser.add_argument("--ml-model-type", default="rf", choices=["rf", "lgbm", "xgb"])
    parser.add_argument("--budget-pct", nargs="+", type=int,
                        default=[10, 25, 40, 55, 70, 85, 99, 100])
    parser.add_argument("--camera-index",   type=int, default=-1)
    parser.add_argument("--camera-indices", nargs="+", type=int, default=None)
    parser.add_argument("--img-w", type=int, default=1600)
    parser.add_argument("--img-h", type=int, default=1600)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()

    if "ml" in args.methods and args.ml_model_dir is None:
        parser.error("--ml-model-dir required for ml method")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  methods={args.methods}", flush=True)

    ply_path    = Path(args.ply)
    tiling_path = Path(args.tiling_cache)
    trace_path  = Path(args.camera_trace)

    # --- Load tiling cache ---
    print("Loading tiling cache...", flush=True)
    tc = np.load(str(tiling_path))
    min_corners   = tc["min_corners"].astype(np.float32)
    max_corners   = tc["max_corners"].astype(np.float32)
    index_offsets = tc["index_offsets"].astype(np.int64)
    flat_indices  = tc["flat_indices"].astype(np.int64)
    n_tiles       = len(index_offsets) - 1
    n_gs_per_tile = (index_offsets[1:] - index_offsets[:-1]).astype(np.int64)
    print(f"  tiles={n_tiles}", flush=True)

    min_corners_t      = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t      = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers       = (min_corners_t + max_corners_t) / 2.0
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices  = torch.tensor(flat_indices,  dtype=torch.long, device=device)

    # --- Load selection PLY ---
    print("Loading selection PLY...", flush=True)
    sel_gs = io_3dgs.GaussianModelV2(str(ply_path))
    bytes_per_gaussian = _bytes_per_gaussian(sel_gs)
    n_total_gs = len(sel_gs.data["x"]["data"])
    print(f"  n_gs={n_total_gs}  bpg={bytes_per_gaussian}", flush=True)

    opacity = sel_gs.data["opacity"]["data"]
    scale_0 = sel_gs.data["scale_0"]["data"]
    scale_1 = sel_gs.data["scale_1"]["data"]
    scale_2 = sel_gs.data["scale_2"]["data"]
    rot_0 = sel_gs.data["rot_0"]["data"] if "rot_0" in sel_gs.data else None
    rot_1 = sel_gs.data["rot_1"]["data"] if "rot_1" in sel_gs.data else None
    rot_2 = sel_gs.data["rot_2"]["data"] if "rot_2" in sel_gs.data else None
    rot_3 = sel_gs.data["rot_3"]["data"] if "rot_3" in sel_gs.data else None
    gs_xyz = np.stack([sel_gs.data["x"]["data"], sel_gs.data["y"]["data"],
                       sel_gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)

    # --- Load renderer (oracle_online only) ---
    rend_gs_full = None
    rend_cameras = None
    if "oracle_online" in args.methods:
        print("Loading renderer PLY (oracle_online)...", flush=True)
        rend_gs_full = GaussianModel(sh_degree=3)
        rend_gs_full.load_ply(str(ply_path))
        frames = _load_trace(trace_path)
        rend_cameras = [
            load_camera_from_streaming_config(f, width=args.img_w, height=args.img_h)
            for f in frames
        ]
        for cam in rend_cameras:
            cam.original_image = None

    # --- Load cameras ---
    print("Loading cameras...", flush=True)
    cam_infos   = visibility_AABB_pytorch.readCamerasFromTransforms(
        str(trace_path), args.img_w, args.img_h)
    sel_cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    n_cams = len(sel_cameras)
    print(f"  cameras={n_cams}", flush=True)

    if args.camera_indices is not None:
        camera_indices = sorted(set(args.camera_indices))
    elif args.camera_index < 0:
        camera_indices = list(range(n_cams))
    else:
        camera_indices = [args.camera_index]

    # --- Load ML model + static features ---
    ml_model = ml_static_feats = ml_feature_names = None
    if "ml" in args.methods:
        print("Loading ML model...", flush=True)
        ml_model_path = Path(args.ml_model_dir)
        ml_model = joblib.load(ml_model_path / f"{args.ml_model_type}.pkl")
        if hasattr(ml_model, "n_jobs"):
            ml_model.n_jobs = 1
        ml_feature_names = json.loads((ml_model_path / "feature_names.json").read_text())
        cache_path = ml_model_path / "feature_cache.npz"
        if cache_path.exists():
            ml_static_feats = ml_features.load_feature_cache(cache_path, index_offsets)
            print(f"  feature cache: {cache_path}", flush=True)
        else:
            ml_static_feats = ml_features.build_static_features(
                sel_gs.data, index_offsets, flat_indices)
            print("  static features built from PLY", flush=True)

    # --- Common params dict for params.yaml ---
    run_params = {
        "ply":            str(ply_path),
        "tiling_cache":   str(tiling_path),
        "camera_trace":   str(trace_path),
        "methods":        args.methods,
        "ml_model_dir":   args.ml_model_dir,
        "ml_model_type":  args.ml_model_type,
        "budget_pct":     args.budget_pct,
        "n_cameras_run":  len(camera_indices),
        "img_w":          args.img_w,
        "img_h":          args.img_h,
        "scene":          args.scene,
        "n_gs":           int(n_total_gs),
        "bpg":            int(bytes_per_gaussian),
        "n_tiles":        int(n_tiles),
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "grid_shape":     [8, 8, 8],
    }

    # --- Run each method ---
    for method in args.methods:
        print(f"\n=== {method} ===", flush=True)
        cam_results = _time_method(
            method=method,
            camera_indices=camera_indices,
            sel_cameras=sel_cameras,
            rend_cameras=rend_cameras,
            sel_gs=sel_gs,
            rend_gs_full=rend_gs_full,
            tile_index_offsets=tile_index_offsets,
            tile_flat_indices=tile_flat_indices,
            min_corners_t=min_corners_t,
            max_corners_t=max_corners_t,
            tile_centers=tile_centers,
            opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
            rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
            gs_xyz_t=gs_xyz_t,
            index_offsets=index_offsets,
            flat_indices=flat_indices,
            n_gs_per_tile=n_gs_per_tile,
            bytes_per_gaussian=bytes_per_gaussian,
            device=device,
            args=args,
            budget_pcts=args.budget_pct,
            ml_model=ml_model,
            ml_static_feats=ml_static_feats,
            ml_feature_names=ml_feature_names,
        )

        # --- Output dirs ---
        out_dir = Path(args.output_root) / args.scene / method
        out_dir.mkdir(parents=True, exist_ok=True)

        # params.yaml
        (out_dir / "params.yaml").write_text(
            yaml.dump({**run_params, "method": method}, default_flow_style=False))

        # timings.json — per-camera stage breakdown + aggregate stats
        times = np.array([r["total_s"] for r in cam_results])
        timings = {
            "scene":   args.scene,
            "method":  method,
            "n_cameras": len(cam_results),
            "median_s":  float(np.median(times)),
            "p25_s":     float(np.percentile(times, 25)),
            "p75_s":     float(np.percentile(times, 75)),
            "mean_s":    float(np.mean(times)),
            "std_s":     float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "cameras":   [{"camera": r["camera"], "total_s": r["total_s"],
                           "stages": r["stages"]} for r in cam_results],
        }
        if method == "oracle_online":
            n_scored = [r["stages"].get("n_tiles_scored", 0) for r in cam_results]
            timings["median_n_tiles_scored"] = float(np.median(n_scored))

        (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))

        # summary.csv — one row per (camera × budget_pct)
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            writer.writeheader()
            for r in cam_results:
                base = {
                    "scene":             args.scene,
                    "method":            method,
                    "camera_index":      r["camera"],
                    "bytes_per_gaussian": bytes_per_gaussian,
                    "total_s":           r["total_s"],
                }
                for stage in ALL_STAGE_COLS:
                    base[f"{stage}_s"] = r["stages"].get(stage, "")
                for b in r["budgets"]:
                    row = {**base,
                           "budget_pct":        b["budget_pct"],
                           "selected_gaussians": b["selected_gaussians"],
                           "used_bytes":         b["used_bytes"]}
                    writer.writerow(row)

        print(f"  saved: {out_dir}/", flush=True)
        print(f"  median={timings['median_s']*1000:.1f} ms  "
              f"p25={timings['p25_s']*1000:.1f} ms  p75={timings['p75_s']*1000:.1f} ms",
              flush=True)


if __name__ == "__main__":
    main()
