#!/usr/bin/env python3
"""Stateful streaming-simulation harness -- metric-vs-time, not metric-vs-budget.

Simulates a PERSISTENT per-client tile buffer along a REAL recorded EyeNavGS head
trajectory (NTHU bicycle dataset), full ~60s session, at 3 bandwidth tiers, for 3
methods. Tile ORDER is recomputed every cadence tick (300ms default) from the current
viewport, but the delivered buffer is NEVER flushed -- a track works one tile at a
time to completion (never abandoned mid-tile), and once free, claims the highest-
priority not-yet-delivered tile from whichever cadence tick is latest as of that
moment (see .claude/PLAN.md streaming-sim stateful-rewrite entry, 2026-07-16).
Single LOD only -- once a tile is fully delivered it is permanently done, no
re-upgrade. Online per-tile pruning (Workstream C) is deliberately out of scope for
this rewrite (circle back later); every tile's full Gaussian set is used.

Run in the gaussian_splatting conda env (same as test_utility_inmem.py).

Usage:
  conda run -n gaussian_splatting python streaming_sim.py \\
    --output-root output/streaming_sim/bicycle_run1 --sanity-check-only

  conda run -n gaussian_splatting python streaming_sim.py \\
    --output-root output/streaming_sim/bicycle_run1

  conda run -n gaussian_splatting python streaming_sim.py \\
    --output-root output/streaming_sim/bicycle_run1 --vmaf-only
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import heapq
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from loguru import logger

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent

sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))
sys.path.insert(0, str(HERE / "experiments"))

import visibility_AABB_pytorch  # noqa: E402
import utility_calculation as uc  # noqa: E402
import io_3dgs  # noqa: E402
from ml import predict as ml_predict, features as ml_features  # noqa: E402
import selection_core as sc  # noqa: E402
from gen_sparse_views import _render_quality  # noqa: E402

import streaming_camera as scam  # noqa: E402

RENDERER_ROOT = WORKSPACE / "LapisGS-object-based-renderer"
sys.path.insert(0, str(RENDERER_ROOT))
import os  # noqa: E402
_DUMMY = WORKSPACE / "exp-dataset" / "chair" / "predictions" / "color" / "test" / "r_0.png"
os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))
from gaussian_renderer_lapisgs import GaussianModel  # noqa: E402

METHODS = ["vd_lod", "v_lod_w", "ml"]
BANDWIDTHS_MBPS = [60, 120, 300]

# Per-method intra-tile Gaussian order divergence (2026-07-15): vd_lod never computes a
# per-GS weight, so it must send a tile's Gaussians in raw PLY order while v_lod_w/ml sort
# by real weight. This is enforced centrally in selection_core.py's greedy_order() via
# NO_WEIGHT_SCHEMES + the scheme=method passed to build_greedy_order() below -- no
# per-caller dict needed here (a duplicate of that dict lived here until this session's
# own antipattern was caught and collapsed into the one, shared enforcement point).

# bicycle's scene_setting.csv correction quaternion (x,y,z,w), scale=1 (moot, not
# applied to the camera -- CAM.py applies scale to the loaded scene object instead).
BICYCLE_SCENE_QUAT_XYZW = (-0.1305, 0.0, 0.0, 0.9914)


def budget_bytes(bw_mbps: float, interval_sec: float) -> int:
    return int(bw_mbps * 1e6 * interval_sec / 8.0)


def latest_tick_leq(t: float, cadence_marks: list) -> int:
    """Index of the latest cadence tick with tick_time <= t. cadence_marks is sorted
    ascending (marks_up_to's output)."""
    i = bisect.bisect_right(cadence_marks, t) - 1
    return max(i, 0)


def tile_gs_order_at_tick(tile_idx: int, method: str, cam, tile_index_offsets, tile_flat_indices,
                           opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3,
                           gs_xyz_t, device, weight_mode, img_w, img_h, gs_order: str = "weight"):
    """Lazily compute ONE tile's own Gaussian send-order at a given cadence tick's camera --
    not a full-scene sort. Deliberately mirrors selection_core.greedy_order()'s per-tile
    ordering exactly (weight-descending for weight-bearing schemes via compute_camera_weights
    restricted to this tile's own GS slice -- same pattern as compute_camera_weights_culled;
    pure PLY/storage order, camera-independent, for NO_WEIGHT_SCHEMES or gs_order="ply"),
    because selection_core.greedy_order proved a tile's internal order is a pure function of
    (gs_order mode, that tile's own w_gi) -- independent of budget/other tiles/its own rank
    (see .claude/PLAN.md streaming-sim stateful-rewrite entry, 2026-07-16, design step 1).
    Safe to compute once, at claim time, and never revisit: the scheduler below never
    abandons a tile mid-send. `method in NO_WEIGHT_SCHEMES` force-overrides gs_order to "ply"
    regardless of the caller's preference, same rule greedy_order() itself enforces."""
    start = int(tile_index_offsets[tile_idx])
    end = int(tile_index_offsets[tile_idx + 1])
    idx = tile_flat_indices[start:end]
    if idx.numel() == 0:
        return np.empty(0, dtype=np.int64)
    if method in sc.NO_WEIGHT_SCHEMES or gs_order == "ply":
        return idx.cpu().numpy().astype(np.int64, copy=False)
    idx_np = idx.cpu().numpy()
    w_tile = sc.compute_camera_weights(
        cam, opacity[idx_np], scale_0[idx_np], scale_1[idx_np], scale_2[idx_np],
        rot_0[idx_np], rot_1[idx_np], rot_2[idx_np], rot_3[idx_np],
        gs_xyz_t[idx], device, weight_mode, img_w, img_h)
    order = torch.sort(w_tile, descending=True, stable=True).indices
    return idx[order].cpu().numpy().astype(np.int64, copy=False)


def build_session_schedule(cadence_marks, cadence_order_pairs, cameras, method,
                            tile_index_offsets, tile_flat_indices,
                            opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3,
                            gs_xyz_t, device, weight_mode, img_w, img_h, bpg,
                            n_tiles: int, track_rates: list, gs_order: str = "weight") -> dict:
    """Whole-session event-driven scheduler (2026-07-16 stateful redesign) -- replaces the
    old per-window schedule_tracks_greedy/partition_into_tracks. Each of len(track_rates)
    tracks works ONE tile at a time to completion (never abandoned mid-tile, see
    tile_gs_order_at_tick docstring); whenever a track frees up, it claims the highest-
    priority tile not yet claimed by ANY track, using whichever cadence tick's ranking is
    latest as of that moment. A tile is claimed at most once, ever, across the whole
    session -- single LOD, no re-upgrade (pruning/downsampling deferred, see module
    docstring). Returns {tile_idx: (start_sec, end_sec, track_idx, gs_order np.int64
    array)}. Ties at the same free_at time resolve by ascending track index (heap tuple
    order), so simultaneous starts pick in a stable, deterministic order.

    O(n_tiles^2) worst case (each claim linear-scans the current tick's ranking for the
    first not-yet-taken tile) -- trivially cheap at this scale (~150 tiles), not worth a
    fancier structure."""
    n_tracks = len(track_rates)
    free_at = [0.0] * n_tracks
    heap = [(0.0, k) for k in range(n_tracks)]
    heapq.heapify(heap)
    taken: dict = {}
    while heap and len(taken) < n_tiles:
        t, k = heapq.heappop(heap)
        i = latest_tick_leq(t, cadence_marks)
        ranking = cadence_order_pairs[i][:, 0]
        pick = None
        for tile_idx in ranking:
            ti = int(tile_idx)
            if ti not in taken:
                pick = ti
                break
        if pick is None:
            continue  # every tile already claimed -- track idle for rest of session
        tile_order = tile_gs_order_at_tick(
            pick, method, cameras[i], tile_index_offsets, tile_flat_indices,
            opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3,
            gs_xyz_t, device, weight_mode, img_w, img_h, gs_order=gs_order)
        n_gs_tile = len(tile_order)
        dur = (n_gs_tile * bpg) / track_rates[k] if n_gs_tile > 0 else 0.0
        end = t + dur
        taken[pick] = (t, end, k, tile_order)
        heapq.heappush(heap, (end, k))
    return taken


# ---------------------------------------------------------------------------
# Trace selection
# ---------------------------------------------------------------------------

def pick_longest_trace(bicycle_dir: Path) -> Path:
    candidates = sorted(glob.glob(str(bicycle_dir / "user*_bicycle.csv")))
    if not candidates:
        raise FileNotFoundError(f"No user*_bicycle.csv found in {bicycle_dir}")
    best_path, best_ts = None, -1
    for path in candidates:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        max_ts = max(int(r["timestep"]) for r in rows)
        if max_ts > best_ts:
            best_path, best_ts = path, max_ts
    logger.info("Auto-picked trace {} (duration {}ms)", best_path, best_ts)
    return Path(best_path)


def load_trace_rows(trace_path: Path) -> list:
    """Filter to ViewIndex==0 (see streaming_camera.py module docstring / PLAN.md Risk
    notes -- ViewIndex 0/1 rows are genuinely distinct-timed samples, not a paired
    stereo snapshot; using one eye is correct de-interleaving, not data loss)."""
    with open(trace_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["ViewIndex"]) == 0]
    rows.sort(key=lambda r: int(r["timestep"]))
    if not rows:
        raise ValueError(f"No ViewIndex==0 rows in {trace_path}")
    return rows


def sample_at_marks(rows: list, mark_sec_list: list) -> list:
    """Nearest-preceding-hold sample of `rows` at each mark (seconds)."""
    sampled = []
    row_i = 0
    for mark_sec in mark_sec_list:
        mark_ms = mark_sec * 1000.0
        while row_i + 1 < len(rows) and int(rows[row_i + 1]["timestep"]) <= mark_ms:
            row_i += 1
        sampled.append(rows[row_i])
    return sampled


def marks_up_to(duration_ms: int, step_sec: float) -> list:
    step_ms = step_sec * 1000.0
    n = int(duration_ms // step_ms) + 1
    return [k * step_sec for k in range(n)]


def nearest_oracle_camera(cam_center: torch.Tensor, eval_centers: torch.Tensor) -> int:
    """Index into `eval_centers` (N,3) of the nearest world-space camera position to
    `cam_center` (3,). Used to approximate oracle_loo (generated at a fixed discrete
    150-camera eval trace) for a continuous streaming pose that never lands exactly on
    one of those 150 poses -- an approximation on top of oracle_loo's own leave-one-tile-
    out approximation of a true RD-optimal oracle. Both must be flagged wherever a result
    using this mapping is reported, not presented as a clean oracle (see --oracle-npz
    help text)."""
    d2 = ((eval_centers - cam_center.unsqueeze(0)) ** 2).sum(dim=1)
    return int(torch.argmin(d2).item())


# ---------------------------------------------------------------------------
# Sanity gate
# ---------------------------------------------------------------------------

def run_sanity_check(args, device, gs=None) -> bool:
    """gs: optional pre-loaded GaussianModel (avoids a second ~8.5s PLY load when called
    right before run_sweep, which needs the same model -- load once in main(), pass it in)."""
    trace_path = Path(args.trace_file) if args.trace_file else pick_longest_trace(
        Path(args.bicycle_trace_dir))
    with open(trace_path, newline="") as f:
        first_row = next(r for r in csv.DictReader(f) if int(r["ViewIndex"]) == 0)

    aspect = scam.frustum_aspect(first_row, swap_top_bottom=args.swap_top_bottom)
    img_w = args.img_w
    img_h = max(1, round(img_w / aspect))
    logger.info("Frustum aspect={:.4f} -> image {}x{}", aspect, img_w, img_h)

    cam = scam.build_streaming_camera(
        first_row, BICYCLE_SCENE_QUAT_XYZW, img_w, img_h,
        znear=args.znear, zfar=args.zfar, device=device,
        swap_top_bottom=args.swap_top_bottom)

    if gs is None:
        gs = GaussianModel(args.sh_degree)
        gs.load_ply(str(args.ply))
    rendered = sc.render_gs(gs, cam, args.white_bg)

    out_dir = Path(args.output_root) / "sanity_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(rendered, str(out_dir / "first_pose.png"))

    ok, stats = _render_quality(rendered, args.max_white_frac, args.max_black_frac,
                                 args.min_edge_var)
    report = {"trace_path": str(trace_path), "image_width": img_w, "image_height": img_h,
              "aspect": aspect, "pass": ok, **stats}
    (out_dir / "sanity_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Sanity check: {} stats={}", "PASS" if ok else "FAIL", stats)
    return ok


# ---------------------------------------------------------------------------
# Frame writer -- raw RGB24 piped straight into a persistent ffmpeg process that does
# the RGB->YUV420p conversion, instead of a per-frame PNG encode (measured ~343ms/frame
# vs ~15ms/frame piped -- PNG disk write was the actual per-mark bottleneck, not
# rendering or selection; see .claude/PLAN.md fast-cadence entry).
# ---------------------------------------------------------------------------

class FrameWriter:
    def __init__(self, out_path: Path, width: int, height: int):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-framerate", "1", "-i", "pipe:0", "-pix_fmt", "yuv420p", "-f", "rawvideo",
             str(out_path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, rendered: torch.Tensor):
        frame = (rendered.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy().tobytes()
        self.proc.stdin.write(frame)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args, device, rend_gs_full=None):
    """rend_gs_full: optional pre-loaded GaussianModel (avoids a second ~8.5s PLY load when
    run_sanity_check already loaded one right before this -- load once in main(), pass it in).

    Stateful persistent-buffer streaming model (2026-07-16 redesign, replaces the earlier
    stateless per-window-reset version):

    - Tile ORDER is recomputed at every cadence tick (--interval-sec), from the viewport AT
      that tick. This ranking is only ever CONSULTED, never enforced wholesale: the client's
      delivered buffer is never flushed.
    - Delivery is modeled by build_session_schedule(): each of --n-tracks tracks works one
      tile at a time to completion (never abandoned mid-tile), and once free, claims the
      highest-priority not-yet-delivered tile using whichever cadence tick is latest as of
      that moment. A tile is claimed at most once, ever -- single LOD, no re-upgrade.
    - The RENDER pose is the actual continuous trace pose at the fine render time
      (--render-interval-sec, independent of --interval-sec).
    This produces a genuine monotonic coverage ramp (used_bytes/selected GS count never
    decreases) with viewport-driven reprioritization of what's picked NEXT, instead of the
    old sawtooth reset-every-window artifact.
    """
    trace_path = Path(args.trace_file) if args.trace_file else pick_longest_trace(
        Path(args.bicycle_trace_dir))
    rows = load_trace_rows(trace_path)
    duration_ms = int(rows[-1]["timestep"])
    if args.duration_sec is not None:
        duration_ms = min(duration_ms, int(args.duration_sec * 1000))

    cadence_marks = marks_up_to(duration_ms, args.interval_sec)
    cadence_rows = sample_at_marks(rows, cadence_marks)
    render_marks = marks_up_to(duration_ms, args.render_interval_sec)
    render_rows = sample_at_marks(rows, render_marks)
    logger.info("Trace {} -> {} cadence ticks @ {}s, {} render frames @ {}s",
                trace_path, len(cadence_rows), args.interval_sec,
                len(render_rows), args.render_interval_sec)

    aspect = scam.frustum_aspect(cadence_rows[0], swap_top_bottom=args.swap_top_bottom)
    img_w = args.img_w
    img_h = max(1, round(img_w / aspect))

    # --- one-time setup (mirrors test_utility_inmem.py:612-797) ---
    sel_gs = io_3dgs.GaussianModelV2(str(args.ply))
    if rend_gs_full is None:
        rend_gs_full = GaussianModel(args.sh_degree)
        rend_gs_full.load_ply(str(args.ply))

    tc = np.load(str(args.tiling_cache), allow_pickle=True)
    min_corners, max_corners = tc["min_corners"], tc["max_corners"]
    index_offsets, flat_indices = tc["index_offsets"], tc["flat_indices"]
    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers = (min_corners_t + max_corners_t) / 2.0
    tile_centers_np = tile_centers.cpu().numpy()
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
    n_gs_per_tile = (index_offsets[1:] - index_offsets[:-1]).astype(np.float32)

    opacity = sel_gs.data["opacity"]["data"]
    scale_0 = sel_gs.data["scale_0"]["data"]
    scale_1 = sel_gs.data["scale_1"]["data"]
    scale_2 = sel_gs.data["scale_2"]["data"]
    rot_0 = sel_gs.data["rot_0"]["data"]
    rot_1 = sel_gs.data["rot_1"]["data"]
    rot_2 = sel_gs.data["rot_2"]["data"]
    rot_3 = sel_gs.data["rot_3"]["data"]
    gs_xyz = np.stack([sel_gs.data["x"]["data"], sel_gs.data["y"]["data"],
                        sel_gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)

    bpg = sc.bytes_per_gaussian(sel_gs)
    full_scene_bytes = len(opacity) * bpg
    for bw in args.bandwidths_mbps:
        gs_per_sec = bw * 1e6 / 8 / bpg
        logger.info("Bandwidth {} Mbps = {:.0f} GS/s ({:.1f} GS/ms)", bw, gs_per_sec, gs_per_sec / 1000)

    ml_model = ml_static_feats = ml_feature_names = None
    if "ml" in args.methods:
        ml_model = ml_predict.load_model(
            args.ml_model_dir, args.ml_model_type,
            expected_n_gs=len(sel_gs.data["x"]["data"]))
        ml_feature_names = json.loads((Path(args.ml_model_dir) / "feature_names.json").read_text())
        ml_static_feats = ml_features.build_static_features(sel_gs.data, index_offsets, flat_indices)

    # --- Oracle data (mirrors test_utility_inmem.py:840-859) + eval-camera positions for
    #     nearest-pose matching (see nearest_oracle_camera() docstring for the caveat) ---
    oracle_data = oracle_eval_centers = None
    if "oracle_loo" in args.methods:
        if args.oracle_npz is None:
            raise ValueError("--oracle-npz required when 'oracle_loo' is in --methods")
        _od = np.load(args.oracle_npz, allow_pickle=False)
        _cam_ids = _od["camera_indices"].astype(np.int32)
        oracle_data = {
            "mse_loo": _od["mse"].astype(np.float64),
            "ssim_loo": _od["ssim"].astype(np.float64),
            "mse_aoi": _od["mse_aoi"].astype(np.float64) if "mse_aoi" in _od else None,
            "mse_blank": _od["mse_blank"].astype(np.float64) if "mse_blank" in _od else None,
            "cam_idx_to_row": {int(c): i for i, c in enumerate(_cam_ids)},
        }
        n_oracle_tiles = oracle_data["mse_loo"].shape[1]
        n_scene_tiles = len(index_offsets) - 1
        if n_oracle_tiles != n_scene_tiles:
            raise ValueError(
                f"--oracle-npz {args.oracle_npz} has {n_oracle_tiles} tiles but the scene "
                f"tiling has {n_scene_tiles} -- use the matching --tiling-cache."
            )
        _eval_cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
            str(args.oracle_eval_trace), img_w, img_h)
        _eval_cams = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(_eval_cam_infos)
        oracle_eval_centers = torch.stack(
            [c.camera_center.to(device) for c in _eval_cams])
        logger.info("Oracle: {} tiles, {} eval cameras (nearest-pose-matched to "
                    "{} cadence ticks)", n_oracle_tiles, len(_eval_cams), len(cadence_rows))

    out_root = Path(args.output_root)
    (out_root / "gt_renders").mkdir(parents=True, exist_ok=True)
    for bw in args.bandwidths_mbps:
        for method in args.methods:
            (out_root / f"renders/bw_{bw}mbps/{method}").mkdir(parents=True, exist_ok=True)
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    vmaf_dir = out_root / "vmaf"

    frame_writers = {"gt": FrameWriter(vmaf_dir / "gt.yuv", img_w, img_h)}
    for bw in args.bandwidths_mbps:
        for method in args.methods:
            key = (bw, method)
            frame_writers[key] = FrameWriter(
                vmaf_dir / f"bw_{bw}mbps" / method / "distorted.yuv", img_w, img_h)
    png_pool = sc.AsyncWritePool(args.png_workers)

    n_tiles = len(index_offsets) - 1
    t0 = time.time()

    # --- Pass 1: tile RANKING per (cadence tick, method) -- viewport-based. Stores only
    #     order_pairs (tiny, (n_tiles,2)) + the tick's camera object, NOT a full-scene
    #     ordered GS array -- avoids holding ~29GB (200 ticks x 3 methods x 48MB) of
    #     redundant full_ordered arrays in memory. Per-tile GS send-order is computed
    #     lazily, once, only for tiles actually claimed (see build_session_schedule /
    #     tile_gs_order_at_tick below). See .claude/PLAN.md stateful-rewrite design step 3.
    orders = {}
    cameras = []
    for cadence_idx, trace_row in enumerate(cadence_rows):
        cam = scam.build_streaming_camera(
            trace_row, BICYCLE_SCENE_QUAT_XYZW, img_w, img_h,
            uid=cadence_idx, znear=args.znear, zfar=args.zfar, device=device,
            swap_top_bottom=args.swap_top_bottom)
        cameras.append(cam)

        distances = uc.calculate_distances(tile_centers, cam.camera_center.to(device))
        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device)
        w_gi = sc.compute_camera_weights(
            cam, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3,
            gs_xyz_t, device, args.weight_mode, img_w, img_h)
        W_k, N_k = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_gi,
            w_norm=args.w_norm, c_norm=args.c_norm)

        oracle_camera_index = None
        if oracle_data is not None:
            oracle_camera_index = nearest_oracle_camera(
                cam.camera_center.to(device), oracle_eval_centers)

        ml_group_a = None
        if "ml" in args.methods:
            cam_w2v = cam.world_view_transform.cpu().numpy()
            cam_center_np = cam.camera_center.cpu().numpy()
            ml_group_a = ml_features.build_group_a(
                cam_center_np, cam_w2v[:3, 2], float(cam.FoVx), float(cam.FoVy),
                tile_centers_np, n_gs_per_tile,
                distances.cpu().numpy(), visibility.cpu().numpy().astype(np.float32))
        ml_predict_kwargs = dict(
            model_dir=args.ml_model_dir, model_type=args.ml_model_type,
            static_features=ml_static_feats, group_a=ml_group_a,
            feature_names=ml_feature_names, model=ml_model)

        for method in args.methods:
            is_oracle = method.startswith("oracle_")
            raw_scores = sc.compute_raw_scores(
                method, oracle_data=oracle_data if is_oracle else None,
                camera_index=oracle_camera_index if is_oracle else cadence_idx,
                n_tiles=n_tiles,
                visibility=visibility, distances=distances, num_lod=1, W_k=W_k, C_k=N_k,
                ml_predict_kwargs=ml_predict_kwargs)
            orders[(cadence_idx, method)] = sc.sort_tiles(
                raw_scores, n_gs_per_tile, bpg, greedy_key="marginal", num_of_level=1)
    logger.info("orders: {} cadence ticks x {} methods computed ({:.1f}s elapsed)",
                len(cadence_rows), len(args.methods), time.time() - t0)

    # --- Pass 1.5: whole-session event-driven delivery schedule, per (method, bandwidth) --
    #     see build_session_schedule() docstring. Also dumps the exact per-tile schedule
    #     (Gantt-chart data, not sampled from the render loop) for a bandwidth-utilization
    #     sanity check -- confirms the scheduler isn't pathologically stalling a track while
    #     unclaimed tiles remain (2026-07-16 user request; see plot_track_utilization.py). ---
    track_rate_total = {bw: budget_bytes(bw, 1.0) for bw in args.bandwidths_mbps}  # bytes/sec
    schedules = {}
    schedule_dir = out_root / "schedule"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        cadence_order_pairs = [orders[(ci, method)] for ci in range(len(cadence_rows))]
        for bw in args.bandwidths_mbps:
            track_rates = [track_rate_total[bw] / args.n_tracks] * args.n_tracks
            schedule = build_session_schedule(
                cadence_marks, cadence_order_pairs, cameras, method,
                tile_index_offsets, tile_flat_indices,
                opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3,
                gs_xyz_t, device, args.weight_mode, img_w, img_h, bpg,
                n_tiles, track_rates, gs_order=args.gs_order)
            schedules[(method, bw)] = schedule
            entries = [
                {"tile_idx": ti, "track": tr, "start_sec": st, "end_sec": en,
                 "n_gs": len(tile_order), "bytes": len(tile_order) * bpg}
                for ti, (st, en, tr, tile_order) in schedule.items()
            ]
            (schedule_dir / f"{method}_bw{bw}mbps.json").write_text(json.dumps({
                "n_tracks": args.n_tracks, "bandwidth_mbps": bw,
                "track_rates_bps": track_rates, "bpg": bpg, "n_tiles": n_tiles,
                "session_duration_sec": duration_ms / 1000.0, "entries": entries,
            }, indent=2))
    logger.info("schedules: {} methods x {} bandwidths, whole-session event-driven "
                "({:.1f}s elapsed)", len(args.methods), len(args.bandwidths_mbps),
                time.time() - t0)

    # --- Pass 2: dense render/metric samples off the persistent schedule -- delivered GS
    #     count per tile only ever grows (no per-window reset), actual continuous camera
    #     pose at each render mark (independent of cadence). PNGs saved at 5 evenly-spaced
    #     representative session-time marks (no more guaranteed-empty reset instants to
    #     dodge, unlike the old per-window picker -- t=0 is legitimately near-empty now,
    #     which is expected and fine to show). ---
    n_frames = len(render_rows)
    save_indices = ({round(f * (n_frames - 1)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)}
                     if n_frames > 1 else {0})
    is_repr_mark = lambda fi: fi in save_indices  # noqa: E731
    metric_rows = []
    for fine_idx, trace_row in enumerate(render_rows):
        t_sec = render_marks[fine_idx]

        cam = scam.build_streaming_camera(
            trace_row, BICYCLE_SCENE_QUAT_XYZW, img_w, img_h,
            uid=fine_idx, znear=args.znear, zfar=args.zfar, device=device,
            swap_top_bottom=args.swap_top_bottom)

        gt_rendered = sc.render_gs(rend_gs_full, cam, args.white_bg)
        frame_writers["gt"].write(gt_rendered)
        if is_repr_mark(fine_idx) and args.png_workers > 0:
            png_pool.submit(
                torchvision.utils.save_image, gt_rendered.cpu().clone(),
                str(out_root / "gt_renders" / f"frame_{fine_idx:04d}.png"))

        for method in args.methods:
            for bw in args.bandwidths_mbps:
                schedule = schedules[(method, bw)]
                rate_k = track_rate_total[bw] / args.n_tracks
                sel_parts, used_bytes = [], 0
                for tile_idx, (start, end, track_idx, tile_order) in schedule.items():
                    n_tile = len(tile_order)
                    if n_tile == 0 or t_sec <= start:
                        continue
                    if t_sec >= end:
                        n_gs = n_tile
                    else:
                        n_gs = min(int(rate_k * (t_sec - start) // bpg), n_tile)
                    if n_gs > 0:
                        sel_parts.append(tile_order[:n_gs])
                        used_bytes += n_gs * bpg
                selected = np.concatenate(sel_parts) if sel_parts else np.array([], dtype=np.int64)
                sub_gs = sc.subset_gaussians(rend_gs_full, selected)
                rendered = sc.render_gs(sub_gs, cam, args.white_bg)
                del sub_gs
                metrics = sc.compute_metrics(rendered, gt_rendered, skip_lpips=True)

                frame_writers[(bw, method)].write(rendered)
                if is_repr_mark(fine_idx) and args.png_workers > 0:
                    png_path = out_root / f"renders/bw_{bw}mbps/{method}/frame_{fine_idx:04d}.png"
                    png_pool.submit(
                        torchvision.utils.save_image, rendered.cpu().clone(), str(png_path))

                metric_rows.append({
                    "frame_idx": fine_idx, "t_sec": t_sec, "method": method,
                    "bandwidth_mbps": bw,
                    "theoretical_max_bytes": int(track_rate_total[bw] * t_sec),
                    "used_bytes": used_bytes,
                    "coverage_frac": used_bytes / full_scene_bytes,
                    "n_tracks": args.n_tracks,
                    "n_selected": len(selected), "n_gs_total": len(opacity),
                    "psnr": metrics["psnr"], "ssim": metrics["ssim"],
                })
        if fine_idx % 10 == 0 or fine_idx == len(render_rows) - 1:
            logger.info("frame {}/{} t={:.1f}s done ({:.1f}s elapsed)",
                        fine_idx + 1, len(render_rows), t_sec, time.time() - t0)

    for fw in frame_writers.values():
        fw.close()
    png_pool.drain()
    png_pool.shutdown()

    rows = metric_rows
    summary_csv = out_root / "metrics" / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_root / "metrics" / "summary.json").write_text(json.dumps(rows, indent=2))
    logger.success("Wrote {} rows to {}", len(rows), summary_csv)

    params = {
        "trace_path": str(trace_path), "n_cadence_ticks": len(cadence_rows),
        "n_frames": len(render_rows), "render_interval_sec": args.render_interval_sec,
        "interval_sec": args.interval_sec, "bandwidths_mbps": args.bandwidths_mbps,
        "methods": args.methods, "img_w": img_w, "img_h": img_h,
        "ply": str(args.ply), "tiling_cache": str(args.tiling_cache),
        "ml_model_dir": args.ml_model_dir, "ml_model_type": args.ml_model_type,
        "weight_mode": args.weight_mode, "w_norm": args.w_norm, "c_norm": args.c_norm,
        "white_bg": args.white_bg, "swap_top_bottom": args.swap_top_bottom,
        "n_tracks": args.n_tracks, "gs_order": args.gs_order,
        "full_scene_bytes": full_scene_bytes,
        "elapsed_sec": time.time() - t0,
    }
    (out_root / "params.yaml").write_text(json.dumps(params, indent=2))


# ---------------------------------------------------------------------------
# VMAF stage
# ---------------------------------------------------------------------------

def run_vmaf(args):
    """Reads the yuv sequences FrameWriter already wrote during run_sweep -- no PNG-sequence
    conversion step here (see FrameWriter class docstring for why)."""
    out_root = Path(args.output_root)
    summary_rows = json.loads((out_root / "metrics" / "summary.json").read_text())
    params = json.loads((out_root / "params.yaml").read_text())
    img_w, img_h = params["img_w"], params["img_h"]

    vmaf_dir = out_root / "vmaf"
    gt_yuv = vmaf_dir / "gt.yuv"

    vmaf_by_key = {}
    for bw in params["bandwidths_mbps"]:
        for method in params["methods"]:
            key_dir = vmaf_dir / f"bw_{bw}mbps" / method
            dist_yuv = key_dir / "distorted.yuv"

            out_json = key_dir / "vmaf.json"
            # This vmaf binary's .y4m path doesn't parse the container header (confirmed
            # empirically this session -- claims .y4m support in --help but silently fails
            # to read frame data from a well-formed y4m file); raw .yuv with explicit
            # -w/-h/-p/-b works.
            # --threads 8: real measured 8.3s->1.35s per call (6.2x) on this 64-core box,
            # load average ~4.5 at measurement time -- plenty of headroom for 8 threads.
            subprocess.run(
                ["vmaf", "-r", str(gt_yuv), "-d", str(dist_yuv),
                 "-w", str(img_w), "-h", str(img_h), "-p", "420", "-b", "8",
                 "--threads", "8", "--json", "-o", str(out_json),
                 "-m", args.vmaf_model],
                check=True)
            scores = [f["metrics"]["vmaf"] for f in json.loads(out_json.read_text())["frames"]]
            for frame_idx, score in enumerate(scores):
                vmaf_by_key[(frame_idx, method, bw)] = score

    for row in summary_rows:
        key = (row["frame_idx"], row["method"], row["bandwidth_mbps"])
        row["vmaf"] = vmaf_by_key.get(key)

    summary_csv = out_root / "metrics" / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    (out_root / "metrics" / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    logger.success("VMAF merged into {}", summary_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--bicycle-trace-dir",
                    default=str(HERE / "dataset/EyeNavGS_NTHU_Dataset/bicycle"))
    p.add_argument("--trace-file", default=None)
    p.add_argument("--ply", default=str(WORKSPACE / "exp-dataset/bicycle/point_cloud.ply"))
    p.add_argument("--tiling-cache",
                    default=str(HERE / "output/oracle_tiling_cache/bicycle_8x8x8.npz"))
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--img-w", type=int, default=1600)
    p.add_argument("--bandwidths-mbps", type=float, nargs="+", default=BANDWIDTHS_MBPS)
    p.add_argument("--interval-sec", type=float, default=0.3,
                    help="Cadence: how often the tile RANKING is recomputed from the current "
                         "viewport. Only ever consulted by whichever track next goes idle -- "
                         "the delivered buffer itself is never flushed on a tick (stateful "
                         "model, 2026-07-16).")
    p.add_argument("--render-interval-sec", type=float, default=0.5,
                    help="How often a frame is rendered+measured -- independent of "
                         "--interval-sec (not required to be in phase with it). Dense, "
                         "continuous camera pose against the persistent delivery schedule -- "
                         "this is what makes the metric-vs-time curve smooth.")
    p.add_argument("--duration-sec", type=float, default=None,
                    help="Cap the trace session length (default: use the full recorded trace "
                         "duration, ~60s for the EyeNavGS bicycle traces). E.g. 30 to only "
                         "simulate the first 30s.")
    p.add_argument("--png-workers", type=int, default=4,
                    help="Thread pool size for the small number of representative PNG saves "
                         "(5 evenly-spaced session-time marks, for visual grounding -- 0 "
                         "disables PNG saving entirely). Every mark still feeds the "
                         "vmaf/*.yuv sequences via FrameWriter regardless of this flag.")
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--oracle-npz", default=None,
                    help="oracle_dq.npz (tile-level LOO delta-MSE, e.g. "
                         "output/oracle/8/eval/bicycle/oracle_dq.npz). Required if 'oracle_loo' "
                         "is in --methods. The npz is indexed by discrete eval-trace camera "
                         "index (--oracle-eval-trace); streaming poses are continuous, so each "
                         "cadence tick's real pose is nearest-matched to the closest of those "
                         "discrete cameras (see nearest_oracle_camera()) -- this is a second "
                         "approximation layered on top of oracle_loo's own leave-one-tile-out "
                         "approximation, not an exact oracle. Flag any oracle_loo result with "
                         "both caveats, don't present as a clean upper bound.")
    p.add_argument("--oracle-eval-trace",
                    default=str(WORKSPACE / "exp-dataset/bicycle/sparse_views_eval.json"),
                    help="Eval-trace JSON the --oracle-npz's camera_indices are keyed against "
                         "(must be the same trace exp4_oracle_dq.py was run with).")
    p.add_argument("--ml-model-dir",
                    default=str(HERE / "output/ml_models_experimental/per_scene/bicycle/AC"))
    p.add_argument("--ml-model-type", default="lgbm")
    p.add_argument("--weight-mode", default="screen_area")
    p.add_argument("--w-norm", default="sum")
    p.add_argument("--c-norm", default="sum")
    p.add_argument("--gs-order", default="weight",
                    help="Intra-tile Gaussian send-order preference for weight-bearing "
                         "schemes: 'weight' (descending per-GS screen_area, default) or "
                         "'ply' (storage order). Force-overridden to 'ply' regardless for "
                         "schemes in selection_core.NO_WEIGHT_SCHEMES (currently vd_lod).")
    # Online per-tile pruning (Workstream C) is deliberately out of scope for the stateful
    # rewrite (2026-07-16, explicit user decision -- circle back later). Every tile's full
    # Gaussian set is delivered. selection_core.prune_tile_gs itself is untouched and stays
    # available for reuse when this is revisited.
    p.add_argument("--n-tracks", type=int, default=1,
                    help="Number of simultaneous QUIC/MOQ-style transmission tracks, equal "
                         "bandwidth share. Each track works one tile at a time to completion "
                         "and, once free, claims the highest-priority not-yet-delivered tile "
                         "(event-driven, work-conserving -- see build_session_schedule()). "
                         "1 = strictly-serial single-track (default).")
    p.add_argument("--white-bg", action="store_true")
    p.add_argument("--znear", type=float, default=0.01)
    p.add_argument("--zfar", type=float, default=100.0)
    p.add_argument("--no-swap-top-bottom", dest="swap_top_bottom", action="store_false",
                    help="FOV3/FOV4 (top/bottom) sign convention empirically needs swapping "
                         "for this trace (confirmed via sanity render 2026-07-14) -- default "
                         "on. Escape hatch to disable if a different trace behaves differently.")
    p.set_defaults(swap_top_bottom=True)
    p.add_argument("--max-white-frac", type=float, default=0.35)
    p.add_argument("--max-black-frac", type=float, default=0.50)
    p.add_argument("--min-edge-var", type=float, default=0.0)
    p.add_argument("--sanity-check-only", action="store_true")
    p.add_argument("--vmaf-only", action="store_true")
    p.add_argument("--vmaf-model", default="version=vmaf_4k_v0.6.1",
                    help="libvmaf -m value. Switched from vmaf_v0.6.1 (1080p-calibrated) to "
                         "vmaf_4k_v0.6.1 (2026-07-16, PLAN.md 'VMAF profile decision') -- "
                         "renders are 1600x1644, resolution-matched to 4K not 1080p.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.vmaf_only:
        run_vmaf(args)
        return

    gs = GaussianModel(args.sh_degree)
    gs.load_ply(str(args.ply))

    ok = run_sanity_check(args, device, gs=gs)
    if not ok and not args.force:
        logger.error("Sanity check FAILED -- refusing to run full sweep (pass --force to override)")
        sys.exit(1)
    if args.sanity_check_only:
        return

    run_sweep(args, device, rend_gs_full=gs)


if __name__ == "__main__":
    main()
