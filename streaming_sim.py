#!/usr/bin/env python3
"""Stateless streaming-simulation harness -- metric-vs-time, not metric-vs-budget.

Reruns the existing single-shot selection pipeline (selection_core.py) once per
10-second time-mark along a REAL recorded EyeNavGS head trajectory (NTHU bicycle
dataset), at 3 bandwidth-derived byte budgets, for 3 methods. Every time-mark
reselects from the FULL scene -- no "already sent" buffer, no accumulation.
Stateless by design (see .claude/PLAN.md streaming-sim entry): this is meant to
generalize to future 4D/dynamic-scene work where each frame's content genuinely
differs, so an incremental client-buffer model wouldn't even apply there.

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
import csv
import glob
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
BANDWIDTHS_MBPS = [40, 80, 120]

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


def tile_chunk_offsets(tile_ranked, offsets, n_avail: int):
    """Cumulative chunk-length boundaries for all_ordered's per-tile-rank chunks, in
    tile_ranked (priority) order, sourced from tile-index-keyed `offsets` (e.g.
    pruned_offsets from sc.prune_tile_gs -- reflects any online per-tile pruning; passing
    the raw unpruned tile_index_offsets instead silently mismatches a pruned all_ordered,
    the exact bug this shared helper exists to avoid now that both partition_into_tracks
    and schedule_tracks_greedy need the same chunk boundaries). Returns (starts, ends)
    arrays clipped to n_avail -- all_ordered may be shorter than the full tile set
    (truncated at max_budget_bytes, always at a whole-tile boundary since tile_partial
    packing never splits a tile mid-chunk at this stage); a tile whose start >= n_avail is
    simply unreachable. `offsets` may be a GPU torch.Tensor (tile_index_offsets/pruned_offsets
    are constructed as such) or a numpy array -- normalized to numpy here since this is cheap
    per-tile (~150) bookkeeping, not worth keeping on-device."""
    if hasattr(offsets, "cpu"):
        offsets = offsets.cpu().numpy()
    tile_ranked = np.asarray(tile_ranked)
    chunk_lens = (offsets[tile_ranked + 1] - offsets[tile_ranked]).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(chunk_lens)])
    starts = np.minimum(cum[:-1], n_avail)
    ends = np.minimum(cum[1:], n_avail)
    return starts, ends


def partition_into_tracks(all_ordered, order_pairs, offsets, n_tracks: int):
    """Split a tile-contiguous priority-ordered GS sequence (from build_greedy_order) into
    n_tracks round-robin sub-sequences: track k owns tile-priority ranks k, k+N, k+2N, ...
    -- so tracks 0..N-1 hold exactly the top-N tiles at the very start of a window."""
    if n_tracks <= 1:
        return [all_ordered]
    tile_ranked = order_pairs[:, 0]
    n_avail = len(all_ordered)
    starts, ends = tile_chunk_offsets(tile_ranked, offsets, n_avail)
    track_chunks = [[] for _ in range(n_tracks)]
    for rank in range(len(tile_ranked)):
        if starts[rank] >= n_avail:
            break
        track_chunks[rank % n_tracks].append(all_ordered[starts[rank]:ends[rank]])
    return [np.concatenate(chunks) if chunks else np.array([], dtype=all_ordered.dtype)
            for chunks in track_chunks]


def schedule_tracks_greedy(tile_ranked, tile_bytes, track_rates):
    """Event-driven greedy list scheduling (2026-07-16, replaces static round-robin when
    --track-schedule greedy): each track is a server with its own byte rate; tiles are
    fixed-priority-order jobs. Whenever a track goes idle, it claims the next unclaimed tile
    -- a finished fast track doesn't sit idle while a slow track is still stuck on a large
    tile, unlike round-robin's permanent per-rank track ownership. tile_ranked/tile_bytes:
    parallel arrays, ALREADY reachable-truncated (start < n_avail from tile_chunk_offsets).
    track_rates: bytes/sec per track (possibly unequal -- weighted split). Returns a list of
    (start_sec, end_sec, track_idx), same order/length as tile_ranked.

    Assigns each tile, in priority order, to whichever track gives that tile the EARLIEST
    COMPLETION time (free_at[k] + tile_bytes/rate[k]) -- not whichever track is free soonest.
    These coincide when all rates are equal (duration is then identical across tracks, so
    ranking by completion == ranking by free_at), which is why a simpler free_at-only min-heap
    looked correct in initial testing (all n_tracks=1 unit tests used equal rates) -- but for
    weighted tracks a slower track can free up sooner and still finish a given tile later than
    a busier, faster one, so free_at-only assignment would send it to the wrong track. No heap
    needed: linear scan per tile, O(n_tiles x n_tracks), trivially cheap at this scale (~150
    tiles x <=8 tracks)."""
    n_tracks = len(track_rates)
    free_at = [0.0] * n_tracks
    schedule = []
    for tb in tile_bytes:
        completions = [free_at[k] + float(tb) / track_rates[k] for k in range(n_tracks)]
        track_idx = min(range(n_tracks), key=lambda k: completions[k])
        start_at = free_at[track_idx]
        end_at = completions[track_idx]
        schedule.append((start_at, end_at, track_idx))
        free_at[track_idx] = end_at
    return schedule


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

    Continuous-viewport streaming model (2026-07-14 redesign, replaces the earlier
    one-shot-per-cadence-tick version):

    - Tile ORDER is recomputed only at cadence ticks (every --interval-sec), from the
      viewport AT that tick -- frozen until the next tick. Models a real scheduler that
      only periodically re-scores/reorders tiles.
    - Byte BUDGET grows continuously within a cadence window: budget(t) = bandwidth *
      (t - last_cadence_tick), reset to 0 at each new tick (no banking, matches spec).
    - The RENDER pose is the actual continuous trace pose at the fine render time, not
      the (stale) cadence-tick pose -- the client keeps moving while still working
      through the frozen order.
    This produces the smooth-ramp-with-stepwise-reorder curve shape (see
    plotting/paper/plot_format_ref/streaming/*.png), not a handful of disconnected
    snapshot points.
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
    max_budget_bytes = budget_bytes(max(args.bandwidths_mbps), args.interval_sec)
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
    png_executor = ThreadPoolExecutor(max_workers=args.png_workers) if args.png_workers > 0 else None
    png_futures: list = []

    n_tiles = len(index_offsets) - 1
    t0 = time.time()

    # --- Pass 1: tile order per (cadence tick, method) -- viewport-based, frozen ---
    orders = {}
    greedy_chunks = {}
    schedules = {}
    for cadence_idx, trace_row in enumerate(cadence_rows):
        cam = scam.build_streaming_camera(
            trace_row, BICYCLE_SCENE_QUAT_XYZW, img_w, img_h,
            uid=cadence_idx, znear=args.znear, zfar=args.zfar, device=device,
            swap_top_bottom=args.swap_top_bottom)

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
            utilities = sc.sort_tiles(raw_scores, n_gs_per_tile, bpg, greedy_key="marginal",
                                       num_of_level=1)
            # Online per-tile prune (Workstream C, 2026-07-15): no-op for vd_lod
            # (NO_WEIGHT_SCHEMES) and at default keep_frac=1.0/no max_gs_per_tile -- see
            # prune_tile_gs() docstring. Only affects the PACKING step below; tile-level
            # ranking (utilities/visibility above) always sees the full, unpruned tiling.
            # Combining with --n-tracks>1 is now safe (2026-07-16): both partition_into_tracks
            # and the greedy scheduler derive chunk boundaries from tile_chunk_offsets() over
            # pruned_offsets, not the raw unpruned tile_index_offsets/n_gs_per_tile -- fixes
            # the mismatch noted here previously.
            pruned_offsets, pruned_flat = sc.prune_tile_gs(
                tile_index_offsets, tile_flat_indices, w_gi,
                args.online_prune_keep_frac, method,
                max_gs_per_tile=args.online_prune_max_gs_per_tile)
            all_ordered, _tile_cum_counts = sc.build_greedy_order(
                "tile_partial", method, utilities, visibility, pruned_offsets,
                pruned_flat, w_gi, bpg, max_budget_bytes, gs_order=args.gs_order)
            if args.track_schedule == "round_robin":
                orders[(cadence_idx, method)] = partition_into_tracks(
                    all_ordered, utilities, pruned_offsets, args.n_tracks)
            else:  # greedy -- schedule depends on bandwidth too (via track_rates), so
                   # pass 1 here only precomputes the bw-independent tile-chunk layout;
                   # per-bw schedules are filled in right below.
                tile_ranked = utilities[:, 0]
                n_avail = len(all_ordered)
                starts, ends = tile_chunk_offsets(tile_ranked, pruned_offsets, n_avail)
                reachable = starts < n_avail
                tile_ranked_r = tile_ranked[reachable]
                starts_r = starts[reachable]
                ends_r = ends[reachable]
                tile_bytes_r = (ends_r - starts_r) * bpg
                greedy_chunks[(cadence_idx, method)] = (all_ordered, starts_r, ends_r)
                for bw in args.bandwidths_mbps:
                    rates = [w / sum(args.track_weights) * budget_bytes(bw, 1.0)
                             for w in args.track_weights]
                    schedules[(cadence_idx, method, bw)] = schedule_tracks_greedy(
                        tile_ranked_r, tile_bytes_r, rates)
    logger.info("orders: {} cadence ticks x {} methods computed ({:.1f}s elapsed)",
                len(cadence_rows), len(args.methods), time.time() - t0)

    # --- Pass 2: dense render/metric samples -- frozen order, continuously growing
    #     budget within the current cadence window, actual continuous camera pose ---
    # PNGs are only saved for the LAST mark of the first and last cadence windows (visual
    # grounding) -- every mark's frame still goes into the vmaf/*.yuv sequences via
    # FrameWriter regardless (see class docstring). Deliberately NOT the trace's first/last
    # render mark: elapsed_sec resets to 0 (budget_bytes=0, guaranteed-empty frame) at every
    # cadence tick, and whenever duration is an exact multiple of interval_sec (the common
    # case) the trace's literal first AND last marks both land on that reset instant -- the
    # two blackest possible frames. The last mark of a window has the window's maximum
    # elapsed_sec (most bytes delivered), which is what's actually worth looking at.
    marks_cadence_idx = [int(t // args.interval_sec) for t in render_marks]
    windows: list[list[int]] = []
    for i, ci in enumerate(marks_cadence_idx):
        if windows and marks_cadence_idx[windows[-1][-1]] == ci:
            windows[-1].append(i)
        else:
            windows.append([i])
    # When duration is an exact multiple of interval_sec (the common case), the trailing
    # window is a singleton (elapsed_sec=0, empty by construction) -- skip it when picking
    # the "last window" representative so both saved marks aren't guaranteed-empty.
    real_windows = [w for w in windows if len(w) > 1] or windows
    repr_marks = {windows[0][-1], real_windows[-1][-1]}
    is_repr_mark = lambda fi: fi in repr_marks  # noqa: E731
    metric_rows = []
    for fine_idx, trace_row in enumerate(render_rows):
        t_sec = render_marks[fine_idx]
        cadence_idx = int(t_sec // args.interval_sec)
        elapsed_sec = t_sec - cadence_idx * args.interval_sec

        cam = scam.build_streaming_camera(
            trace_row, BICYCLE_SCENE_QUAT_XYZW, img_w, img_h,
            uid=fine_idx, znear=args.znear, zfar=args.zfar, device=device,
            swap_top_bottom=args.swap_top_bottom)

        gt_rendered = sc.render_gs(rend_gs_full, cam, args.white_bg)
        frame_writers["gt"].write(gt_rendered)
        if is_repr_mark(fine_idx) and png_executor:
            png_futures.append(png_executor.submit(
                torchvision.utils.save_image, gt_rendered.cpu().clone(),
                str(out_root / "gt_renders" / f"frame_{fine_idx:04d}.png")))

        for method in args.methods:
            for bw in args.bandwidths_mbps:
                bb = budget_bytes(bw, elapsed_sec)
                sel_parts, used_bytes = [], 0
                if args.track_schedule == "round_robin":
                    track_bb = budget_bytes(bw / args.n_tracks, elapsed_sec)
                    for track_ordered in orders[(cadence_idx, method)]:
                        sel_k, used_k = sc.select_at_budget(track_ordered, track_bb, bpg, None)
                        sel_parts.append(sel_k)
                        used_bytes += used_k
                else:  # greedy -- per-tile partial cutoff from the precomputed schedule,
                       # not a flat per-track byte-budget truncation.
                    all_ordered, starts_r, ends_r = greedy_chunks[(cadence_idx, method)]
                    schedule = schedules[(cadence_idx, method, bw)]
                    rates = [w / sum(args.track_weights) * budget_bytes(bw, 1.0)
                             for w in args.track_weights]
                    for j, (start_sec, end_sec, track_idx) in enumerate(schedule):
                        if elapsed_sec <= start_sec:
                            continue
                        chunk_start, chunk_end = int(starts_r[j]), int(ends_r[j])
                        if elapsed_sec >= end_sec:
                            n_gs = chunk_end - chunk_start
                        else:
                            bytes_sent = rates[track_idx] * (elapsed_sec - start_sec)
                            n_gs = min(int(bytes_sent // bpg), chunk_end - chunk_start)
                        if n_gs > 0:
                            sel_parts.append(all_ordered[chunk_start:chunk_start + n_gs])
                            used_bytes += n_gs * bpg
                selected = np.concatenate(sel_parts) if sel_parts else np.array([], dtype=np.int64)
                sub_gs = sc.subset_gaussians(rend_gs_full, selected)
                rendered = sc.render_gs(sub_gs, cam, args.white_bg)
                del sub_gs
                metrics = sc.compute_metrics(rendered, gt_rendered, skip_lpips=True)

                frame_writers[(bw, method)].write(rendered)
                if is_repr_mark(fine_idx) and png_executor:
                    png_path = out_root / f"renders/bw_{bw}mbps/{method}/frame_{fine_idx:04d}.png"
                    png_futures.append(png_executor.submit(
                        torchvision.utils.save_image, rendered.cpu().clone(), str(png_path)))

                metric_rows.append({
                    "frame_idx": fine_idx, "t_sec": t_sec, "cadence_idx": cadence_idx,
                    "elapsed_sec": elapsed_sec, "method": method,
                    "bandwidth_mbps": bw, "budget_bytes": bb, "used_bytes": used_bytes,
                    "n_tracks": args.n_tracks, "track_schedule": args.track_schedule,
                    "track_weights": ",".join(str(w) for w in args.track_weights),
                    "n_selected": len(selected), "n_gs_total": len(opacity),
                    "psnr": metrics["psnr"], "ssim": metrics["ssim"],
                })
        if fine_idx % 10 == 0 or fine_idx == len(render_rows) - 1:
            logger.info("frame {}/{} t={:.1f}s done ({:.1f}s elapsed)",
                        fine_idx + 1, len(render_rows), t_sec, time.time() - t0)

    for fw in frame_writers.values():
        fw.close()
    if png_executor:
        for fut in png_futures:
            fut.result()
        png_executor.shutdown(wait=True)

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
        "track_schedule": args.track_schedule, "track_weights": args.track_weights,
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
    p.add_argument("--interval-sec", type=float, default=10.0,
                    help="Cadence: how often the tile ORDER is recomputed from the current "
                         "viewport (frozen in between).")
    p.add_argument("--render-interval-sec", type=float, default=0.5,
                    help="How often a frame is rendered+measured within a cadence window "
                         "(dense, continuous camera pose + continuously growing byte budget "
                         "against the frozen order) -- this is what makes the metric-vs-time "
                         "curve smooth rather than a handful of disconnected snapshots.")
    p.add_argument("--duration-sec", type=float, default=None,
                    help="Cap the trace session length (default: use the full recorded trace "
                         "duration). E.g. 30 to only simulate the first 30s.")
    p.add_argument("--png-workers", type=int, default=4,
                    help="Thread pool size for the small number of representative PNG saves "
                         "(first + last render mark only, for visual grounding -- 0 disables "
                         "PNG saving entirely). Every mark still feeds the vmaf/*.yuv "
                         "sequences via FrameWriter regardless of this flag.")
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
    p.add_argument("--gs-order", default="weight")
    p.add_argument("--online-prune-keep-frac", type=float, default=1.0,
                    help="Within each tile, keep only the top keep-frac of that tile's own "
                         "Gaussians by w(g_i) before packing -- the tile 'finishes' after "
                         "fewer bytes, so the schedule reaches more tiles per cadence window "
                         "at coarser per-tile fidelity. 1.0 (default) = no-op. Never applied "
                         "to vd_lod (no real per-GS weight -- see NO_WEIGHT_SCHEMES). "
                         "Orthogonal to --n-tracks and to any offline scene-size reduction; "
                         "test separately, don't combine with either in one run.")
    p.add_argument("--online-prune-max-gs-per-tile", type=int, default=None,
                    help="Starvation backstop: absolute per-tile Gaussian cap applied AFTER "
                         "--online-prune-keep-frac, so one oversized tile (bicycle has a "
                         "1.37M-GS outlier vs. a 850-GS scene median) can't alone consume an "
                         "entire cadence window's budget. Applies even if keep-frac=1.0. "
                         "None (default) = no cap.")
    p.add_argument("--n-tracks", type=int, default=1,
                    help="Number of simultaneous QUIC/MOQ-style transmission tracks. "
                         "1 = current strictly-serial single-track behavior (default). "
                         "See --track-schedule for how tiles are assigned to tracks.")
    p.add_argument("--track-schedule", choices=["round_robin", "greedy"], default="round_robin",
                    help="round_robin (default): track k permanently owns tile-priority ranks "
                         "k, k+N, k+2N, ..., equal bandwidth share -- preserves the already-"
                         "reported n_tracks=1/4/8 numbers exactly. greedy (2026-07-16): "
                         "event-driven work-conserving list scheduling -- a track that "
                         "finishes its current tile claims the next unclaimed tile instead of "
                         "sitting idle; rates set via --track-weights.")
    p.add_argument("--track-weights", type=float, nargs="+", default=None,
                    help="Only consulted when --track-schedule greedy. Per-track relative "
                         "bandwidth share, e.g. '4 3 2 1' for n_tracks=4. Must have exactly "
                         "--n-tracks entries and be non-increasing (track 0 = highest-"
                         "bandwidth slot by convention -- w_i >= w_j for i < j). Default: "
                         "equal weights ([1.0]*n_tracks).")
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
    if args.track_weights is None:
        args.track_weights = [1.0] * args.n_tracks
    if len(args.track_weights) != args.n_tracks:
        raise ValueError(f"--track-weights has {len(args.track_weights)} entries, "
                          f"expected --n-tracks={args.n_tracks}")
    for i in range(len(args.track_weights) - 1):
        if args.track_weights[i] < args.track_weights[i + 1]:
            raise ValueError(
                f"--track-weights must be non-increasing (track index = priority slot, "
                f"w_i >= w_j for i < j) -- got {args.track_weights}, "
                f"w[{i}]={args.track_weights[i]} < w[{i + 1}]={args.track_weights[i + 1]}")
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
