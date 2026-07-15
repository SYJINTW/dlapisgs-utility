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

# bicycle's scene_setting.csv correction quaternion (x,y,z,w), scale=1 (moot, not
# applied to the camera -- CAM.py applies scale to the loaded scene object instead).
BICYCLE_SCENE_QUAT_XYZW = (-0.1305, 0.0, 0.0, 0.9914)


def budget_bytes(bw_mbps: float, interval_sec: float) -> int:
    return int(bw_mbps * 1e6 * interval_sec / 8.0)


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


# ---------------------------------------------------------------------------
# Sanity gate
# ---------------------------------------------------------------------------

def run_sanity_check(args, device) -> bool:
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

def run_sweep(args, device):
    """Continuous-viewport streaming model (2026-07-14 redesign, replaces the earlier
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
            raw_scores = sc.compute_raw_scores(
                method, oracle_data=None, camera_index=cadence_idx, n_tiles=n_tiles,
                visibility=visibility, distances=distances, num_lod=1, W_k=W_k, C_k=N_k,
                ml_predict_kwargs=ml_predict_kwargs)
            utilities = sc.sort_tiles(raw_scores, n_gs_per_tile, bpg, greedy_key="marginal",
                                       num_of_level=1)
            all_ordered, tile_cum_counts = sc.build_greedy_order(
                "tile_partial", method, utilities, visibility, tile_index_offsets,
                tile_flat_indices, w_gi, bpg, max_budget_bytes, gs_order=args.gs_order)
            orders[(cadence_idx, method)] = (all_ordered, tile_cum_counts)
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
            all_ordered, tile_cum_counts = orders[(cadence_idx, method)]
            for bw in args.bandwidths_mbps:
                bb = budget_bytes(bw, elapsed_sec)
                selected, used_bytes = sc.select_at_budget(all_ordered, bb, bpg, tile_cum_counts)
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
            subprocess.run(
                ["vmaf", "-r", str(gt_yuv), "-d", str(dist_yuv),
                 "-w", str(img_w), "-h", str(img_h), "-p", "420", "-b", "8",
                 "--json", "-o", str(out_json)],
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
    p.add_argument("--ml-model-dir",
                    default=str(HERE / "output/ml_models_experimental/per_scene/bicycle/AC"))
    p.add_argument("--ml-model-type", default="lgbm")
    p.add_argument("--weight-mode", default="screen_area")
    p.add_argument("--w-norm", default="sum")
    p.add_argument("--c-norm", default="sum")
    p.add_argument("--gs-order", default="weight")
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
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.vmaf_only:
        run_vmaf(args)
        return

    ok = run_sanity_check(args, device)
    if not ok and not args.force:
        logger.error("Sanity check FAILED -- refusing to run full sweep (pass --force to override)")
        sys.exit(1)
    if args.sanity_check_only:
        return

    run_sweep(args, device)


if __name__ == "__main__":
    main()
