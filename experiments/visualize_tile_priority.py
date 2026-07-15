#!/usr/bin/env python3
"""Top-down (X-Z) map of tile send-priority rank per method, at a handful of cadence ticks
along a real trace -- shows (a) how vd_lod/v_lod_w/ml disagree on which tiles to send first,
and (b) how the camera keeps moving tick to tick while each tick's order is frozen. Diagnostic
only (like experiments/visualize_trace_views.py), not called from any pipeline. Tile-priority
ranks come from selection_core.sort_tiles (same call streaming_sim.py's Pass 1 makes) -- no
rendering needed, so this skips loading the renderable GaussianModel entirely.

Usage:
  conda run -n gaussian_splatting python experiments/visualize_tile_priority.py \\
    --interval-sec 3.0 --n-ticks 6 --output output/.../tile_priority.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
WORKSPACE = HERE.parent

sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

import visibility_AABB_pytorch  # noqa: E402
import utility_calculation as uc  # noqa: E402
import io_3dgs  # noqa: E402
from ml import predict as ml_predict, features as ml_features  # noqa: E402
import selection_core as sc  # noqa: E402
import streaming_camera as scam  # noqa: E402
from streaming_sim import (pick_longest_trace, load_trace_rows, sample_at_marks,
                            marks_up_to, BICYCLE_SCENE_QUAT_XYZW)  # noqa: E402

METHODS = ["vd_lod", "v_lod_w", "ml"]
_METHOD_LABEL = {"vd_lod": "Heuristic (baseline)", "v_lod_w": "Heuristic (ours)", "ml": "Learned (ours)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bicycle-trace-dir",
                    default=str(HERE / "dataset/EyeNavGS_NTHU_Dataset/bicycle"))
    p.add_argument("--trace-file", default=None)
    p.add_argument("--ply", default=str(WORKSPACE / "exp-dataset/bicycle/point_cloud.ply"))
    p.add_argument("--tiling-cache",
                    default=str(HERE / "output/oracle_tiling_cache/bicycle_8x8x8.npz"))
    p.add_argument("--img-w", type=int, default=1600)
    p.add_argument("--interval-sec", type=float, default=3.0)
    p.add_argument("--duration-sec", type=float, default=30.0)
    p.add_argument("--n-ticks", type=int, default=6)
    p.add_argument("--weight-mode", default="screen_area")
    p.add_argument("--w-norm", default="sum")
    p.add_argument("--c-norm", default="sum")
    p.add_argument("--ml-model-dir",
                    default=str(HERE / "output/ml_models_experimental/per_scene/bicycle/AC"))
    p.add_argument("--ml-model-type", default="lgbm")
    p.add_argument("--no-swap-top-bottom", dest="swap_top_bottom", action="store_false")
    p.set_defaults(swap_top_bottom=True)
    p.add_argument("--znear", type=float, default=0.01)
    p.add_argument("--zfar", type=float, default=100.0)
    p.add_argument("--output", required=True)
    p.add_argument("--output-3d", default=None,
                    help="If set, also render a 3D (oblique-angle, zoomed) version to this path.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trace_path = Path(args.trace_file) if args.trace_file else pick_longest_trace(
        Path(args.bicycle_trace_dir))
    rows = load_trace_rows(trace_path)
    duration_ms = min(int(rows[-1]["timestep"]), int(args.duration_sec * 1000))
    cadence_marks = marks_up_to(duration_ms, args.interval_sec)
    cadence_rows = sample_at_marks(rows, cadence_marks)

    tick_idx = np.linspace(0, len(cadence_rows) - 1, args.n_ticks).round().astype(int)
    tick_idx = sorted(set(tick_idx.tolist()))

    aspect = scam.frustum_aspect(cadence_rows[0], swap_top_bottom=args.swap_top_bottom)
    img_w = args.img_w
    img_h = max(1, round(img_w / aspect))

    sel_gs = io_3dgs.GaussianModelV2(str(args.ply))
    tc = np.load(str(args.tiling_cache), allow_pickle=True)
    min_corners, max_corners = tc["min_corners"], tc["max_corners"]
    index_offsets, flat_indices = tc["index_offsets"], tc["flat_indices"]
    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers = ((min_corners_t + max_corners_t) / 2.0).cpu().numpy()
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
    n_gs_per_tile = (index_offsets[1:] - index_offsets[:-1]).astype(np.float32)
    n_tiles = len(index_offsets) - 1

    opacity = sel_gs.data["opacity"]["data"]
    scale_0, scale_1, scale_2 = (sel_gs.data[k]["data"] for k in ("scale_0", "scale_1", "scale_2"))
    rot_0, rot_1, rot_2, rot_3 = (sel_gs.data[k]["data"] for k in ("rot_0", "rot_1", "rot_2", "rot_3"))
    gs_xyz = np.stack([sel_gs.data[k]["data"] for k in ("x", "y", "z")], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)
    bpg = sc.bytes_per_gaussian(sel_gs)

    ml_model = ml_predict.load_model(args.ml_model_dir, args.ml_model_type,
                                      expected_n_gs=len(sel_gs.data["x"]["data"]))
    ml_feature_names = json.loads((Path(args.ml_model_dir) / "feature_names.json").read_text())
    ml_static_feats = ml_features.build_static_features(sel_gs.data, index_offsets, flat_indices)

    # --- per selected tick: camera pose + tile priority rank per method ---
    tick_data = []  # list of dicts: t_sec, cam_xz, cam_fwd_xz, ranks={method: (n_tiles,)}
    for cadence_idx in tick_idx:
        trace_row = cadence_rows[cadence_idx]
        cam = scam.build_streaming_camera(
            trace_row, BICYCLE_SCENE_QUAT_XYZW, img_w, img_h,
            uid=cadence_idx, znear=args.znear, zfar=args.zfar, device=device,
            swap_top_bottom=args.swap_top_bottom)

        distances = uc.calculate_distances(
            torch.tensor(tile_centers, dtype=torch.float32, device=device),
            cam.camera_center.to(device))
        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device)
        w_gi = sc.compute_camera_weights(
            cam, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3,
            gs_xyz_t, device, args.weight_mode, img_w, img_h)
        W_k, N_k = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_gi, w_norm=args.w_norm, c_norm=args.c_norm)

        cam_w2v = cam.world_view_transform.cpu().numpy()
        cam_center_np = cam.camera_center.cpu().numpy()
        ml_group_a = ml_features.build_group_a(
            cam_center_np, cam_w2v[:3, 2], float(cam.FoVx), float(cam.FoVy),
            tile_centers, n_gs_per_tile, distances.cpu().numpy(),
            visibility.cpu().numpy().astype(np.float32))
        ml_predict_kwargs = dict(
            model_dir=args.ml_model_dir, model_type=args.ml_model_type,
            static_features=ml_static_feats, group_a=ml_group_a,
            feature_names=ml_feature_names, model=ml_model)

        ranks = {}
        for method in METHODS:
            raw_scores = sc.compute_raw_scores(
                method, oracle_data=None, camera_index=cadence_idx, n_tiles=n_tiles,
                visibility=visibility, distances=distances, num_lod=1, W_k=W_k, C_k=N_k,
                ml_predict_kwargs=ml_predict_kwargs)
            utilities = sc.sort_tiles(raw_scores, n_gs_per_tile, bpg, greedy_key="marginal",
                                       num_of_level=1)
            order = utilities[:, 0]
            rank = np.empty(n_tiles, dtype=np.int64)
            rank[order] = np.arange(n_tiles)
            ranks[method] = rank

        tick_data.append(dict(
            t_sec=cadence_marks[cadence_idx],
            cam_xz=(cam_center_np[0], cam_center_np[2]),
            cam_fwd_xz=(cam_w2v[0, 2], cam_w2v[2, 2]),
            cam_xyz=tuple(cam_center_np[:3]),
            cam_fwd_xyz=(cam_w2v[0, 2], cam_w2v[1, 2], cam_w2v[2, 2]),
            ranks=ranks,
        ))

    x, z = tile_centers[:, 0], tile_centers[:, 2]
    cam_path_x = [d["cam_xz"][0] for d in tick_data]
    cam_path_z = [d["cam_xz"][1] for d in tick_data]
    # Real head trajectory spans only ~2-3 units (near scene origin); the tiled scene spans
    # ~130 units -- a full-scene view makes the camera look almost static. Render both: the
    # full-scene view for global prioritization pattern, and a zoomed view (camera-path bbox
    # + margin) where the actual local motion and rank changes are visible.
    margin = 20.0
    zoom_xlim = (min(cam_path_x) - margin, max(cam_path_x) + margin)
    zoom_zlim = (min(cam_path_z) - margin, max(cam_path_z) + margin)

    def render(out_path: Path, xlim=None, zlim=None):
        n_ticks = len(tick_data)
        fig, axes = plt.subplots(len(METHODS), n_ticks, figsize=(3.2 * n_ticks, 3.2 * len(METHODS)),
                                  sharex=True, sharey=True, constrained_layout=True)
        if n_ticks == 1:
            axes = axes[:, None]
        for row, method in enumerate(METHODS):
            for col, d in enumerate(tick_data):
                ax = axes[row, col]
                sc_plot = ax.scatter(x, z, c=d["ranks"][method], cmap="viridis_r", s=28,
                                      vmin=0, vmax=n_tiles - 1)
                ax.plot(cam_path_x, cam_path_z, "-", color="gray", linewidth=0.8, alpha=0.5, zorder=1)
                cx, cz = d["cam_xz"]
                fx, fz = d["cam_fwd_xz"]
                ax.plot(cx, cz, marker="*", color="red", markersize=16, zorder=3)
                ax.annotate("", xy=(cx + fx * 5, cz + fz * 5), xytext=(cx, cz),
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.5), zorder=3)
                if row == 0:
                    ax.set_title(f"t={d['t_sec']:.1f}s", fontsize=11)
                if col == 0:
                    ax.set_ylabel(_METHOD_LABEL[method], fontsize=10)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)
                if xlim:
                    ax.set_xlim(*xlim)
                if zlim:
                    ax.set_ylim(*zlim)
        fig.colorbar(sc_plot, ax=axes, shrink=0.6, label="send-priority rank (0 = sent first)")
        fig.suptitle(f"Tile send-priority by method vs. camera motion -- {trace_path.name}"
                     f"{' (zoomed on head trajectory)' if xlim else ' (full scene)'}", fontsize=13)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        print(f"wrote {out_path}")
        plt.close(fig)

    out_path = Path(args.output)
    render(out_path)
    zoomed_path = out_path.parent / f"{out_path.stem}_zoomed{out_path.suffix}"
    render(zoomed_path, xlim=zoom_xlim, zlim=zoom_zlim)

    if args.output_3d:
        render_3d(Path(args.output_3d), tick_data, tile_centers, n_tiles, zoom_xlim, zoom_zlim)


def render_3d(out_path: Path, tick_data: list, tile_centers: np.ndarray, n_tiles: int,
              zoom_xlim: tuple, zoom_zlim: tuple):
    """One 3D panel per (method, tick) -- real oblique angle, not a flattened top-down view,
    zoomed to the head-trajectory region (the full-scene view is dominated by tiles the
    camera's small motion envelope never gets near -- see the 2D zoomed figure's docstring)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    x, y, z = tile_centers[:, 0], tile_centers[:, 1], tile_centers[:, 2]
    in_view = (x >= zoom_xlim[0]) & (x <= zoom_xlim[1]) & (z >= zoom_zlim[0]) & (z <= zoom_zlim[1])
    cam_path = np.array([d["cam_xyz"] for d in tick_data])

    n_ticks = len(tick_data)
    fig = plt.figure(figsize=(5.5 * n_ticks, 5.5 * len(METHODS)))
    for row, method in enumerate(METHODS):
        for col, d in enumerate(tick_data):
            ax = fig.add_subplot(len(METHODS), n_ticks, row * n_ticks + col + 1, projection="3d")
            rank = d["ranks"][method]
            sc_plot = ax.scatter(x[in_view], z[in_view], y[in_view], c=rank[in_view],
                                  cmap="viridis_r", s=90, vmin=0, vmax=n_tiles - 1,
                                  edgecolors="k", linewidths=0.3)
            ax.plot(cam_path[:, 0], cam_path[:, 2], cam_path[:, 1], "-", color="gray",
                    linewidth=1.0, alpha=0.6)
            cx, cy, cz = d["cam_xyz"]
            fx, fy, fz = d["cam_fwd_xyz"]
            ax.scatter([cx], [cz], [cy], color="red", s=180, marker="*", depthshade=False)
            ax.quiver(cx, cz, cy, fx, fz, fy, length=6.0, color="red", linewidth=2.0,
                       arrow_length_ratio=0.3)
            ax.set_xlim(*zoom_xlim)
            ax.set_ylim(*zoom_zlim)
            ax.view_init(elev=28, azim=-60)
            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Z", fontsize=8)
            ax.set_zlabel("Y (up)", fontsize=8)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"t={d['t_sec']:.1f}s", fontsize=12)
            if col == 0:
                ax.text2D(-0.15, 0.5, _METHOD_LABEL[method], fontsize=11, rotation=90,
                          transform=ax.transAxes, va="center")
    fig.colorbar(sc_plot, ax=fig.axes, shrink=0.5, label="send-priority rank (0 = sent first)")
    fig.suptitle("Tile send-priority by method vs. camera motion -- 3D, zoomed on head trajectory",
                 fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
