#!/usr/bin/env python3
"""Visualize a camera trace as a 4-panel inspection figure:

  (a) 3D iso view with full camera right/up/forward axes and a viridis-colored
      faux trajectory line between consecutive frames (the views are discrete,
      but the connection helps eyeball spatial coverage).
  (b) Top-down (XY) — primary axes the viewer can navigate in.
  (c) Front (XZ).
  (d) Side  (YZ).

Scene context: tile AABBs from a tiling NPZ, or a single scene AABB from a PLY.

Usage:
    python experiments/visualize_trace_views.py \
        --trace exp-dataset/bicycle/sparse_views_100.json \
        --out   plotting/traces/bicycle_views.png \
        [--tiling-npz path/to/tiling.npz]
        [--ply path/to/point_cloud.ply]
        [--connect-trajectory 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "dlapisgs-tiling"))
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))

from headless_visualize_tiles import draw_aabb, set_axes_equal  # type: ignore
from camera.blender_camera import readCamerasFromTransforms       # type: ignore


def _camera_axes(cams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns centers, forward, right, up — each (N, 3) in world coords."""
    centers, fwds, rights, ups = [], [], [], []
    for ci in cams:
        center = -ci.R @ ci.T
        fwds.append(ci.R @ np.array([0.0, 0.0, 1.0]))   # COLMAP camera +Z = forward
        rights.append(ci.R @ np.array([1.0, 0.0, 0.0])) # +X = right
        ups.append(ci.R @ np.array([0.0, -1.0, 0.0]))   # -Y = up (COLMAP Y-down)
        centers.append(center)
    return (np.asarray(centers), np.asarray(fwds), np.asarray(rights), np.asarray(ups))


def _load_ply_xyz(ply_path: Path, max_points: int = 200_000) -> np.ndarray:
    from plyfile import PlyData
    pd = PlyData.read(str(ply_path))
    el = pd.elements[0]
    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=-1).astype(np.float64)
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
    return xyz


def _draw_scene(ax_or_axes, scene_min, scene_max, tiling_mins=None, tiling_maxs=None,
                two_d: tuple[int, int] | None = None) -> None:
    """Draw tile AABBs (if given) or a single scene AABB.

    If `two_d` is None, the axis is 3D and we use draw_aabb. Otherwise it's a
    matplotlib 2D axis and we plot rectangles in the (axis_a, axis_b) plane.
    """
    if two_d is None:
        if tiling_mins is not None:
            for mn, mx in zip(tiling_mins, tiling_maxs):
                draw_aabb(ax_or_axes, mn, mx, color="gray", alpha=0.25, linewidth=0.3)
        elif scene_min is not None:
            draw_aabb(ax_or_axes, scene_min, scene_max, color="gray", alpha=0.6, linewidth=1.0)
        return

    a, b = two_d
    if tiling_mins is not None:
        for mn, mx in zip(tiling_mins, tiling_maxs):
            rect = plt.Rectangle((mn[a], mn[b]), mx[a] - mn[a], mx[b] - mn[b],
                                  fill=False, edgecolor="gray", linewidth=0.3, alpha=0.4)
            ax_or_axes.add_patch(rect)
    elif scene_min is not None:
        rect = plt.Rectangle((scene_min[a], scene_min[b]),
                              scene_max[a] - scene_min[a], scene_max[b] - scene_min[b],
                              fill=False, edgecolor="gray", linewidth=1.0, alpha=0.7)
        ax_or_axes.add_patch(rect)


def _set_equal_2d(ax, x, y) -> None:
    lo_x, hi_x = float(x.min()), float(x.max())
    lo_y, hi_y = float(y.min()), float(y.max())
    cx, cy = 0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y)
    r = 0.55 * max(hi_x - lo_x, hi_y - lo_y) or 1.0
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--out",   required=True, type=Path)
    p.add_argument("--tiling-npz", type=Path, default=None)
    p.add_argument("--ply", type=Path, default=None)
    p.add_argument("--arrow-frac", type=float, default=0.07,
                   help="Camera-axis arrow length as fraction of scene diagonal.")
    p.add_argument("--connect-trajectory", type=int, default=1,
                   help="1: draw a faded line connecting consecutive cameras (faux trajectory).")
    p.add_argument("--split-at", type=int, default=150,
                   help="Color cameras [0, N) as train (blue) and [N, end) as eval (orange). 0 = off (viridis by index).")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--width",  type=int, default=800)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    cams = readCamerasFromTransforms(args.trace, args.height, args.width)
    centers, fwds, rights, ups = _camera_axes(cams)
    print(f"[viz] {len(cams)} cameras  centroid={centers.mean(axis=0)}  spread={centers.std(axis=0)}")

    # Scene context
    scene_min = scene_max = None
    tmin = tmax = None
    if args.tiling_npz is not None and args.tiling_npz.exists():
        meta = np.load(args.tiling_npz)
        tmin, tmax = meta["min_corners"], meta["max_corners"]
        scene_min, scene_max = tmin.min(axis=0), tmax.max(axis=0)
    elif args.ply is not None and args.ply.exists():
        xyz = _load_ply_xyz(args.ply)
        scene_min, scene_max = xyz.min(axis=0), xyz.max(axis=0)

    # Diagonal for arrow scaling
    if scene_min is not None:
        diag = float(np.linalg.norm(scene_max - scene_min))
    else:
        diag = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    L = args.arrow_frac * diag

    # Camera colors: split train/eval or viridis by index
    split_at = args.split_at if 0 < args.split_at < len(cams) else 0
    if split_at:
        TRAIN_COLOR = "#1f77b4"   # matplotlib tab:blue
        EVAL_COLOR  = "#ff7f0e"   # matplotlib tab:orange
        point_colors = [TRAIN_COLOR if i < split_at else EVAL_COLOR for i in range(len(cams))]
    else:
        cmap = matplotlib.colormaps[args.cmap]
        point_colors = list(cmap(np.linspace(0.0, 1.0, len(cams))))

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
                          hspace=0.25, wspace=0.20)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_yz = fig.add_subplot(gs[1, 1])

    # ---- 3D iso ---- #
    # Clip scene context to the camera-region bbox so a single far-field
    # outlier doesn't zoom the whole panel out (matters a lot for mipnerf360).
    cam_lo = centers.min(axis=0); cam_hi = centers.max(axis=0)
    cam_diag = float(np.linalg.norm(cam_hi - cam_lo)) or 1.0
    view_lo = cam_lo - 0.10 * cam_diag
    view_hi = cam_hi + 0.10 * cam_diag
    if scene_min is not None:
        clipped_min = np.maximum(scene_min, view_lo)
        clipped_max = np.minimum(scene_max, view_hi)
        if np.all(clipped_max > clipped_min):
            if tmin is not None:
                # Filter tile AABBs to those that intersect the camera region.
                keep = np.all(tmax > view_lo, axis=1) & np.all(tmin < view_hi, axis=1)
                for mn, mx in zip(tmin[keep], tmax[keep]):
                    draw_aabb(ax3d, np.maximum(mn, view_lo), np.minimum(mx, view_hi),
                              color="gray", alpha=0.25, linewidth=0.3)
            else:
                draw_aabb(ax3d, clipped_min, clipped_max, color="gray", alpha=0.6, linewidth=1.0)
    if args.connect_trajectory:
        ax3d.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                  color="black", alpha=0.25, linewidth=0.7)
    ax3d.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
                fwds[:, 0], fwds[:, 1], fwds[:, 2],
                length=L * 0.25, color="crimson", linewidth=0.4, alpha=0.3, normalize=False)
    ax3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                 c=point_colors, s=22, depthshade=True, alpha=0.95, zorder=5)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title(f"{args.trace.name}  —  {len(cams)} views  (3D iso)")
    # Force the 3D box to the camera region (set_axes_equal would otherwise use full data extent)
    span = 0.55 * max(view_hi - view_lo)
    cx, cy, cz = 0.5 * (view_lo + view_hi)
    ax3d.set_xlim3d(cx - span, cx + span)
    ax3d.set_ylim3d(cy - span, cy + span)
    ax3d.set_zlim3d(cz - span, cz + span)

    # ---- 2D top-down / side panels ---- #
    def _panel(ax, a: int, b: int, label_a: str, label_b: str, title: str) -> None:
        _draw_scene(ax, scene_min, scene_max, tmin, tmax, two_d=(a, b))
        if args.connect_trajectory:
            ax.plot(centers[:, a], centers[:, b],
                    color="black", alpha=0.25, linewidth=0.6)
        ax.scatter(centers[:, a], centers[:, b],
                   c=point_colors, s=20, alpha=0.9, zorder=3)
        ax.quiver(centers[:, a], centers[:, b], fwds[:, a], fwds[:, b],
                  angles="xy", scale_units="xy", scale=5.0/L, color="crimson",
                  width=0.002, alpha=0.25)
        ax.set_xlabel(label_a); ax.set_ylabel(label_b); ax.set_title(title)
        _set_equal_2d(ax, centers[:, a], centers[:, b])

    _panel(ax_xy, 0, 1, "X", "Y", "Top-down (XY)")
    _panel(ax_xz, 0, 2, "X", "Z", "Front (XZ)")
    _panel(ax_yz, 1, 2, "Y", "Z", "Side  (YZ)")

    # Legend
    import matplotlib.patches as mpatches
    leg_x = 0.55; leg_y = 0.03
    if split_at:
        fig.text(leg_x, leg_y + 0.088,
                 f"● train  (cams 0–{split_at-1})", fontsize=9, color=TRAIN_COLOR)
        fig.text(leg_x, leg_y + 0.066,
                 f"● eval   (cams {split_at}–{len(cams)-1})", fontsize=9, color=EVAL_COLOR)
    else:
        sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps[args.cmap],
                                   norm=plt.Normalize(0, len(cams) - 1))
        sm.set_array([])
        cbar_ax = fig.add_axes((0.93, 0.15, 0.012, 0.7))
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label("frame_index", rotation=270, labelpad=15)
        fig.text(leg_x, leg_y + 0.066, "● scatter = camera center (color = frame_index)", fontsize=9)
    fig.text(leg_x, leg_y + 0.022, "→ red = forward direction", fontsize=9, color="crimson")
    fig.text(leg_x, leg_y,         "— faded line = consecutive-frame trajectory", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
