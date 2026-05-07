#!/usr/bin/env python3
"""Visual debugging suite for a single camera viewpoint.

For a given camera index this script produces a multi-panel figure:
  Panel 1  – Tile map (bird's-eye 2D, top-down XZ) showing visible vs culled tiles
  Panel 2  – Screen projection: tile centres projected to NDC, sized by GS count,
              coloured by visibility / distance
  Panels 3–6 – Per-scheme utility heatmaps on the same screen projection
              (vd_lod, vd_lod_w, vd_lod_c, vd_lod_w_c)

Runs in the *gsquic* conda environment (numpy, torch, matplotlib).
The --ply flag is optional; omitting it disables the W_k / C_k factors
(vd_lod_w and vd_lod_w_c panels will show a "needs PLY" message).

Usage:
  conda run -n gsquic python3 experiments/debug_viewport.py \
      --camera-index 24 \
      --vis-npz output/0507_budget_sweep/budget_100mb/vd_lod_w_c/camera_024/selected.vis.npz \
      --tile-npz output/0507_budget_sweep/budget_100mb/vd_lod_w_c/camera_024/selected.npz \
      [--ply /path/to/point_cloud.ply] \
      --out output/0507_budget_sweep/inspect/debug_cam024.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "dlapisgs-utility"))
from utility_calculation import (
    compute_gaussian_weights,
    compute_tile_weights_and_counts,
    INVISIBLE_PRIORITY_EPS,
    DISTANCE_EPS,
    LAYER_WEIGHT_BASE,
)

SCHEME_LABELS = {
    "vd_lod":     "VD+LOD",
    "vd_lod_w":   "VD+LOD+W",
    "vd_lod_c":   "VD+LOD+C",
    "vd_lod_w_c": "VD+LOD+W+C (proposed)",
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _project_points(points_world: np.ndarray,
                    world_view: np.ndarray,
                    proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points → NDC xy.  Returns (ndc_xy, in_front) arrays."""
    N = len(points_world)
    ones = np.ones((N, 1), dtype=np.float32)
    p4 = np.concatenate([points_world, ones], axis=1)          # (N,4)
    cam = p4 @ world_view.T                                      # (N,4)
    clip = cam @ proj.T                                          # (N,4)
    w = clip[:, 3:4]
    in_front = w[:, 0] > 0
    w_safe = np.where(np.abs(w) < 1e-6, 1e-6, w)
    ndc = clip[:, :2] / w_safe                                   # (N,2) in [-1,1]
    return ndc, in_front


def _tile_centers(min_corners: np.ndarray, max_corners: np.ndarray) -> np.ndarray:
    return 0.5 * (min_corners + max_corners)


# ---------------------------------------------------------------------------
# Utility computation (raw scores, not ranked)
# ---------------------------------------------------------------------------

def _raw_scores(vis: np.ndarray, dist: np.ndarray,
                W_k: np.ndarray | None, C_k: np.ndarray | None
                ) -> dict[str, np.ndarray]:
    vis_f = np.where(vis, 1.0, INVISIBLE_PRIORITY_EPS).astype(np.float32)
    dist_f = 1.0 / (dist + DISTANCE_EPS)
    base = vis_f * dist_f

    scores: dict[str, np.ndarray] = {}
    scores["vd_lod"] = base.copy()
    scores["vd_lod_w"] = base * W_k if W_k is not None else None
    scores["vd_lod_c"] = base * C_k if C_k is not None else None
    scores["vd_lod_w_c"] = base * W_k * C_k if (W_k is not None and C_k is not None) else None
    return scores


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter_tiles(ax: plt.Axes,
                   ndc: np.ndarray,
                   values: np.ndarray,
                   visible: np.ndarray,
                   gs_counts: np.ndarray,
                   title: str,
                   cmap: str = "plasma",
                   label: str = "score") -> None:
    """Scatter tile centres in NDC space, sized by GS count, coloured by values."""
    sizes = 40 + 400 * (gs_counts / gs_counts.max())
    norm = Normalize(vmin=values.min(), vmax=values.max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(values)

    # Invisible tiles dimmed
    alpha = np.where(visible, 1.0, 0.25)
    for i in range(len(ndc)):
        ax.scatter(ndc[i, 0], ndc[i, 1], s=sizes[i],
                   color=colors[i], alpha=float(alpha[i]),
                   edgecolors="white" if visible[i] else "grey",
                   linewidths=0.5, zorder=3)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    ax.set_xlabel("NDC x", fontsize=8)
    ax.set_ylabel("NDC y", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_facecolor("#111111")

    plt.colorbar(sm, ax=ax, label=label, fraction=0.046, pad=0.04)

    vis_patch = mpatches.Patch(color="white", alpha=1.0, label="visible")
    cull_patch = mpatches.Patch(color="grey",  alpha=0.4, label="culled")
    ax.legend(handles=[vis_patch, cull_patch], fontsize=7, loc="lower right")


def _birdseye(ax: plt.Axes,
              centers: np.ndarray,
              visible: np.ndarray,
              cam_center: np.ndarray,
              gs_counts: np.ndarray) -> None:
    """Top-down XZ view of tile grid."""
    sizes = 60 + 600 * (gs_counts / gs_counts.max())
    colors = ["#44cc44" if v else "#cc4444" for v in visible]
    ax.scatter(centers[:, 0], centers[:, 2], s=sizes, c=colors,
               edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3)
    ax.scatter(cam_center[0], cam_center[2], marker="*", s=300,
               color="gold", edgecolors="black", linewidths=0.8, zorder=5, label="camera")
    for i, (cx, cz) in enumerate(zip(centers[:, 0], centers[:, 2])):
        ax.annotate(str(i), (cx, cz), fontsize=5, ha="center", va="center", color="white")
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Z", fontsize=8)
    ax.set_title("Bird's-eye tile map (XZ)", fontsize=9, fontweight="bold")
    ax.set_facecolor("#111111")
    ax.legend(fontsize=7)
    vis_p = mpatches.Patch(color="#44cc44", label=f"visible ({visible.sum()})")
    cull_p = mpatches.Patch(color="#cc4444", label=f"culled ({(~visible).sum()})")
    ax.legend(handles=[vis_p, cull_p], fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Viewport utility debug visualizer")
    parser.add_argument("--camera-index", type=int, required=True)
    parser.add_argument("--vis-npz",  type=str, required=True,
                        help="selected.vis.npz for this camera")
    parser.add_argument("--tile-npz", type=str, required=True,
                        help="selected.npz (tile metadata) for this camera")
    parser.add_argument("--ply", type=str, default=None,
                        help="Full-scene PLY for computing W_k / C_k (optional, slow)")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: same dir as vis-npz). Ignored in interactive mode.")
    parser.add_argument("--interactive", action="store_true",
                        help="Show plot in a window instead of saving (run locally, no GPU needed)")
    args = parser.parse_args()

    if args.interactive:
        matplotlib.use("TkAgg")   # works on macOS/Linux with display
    else:
        matplotlib.use("Agg")

    vis_data  = np.load(args.vis_npz)
    tile_data = np.load(args.tile_npz, allow_pickle=True)

    min_corners = vis_data["min_corners"]    # (N,3)
    max_corners = vis_data["max_corners"]    # (N,3)
    visibility  = vis_data["visibility"]     # (N,) bool
    distances   = vis_data["distances"]      # (N,) float
    cam_center  = vis_data["camera_center"]  # (3,)
    wv_transform = vis_data["world_view_transform"]   # (4,4)
    proj_matrix  = vis_data["projection_matrix"]      # (4,4)

    offsets     = tile_data["index_offsets"]  # (N+1,)
    flat_idx    = tile_data["flat_indices"]   # (M,)
    gs_counts   = np.diff(offsets).astype(np.float32)
    centers     = _tile_centers(min_corners, max_corners)
    N           = len(centers)

    # Screen-space projection
    ndc, in_front = _project_points(centers, wv_transform, proj_matrix)

    # Optional: load PLY for W_k / C_k
    W_k_np: np.ndarray | None = None
    C_k_np: np.ndarray | None = None
    if args.ply:
        print(f"Loading PLY: {args.ply} ...")
        sys.path.insert(0, str(WORKSPACE_ROOT / "GS-Interface"))
        from io_3dgs import GaussianModelV2  # type: ignore
        gs_model = GaussianModelV2()
        gs_model.load_ply(args.ply)
        opacity = gs_model.get_opacity.squeeze().detach()
        scales  = gs_model.get_scaling.detach()
        w_gi = compute_gaussian_weights(opacity, scales[:, 0], scales[:, 1], scales[:, 2], args.gamma)
        W_k, C_k = compute_tile_weights_and_counts(
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(flat_idx,  dtype=torch.long),
            w_gi,
        )
        W_k_np = W_k.cpu().numpy()
        C_k_np = C_k.cpu().numpy()
        print("W_k / C_k computed.")

    scores = _raw_scores(visibility, distances, W_k_np, C_k_np)

    # -----------------------------------------------------------------------
    # Figure layout: 2 rows × 4 cols
    #   Row 0: [bird's-eye | screen vis/dist | vd_lod | vd_lod_w]
    #   Row 1: [blank      | blank           | vd_lod_c | vd_lod_w_c]
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(f"Camera {args.camera_index:03d}  —  "
                 f"{visibility.sum()}/{N} tiles visible  |  "
                 f"{int(gs_counts[visibility].sum()):,} visible GS  |  "
                 f"{int(gs_counts.sum()):,} total GS",
                 fontsize=13, fontweight="bold")

    gs_fig = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

    # Panel 0: bird's-eye
    ax0 = fig.add_subplot(gs_fig[0, 0])
    _birdseye(ax0, centers, visibility, cam_center, gs_counts)

    # Panel 1: screen projection coloured by distance (visible only)
    ax1 = fig.add_subplot(gs_fig[0, 1])
    _scatter_tiles(ax1, ndc, distances, visibility, gs_counts,
                   title="Screen projection (distance)", cmap="cool", label="dist")

    # Panels 2-5: one per scheme
    positions = [(0, 2), (0, 3), (1, 2), (1, 3)]
    scheme_order = ["vd_lod", "vd_lod_w", "vd_lod_c", "vd_lod_w_c"]
    for (row, col), scheme in zip(positions, scheme_order):
        ax = fig.add_subplot(gs_fig[row, col])
        s = scores[scheme]
        if s is None:
            ax.text(0.5, 0.5, "Needs --ply", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            ax.set_title(SCHEME_LABELS[scheme], fontsize=9, fontweight="bold")
            ax.set_facecolor("#111111")
            continue
        _scatter_tiles(ax, ndc, s, visibility, gs_counts,
                       title=SCHEME_LABELS[scheme], cmap="plasma", label="utility")

    # Blank panels in row 1 cols 0-1
    for col in [0, 1]:
        ax_blank = fig.add_subplot(gs_fig[1, col])
        ax_blank.axis("off")

        if col == 0:
            # Summary table
            visible_gs = int(gs_counts[visibility].sum())
            total_gs   = int(gs_counts.sum())
            lines = [
                f"Camera index : {args.camera_index}",
                f"Tiles        : {N} total, {visibility.sum()} visible",
                f"Visible GS   : {visible_gs:,}",
                f"Total GS     : {total_gs:,}",
                f"Visible frac : {100*visible_gs/total_gs:.1f}%",
                f"Budget needed: ~{visible_gs*248/1e6:.0f} MB (full quality)",
            ]
            ax_blank.text(0.05, 0.95, "\n".join(lines),
                          transform=ax_blank.transAxes,
                          va="top", ha="left", fontsize=9,
                          fontfamily="monospace",
                          bbox=dict(boxstyle="round", fc="#1a1a1a", ec="grey", alpha=0.8),
                          color="white")

    if args.interactive:
        plt.show()
    else:
        out_path = Path(args.out) if args.out else \
            Path(args.vis_npz).parent / f"debug_cam{args.camera_index:03d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
