#!/usr/bin/env python3
"""Generate N sparse random camera views around a 3DGS PLY scene.

Two pose modes are mixed (controlled by --inside-out-frac):
  * inside_out: camera position is sampled INSIDE the dense content region,
                forward direction is a random unit vector (random-walk viewer).
  * close_orbit: camera position is sampled on a shell at a tight radius
                 around the scene centroid, looking back at the centroid
                 (training-style; matches what the model was trained on).

Acceptance criterion (rejects "looking-into-the-void" framings):
  - At least min-visible-frac of subsampled Gaussian centers project inside
    the camera frustum.
  - AND the visible centers' NDC bounding box spans at least min-ndc-spread
    in BOTH x and y (so the scene fills a meaningful chunk of the frame,
    not a corner-dot).

Output JSON matches Frustum-for-3DGS/camera/blender_camera.readCamerasFromTransforms:
    {
      "camera_angle_x": <fov_x in radians>,
      "frames": [{"frame_index": i, "transform_matrix": <4x4 c2w, Blender>}, ...],
      "generation": {...}     # date/host/args/stats for reproducibility
    }
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import socket
import subprocess
from pathlib import Path

import numpy as np
from plyfile import PlyData


# ---------------------------------------------------------------------------- #
# PLY loading + robust scene geometry                                           #
# ---------------------------------------------------------------------------- #
def load_xyz(ply_path: Path, max_points: int, seed: int) -> np.ndarray:
    plydata = PlyData.read(str(ply_path))
    el = plydata.elements[0]
    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])],
                   axis=-1).astype(np.float64)
    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
    return xyz


def robust_scene_geometry(xyz: np.ndarray, radius_pct: float) -> tuple[np.ndarray, float]:
    """Use median + percentile-of-distance to define the dense-content region.

    Median centroid is robust to outlier Gaussians (e.g. far-field background).
    radius = `radius_pct`-th percentile of |xyz - centroid|.
    """
    centroid = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=-1)
    radius = float(np.percentile(dists, radius_pct))
    return centroid, radius


# ---------------------------------------------------------------------------- #
# Pose construction                                                              #
# ---------------------------------------------------------------------------- #
def sample_unit_vec(rng: np.random.Generator, up_axis: int,
                    sin_lo: float, sin_hi: float) -> np.ndarray:
    """Random unit vector with up-axis component (=sin(elevation)) in [sin_lo, sin_hi]."""
    sin_e = rng.uniform(sin_lo, sin_hi)
    horiz = math.sqrt(max(0.0, 1.0 - sin_e ** 2))
    theta = rng.uniform(0.0, 2.0 * math.pi)
    vec = np.zeros(3)
    others = [a for a in (0, 1, 2) if a != up_axis]
    vec[others[0]] = horiz * math.cos(theta)
    vec[others[1]] = horiz * math.sin(theta)
    vec[up_axis] = sin_e
    return vec


def look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 camera-to-world matrix in Blender convention (camera looks down -Z)."""
    forward = target - eye
    fn = np.linalg.norm(forward)
    if fn < 1e-8:
        return np.eye(4)
    forward = forward / fn
    z_axis = -forward                     # Blender +Z points behind camera
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(alt, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye
    return c2w


def forward_to_c2w(eye: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 c2w matrix built from an explicit forward direction (for inside-out)."""
    forward = forward / max(1e-8, np.linalg.norm(forward))
    target = eye + forward
    return look_at_c2w(eye, target, up)


# ---------------------------------------------------------------------------- #
# Visibility / void-rejection                                                    #
# ---------------------------------------------------------------------------- #
def evaluate_view(xyz: np.ndarray, c2w: np.ndarray, fov_x: float, fov_y: float,
                  znear: float = 0.01) -> tuple[float, float, float]:
    """Return (visible_fraction, ndc_spread_x, ndc_spread_y).

    `ndc_spread_*` is the span of visible points' NDC coords in each axis,
    measured as max-min (range [0, 2] since NDC is [-1, 1]).
    """
    c2w_colmap = c2w.copy()
    c2w_colmap[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w_colmap)

    pts_h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=-1)
    view = pts_h @ w2c.T
    z = view[:, 2]
    in_front = z > znear
    if not in_front.any():
        return 0.0, 0.0, 0.0

    tan_hx = math.tan(fov_x / 2.0)
    tan_hy = math.tan(fov_y / 2.0)
    x_ndc = view[:, 0] / (z * tan_hx)
    y_ndc = view[:, 1] / (z * tan_hy)
    inside = in_front & (np.abs(x_ndc) <= 1.0) & (np.abs(y_ndc) <= 1.0)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return 0.0, 0.0, 0.0
    frac = n_inside / len(xyz)
    spread_x = float(x_ndc[inside].max() - x_ndc[inside].min())
    spread_y = float(y_ndc[inside].max() - y_ndc[inside].min())
    return frac, spread_x, spread_y


# ---------------------------------------------------------------------------- #
# Render quality check                                                           #
# ---------------------------------------------------------------------------- #
def _render_quality(img_tensor, max_white: float, max_black: float,
                    min_edge_var: float) -> tuple[bool, dict]:
    """Return (passes: bool, stats: dict). img_tensor: float32 CHW [0,1] CPU/CUDA."""
    img = img_tensor.detach().cpu().numpy()   # CHW float [0,1]
    rgb = img.transpose(1, 2, 0)              # HWC
    mean_c = rgb.mean(axis=2)                 # HW float
    white_frac = float((mean_c > 0.92).mean())
    black_frac = float((mean_c < 0.08).mean())
    gray_u8 = (mean_c * 255).astype(np.float32)
    gp = np.pad(gray_u8, 1, mode="edge")
    lap = gp[:-2, 1:-1] + gp[2:, 1:-1] + gp[1:-1, :-2] + gp[1:-1, 2:] - 4 * gray_u8
    edge_var = float(lap.var())
    ok = white_frac <= max_white and black_frac <= max_black and edge_var >= min_edge_var
    return ok, {"white_frac": white_frac, "black_frac": black_frac, "edge_var": edge_var}


# ---------------------------------------------------------------------------- #
# QA: trace position scatter plot                                                #
# ---------------------------------------------------------------------------- #
def _plot_trace_positions(frames: list, centroid: np.ndarray, radius: float,
                          up_axis: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = np.array([f["transform_matrix"] for f in frames])[:, :3, 3]
    modes = [f.get("pose_mode", "inward") for f in frames]
    color_map = {"inward": "steelblue", "outward": "limegreen"}
    colors = [color_map.get(m, "gray") for m in modes]

    # Scale sphere and axes to actual camera spread, not robust_radius
    dists = np.linalg.norm(positions - centroid, axis=1)
    display_r = float(np.percentile(dists, 95)) * 1.1
    display_r = max(display_r, 0.1)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=colors, s=10, alpha=0.7)
    ax.scatter(*centroid, c="gold", s=80, marker="*", zorder=5, label="centroid")

    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
    sx = centroid[0] + display_r * np.cos(u) * np.sin(v)
    sy = centroid[1] + display_r * np.sin(u) * np.sin(v)
    sz = centroid[2] + display_r * np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color="gray", alpha=0.15, linewidth=0.5)

    from matplotlib.lines import Line2D
    legend = [Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue", label="inward"),
              Line2D([0],[0], marker="o", color="w", markerfacecolor="limegreen", label="outward"),
              Line2D([0],[0], marker="*", color="w", markerfacecolor="gold", label="centroid")]
    ax.legend(handles=legend, fontsize=8)
    ax.set_title(f"{len(frames)} accepted cameras  (robust_r={radius:.2f})", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[trace-plot] {out_path}")


# ---------------------------------------------------------------------------- #
# Position samplers                                                              #
# ---------------------------------------------------------------------------- #
def sample_inside_out(rng: np.random.Generator, centroid: np.ndarray, radius: float,
                       up_axis: int, position_frac: tuple[float, float],
                       elev_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Camera inside the dense region; forward direction is random.

    position_frac: (lo, hi) — sample radial offset uniformly in
                   [lo, hi] * radius from the centroid.
    elev_range:   (sin_lo, sin_hi) for the forward direction's up-axis component.
    """
    r = rng.uniform(*position_frac) * radius
    direction = sample_unit_vec(rng, up_axis, -1.0, 1.0)
    eye = centroid + r * direction
    forward = sample_unit_vec(rng, up_axis, elev_range[0], elev_range[1])
    return eye, forward


def sample_close_orbit(rng: np.random.Generator, centroid: np.ndarray, radius: float,
                        up_axis: int, position_frac: tuple[float, float],
                        elev_range: tuple[float, float],
                        lookat_jitter_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera on a shell around centroid, looking back at (centroid + jitter)."""
    r = rng.uniform(*position_frac) * radius
    direction = sample_unit_vec(rng, up_axis, elev_range[0], elev_range[1])
    eye = centroid + r * direction
    jitter = rng.normal(scale=lookat_jitter_frac * radius, size=3)
    forward = (centroid + jitter) - eye
    return eye, forward


# ---------------------------------------------------------------------------- #
# Main                                                                           #
# ---------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-views",   type=int, default=100)
    p.add_argument("--fov-deg",   type=float, default=50.0)
    p.add_argument("--width",     type=int, default=800)
    p.add_argument("--height",    type=int, default=800)
    p.add_argument("--scene-type", choices=("synthetic", "mipnerf360"), default="synthetic",
                   help="Auto-sets up-axis, pose mix, position-radius range. Overridable by other flags.")
    p.add_argument("--inside-out-frac", type=float, default=None,
                   help="Fraction of views sampled inside-out vs close-orbit. "
                        "Default: 0.0 for synthetic, 0.7 for mipnerf360.")
    p.add_argument("--robust-radius-pct", type=float, default=80.0,
                   help="Percentile of distance-from-centroid that defines the 'robust radius'.")
    p.add_argument("--orbit-radius-min", type=float, default=None,
                   help="Min position-radius for close-orbit, in robust-radius units. "
                        "Default: 1.0 (mipnerf360) / 1.2 (synthetic).")
    p.add_argument("--orbit-radius-max", type=float, default=None,
                   help="Max position-radius for close-orbit. Default: 1.5 / 2.0.")
    p.add_argument("--inside-radius-min", type=float, default=0.0,
                   help="Min position-radius for inside-out, in robust-radius units.")
    p.add_argument("--inside-radius-max", type=float, default=0.6,
                   help="Max position-radius for inside-out.")
    p.add_argument("--min-visible-frac", type=float, default=0.15,
                   help="Reject views where fewer than this fraction of subsampled "
                        "Gaussian centers fall in the frustum.")
    p.add_argument("--min-ndc-spread",   type=float, default=0.4,
                   help="Reject views where visible centers span less than this in NDC "
                        "(either axis). Forces 'scene fills frame'.")
    p.add_argument("--lookat-jitter-frac", type=float, default=0.1)
    p.add_argument("--orbit-radius-abs-min", type=float, default=None,
                   help="Close-orbit min radius in world units (overrides --orbit-radius-min × robust_radius).")
    p.add_argument("--orbit-radius-abs-max", type=float, default=None,
                   help="Close-orbit max radius in world units.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-proposals", type=int, default=20_000)
    p.add_argument("--subsample-points", type=int, default=50_000)
    # Render-quality filter (requires gaussian_splatting env when enabled)
    p.add_argument("--render-check", action="store_true",
                   help="Render each proposal and reject degenerate frames.")
    p.add_argument("--ply-renderer", type=Path, default=None,
                   help="PLY for GaussianModel renderer. Defaults to --ply.")
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--white-bg", action="store_true")
    p.add_argument("--gt-renders-dir", type=Path, default=None,
                   help="Save accepted-camera PNGs here (reusable as --gt-renders-cache).")
    p.add_argument("--max-white-frac", type=float, default=0.40)
    p.add_argument("--max-black-frac", type=float, default=0.40)
    p.add_argument("--min-edge-var", type=float, default=50.0,
                   help="Laplacian variance threshold (uint8 scale).")
    p.add_argument("--add-outward", type=int, default=0,
                   help="After generating --n-views inward cameras, add this many outward-facing "
                        "cameras by inverting the look direction of evenly-spaced accepted positions. "
                        "Render-checked when --render-check is set. "
                        "NOTE: skip for synthetic/object-only scenes — outward views see black background.")
    p.add_argument("--full-sphere", action="store_true",
                   help="Sample orbit cameras over the full sphere (elevation -90 to +90 deg) "
                        "instead of the scene-type default hemisphere/band.")
    # Recommended trace policy:
    #   synthetic (object-only): --n-views 300, --full-sphere, no --add-outward → 300 inward views
    #     outward skipped: synthetic bg is black → all outward views fail black_frame gate
    #   indoor real:             --full-sphere, --add-outward 100          → ~300 views
    #   outdoor real:            --orbit-radius-abs-min/max, --add-outward 100  → ~300 views
    args = p.parse_args()

    # Per-scene-type defaults
    if args.inside_out_frac is None:
        args.inside_out_frac = 0.7 if args.scene_type == "mipnerf360" else 0.0
    if args.orbit_radius_min is None:
        args.orbit_radius_min = 1.0 if args.scene_type == "mipnerf360" else 1.2
    if args.orbit_radius_max is None:
        args.orbit_radius_max = 1.5 if args.scene_type == "mipnerf360" else 2.0

    if args.scene_type == "synthetic":
        up_axis = 1
        up_world = np.array([0.0, 1.0, 0.0])
        elev_range_orbit = (0.05, 0.95)
        elev_range_inout = (-0.7, 0.9)
    else:  # mipnerf360
        up_axis = 2
        up_world = np.array([0.0, 0.0, 1.0])
        elev_range_orbit = (-0.4, 0.9)
        elev_range_inout = (-0.5, 0.7)

    if args.full_sphere:
        elev_range_orbit = (-1.0, 1.0)
        elev_range_inout = (-1.0, 1.0)

    rng = np.random.default_rng(args.seed)

    print(f"[gen_sparse_views] loading PLY: {args.ply}")
    xyz = load_xyz(args.ply, max_points=args.subsample_points, seed=args.seed)
    centroid, radius = robust_scene_geometry(xyz, args.robust_radius_pct)
    print(f"[gen_sparse_views] N={len(xyz)} sampled  centroid={centroid}  robust_radius(p{args.robust_radius_pct:.0f})={radius:.3f}")

    # Absolute orbit radius overrides (world units, bypass robust_radius scaling)
    if args.orbit_radius_abs_min is not None:
        args.orbit_radius_min = args.orbit_radius_abs_min / radius
    if args.orbit_radius_abs_max is not None:
        args.orbit_radius_max = args.orbit_radius_abs_max / radius

    fov_x = math.radians(args.fov_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / (args.width / args.height))

    # --- Renderer init (only when --render-check) ---
    rend_gaussians = _gs_render_fn = _cam_loader = rend_pipeline = None
    if args.render_check:
        import sys, os, torch
        _HERE = Path(__file__).resolve().parent
        _WS = _HERE.parent.parent
        _RROOT = _WS / "LapisGS-object-based-renderer"
        sys.path.insert(0, str(_RROOT))
        _DUMMY = _WS / "exp-dataset/chair/predictions/color/test/r_0.png"
        os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))
        from gaussian_renderer_lapisgs import GaussianModel, render as _gs_render_fn  # type: ignore
        from streaming_utils.camera_loader import load_camera_from_streaming_config as _cam_loader  # type: ignore
        import torchvision.utils as _tvu  # type: ignore

        class _Pipe:
            convert_SHs_python = compute_cov3D_python = debug = antialiasing = False
        rend_pipeline = _Pipe()

        ply_rend = args.ply_renderer or args.ply
        print(f"[render-check] loading GaussianModel from {ply_rend} ...")
        rend_gaussians = GaussianModel(args.sh_degree)
        rend_gaussians.load_ply(str(ply_rend))
        if args.gt_renders_dir:
            args.gt_renders_dir.mkdir(parents=True, exist_ok=True)
        _bg = [1, 1, 1] if args.white_bg else [0, 0, 0]
        bg_t = torch.tensor(_bg, dtype=torch.float32, device="cuda").view(3, 1, 1).expand(3, args.height, args.width).contiguous()
        bg_d = torch.zeros(1, args.height, args.width, device="cuda")
        gs_res = torch.ones(len(rend_gaussians.get_xyz), device="cuda")

    frames = []
    counts = {"inward": 0}
    rejects = {"low_visible": 0, "low_spread": 0}
    if args.render_check:
        rejects.update({"white_blob": 0, "black_frame": 0, "low_edge": 0})
    proposals = 0
    accepted = 0
    while accepted < args.n_views and proposals < args.max_proposals:
        proposals += 1
        use_inside = rng.random() < args.inside_out_frac
        if use_inside:
            eye, fwd = sample_inside_out(rng, centroid, radius, up_axis,
                                          (args.inside_radius_min, args.inside_radius_max),
                                          elev_range_inout)
            c2w = forward_to_c2w(eye, fwd, up_world)
            mode = "inward"
        else:
            eye, fwd = sample_close_orbit(rng, centroid, radius, up_axis,
                                           (args.orbit_radius_min, args.orbit_radius_max),
                                           elev_range_orbit,
                                           args.lookat_jitter_frac)
            c2w = forward_to_c2w(eye, fwd, up_world)
            mode = "inward"

        frac, sx, sy = evaluate_view(xyz, c2w, fov_x, fov_y)
        if frac < args.min_visible_frac:
            rejects["low_visible"] += 1
            continue
        if min(sx, sy) < args.min_ndc_spread:
            rejects["low_spread"] += 1
            continue

        # --- Render quality check ---
        if args.render_check:
            frame_tmp = {"file_path": "./test/r_tmp", "frame_index": 0,
                         "transform_matrix": c2w.tolist(), "camera_angle_x": fov_x}
            cam = _cam_loader(frame_tmp, width=args.width, height=args.height)
            cam.original_image = None
            with torch.no_grad():
                rendered = _gs_render_fn(cam, rend_gaussians, rend_pipeline,
                                         bg_t, bg_d, gs_res=gs_res)["render"].clamp(0, 1)
            ok, qstats = _render_quality(rendered, args.max_white_frac,
                                         args.max_black_frac, args.min_edge_var)
            if not ok:
                if qstats["white_frac"] > args.max_white_frac:
                    rejects["white_blob"] += 1
                elif qstats["black_frac"] > args.max_black_frac:
                    rejects["black_frame"] += 1
                else:
                    rejects["low_edge"] += 1
                continue
            if args.gt_renders_dir:
                gt_png = args.gt_renders_dir / f"camera_{accepted:03d}.png"
                _tvu.save_image(rendered.cpu(), str(gt_png))

        frames.append({
            "file_path": f"./test/r_{accepted}",
            "frame_index": accepted,
            "transform_matrix": c2w.tolist(),
            "pose_mode": mode,
        })
        counts[mode] += 1
        accepted += 1

    accept_rate = accepted / max(1, proposals)
    print(f"[gen_sparse_views] accepted {accepted}/{proposals}  rate={accept_rate:.1%}")
    print(f"[gen_sparse_views] modes: {counts}   rejects: {rejects}")
    if accepted < args.n_views:
        print(f"[gen_sparse_views] WARNING: only {accepted}/{args.n_views} accepted. "
              "Loosen --min-visible-frac / --min-ndc-spread or raise --max-proposals.")

    # --- Outward cameras: invert look direction, with sign-flip fallback ---
    if args.add_outward > 0 and frames:
        n_target = args.add_outward
        out_rejects = {"white_blob": 0, "black_frame": 0, "low_edge": 0}
        out_accepted = 0

        def _try_outward(eye: np.ndarray, fwd: np.ndarray) -> bool:
            nonlocal out_accepted
            c2w_out = forward_to_c2w(eye, fwd, up_world)
            if args.render_check:
                frame_tmp = {"file_path": "./test/r_tmp", "frame_index": 0,
                             "transform_matrix": c2w_out.tolist(), "camera_angle_x": fov_x}
                cam = _cam_loader(frame_tmp, width=args.width, height=args.height)
                cam.original_image = None
                with torch.no_grad():
                    rendered = _gs_render_fn(cam, rend_gaussians, rend_pipeline,
                                             bg_t, bg_d, gs_res=gs_res)["render"].clamp(0, 1)
                ok, qstats = _render_quality(rendered, args.max_white_frac,
                                             args.max_black_frac, args.min_edge_var)
                if not ok:
                    if qstats["white_frac"] > args.max_white_frac:
                        out_rejects["white_blob"] += 1
                    elif qstats["black_frac"] > args.max_black_frac:
                        out_rejects["black_frame"] += 1
                    else:
                        out_rejects["low_edge"] += 1
                    return False
                if args.gt_renders_dir:
                    gt_png = args.gt_renders_dir / f"camera_{accepted + out_accepted:03d}.png"
                    _tvu.save_image(rendered.cpu(), str(gt_png))
            frames.append({
                "file_path": f"./test/r_{accepted + out_accepted}",
                "frame_index": accepted + out_accepted,
                "transform_matrix": c2w_out.tolist(),
                "pose_mode": "outward",
            })
            out_accepted += 1
            return True

        # Pass 1: exact outward (eye - centroid) for ALL inward positions
        inward_snapshot = list(frames)
        for f in inward_snapshot:
            if out_accepted >= n_target:
                break
            eye = np.array(f["transform_matrix"])[:3, 3]
            _try_outward(eye, eye - centroid)

        # Pass 2: sign-flip fallback — flip each axis of the outward direction
        if out_accepted < n_target:
            for f in inward_snapshot:
                for axis in range(3):
                    if out_accepted >= n_target:
                        break
                    eye = np.array(f["transform_matrix"])[:3, 3]
                    fwd = eye - centroid
                    fwd[axis] *= -1
                    _try_outward(eye, fwd)
                if out_accepted >= n_target:
                    break

        n_tried = len(inward_snapshot)
        print(f"[gen_sparse_views] outward: {out_accepted}/{n_target} accepted  "
              f"tried={n_tried}+fallback  rejects: {out_rejects}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "camera_angle_x": fov_x,
        "frames": frames,
        "generation": {
            "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "script": str(Path(__file__).resolve()),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "scene": {
                "centroid": centroid.tolist(),
                "robust_radius": radius,
                "robust_radius_pct": args.robust_radius_pct,
                "subsample_n": int(len(xyz)),
            },
            "stats": {
                "proposals": proposals, "accepted": accepted,
                "accept_rate": accept_rate, "mode_counts": counts,
                "rejects": rejects, "image_size": [args.width, args.height],
                "fov_x_rad": fov_x, "fov_y_rad": fov_y,
            },
        },
    }, indent=2))
    print(f"[gen_sparse_views] wrote {args.out}")

    # --- QA: trace position scatter plot (alongside review video) ---
    if frames:
        _HERE2 = Path(__file__).resolve().parent
        _scene_tag = f"{args.out.parent.name}_{args.out.stem}"
        _qa_dir = _HERE2.parent / "output" / "camera_rendered" / _scene_tag
        _qa_dir.mkdir(parents=True, exist_ok=True)
        trace_out = _qa_dir / f"{_scene_tag}_trace.png"
        _plot_trace_positions(frames, centroid, radius, up_axis, trace_out)

    # --- QA: render-check spatial stats + review video ---
    if args.render_check and frames:
        positions = np.array([f["transform_matrix"] for f in frames])[:, :3, 3]
        print(f"[spatial] position std: {positions.std(axis=0).round(3)}")
        vecs = positions - centroid
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-8)
        elevs = np.degrees(np.arcsin((vecs / norms)[:, up_axis].clip(-1, 1)))
        print(f"[spatial] elevation deg: min={elevs.min():.1f} max={elevs.max():.1f} "
              f"mean={elevs.mean():.1f} std={elevs.std():.1f}")
        if len(positions) >= 2:
            diffs = positions[:, None, :] - positions[None, :, :]
            pairwise = np.sqrt((diffs ** 2).sum(axis=2))
            mean_pdist = pairwise[np.triu_indices(len(positions), k=1)].mean()
            print(f"[spatial] mean pairwise dist: {mean_pdist:.3f}")
        bins = np.linspace(-90, 90, 7)
        hist, _ = np.histogram(elevs, bins=bins)
        bin_labels = [f"[{bins[i]:.0f},{bins[i+1]:.0f})" for i in range(len(bins) - 1)]
        print(f"[spatial] elev histogram: { {b: int(h) for b, h in zip(bin_labels, hist)} }")
        unit_vecs = vecs / norms
        octant_signs = (unit_vecs > 0).astype(int)
        octants = set(map(tuple, octant_signs))
        hemi_fill = len(octants) / 8.0
        print(f"[spatial] hemisphere fill: {hemi_fill:.0%}  ({len(octants)}/8 octants)")

        if args.gt_renders_dir and accepted > 0:
            video_out = _qa_dir / "review.mp4"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-framerate", "5",
                    "-i", str(args.gt_renders_dir / "camera_%03d.png"),
                    "-vf", "scale=800:-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(video_out),
                ], check=True, capture_output=True)
                print(f"[review-video] {video_out}")
            except Exception as exc:
                print(f"[review-video] ffmpeg failed: {exc}")


if __name__ == "__main__":
    main()
