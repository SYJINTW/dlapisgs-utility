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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-proposals", type=int, default=20_000)
    p.add_argument("--subsample-points", type=int, default=50_000)
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
        elev_range_orbit = (0.05, 0.95)     # upper hemisphere
        elev_range_inout = (-0.7, 0.9)      # mostly upper, allow some look-down
    else:  # mipnerf360
        up_axis = 2
        up_world = np.array([0.0, 0.0, 1.0])
        elev_range_orbit = (-0.4, 0.9)
        elev_range_inout = (-0.5, 0.7)

    rng = np.random.default_rng(args.seed)

    print(f"[gen_sparse_views] loading PLY: {args.ply}")
    xyz = load_xyz(args.ply, max_points=args.subsample_points, seed=args.seed)
    centroid, radius = robust_scene_geometry(xyz, args.robust_radius_pct)
    print(f"[gen_sparse_views] N={len(xyz)} sampled  centroid={centroid}  robust_radius(p{args.robust_radius_pct:.0f})={radius:.3f}")

    fov_x = math.radians(args.fov_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / (args.width / args.height))

    frames = []
    counts = {"inside_out": 0, "close_orbit": 0}
    rejects = {"low_visible": 0, "low_spread": 0}
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
            mode = "inside_out"
        else:
            eye, fwd = sample_close_orbit(rng, centroid, radius, up_axis,
                                           (args.orbit_radius_min, args.orbit_radius_max),
                                           elev_range_orbit,
                                           args.lookat_jitter_frac)
            c2w = forward_to_c2w(eye, fwd, up_world)
            mode = "close_orbit"

        frac, sx, sy = evaluate_view(xyz, c2w, fov_x, fov_y)
        if frac < args.min_visible_frac:
            rejects["low_visible"] += 1
            continue
        if min(sx, sy) < args.min_ndc_spread:
            rejects["low_spread"] += 1
            continue

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


if __name__ == "__main__":
    main()
