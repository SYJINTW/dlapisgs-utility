#!/usr/bin/env python3
"""Generate N sparse random camera views around a 3DGS PLY scene.

Filters out "void" views by checking that a min fraction of subsampled
Gaussian centers projects inside the camera frustum.

Output JSON matches Frustum-for-3DGS/camera/blender_camera.readCamerasFromTransforms:
    {
      "camera_angle_x": <fov_x in radians>,
      "frames": [
        {"frame_index": i, "transform_matrix": <4x4 c2w, Blender convention>},
        ...
      ]
    }

Blender convention: camera looks down -Z, +Y up, +X right (camera local frame).
readCamerasFromTransforms() then flips Y/Z to COLMAP for downstream use.
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


def load_xyz(ply_path: Path, max_points: int = 200_000, seed: int = 0) -> np.ndarray:
    plydata = PlyData.read(str(ply_path))
    el = plydata.elements[0]
    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=-1).astype(np.float64)
    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
    return xyz


def compute_aabb(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Return scene centroid and bounding-sphere radius."""
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = float(np.linalg.norm(hi - center))
    return center, radius


def sample_unit_vec(rng: np.random.Generator, up_axis: int, sin_lo: float, sin_hi: float) -> np.ndarray:
    """Sample a unit vector with the up-axis component (= sin(elevation)) in [sin_lo, sin_hi]."""
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
    """Build a 4x4 camera-to-world matrix (Blender convention: camera looks down -Z)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    # Blender: camera +Z points behind the camera (away from target)
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # parallel — fall back to an alternate up
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


def fraction_in_frustum(xyz: np.ndarray, c2w: np.ndarray, fov_x: float, fov_y: float,
                         znear: float = 0.01) -> tuple[float, float]:
    """Project points through (Blender c2w → COLMAP w2c) and count NDC-inside fraction.

    Returns (visible_fraction, mean_distance_to_visible_centroid).
    """
    # Blender → COLMAP axis flip on Y, Z columns (matches readCamerasFromTransforms)
    c2w_colmap = c2w.copy()
    c2w_colmap[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w_colmap)

    pts_h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=-1)  # [N,4]
    view = pts_h @ w2c.T  # [N,4]  (now in camera frame, +Z forward)
    z = view[:, 2]
    in_front = z > znear
    if not in_front.any():
        return 0.0, float("inf")

    tan_hx = math.tan(fov_x / 2.0)
    tan_hy = math.tan(fov_y / 2.0)
    x_ndc = view[:, 0] / (z * tan_hx)
    y_ndc = view[:, 1] / (z * tan_hy)
    inside = in_front & (np.abs(x_ndc) <= 1.0) & (np.abs(y_ndc) <= 1.0)
    visible_count = int(inside.sum())
    frac = visible_count / len(xyz)
    mean_dist = float(np.linalg.norm(view[inside, :3], axis=-1).mean()) if visible_count > 0 else float("inf")
    return frac, mean_dist


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-views", type=int, default=100)
    p.add_argument("--fov-deg", type=float, default=50.0)
    p.add_argument("--width",  type=int, default=800)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--scene-type", choices=("synthetic", "mipnerf360"), default="synthetic",
                   help="synthetic: +Y up, upper hemisphere only. mipnerf360: +Z up, allow all elevations.")
    p.add_argument("--min-visible-frac", type=float, default=0.05,
                   help="Minimum subsampled-Gaussian visibility for an accepted view.")
    p.add_argument("--max-dist-mult", type=float, default=5.0,
                   help="Reject views whose mean-visible-distance > max_dist_mult * scene_radius.")
    p.add_argument("--radius-min-mult", type=float, default=1.5)
    p.add_argument("--radius-max-mult", type=float, default=3.0)
    p.add_argument("--lookat-jitter-frac", type=float, default=0.2,
                   help="Random look-at offset, in units of scene_radius.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-proposals", type=int, default=10_000)
    p.add_argument("--subsample-points", type=int, default=50_000)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[gen_sparse_views] loading PLY: {args.ply}")
    xyz = load_xyz(args.ply, max_points=args.subsample_points, seed=args.seed)
    center, radius = compute_aabb(xyz)
    print(f"[gen_sparse_views] N={len(xyz)} subsampled  centroid={center}  radius={radius:.3f}")

    if args.scene_type == "synthetic":
        up_axis = 1                         # +Y up
        up_world = np.array([0.0, 1.0, 0.0])
        elev_range = (0.05, 0.95)           # upper hemisphere
    else:  # mipnerf360
        up_axis = 2                         # +Z up
        up_world = np.array([0.0, 0.0, 1.0])
        elev_range = (-0.6, 0.95)           # mostly above ground

    fov_x = math.radians(args.fov_deg)
    aspect = args.width / args.height
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)

    frames = []
    proposals = 0
    accepted = 0
    while accepted < args.n_views and proposals < args.max_proposals:
        proposals += 1
        r = rng.uniform(args.radius_min_mult, args.radius_max_mult) * radius
        direction = sample_unit_vec(rng, up_axis, elev_range[0], elev_range[1])
        eye = center + r * direction

        jitter = rng.normal(scale=args.lookat_jitter_frac * radius, size=3)
        target = center + jitter

        c2w = look_at_c2w(eye, target, up_world)
        frac, mean_dist = fraction_in_frustum(xyz, c2w, fov_x, fov_y)
        if frac < args.min_visible_frac:
            continue
        if mean_dist > args.max_dist_mult * radius:
            continue
        frames.append({
            "file_path": f"./test/r_{accepted}",
            "frame_index": accepted,
            "transform_matrix": c2w.tolist(),
        })
        accepted += 1

    accept_rate = accepted / max(1, proposals)
    print(f"[gen_sparse_views] accepted {accepted}/{proposals}  rate={accept_rate:.1%}")
    if accepted < args.n_views:
        print(f"[gen_sparse_views] WARNING: only got {accepted} of {args.n_views} requested views. "
              f"Loosen --min-visible-frac or increase --max-proposals.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "camera_angle_x": fov_x,
        "frames": frames,
        "generation": {
            "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "script": str(Path(__file__).resolve()),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "scene_aabb": {
                "centroid": center.tolist(),
                "radius": radius,
                "subsample_n": int(len(xyz)),
            },
            "stats": {
                "proposals": proposals,
                "accepted": accepted,
                "accept_rate": accept_rate,
                "image_size": [args.width, args.height],
                "fov_x_rad": fov_x,
                "fov_y_rad": fov_y,
            },
        },
    }, indent=2))
    print(f"[gen_sparse_views] wrote {args.out}")


if __name__ == "__main__":
    main()
