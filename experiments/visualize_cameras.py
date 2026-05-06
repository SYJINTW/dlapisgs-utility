#!/usr/bin/env python3
"""
Visualize camera positions and orientations from a NeRF-style camera trace JSON.
Saves a headless PNG showing camera centers and frustums/arrows for orientation.

Usage:
    python visualize_cameras.py --trace /path/to/trace1.json --out cameras.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# reuse the reader from Frustum-for-3DGS
import sys
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'Frustum-for-3DGS'))
from camera.blender_camera import readCamerasFromTransforms


def draw_camera(ax, center, R, scale=5.0, color='blue'):
    """Draw camera axes at center. R should be a 3x3 rotation (camera->world)."""
    # basis in camera coordinates: x-right, y-up, z-forward
    right = R @ np.array([1.0, 0.0, 0.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    forward = R @ np.array([0.0, 0.0, 1.0])

    ax.quiver(center[0], center[1], center[2],
              right[0], right[1], right[2], length=scale, color=color, linewidth=1.0)
    ax.quiver(center[0], center[1], center[2],
              up[0], up[1], up[2], length=scale, color=color, linewidth=1.0)
    ax.quiver(center[0], center[1], center[2],
              forward[0], forward[1], forward[2], length=scale, color='red', linewidth=1.0)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True, help='Camera trace JSON (NeRF/Blender format)')
    p.add_argument('--out', required=True, help='Output PNG path')
    p.add_argument('--scale', type=float, default=5.0, help='Arrow scale for orientation')
    p.add_argument('--skip', type=int, default=1, help='Skip frames to reduce clutter (1 = all)')
    args = p.parse_args()

    cams = readCamerasFromTransforms(args.trace, 800, 800)

    centers = []
    Rs = []
    for i, ci in enumerate(cams):
        # ci.R as returned is a rotation matrix; compute camera center in world coords
        # In readCamerasFromTransforms, R = transpose(w2c[:3,:3]), T = w2c[:3,3]
        # camera_center = - R @ T
        # Note: this yields a 3-vector
        # Safe compute using ci.R and ci.T
        T = ci.T
        R = ci.R
        center = - R @ T
        centers.append(center)
        # For drawing axes, we use R (camera->world)
        Rs.append(R)

    centers = np.asarray(centers)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera centers
    ax.scatter(centers[:,0], centers[:,1], centers[:,2], c='blue', s=20)

    # Draw orientation arrows (skip to avoid clutter)
    # for i in range(0, len(centers), args.skip):
    #     draw_camera(ax, centers[i], Rs[i], scale=args.scale, color='blue')
    #     ax.text(centers[i][0], centers[i][1], centers[i][2], str(i), color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera positions ({len(centers)} views)')
    set_axes_equal(ax)
    plt.tight_layout()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outp), dpi=200)
    print('Saved camera visualization to', outp)


if __name__ == '__main__':
    main()
