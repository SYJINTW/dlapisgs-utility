#!/usr/bin/env python3
"""Camera/pose module for the EyeNavGS streaming-sim harness.

Builds a renderable camera directly from an EyeNavGS trace row (position +
quaternion + asymmetric per-eye FOV), without touching the read-only
Frustum-for-3DGS repo. Ports NTHU's own Blender reference-replay pose
composition (dataset/EyeNavGS_NTHU_Dataset/_code/CAM.py:14-27) into
numpy/scipy, then reuses this codebase's existing Blender->COLMAP axis
convention (Frustum-for-3DGS/camera/blender_camera.py:29-36) to build R/T.

Risk (see .claude/PLAN.md streaming-sim entry): CAM.py's own replay script
composes orig_quat @ data -- this reopens the spec's earlier "no correction
needed" conclusion. Followed here, not the spec. Verify with the mandatory
sanity render before trusting the full sweep.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
_FRUSTUM_ROOT = str(WORKSPACE / "Frustum-for-3DGS")
if _FRUSTUM_ROOT not in sys.path:
    sys.path.insert(0, _FRUSTUM_ROOT)

from utils.graphics_utils import getWorld2View2  # noqa: E402


def get_projection_matrix_asymmetric(znear, zfar, left, right, top, bottom) -> torch.Tensor:
    """Same math as Frustum-for-3DGS/utils/graphics_utils.py::getProjectionMatrix
    (graphics_utils.py:51-71), generalized to explicit frustum bounds instead of
    deriving symmetric ones from a single fovX/fovY."""
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def compose_pose(scene_quat_xyzw, data_pos_xyz, data_quat_xyzw):
    """Port of CAM.py:26-27 (final_quat = orig_quat @ data_quat, final_pos = orig_quat
    @ data_pos) using scipy in place of Blender's mathutils. Returns (position (3,),
    rotation_matrix (3,3)) still in Blender/OpenGL convention (same as CAM.py builds)."""
    orig = Rotation.from_quat(scene_quat_xyzw)
    data = Rotation.from_quat(data_quat_xyzw)
    final_rot = orig * data
    final_pos = orig.apply(np.asarray(data_pos_xyz, dtype=np.float64))
    return final_pos, final_rot.as_matrix()


def blender_pose_to_RT(position, rotation_matrix):
    """camera-to-world -> COLMAP R,T (invert + transpose, glm/CUDA convention).

    Deliberately does NOT apply the Blender/OpenGL->COLMAP axis flip
    (`c2w[:3,1:3] *= -1`) that Frustum-for-3DGS/camera/blender_camera.py:29-36 applies
    to NeRF-transforms-format c2w matrices. Empirically determined this session (sanity
    render): applying that flip to CAM.py-composed EyeNavGS poses produces a consistent
    ~90 degree roll (confirmed via MiniCam + symmetric FOV isolation, independent of the
    asymmetric-frustum code); omitting it renders an upright, correctly-framed scene
    (grass at bottom, sky/trees/house at top). I.e. CAM.py's composed rotation matrix is
    already in the axis convention this codebase's R/T expects, unlike a raw NeRF
    transforms.json c2w -- the two pose sources are NOT interchangeable through the same
    conversion path. Function name kept for continuity with the (very similar) sibling
    convention despite this difference.
    """
    c2w = np.eye(4)
    c2w[:3, :3] = rotation_matrix
    c2w[:3, 3] = position
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    return R, T


class StreamingCamera:
    """MiniCam-equivalent built from an asymmetric frustum. Exposes exactly the
    attribute set selection_core.render_gs / compute_camera_weights /
    visibility_AABB_pytorch.batched_check_tiles_visible read: uid, image_width,
    image_height, R, T, FoVx, FoVy, world_view_transform, projection_matrix,
    full_proj_transform, camera_center."""

    def __init__(self, uid, width, height, R, T, left, right, top, bottom,
                 znear=0.01, zfar=100.0, device="cuda"):
        self.uid = uid
        self.image_width = width
        self.image_height = height
        self.R = R
        self.T = T
        self.znear = znear
        self.zfar = zfar
        self.left, self.right, self.top, self.bottom = left, right, top, bottom

        # Real asymmetric bounds drive projection_matrix (culling + rendering geometry).
        # FoVx/FoVy are an AVERAGED symmetric equivalent used only for the rasterizer's
        # internal tanfovx/tanfovy scalar (gaussian_renderer_lapisgs/__init__.py:39-40) --
        # an approximation whose error grows with frustum asymmetry. Flagged, not silent.
        self.FoVx = 2.0 * math.atan((right - left) / (2.0 * znear))
        self.FoVy = 2.0 * math.atan((top - bottom) / (2.0 * znear))

        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(device)
        self.projection_matrix = get_projection_matrix_asymmetric(
            znear, zfar, left, right, top, bottom).transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def build_streaming_camera(trace_row, scene_quat_xyzw, width, height, uid=0,
                            znear=0.01, zfar=100.0, device="cuda",
                            swap_top_bottom=False) -> StreamingCamera:
    """Single entry point: EyeNavGS trace row (dict-like, string-or-float values) ->
    StreamingCamera. swap_top_bottom: escape hatch for Risk 2 (FOV3/FOV4 sign
    convention doesn't trivially match naive top>0>bottom -- flip if the sanity
    render comes out vertically inverted)."""
    data_quat_xyzw = (float(trace_row["QuaternionX"]), float(trace_row["QuaternionY"]),
                       float(trace_row["QuaternionZ"]), float(trace_row["QuaternionW"]))
    data_pos = (float(trace_row["PositionX"]), float(trace_row["PositionY"]),
                float(trace_row["PositionZ"]))
    position, rot_mat = compose_pose(scene_quat_xyzw, data_pos, data_quat_xyzw)
    R, T = blender_pose_to_RT(position, rot_mat)

    fov_left, fov_right = float(trace_row["FOV1"]), float(trace_row["FOV2"])
    fov_top, fov_bottom = float(trace_row["FOV3"]), float(trace_row["FOV4"])
    if swap_top_bottom:
        fov_top, fov_bottom = fov_bottom, fov_top

    left = math.tan(fov_left) * znear
    right = math.tan(fov_right) * znear
    top = math.tan(fov_top) * znear
    bottom = math.tan(fov_bottom) * znear

    return StreamingCamera(uid, width, height, R, T, left, right, top, bottom,
                            znear=znear, zfar=zfar, device=device)


def frustum_aspect(trace_row, swap_top_bottom=False) -> float:
    """(right-left)/(top-bottom) at znear -- independent of znear's actual value.
    Used once at startup to pick a non-distorting image_height from image_width."""
    fov_left, fov_right = float(trace_row["FOV1"]), float(trace_row["FOV2"])
    fov_top, fov_bottom = float(trace_row["FOV3"]), float(trace_row["FOV4"])
    if swap_top_bottom:
        fov_top, fov_bottom = fov_bottom, fov_top
    width_ratio = math.tan(fov_right) - math.tan(fov_left)
    height_ratio = math.tan(fov_top) - math.tan(fov_bottom)
    return width_ratio / height_ratio
