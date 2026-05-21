"""Build per-(camera, tile) feature matrix from oracle_dq.npz."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent.parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))

import visibility_AABB_pytorch  # noqa: E402
import io_3dgs  # noqa: E402
import utility_calculation as uc  # noqa: E402


def build_feature_matrix(oracle_npz_path: str) -> pd.DataFrame:
    """Build long-form feature DataFrame from oracle_dq.npz.

    Returns DataFrame with columns: camera, tile, lod, v_k, d_k, C_k, W_k.
    One row per (camera, tile) pair for all cameras × all tiles.
    PLY path, trace path, and image dimensions are read from gen_meta in the npz.
    """
    oracle_npz_path = Path(oracle_npz_path)
    npz = np.load(str(oracle_npz_path), allow_pickle=True)

    min_corners = npz["min_corners"]        # (N_tiles, 3)
    max_corners = npz["max_corners"]        # (N_tiles, 3)
    index_offsets = npz["index_offsets"]    # (N_tiles+1,)
    flat_indices = npz["flat_indices"]      # (N_total_gs,)
    n_gs_per_tile = npz["n_gs_per_tile"].astype(np.float32)  # (N_tiles,)
    camera_indices = npz["camera_indices"]  # (N_cams,)

    meta = json.loads(str(npz["gen_meta"].item()))
    ply_path = meta["scene_ply"]
    trace_path = meta["trace"]
    img_w = int(meta["width"])
    img_h = int(meta["height"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gs = io_3dgs.GaussianModelV2(ply_path)
    opacity = gs.data["opacity"]["data"]
    scale_0 = gs.data["scale_0"]["data"]
    scale_1 = gs.data["scale_1"]["data"]
    scale_2 = gs.data["scale_2"]["data"]
    rot_0 = gs.data["rot_0"]["data"]
    rot_1 = gs.data["rot_1"]["data"]
    rot_2 = gs.data["rot_2"]["data"]
    rot_3 = gs.data["rot_3"]["data"]
    gs_xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"], gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers = (min_corners_t + max_corners_t) / 2.0
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)

    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(trace_path, img_w, img_h)
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)

    N_tiles = len(n_gs_per_tile)
    tile_col = np.arange(N_tiles)

    rows = []
    for i, cam_idx in enumerate(camera_indices):
        cam = cameras[i]
        cam_center = cam.camera_center.to(device)

        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device
        )
        distances = uc.calculate_distances(tile_centers, cam_center)

        w_gi = uc.compute_gaussian_weights_v2(
            "screen_area",
            opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
            xyz=gs_xyz_t, cam_center=cam_center,
            rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
            world_view=cam.world_view_transform, proj=cam.projection_matrix,
            img_w=img_w, img_h=img_h,
            fov_x=getattr(cam, "FoVx", None), fov_y=getattr(cam, "FoVy", None),
        ).to(device)

        W_k, _ = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_gi,
            w_norm="sum", c_norm="sum",
        )

        rows.append(pd.DataFrame({
            "camera": int(cam_idx),
            "tile": tile_col,
            "lod": np.int8(0),
            "v_k": visibility.float().cpu().numpy(),
            "d_k": distances.cpu().numpy(),
            "C_k": n_gs_per_tile,
            "W_k": W_k.cpu().numpy(),
        }))

    return pd.concat(rows, ignore_index=True)
