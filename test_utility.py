"""
Offline utility runner.

Steps:
1) Load a full PLY scene.
2) Tile it using GGSP tiling.
3) Compute view-conditioned utilities per scheme.
4) Select Gaussians under a byte budget and export a subset PLY.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GGSP"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))

import visibility_AABB_pytorch  # noqa: E402
import tiling as ggsp_tiling  # noqa: E402
import utility_calculation as uc  # noqa: E402
import io_3dgs  # noqa: E402


def _build_tile_arrays(tile_aabbs, tile_indices, layer_idx=0):
    sorted_tile_keys = sorted(tile_aabbs.keys())
    min_corners_list = []
    max_corners_list = []
    flat_indices_list = []
    index_offsets = [0]

    for tile_key in sorted_tile_keys:
        aabb = tile_aabbs[tile_key]
        gs_idx = tile_indices[tile_key][layer_idx]
        min_corners_list.append(aabb["min_corner"])
        max_corners_list.append(aabb["max_corner"])
        flat_indices_list.append(gs_idx.astype(np.int64))
        index_offsets.append(index_offsets[-1] + len(gs_idx))

    min_corners = np.asarray(min_corners_list, dtype=np.float32)
    max_corners = np.asarray(max_corners_list, dtype=np.float32)
    index_offsets = np.asarray(index_offsets, dtype=np.int64)
    flat_indices = np.concatenate(flat_indices_list).astype(np.int64) if flat_indices_list else np.empty((0,), dtype=np.int64)
    return min_corners, max_corners, index_offsets, flat_indices, sorted_tile_keys


def _bytes_per_gaussian(gs):
    per_attr = []
    for key, value in gs.data.items():
        per_attr.append(np.dtype(value["val_dtype"]).itemsize)
    return int(np.sum(per_attr))


def _select_gaussians(order_pairs, tile_index_offsets, tile_flat_indices, w_gi, bytes_per_gaussian, budget_bytes):
    selected_indices = []
    used_bytes = 0

    for tile_idx, _lod in order_pairs:
        start = tile_index_offsets[tile_idx]
        end = tile_index_offsets[tile_idx + 1]
        indices_for_tile = tile_flat_indices[start:end]
        if len(indices_for_tile) == 0:
            continue

        tile_weights = w_gi[indices_for_tile]
        sorted_tile = indices_for_tile[torch.argsort(tile_weights, descending=True).cpu().numpy()]

        for idx in sorted_tile:
            if used_bytes + bytes_per_gaussian > budget_bytes:
                return selected_indices, used_bytes
            selected_indices.append(int(idx))
            used_bytes += bytes_per_gaussian

    return selected_indices, used_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline 3DGS utility selection")
    parser.add_argument("--ply", required=True, type=str,
                        help="Input full-scene PLY (original 3DGS model)")
    parser.add_argument("--output", required=True, type=str,
                        help="Output PLY path for selected Gaussians")
    parser.add_argument("--camera-trace", required=True, type=str,
                        help="Camera trace JSON (Frustum-for-3DGS format)")
    parser.add_argument("--grid-shape", nargs=3, type=int, default=[4, 4, 4],
                        help="Tiling grid as Nx Ny Nz (e.g., 4 4 4)")
    parser.add_argument("--budget-mb", type=float, default=100.0,
                        help="Byte budget in MB for the output subset")
    parser.add_argument("--num-lod", type=int, default=1,
                        help="Number of LOD layers (1 means plain 3DGS)")
    parser.add_argument("--scheme", type=str, default="vd",
                        choices=["vd", "vd_lod", "vd_lod_w", "vd_lod_w_c"],
                        help="Utility scheme: vd | vd_lod | vd_lod_w | vd_lod_w_c")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Exponent for Gaussian weight: w = o * det(Sigma)^gamma")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Index of the camera in the trace to evaluate")
    parser.add_argument("--img-w", type=int, default=800,
                        help="Camera image width for visibility checks")
    parser.add_argument("--img-h", type=int, default=800,
                        help="Camera image height for visibility checks")
    parser.add_argument("--ascii-ply", action="store_true",
                        help="Write PLY in ASCII instead of binary (larger, human-readable)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ply_path = Path(args.ply)
    output_path = Path(args.output)
    camera_trace = Path(args.camera_trace)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if not camera_trace.exists():
        raise FileNotFoundError(camera_trace)

    log_path = output_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="INFO")
    logger.info("device={}", device)
    logger.info("ply={} output={} trace={}", ply_path, output_path, camera_trace)
    logger.info("grid_shape={} budget_mb={} num_lod={} scheme={}",
                args.grid_shape, args.budget_mb, args.num_lod, args.scheme)

    gs = io_3dgs.GaussianModelV2(str(ply_path))
    tile_aabbs, tile_indices, scene_min, scene_max = ggsp_tiling.tiling_uniform_layered_gs([gs], grid_shape=tuple(args.grid_shape))

    min_corners, max_corners, index_offsets, flat_indices, sorted_tile_keys = _build_tile_arrays(
        tile_aabbs, tile_indices, layer_idx=0
    )

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers = (min_corners_t + max_corners_t) / 2.0

    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
        str(camera_trace), args.img_w, args.img_h
    )
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    if args.camera_index < 0 or args.camera_index >= len(cameras):
        raise ValueError("camera-index out of range")
    cam = cameras[args.camera_index]
    logger.info("cameras={} using_index={}", len(cameras), args.camera_index)

    distances = uc.calculate_distances(tile_centers, cam.camera_center.to(device))
    visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
        min_corners_t, max_corners_t, cam, device=device
    )

    # Save visibility and camera info for headless visualization
    try:
        vis_npz_path = output_path.with_suffix('.vis.npz')
        # Ensure numpy arrays on CPU
        np_min = min_corners
        np_max = max_corners
        np_visibility_all = visibility.cpu().numpy() if hasattr(visibility, 'cpu') else np.asarray(visibility)
        np_distances = distances.cpu().numpy() if hasattr(distances, 'cpu') else np.asarray(distances)
        # Camera matrices (world->view and projection) and camera center
        cam_w2v = cam.world_view_transform.cpu().numpy()
        cam_proj = cam.projection_matrix.cpu().numpy()
        cam_center = cam.camera_center.cpu().numpy()

        # Align visibility to the metadata ordering used by save_tiles_to_npz()
        # save_tiles_to_npz only writes non-empty tiles (it filters empty tiles out).
        # Build positions of non-empty tiles in the sorted_tile_keys order.
        meta_positions = [i for i, tk in enumerate(sorted_tile_keys) if len(tile_indices[tk][0]) > 0]
        visibility_meta = np_visibility_all[meta_positions] if len(meta_positions) > 0 else np.zeros((0,), dtype=bool)

        np.savez(str(vis_npz_path),
                 min_corners=np_min,
                 max_corners=np_max,
                 visibility_all=np_visibility_all,
                 visibility=visibility_meta,
                 distances=np_distances,
                 tile_centers=tile_centers.cpu().numpy(),
                 camera_center=cam_center,
                 world_view_transform=cam_w2v,
                 projection_matrix=cam_proj)
        logger.info("saved visibility metadata to={}", vis_npz_path)
    except Exception as e:
        logger.warning("Failed saving visibility NPZ: {}", e)

    opacity = gs.data["opacity"]["data"]
    scale_0 = gs.data["scale_0"]["data"]
    scale_1 = gs.data["scale_1"]["data"]
    scale_2 = gs.data["scale_2"]["data"]
    w_gi = uc.compute_gaussian_weights(opacity, scale_0, scale_1, scale_2, gamma=args.gamma).to(device)

    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
    W_k, C_k = uc.compute_tile_weights_and_counts(tile_index_offsets, tile_flat_indices, w_gi)

    include_lod = args.scheme != "vd"
    include_w = args.scheme in ("vd_lod_w", "vd_lod_w_c")
    include_c = args.scheme == "vd_lod_w_c"

    utilities = uc.calculate_utility_param(
        visibility,
        distances,
        num_of_level=args.num_lod,
        weight_sum_tensor=W_k if include_w else None,
        complexity_tensor=C_k if include_c else None,
        include_lod=include_lod,
        include_w=include_w,
        include_c=include_c,
    )

    top_k = min(20, utilities.shape[0])
    logger.info("top_utility_pairs (tile_idx, lod): {}", utilities[:top_k].tolist())

    budget_bytes = int(args.budget_mb * 1024 * 1024)
    bytes_per_gaussian = _bytes_per_gaussian(gs)
    selected_indices, used_bytes = _select_gaussians(
        utilities,
        tile_index_offsets,
        tile_flat_indices,
        w_gi,
        bytes_per_gaussian,
        budget_bytes,
    )

    selected_gs = gs.extract_gaussians(selected_indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_gs.export_gs_to_ply(str(output_path), ascii=args.ascii_ply)

    # Save tiling metadata for visualization/debug
    npz_output_path = output_path.with_suffix(".npz")
    ggsp_tiling.save_tiles_to_npz(
        tile_aabbs,
        tile_indices,
        str(npz_output_path),
        grid_shape=tuple(args.grid_shape),
        scene_min=scene_min,
        scene_max=scene_max,
        layer_idx=0
    )

    logger.info("tiles={}", len(index_offsets) - 1)
    logger.info("selected_gaussians={}", len(selected_indices))
    logger.info("approx_bytes_used={}/{}", used_bytes, budget_bytes)
    logger.info("output={}", output_path)
    logger.info("tiling_metadata_npz={}", npz_output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("utility run failed")
        raise
