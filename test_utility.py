"""
Offline utility runner.

Steps:
1) Load a full PLY scene.
2) Tile it using GGSP tiling.
3) Compute view-conditioned utilities per scheme.
4) Select Gaussians under a byte budget and export a subset PLY.

Optimizations:
- PLY exports run in a thread pool (overlaps GPU compute with disk writes).
- tiling.npz written once per run.
- vis.npz written once per camera to camera_viz/{NNN}.npz.
- w_gi recomputed per camera using view-dependent 1/d^2 weighting.
- Prefix-slice greedy: run once at max budget, slice for smaller ones.

Output layout (--output-root mode):
  output_root/
  ├── tiling.npz
  ├── utility.log
  ├── camera_viz/{NNN}.npz
  └── ply/budget_{B}mb/{scheme}/camera_{NNN}.ply + .json


[TODO]
- test utility scoring scheme
- test scheduler without GS-level or without Tile-level vs. Ours Full
(pure progressive v. pure adaptive v. our two-level design)


"""


import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import io_3dgs  # noqa: E402  # pyright: ignore[reportMissingImports]


VALID_SCHEMES = ["vd", "vd_lod", "vd_lod_w", "vd_lod_c", "vd_lod_w_c"]
PLY_WORKERS = 4


def _build_tile_arrays(tile_aabbs, tile_indices, layer_idx=0):
    sorted_tile_keys = sorted(tile_aabbs.keys())
    min_corners_list, max_corners_list, flat_indices_list = [], [], []
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
    return int(np.sum([np.dtype(v["val_dtype"]).itemsize for v in gs.data.values()]))


def _budget_tag(budget_mb: float) -> str:
    return f"{budget_mb:g}".replace(".", "p") + "mb"


def _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices, w_gi, bytes_per_gaussian, max_budget_bytes):
    """Returns Gaussian indices as a numpy array in greedy priority order up to max_budget_bytes."""
    max_count = max_budget_bytes // bytes_per_gaussian
    chunks = []
    count = 0
    for tile_idx, _lod in order_pairs:
        if count >= max_count:
            break
        start = tile_index_offsets[tile_idx]
        end = tile_index_offsets[tile_idx + 1]
        indices_for_tile = tile_flat_indices[start:end]
        if len(indices_for_tile) == 0:
            continue
        tile_weights = w_gi[indices_for_tile]
        sorted_tile = indices_for_tile[torch.argsort(tile_weights, descending=True)].cpu().numpy()
        take = min(len(sorted_tile), max_count - count)
        chunks.append(sorted_tile[:take])
        count += take
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _select_at_budget(all_ordered, budget_bytes, bytes_per_gaussian):
    count = budget_bytes // bytes_per_gaussian
    selected = all_ordered[:count]
    return selected, len(selected) * bytes_per_gaussian


def _write_ply(gs, selected_indices, output_path, ascii_ply):
    selected_gs = gs.extract_gaussians(selected_indices)
    selected_gs.export_gs_to_ply(str(output_path), ascii=ascii_ply)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline 3DGS utility selection")
    parser.add_argument("--ply", required=True, type=str,
                        help="Input full-scene PLY (original 3DGS model)")
    parser.add_argument("--output", type=str, default=None,
                        help="Legacy single-output PLY path (single budget/scheme only)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Directory root for trace-wide outputs")
    parser.add_argument("--camera-trace", required=True, type=str,
                        help="Camera trace JSON (Frustum-for-3DGS format)")
    parser.add_argument("--grid-shape", nargs=3, type=int, default=[4, 4, 4],
                        help="Number of tiles along each axis as Nx Ny Nz (e.g., 4 4 4)")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="Single byte budget in MB (legacy; use --budgets-mb for sweeps)")
    parser.add_argument("--budgets-mb", nargs="+", type=float, default=None,
                        help="One or more budgets in MB; overrides --budget-mb")
    parser.add_argument("--num-lod", type=int, default=1,
                        help="Number of LOD layers (1 means plain 3DGS)")
    parser.add_argument("--scheme", type=str, default=None, choices=VALID_SCHEMES,
                        help="Single utility scheme (legacy; use --schemes for sweeps)")
    parser.add_argument("--schemes", nargs="+", type=str, default=None,
                        help="One or more schemes; overrides --scheme")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Exponent for Gaussian weight: w = sigmoid(o) * det(Sigma)^gamma / d^2")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Index of the camera in the trace to evaluate; use -1 to process all cameras")
    parser.add_argument("--img-w", type=int, default=800,
                        help="Camera image width for visibility checks")
    parser.add_argument("--img-h", type=int, default=800,
                        help="Camera image height for visibility checks")
    parser.add_argument("--ascii-ply", action="store_true",
                        help="Write PLY in ASCII instead of binary (larger, human-readable)")
    parser.add_argument("--ply-workers", type=int, default=PLY_WORKERS,
                        help="Thread pool size for concurrent PLY writes")
    args = parser.parse_args()

    if args.budgets_mb is not None:
        budget_list = sorted(args.budgets_mb)
    elif args.budget_mb is not None:
        budget_list = [args.budget_mb]
    else:
        raise ValueError("Either --budgets-mb or --budget-mb must be provided")

    if args.schemes is not None:
        for s in args.schemes:
            if s not in VALID_SCHEMES:
                raise ValueError(f"Unknown scheme '{s}'. Valid: {VALID_SCHEMES}")
        scheme_list = args.schemes
    elif args.scheme is not None:
        scheme_list = [args.scheme]
    else:
        raise ValueError("Either --schemes or --scheme must be provided")

    if args.output_root is None and args.output is None:
        raise ValueError("Either --output-root or --output must be provided")

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ply_path = Path(args.ply)
    camera_trace = Path(args.camera_trace)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if not camera_trace.exists():
        raise FileNotFoundError(camera_trace)

    base_output_path = Path(args.output_root) if args.output_root is not None else Path(args.output)
    log_path = base_output_path.with_suffix(".log") if base_output_path.suffix else base_output_path / "utility.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="INFO")
    logger.info("device={} ply_workers={}", device, args.ply_workers)
    logger.info("ply={} output={} trace={}", ply_path, base_output_path, camera_trace)
    logger.info("grid_shape={} budgets_mb={} num_lod={} schemes={}",
                args.grid_shape, budget_list, args.num_lod, scheme_list)

    # ---- Shared preprocessing (once per run) ----
    logger.success("stage=ply_load start")
    _t0 = time.perf_counter()
    gs = io_3dgs.GaussianModelV2(str(ply_path))
    logger.success("stage=ply_load done t={:.3f}s", time.perf_counter() - _t0)

    logger.success("stage=tiling start")
    _t0 = time.perf_counter()
    tile_aabbs, tile_indices, scene_min, scene_max = ggsp_tiling.tiling_uniform_layered_gs([gs], grid_shape=tuple(args.grid_shape))
    min_corners, max_corners, index_offsets, flat_indices, sorted_tile_keys = _build_tile_arrays(
        tile_aabbs, tile_indices, layer_idx=0
    )
    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers = (min_corners_t + max_corners_t) / 2.0
    logger.success("stage=tiling done t={:.3f}s", time.perf_counter() - _t0)

    _t0 = time.perf_counter()
    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(str(camera_trace), args.img_w, args.img_h)
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    logger.success("stage=camera_load done t={:.3f}s", time.perf_counter() - _t0)

    _t0 = time.perf_counter()
    opacity = gs.data["opacity"]["data"]
    scale_0 = gs.data["scale_0"]["data"]
    scale_1 = gs.data["scale_1"]["data"]
    scale_2 = gs.data["scale_2"]["data"]
    gs_xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"], gs.data["z"]["data"]], axis=1)
    gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
    logger.success("stage=gs_attrs loaded t={:.3f}s", time.perf_counter() - _t0)

    bytes_per_gaussian = _bytes_per_gaussian(gs)
    max_budget_bytes = int(max(budget_list) * 1024 * 1024)

    if args.camera_index < 0:
        camera_indices = list(range(len(cameras)))
    else:
        if args.camera_index >= len(cameras):
            raise ValueError("camera-index out of range")
        camera_indices = [args.camera_index]

    logger.info("tiles={} cameras={} selected_camera_indices={} bytes_per_gaussian={}",
                len(index_offsets) - 1, len(cameras), camera_indices, bytes_per_gaussian)

    # ---- tiling.npz: write once per run ----
    _t0 = time.perf_counter()
    shared_tiling_npz = base_output_path / "tiling.npz"
    shared_tiling_npz.parent.mkdir(parents=True, exist_ok=True)
    ggsp_tiling.save_tiles_to_npz(
        tile_aabbs, tile_indices, str(shared_tiling_npz),
        grid_shape=tuple(args.grid_shape), scene_min=scene_min, scene_max=scene_max, layer_idx=0
    )
    logger.success("stage=tile_npz done t={:.3f}s path={}", time.perf_counter() - _t0, shared_tiling_npz)

    executor = ThreadPoolExecutor(max_workers=args.ply_workers)

    # ---- Per-camera loop ----
    for camera_index in camera_indices:
        cam = cameras[camera_index]
        cam_futures = []

        _t0 = time.perf_counter()
        distances = uc.calculate_distances(tile_centers, cam.camera_center.to(device))
        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device
        )
        logger.success("camera={} stage=visibility done t={:.3f}s", camera_index, time.perf_counter() - _t0)

        # w_gi is view-dependent: recomputed per camera with 1/d^2 weighting
        _t0 = time.perf_counter()
        cam_center = cam.camera_center.to(device)
        w_gi = uc.compute_gaussian_weights(
            opacity, scale_0, scale_1, scale_2, gamma=args.gamma,
            xyz=gs_xyz_t, cam_center=cam_center,
        ).to(device)
        W_k, C_k = uc.compute_tile_weights_and_counts(tile_index_offsets, tile_flat_indices, w_gi)
        logger.success("camera={} stage=gaussian_weights done t={:.3f}s", camera_index, time.perf_counter() - _t0)

        # vis.npz: write once per camera
        np_visibility_all = visibility.cpu().numpy() if hasattr(visibility, 'cpu') else np.asarray(visibility)
        np_distances = distances.cpu().numpy() if hasattr(distances, 'cpu') else np.asarray(distances)
        cam_w2v = cam.world_view_transform.cpu().numpy()
        cam_proj = cam.projection_matrix.cpu().numpy()
        cam_center_np = cam.camera_center.cpu().numpy()
        meta_positions = [i for i, tk in enumerate(sorted_tile_keys) if len(tile_indices[tk][0]) > 0]
        visibility_meta = np_visibility_all[meta_positions] if len(meta_positions) > 0 else np.zeros((0,), dtype=bool)
        tile_centers_np = tile_centers.cpu().numpy()

        shared_vis_dir = base_output_path / "camera_viz"
        shared_vis_dir.mkdir(parents=True, exist_ok=True)
        shared_vis_npz = shared_vis_dir / f"{camera_index:03d}.npz"
        _t0 = time.perf_counter()
        np.savez(str(shared_vis_npz),
                 min_corners=min_corners, max_corners=max_corners,
                 visibility_all=np_visibility_all, visibility=visibility_meta,
                 distances=np_distances, tile_centers=tile_centers_np,
                 camera_center=cam_center_np, world_view_transform=cam_w2v,
                 projection_matrix=cam_proj)
        logger.success("camera={} stage=vis_npz done t={:.3f}s", camera_index, time.perf_counter() - _t0)

        # ---- Per-scheme loop ----
        for scheme in scheme_list:
            include_lod = scheme != "vd"
            include_w = scheme in ("vd_lod_w", "vd_lod_w_c")
            include_c = scheme in ("vd_lod_c", "vd_lod_w_c")

            _t0 = time.perf_counter()
            utilities = uc.calculate_utility_param(
                visibility, distances,
                num_of_level=args.num_lod,
                weight_sum_tensor=W_k if include_w else None,
                complexity_tensor=C_k if include_c else None,
                include_lod=include_lod, include_w=include_w, include_c=include_c,
            )
            logger.success("camera={} scheme={} stage=utility done t={:.3f}s",
                           camera_index, scheme, time.perf_counter() - _t0)

            _t0 = time.perf_counter()
            all_ordered = _greedy_order(
                utilities, tile_index_offsets, tile_flat_indices, w_gi, bytes_per_gaussian, max_budget_bytes
            )
            logger.success("camera={} scheme={} stage=greedy done t={:.3f}s n_ordered={}",
                           camera_index, scheme, time.perf_counter() - _t0, len(all_ordered))

            # ---- Per-budget loop: submit PLY writes to thread pool ----
            for budget_mb in budget_list:
                budget_bytes = int(budget_mb * 1024 * 1024)
                selected_indices, used_bytes = _select_at_budget(all_ordered, budget_bytes, bytes_per_gaussian)

                if args.output_root is not None:
                    camera_dir = base_output_path / "ply" / f"budget_{_budget_tag(budget_mb)}" / scheme
                    output_path = camera_dir / f"camera_{camera_index:03d}.ply"
                else:
                    legacy_output = Path(args.output)
                    if len(camera_indices) == 1 and len(budget_list) == 1 and len(scheme_list) == 1:
                        output_path = legacy_output
                    else:
                        output_path = legacy_output.with_name(
                            f"{legacy_output.stem}_{scheme}_{_budget_tag(budget_mb)}_camera_{camera_index:03d}{legacy_output.suffix}"
                        )
                    camera_dir = output_path.parent

                camera_dir.mkdir(parents=True, exist_ok=True)

                fut = executor.submit(_write_ply, gs, selected_indices, output_path, args.ascii_ply)
                cam_futures.append(fut)

                manifest = {
                    "scheme": scheme,
                    "camera_index": camera_index,
                    "budget_mb": budget_mb,
                    "budget_bytes": budget_bytes,
                    "used_bytes": used_bytes,
                    "selected_gaussians": len(selected_indices),
                    "bytes_per_gaussian": bytes_per_gaussian,
                    "output_path": str(output_path),
                    "tiling_metadata_npz": str(shared_tiling_npz),
                    "visibility_npz": str(shared_vis_npz),
                    "camera_trace": str(camera_trace),
                    "grid_shape": list(args.grid_shape),
                    "num_lod": args.num_lod,
                }
                output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                logger.debug("camera={} scheme={} budget_mb={} selected={} submitted_ply_write",
                             camera_index, scheme, budget_mb, len(selected_indices))

        _t0 = time.perf_counter()
        for fut in as_completed(cam_futures):
            fut.result()
        logger.success("camera={} stage=ply_writes_drained t={:.3f}s ({} files)",
                       camera_index, time.perf_counter() - _t0, len(cam_futures))

    executor.shutdown(wait=True)
    logger.info("done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("utility run failed")
        raise
