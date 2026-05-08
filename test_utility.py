"""
Offline utility runner.

Steps:
1) Load a full PLY scene.
2) Tile it using GGSP tiling.
3) Compute view-conditioned utilities per scheme.
4) Select Gaussians under a byte budget and export a subset PLY.

[TODO]
- test utility scoring scheme
- test scheduler without GS-level or without Tile-level vs. Ours Full
(pure progressive v. pure adaptive v. our two-level design)


"""
import argparse
import json
import sys
import time
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
    # should be 236 bytes or more depending on attributes; we can also compute it from the PLY header for accuracy
    per_attr = []
    for key, value in gs.data.items():
        per_attr.append(np.dtype(value["val_dtype"]).itemsize)
    return int(np.sum(per_attr))


def _budget_tag(budget_mb: float) -> str:
    return f"{budget_mb:g}".replace(".", "p") + "mb"


def _camera_output_dir(output_root: Path, budget_mb: float, scheme: str, camera_index: int) -> Path:
    return output_root / f"budget_{_budget_tag(budget_mb)}" / scheme / f"camera_{camera_index:03d}"


def _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices, w_gi, bytes_per_gaussian, max_budget_bytes):
    """Returns Gaussian indices in greedy priority order up to max_budget_bytes."""
    selected = []
    used = 0
    for tile_idx, _lod in order_pairs:
        start = tile_index_offsets[tile_idx]
        end = tile_index_offsets[tile_idx + 1]
        indices_for_tile = tile_flat_indices[start:end]
        
        if len(indices_for_tile) == 0: # empty tile, skip
            continue
        
        tile_weights = w_gi[indices_for_tile]
        sorted_tile = indices_for_tile[torch.argsort(tile_weights, descending=True).cpu().numpy()]
        for idx in sorted_tile:
            if used + bytes_per_gaussian > max_budget_bytes:
                return selected
            selected.append(int(idx))
            used += bytes_per_gaussian
    return selected


def _select_at_budget(all_ordered, budget_bytes, bytes_per_gaussian):
    count = budget_bytes // bytes_per_gaussian
    selected = all_ordered[:count]
    return selected, len(selected) * bytes_per_gaussian


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
    # Budget args: prefer --budgets-mb for multi-budget sweeps
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="Single byte budget in MB (this is legacy; use --budgets-mb for sweeps)")
    parser.add_argument("--budgets-mb", nargs="+", type=float, default=None,
                        help="One or more budgets in MB; overrides --budget-mb")
    parser.add_argument("--num-lod", type=int, default=1,
                        help="Number of LOD layers (1 means plain 3DGS)")
    # Scheme args: prefer --schemes for multi-scheme sweeps
    parser.add_argument("--scheme", type=str, default=None, choices=VALID_SCHEMES,
                        help="Single utility scheme (legacy; use --schemes for sweeps)")
    parser.add_argument("--schemes", nargs="+", type=str, default=None,
                        help="One or more schemes; overrides --scheme")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Exponent for Gaussian weight: w = o * det(Sigma)^gamma")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Index of the camera in the trace to evaluate; use -1 to process all cameras")
    parser.add_argument("--img-w", type=int, default=800,
                        help="Camera image width for visibility checks")
    parser.add_argument("--img-h", type=int, default=800,
                        help="Camera image height for visibility checks")
    parser.add_argument("--ascii-ply", action="store_true",
                        help="Write PLY in ASCII instead of binary (larger, human-readable)")
    args = parser.parse_args()

    # Resolve budget list
    if args.budgets_mb is not None:
        budget_list = sorted(args.budgets_mb)
    elif args.budget_mb is not None:
        budget_list = [args.budget_mb]
    else:
        raise ValueError("Either --budgets-mb or --budget-mb must be provided")

    # Resolve scheme list
    if args.schemes is not None:
        for s in args.schemes:
            if s not in VALID_SCHEMES:
                raise ValueError(f"Unknown scheme '{s}'. Valid: {VALID_SCHEMES}")
        scheme_list = args.schemes
    elif args.scheme is not None:
        scheme_list = [args.scheme]
    else:
        raise ValueError("Either --schemes or --scheme must be provided")

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ply_path = Path(args.ply)
    camera_trace = Path(args.camera_trace)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if not camera_trace.exists():
        raise FileNotFoundError(camera_trace)

    if args.output_root is None and args.output is None:
        raise ValueError("Either --output-root or --output must be provided")

    base_output_path = Path(args.output_root) if args.output_root is not None else Path(args.output)
    log_path = base_output_path.with_suffix(".log") if base_output_path.suffix else base_output_path / "utility.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="INFO")
    logger.info("device={}", device)
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
    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
        str(camera_trace), args.img_w, args.img_h
    )
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    logger.success("stage=camera_load done t={:.3f}s", time.perf_counter() - _t0)

    logger.success("stage=gaussian_weights start")
    _t0 = time.perf_counter()
    opacity = gs.data["opacity"]["data"]
    scale_0 = gs.data["scale_0"]["data"]
    scale_1 = gs.data["scale_1"]["data"]
    scale_2 = gs.data["scale_2"]["data"]
    w_gi = uc.compute_gaussian_weights(opacity, scale_0, scale_1, scale_2, gamma=args.gamma).to(device)
    tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
    W_k, C_k = uc.compute_tile_weights_and_counts(tile_index_offsets, tile_flat_indices, w_gi)
    logger.success("stage=gaussian_weights done t={:.3f}s", time.perf_counter() - _t0)

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

    # ---- Per-camera loop ----
    for camera_index in camera_indices:
        cam = cameras[camera_index]

        _t0 = time.perf_counter()
        distances = uc.calculate_distances(tile_centers, cam.camera_center.to(device))
        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device
        )
        logger.success("camera={} stage=visibility done t={:.3f}s", camera_index, time.perf_counter() - _t0)

        # Precompute visibility arrays for NPZ (same for all schemes/budgets of this camera)
        np_visibility_all = visibility.cpu().numpy() if hasattr(visibility, 'cpu') else np.asarray(visibility)
        np_distances = distances.cpu().numpy() if hasattr(distances, 'cpu') else np.asarray(distances)
        cam_w2v = cam.world_view_transform.cpu().numpy()
        cam_proj = cam.projection_matrix.cpu().numpy()
        cam_center_np = cam.camera_center.cpu().numpy()
        meta_positions = [i for i, tk in enumerate(sorted_tile_keys) if len(tile_indices[tk][0]) > 0]
        visibility_meta = np_visibility_all[meta_positions] if len(meta_positions) > 0 else np.zeros((0,), dtype=bool)
        tile_centers_np = tile_centers.cpu().numpy()

        # ---- Per-scheme loop (utilities computed once per scheme, shared across budgets) ----
        for scheme in scheme_list:
            include_lod = scheme != "vd"
            include_w = scheme in ("vd_lod_w", "vd_lod_w_c")
            include_c = scheme in ("vd_lod_c", "vd_lod_w_c")

            _t0 = time.perf_counter()
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
            logger.success("camera={} scheme={} stage=utility done t={:.3f}s",
                           camera_index, scheme, time.perf_counter() - _t0)

            top_k = min(20, utilities.shape[0])
            logger.debug("camera={} scheme={} top_utility_pairs (tile_idx, lod): {}",
                         camera_index, scheme, utilities[:top_k].tolist())

            _t0 = time.perf_counter()
            all_ordered = _greedy_order(
                utilities, tile_index_offsets, tile_flat_indices, w_gi, bytes_per_gaussian, max_budget_bytes
            )
            logger.success("camera={} scheme={} stage=greedy done t={:.3f}s n_ordered={}",
                           camera_index, scheme, time.perf_counter() - _t0, len(all_ordered))

            # ---- Per-budget loop (free: just slice the prefix) ----
            _t_ply = _t_vis_npz = _t_tile_npz = 0.0
            for budget_mb in budget_list:
                budget_bytes = int(budget_mb * 1024 * 1024)
                selected_indices, used_bytes = _select_at_budget(all_ordered, budget_bytes, bytes_per_gaussian)

                if args.output_root is not None:
                    camera_dir = _camera_output_dir(Path(args.output_root), budget_mb, scheme, camera_index)
                    output_path = camera_dir / "selected.ply"
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

                vis_npz_path = output_path.with_suffix('.vis.npz')
                _t1 = time.perf_counter()
                try:
                    np.savez(str(vis_npz_path),
                             min_corners=min_corners,
                             max_corners=max_corners,
                             visibility_all=np_visibility_all,
                             visibility=visibility_meta,
                             distances=np_distances,
                             tile_centers=tile_centers_np,
                             camera_center=cam_center_np,
                             world_view_transform=cam_w2v,
                             projection_matrix=cam_proj)
                except Exception as e:
                    logger.warning("Failed saving visibility NPZ camera={} scheme={} budget={}: {}",
                                   camera_index, scheme, budget_mb, e)
                _t_vis_npz += time.perf_counter() - _t1

                _t1 = time.perf_counter()
                selected_gs = gs.extract_gaussians(selected_indices)
                selected_gs.export_gs_to_ply(str(output_path), ascii=args.ascii_ply)
                _t_ply += time.perf_counter() - _t1

                npz_output_path = output_path.with_suffix(".npz")
                _t1 = time.perf_counter()
                ggsp_tiling.save_tiles_to_npz(
                    tile_aabbs,
                    tile_indices,
                    str(npz_output_path),
                    grid_shape=tuple(args.grid_shape),
                    scene_min=scene_min,
                    scene_max=scene_max,
                    layer_idx=0
                )
                _t_tile_npz += time.perf_counter() - _t1

                manifest_path = output_path.with_suffix(".json")
                manifest = {
                    "scheme": scheme,
                    "camera_index": camera_index,
                    "budget_mb": budget_mb,
                    "budget_bytes": budget_bytes,
                    "used_bytes": used_bytes,
                    "selected_gaussians": len(selected_indices),
                    "bytes_per_gaussian": bytes_per_gaussian,
                    "output_path": str(output_path),
                    "tiling_metadata_npz": str(npz_output_path),
                    "visibility_npz": str(vis_npz_path),
                    "camera_trace": str(camera_trace),
                    "grid_shape": list(args.grid_shape),
                    "num_lod": args.num_lod,
                }
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                logger.debug("camera={} scheme={} budget_mb={} selected={} used={}/{} out={}",
                             camera_index, scheme, budget_mb, len(selected_indices), used_bytes, budget_bytes, output_path)

            logger.success("camera={} scheme={} stage=io done  "
                           "ply_export={:.3f}s  vis_npz={:.3f}s  tile_npz={:.3f}s  ({} budgets)",
                           camera_index, scheme, _t_ply, _t_vis_npz, _t_tile_npz, len(budget_list))

    logger.info("done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("utility run failed")
        raise
