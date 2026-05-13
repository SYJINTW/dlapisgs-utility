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
(pure progressive vs. pure adaptive vs. our two-level design)


"""


import argparse
import datetime as _dt
import json
import os
import platform
import socket
import sys
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
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


@contextmanager
def _timed(name, store, **labels):
    """Log + record the wall time of a code block.

    Emits the same `stage=… done t=…s` line as the legacy logger.success calls,
    AND appends a structured row to `store` for later JSON dump.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        label_str = " ".join(f"{k}={v}" for k, v in labels.items())
        logger.success("{} stage={} done t={:.3f}s", label_str, name, dt)
        row = {"stage": name, "t_sec": dt}
        row.update(labels)
        store.append(row)


def _yaml_safe(obj):
    """Recursively coerce values into PyYAML-safe primitives.

    Some libraries return str subclasses (e.g. torch.__version__) which
    yaml.safe_dump refuses; pathlib.Path is also not handled. This walks the
    structure once and casts each leaf to a built-in type.
    """
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):
        return str(obj)  # collapse str subclasses to plain str
    return str(obj)


def _dump_run_params(output_root: Path, args: argparse.Namespace, device: str) -> None:
    """Write a YAML manifest of the run's configuration and execution context."""
    payload = {
        "run": {
            "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "device": device,
            "python": platform.python_version(),
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cwd": os.getcwd(),
            "script": str(Path(__file__).resolve()),
        },
        "args": vars(args),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "params.yaml").write_text(
        yaml.safe_dump(_yaml_safe(payload), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


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


def _greedy_order_progressive(visibility_tile, tile_index_offsets, tile_flat_indices,
                              w_gi, bytes_per_gaussian, max_budget_bytes):
    """Mode `progressive`: flatten visible-tile GS, sort by w_gi, trim to budget.

    `visibility_tile` is a length-num_tiles bool tensor. Returns a numpy int64
    array of Gaussian indices in priority order, same shape as the other
    packers' outputs.
    """
    max_count = max_budget_bytes // bytes_per_gaussian
    device = w_gi.device
    vis = visibility_tile.to(device=device, dtype=torch.bool)
    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    per_gs_vis = torch.repeat_interleave(vis, sizes)
    if per_gs_vis.numel() == 0 or not per_gs_vis.any():
        return np.empty(0, dtype=np.int64)
    visible_gs = tile_flat_indices[per_gs_vis]
    visible_w = w_gi[visible_gs]
    order = torch.argsort(visible_w, descending=True)
    ordered = visible_gs[order]
    take = int(min(ordered.numel(), max_count))
    return ordered[:take].detach().cpu().numpy().astype(np.int64, copy=False)


def _greedy_order_tile_strict(order_pairs, tile_index_offsets, tile_flat_indices,
                              w_gi, bytes_per_gaussian, max_budget_bytes):
    """Mode `tile_strict`: pick tiles by utility; drop any tile that doesn't fit whole.

    Same priority order as `tile_partial`, but instead of partial-filling the
    final tile we skip every tile whose count exceeds the remaining budget.
    """
    max_count = max_budget_bytes // bytes_per_gaussian
    chunks = []
    count = 0
    for tile_idx, _lod in order_pairs:
        remaining = max_count - count
        if remaining <= 0:
            break
        start = tile_index_offsets[tile_idx]
        end = tile_index_offsets[tile_idx + 1]
        indices_for_tile = tile_flat_indices[start:end]
        n = len(indices_for_tile)
        if n == 0 or n > remaining:
            continue
        tile_weights = w_gi[indices_for_tile]
        sorted_tile = indices_for_tile[torch.argsort(tile_weights, descending=True)].cpu().numpy()
        chunks.append(sorted_tile)
        count += n
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _greedy_order(order_pairs, tile_index_offsets, tile_flat_indices, w_gi, bytes_per_gaussian, max_budget_bytes):
    """Mode `tile_partial` (default, our proposed method).

    Greedy by (tile, LOD) utility; on the tile that overflows the budget,
    sort GS inside by weight and partial-send to fill the remainder.
    Returns a numpy int64 array of Gaussian indices in priority order.
    """
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
    parser.add_argument("--w-norm", type=str, default="none", choices=list(uc.NORM_MODES),
                        help="Normalization for W_k (tile aggregate weight). Default: none (legacy).")
    parser.add_argument("--c-norm", type=str, default="max", choices=list(uc.NORM_MODES),
                        help="Normalization for C_k (tile complexity). Default: max (legacy).")
    parser.add_argument("--packing-mode", type=str, default="tile_partial",
                        choices=["tile_partial", "tile_strict", "progressive"],
                        help="tile_partial (proposed, default): tile-greedy + partial last tile. "
                             "tile_strict: tile-greedy, drop tiles that don't fit whole. "
                             "progressive: ignore tiles at selection; sort all visible GS by w_gi.")
    parser.add_argument("--weight-mode", type=str, default="det_gamma_over_d2",
                        choices=list(uc.WEIGHT_MODES),
                        help="Per-Gaussian weight formula. "
                             "det_gamma_over_d2 (default): sigmoid(o)*det(Σ)^gamma/d^2 (gamma via --gamma). "
                             "volume: sigmoid(o)*det(Σ)^0.5 (true volume proxy s_x·s_y·s_z). "
                             "volume_over_d2: same / d^2. "
                             "screen_area: sigmoid(o)*π·sqrt(det(Σ_2D)) (true projected footprint).")
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
    logger.info("w_norm={} c_norm={} packing_mode={} weight_mode={}",
                args.w_norm, args.c_norm, args.packing_mode, args.weight_mode)

    # ---- Run-metadata + timings setup ----
    output_root_for_meta = base_output_path if base_output_path.suffix == "" else base_output_path.parent
    _dump_run_params(output_root_for_meta, args, device)
    timings: list = []

    # ---- Shared preprocessing (once per run) ----
    with _timed("ply_load", timings):
        gs = io_3dgs.GaussianModelV2(str(ply_path))

    with _timed("tiling", timings):
        tile_aabbs, tile_indices, scene_min, scene_max = ggsp_tiling.tiling_uniform_layered_gs(
            [gs], grid_shape=tuple(args.grid_shape)
        )
        min_corners, max_corners, index_offsets, flat_indices, sorted_tile_keys = _build_tile_arrays(
            tile_aabbs, tile_indices, layer_idx=0
        )
        min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
        max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
        tile_centers = (min_corners_t + max_corners_t) / 2.0

    with _timed("camera_load", timings):
        cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
            str(camera_trace), args.img_w, args.img_h
        )
        cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)

    with _timed("gs_attrs", timings):
        opacity = gs.data["opacity"]["data"]
        scale_0 = gs.data["scale_0"]["data"]
        scale_1 = gs.data["scale_1"]["data"]
        scale_2 = gs.data["scale_2"]["data"]
        rot_0 = gs.data["rot_0"]["data"] if "rot_0" in gs.data else None
        rot_1 = gs.data["rot_1"]["data"] if "rot_1" in gs.data else None
        rot_2 = gs.data["rot_2"]["data"] if "rot_2" in gs.data else None
        rot_3 = gs.data["rot_3"]["data"] if "rot_3" in gs.data else None
        gs_xyz = np.stack([gs.data["x"]["data"], gs.data["y"]["data"], gs.data["z"]["data"]], axis=1)
        gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)
        tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
        tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)

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
    with _timed("tile_npz", timings):
        shared_tiling_npz = base_output_path / "tiling.npz"
        shared_tiling_npz.parent.mkdir(parents=True, exist_ok=True)
        ggsp_tiling.save_tiles_to_npz(
            tile_aabbs, tile_indices, str(shared_tiling_npz),
            grid_shape=tuple(args.grid_shape), scene_min=scene_min, scene_max=scene_max, layer_idx=0
        )

    executor = ThreadPoolExecutor(max_workers=args.ply_workers)

    # ---- Per-camera loop ----
    for camera_index in camera_indices:
        cam = cameras[camera_index]
        cam_futures = []

        with _timed("visibility", timings, camera=camera_index):
            distances = uc.calculate_distances(tile_centers, cam.camera_center.to(device))
            visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
                min_corners_t, max_corners_t, cam, device=device
            )

        with _timed("gaussian_weights", timings, camera=camera_index,
                    weight_mode=args.weight_mode):
            cam_center = cam.camera_center.to(device)
            if args.weight_mode == "det_gamma_over_d2":
                # sigmoid(o) * det(Σ)^gamma / d^2 — preserves current default behavior.
                w_gi = uc.compute_gaussian_weights(
                    opacity, scale_0, scale_1, scale_2, gamma=args.gamma,
                    xyz=gs_xyz_t, cam_center=cam_center,
                ).to(device)
            else:
                kw = dict(
                    opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
                    xyz=gs_xyz_t, cam_center=cam_center,
                )
                if args.weight_mode == "screen_area":
                    if any(r is None for r in (rot_0, rot_1, rot_2, rot_3)):
                        raise RuntimeError("weight_mode=screen_area requires rot_0..rot_3 in PLY")
                    kw.update(
                        rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
                        world_view=cam.world_view_transform, proj=cam.projection_matrix,
                        img_w=args.img_w, img_h=args.img_h,
                        fov_x=getattr(cam, "FoVx", None), fov_y=getattr(cam, "FoVy", None),
                    )
                w_gi = uc.compute_gaussian_weights_v2(args.weight_mode, **kw).to(device)

        with _timed("tile_weights", timings, camera=camera_index,
                    w_norm=args.w_norm, c_norm=args.c_norm):
            W_k, C_k = uc.compute_tile_weights_and_counts(
                tile_index_offsets, tile_flat_indices, w_gi,
                w_norm=args.w_norm, c_norm=args.c_norm,
            )

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
        with _timed("vis_npz", timings, camera=camera_index):
            np.savez(str(shared_vis_npz),
                     min_corners=min_corners, max_corners=max_corners,
                     visibility_all=np_visibility_all, visibility=visibility_meta,
                     distances=np_distances, tile_centers=tile_centers_np,
                     camera_center=cam_center_np, world_view_transform=cam_w2v,
                     projection_matrix=cam_proj)

        # ---- Per-scheme loop ----
        for scheme in scheme_list:
            include_lod = scheme != "vd"
            include_w = scheme in ("vd_lod_w", "vd_lod_w_c")
            include_c = scheme in ("vd_lod_c", "vd_lod_w_c")

            with _timed("utility", timings, camera=camera_index, scheme=scheme):
                utilities = uc.calculate_utility_param(
                    visibility, distances,
                    num_of_level=args.num_lod,
                    weight_sum_tensor=W_k if include_w else None,
                    complexity_tensor=C_k if include_c else None,
                    include_lod=include_lod, include_w=include_w, include_c=include_c,
                )

            with _timed("greedy", timings, camera=camera_index, scheme=scheme,
                        packing_mode=args.packing_mode):
                if args.packing_mode == "progressive":
                    all_ordered = _greedy_order_progressive(
                        visibility, tile_index_offsets, tile_flat_indices,
                        w_gi, bytes_per_gaussian, max_budget_bytes,
                    )
                elif args.packing_mode == "tile_strict":
                    all_ordered = _greedy_order_tile_strict(
                        utilities, tile_index_offsets, tile_flat_indices,
                        w_gi, bytes_per_gaussian, max_budget_bytes,
                    )
                else:  # tile_partial (default, our proposed method)
                    all_ordered = _greedy_order(
                        utilities, tile_index_offsets, tile_flat_indices,
                        w_gi, bytes_per_gaussian, max_budget_bytes,
                    )
                logger.info("camera={} scheme={} n_ordered={}",
                            camera_index, scheme, len(all_ordered))

            # ---- Per-budget loop: submit PLY writes to thread pool ----
            for budget_mb in budget_list:
                budget_bytes = int(budget_mb * 1024 * 1024)
                with _timed("select_at_budget", timings, camera=camera_index,
                            scheme=scheme, budget_mb=budget_mb):
                    selected_indices, used_bytes = _select_at_budget(
                        all_ordered, budget_bytes, bytes_per_gaussian,
                    )

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

                submitted_at = time.perf_counter()
                fut = executor.submit(_write_ply, gs, selected_indices, output_path, args.ascii_ply)
                cam_futures.append((fut, submitted_at))

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
                    "w_norm": args.w_norm,
                    "c_norm": args.c_norm,
                    "packing_mode": args.packing_mode,
                    "weight_mode": args.weight_mode,
                    "gamma": args.gamma,
                }
                output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                logger.debug("camera={} scheme={} budget_mb={} selected={} submitted_ply_write",
                             camera_index, scheme, budget_mb, len(selected_indices))

        with _timed("ply_writes_drained", timings, camera=camera_index, n=len(cam_futures)):
            per_write_times = []
            for fut, t_submit in cam_futures:
                fut.result()
                per_write_times.append(time.perf_counter() - t_submit)
        if per_write_times:
            timings.append({
                "stage": "ply_write_summary",
                "camera": camera_index,
                "n": len(per_write_times),
                "t_sec_mean": float(np.mean(per_write_times)),
                "t_sec_max": float(np.max(per_write_times)),
                "t_sec_total": float(np.sum(per_write_times)),
            })

    executor.shutdown(wait=True)

    timings_path = output_root_for_meta / "timings.json"
    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    logger.info("done; timings -> {}", timings_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("utility run failed")
        raise
