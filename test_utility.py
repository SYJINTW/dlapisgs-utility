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
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GGSP"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))

import visibility_AABB_pytorch  # noqa: E402
import tiling as ggsp_tiling  # noqa: E402
import utility_calculation as uc  # noqa: E402
import io_3dgs  # noqa: E402  # pyright: ignore[reportMissingImports]
from ml import predict as ml_predict  # noqa: E402


VALID_SCHEMES = ["vd", "vd_lod", "vd_lod_w", "vd_lod_c", "vd_lod_w_c",
                 "ml_lgbm_raw", "ml_lgbm_resid",
                 "oracle_loo", "oracle_aoi", "oracle_combined"]
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
    """Mode `progressive`: visible-tile GS first (sorted by w_gi), then invisible.

    Two-pass selection: visible-tile Gaussians always outrank invisible-tile
    Gaussians, but if budget remains after the visible pool is drained, the
    invisible pool fills it (sorted by w_gi descending). Multiplicative ε
    softening doesn't work here because w(g_i) spans ~30 orders of magnitude
    (see output/0513_histogram_bicycle/*.png) — a two-tier sort is the only
    numerically clean way to guarantee identity at byte_budget ≥ scene_size.
    """
    max_count = max_budget_bytes // bytes_per_gaussian
    device = w_gi.device
    vis = visibility_tile.to(device=device, dtype=torch.bool)
    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    per_gs_vis = torch.repeat_interleave(vis, sizes)
    if per_gs_vis.numel() == 0:
        return np.empty(0, dtype=np.int64)

    visible_gs = tile_flat_indices[per_gs_vis]
    visible_sorted = visible_gs[torch.argsort(w_gi[visible_gs], descending=True)]

    if visible_sorted.numel() >= max_count:
        return visible_sorted[:max_count].detach().cpu().numpy().astype(np.int64, copy=False)

    invisible_gs = tile_flat_indices[~per_gs_vis]
    invisible_sorted = invisible_gs[torch.argsort(w_gi[invisible_gs], descending=True)]
    remaining = max_count - int(visible_sorted.numel())
    ordered = torch.cat([visible_sorted, invisible_sorted[:remaining]])
    return ordered.detach().cpu().numpy().astype(np.int64, copy=False)


def _greedy_order_tile_strict(order_pairs, tile_index_offsets, tile_flat_indices, w_gi):
    """Mode `tile_strict`: emit tiles by utility with cumulative-count boundaries.

    All-or-nothing per tile. Emits every non-empty tile in utility order (within
    each tile, GS sorted by w_gi descending) and records the cumulative GS count
    after each tile. The per-budget selector (`_select_at_budget`) keeps the
    first k tiles whose cumulative count fits and STOPS at the first overflow —
    no skip-and-continue, no mid-tile partial fill.

    Returns:
      all_ordered:     flat int64 array, GS indices in tile-priority order
      tile_cum_counts: int64 array; cum_counts[k] = total #GS after emitting tile k
    """
    chunks = []
    cum_counts = []
    count = 0
    for tile_idx, _lod in order_pairs:
        start = tile_index_offsets[tile_idx]
        end = tile_index_offsets[tile_idx + 1]
        indices_for_tile = tile_flat_indices[start:end]
        n = len(indices_for_tile)
        if n == 0:
            continue
        tile_weights = w_gi[indices_for_tile]
        sorted_tile = indices_for_tile[torch.argsort(tile_weights, descending=True)].cpu().numpy()
        chunks.append(sorted_tile)
        count += n
        cum_counts.append(count)
    if not chunks:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(chunks), np.asarray(cum_counts, dtype=np.int64)


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


def _select_at_budget(all_ordered, budget_bytes, bytes_per_gaussian, tile_cum_counts=None):
    """Cut `all_ordered` at the per-budget GS count.

    Default cut: flat prefix at `budget_bytes // bpg` (tile_partial / progressive —
    these orderings are designed to be cut anywhere). When `tile_cum_counts` is
    provided (tile_strict), the cut is snapped down to the last tile boundary
    that fits — all-or-nothing per tile, stop at first overflow.
    """
    count = budget_bytes // bytes_per_gaussian
    if tile_cum_counts is not None and len(tile_cum_counts) > 0:
        fit = int(np.searchsorted(tile_cum_counts, count, side="right"))
        count = int(tile_cum_counts[fit - 1]) if fit > 0 else 0
    selected = all_ordered[:count]
    # Sort ascending to enforce source-PLY order in the output PLY. This is a
    # *reproducibility* policy, not a transport-ordering decision:
    # diff_gaussian_rasterization_lapisgs is order-sensitive (depth-sort
    # tie-breaking + non-associative alpha blending), so two PLYs with the same
    # gaussian SET in different orders render to slightly different pixels.
    # Without this sort, the inf-rate at 100% budget varied by scheme on hotdog
    # (vd_lod 28/100, vd_lod_w 23, vd_lod_c 22, vd_lod_w_c 21) and mid-budget
    # cross-scheme PSNR mixed "set chosen" with "ordering noise". Sorting source-
    # ascending decouples those signals. Selection rank-order is preserved inside
    # _greedy_order_* for any consumer (priority streaming, logging) that needs it.
    selected = np.sort(selected)
    return selected, len(selected) * bytes_per_gaussian


def _oracle_utilities(scheme, oracle_data, camera_index, n_tiles):
    """Build sorted (tile_idx, lod=0) pairs from oracle_dq.npz scores.

    Returns (n_tiles, 2) int64 array matching calculate_utility_param's output format.
    Tiles not in the oracle (NaN scores) are sorted to the end.
    """
    cam_idx_to_row = oracle_data["cam_idx_to_row"]
    if camera_index not in cam_idx_to_row:
        logger.warning("camera {} not in oracle NPZ; {} falls back to tile-index order",
                       camera_index, scheme)
        tile_order = np.arange(n_tiles, dtype=np.int64)
        return np.stack([tile_order, np.zeros(n_tiles, dtype=np.int64)], axis=1)

    row = cam_idx_to_row[camera_index]
    mse_loo = oracle_data["mse_loo"][row]   # (N_tiles,), higher = more important

    if scheme == "oracle_loo":
        scores = np.where(np.isfinite(mse_loo), mse_loo, -np.inf)
        tile_order = np.argsort(scores)[::-1].astype(np.int64)

    elif scheme == "oracle_aoi":
        mse_aoi = oracle_data["mse_aoi"]
        if mse_aoi is None:
            raise ValueError("oracle_aoi requires mse_aoi in oracle NPZ (re-run exp4 with --compute-aoi)")
        aoi_row = mse_aoi[row]              # lower = more important
        scores = np.where(np.isfinite(aoi_row), -aoi_row, -np.inf)
        tile_order = np.argsort(scores)[::-1].astype(np.int64)

    else:  # oracle_combined
        mse_aoi = oracle_data["mse_aoi"]
        if mse_aoi is None:
            raise ValueError("oracle_combined requires mse_aoi in oracle NPZ (re-run exp4 with --compute-aoi)")
        aoi_row = mse_aoi[row]
        # Borda count: rank each by importance (higher rank = more important), average ranks.
        # NaN tiles get rank 0 (least important).
        def _rank_safe(arr, ascending=False):
            finite = np.isfinite(arr)
            ranks = np.zeros(len(arr), dtype=np.float64)
            sub = arr[finite]
            order = np.argsort(sub) if ascending else np.argsort(sub)[::-1]
            dense = np.empty(len(sub), dtype=np.float64)
            dense[order] = np.arange(1, len(sub) + 1, dtype=np.float64)
            ranks[finite] = dense
            return ranks
        loo_rank = _rank_safe(mse_loo, ascending=False)   # higher loo MSE → rank 1
        aoi_rank = _rank_safe(aoi_row, ascending=True)    # lower aoi MSE → rank 1
        combined = (loo_rank + aoi_rank) / 2.0
        # rank 1 = most important; argsort ascending puts rank-1 tiles first
        tile_order = np.argsort(combined).astype(np.int64)

    return np.stack([tile_order, np.zeros(n_tiles, dtype=np.int64)], axis=1)


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
    parser.add_argument("--budget-pct", nargs="+", type=float, default=None,
                        help="One or more budgets as percent of full-scene size "
                             "(N * bytes_per_gaussian). 100 → exact identity at saturation.")
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
    parser.add_argument("--camera-indices", nargs="+", type=int, default=None,
                        help="Explicit list of camera indices to process (overrides --camera-index)")
    parser.add_argument("--img-w", type=int, default=800,
                        help="Camera image width for visibility checks")
    parser.add_argument("--img-h", type=int, default=800,
                        help="Camera image height for visibility checks")
    parser.add_argument("--ascii-ply", action="store_true",
                        help="Write PLY in ASCII instead of binary (larger, human-readable)")
    parser.add_argument("--ply-workers", type=int, default=PLY_WORKERS,
                        help="Thread pool size for concurrent PLY writes")
    parser.add_argument("--w-norm", type=str, default="sum", choices=list(uc.NORM_MODES),
                        help="Normalization for W_k (tile aggregate weight). Default: sum.")
    parser.add_argument("--c-norm", type=str, default="sum", choices=list(uc.NORM_MODES),
                        help="Normalization for C_k (tile complexity). Default: sum.")
    parser.add_argument("--w-mode", type=str, default="sum", choices=list(uc.W_MODES),
                        help="Tile aggregate weight reduction for W_k. "
                             "sum (default): W_k = Σ w(g_i), scales with tile size. "
                             "mean: W_k = Σ w(g_i) / N_k, size-invariant mean quality.")
    parser.add_argument("--c-kind", type=str, default="count",
                        choices=list(uc.COMPLEXITY_KINDS) + ["count"],
                        help="Tile complexity descriptor for C_k. "
                             "count (default): backward-compat #GS count. "
                             "eigenentropy: entropy of eigenvalue spectrum of weighted centroid covariance. "
                             "voxel_entropy: Shannon entropy of 8^3 weighted voxel occupancy.")
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
    parser.add_argument("--tiling-cache", type=str, default=None,
                        help="Shared tiling cache npz. If it exists, skip tiling and load from it; "
                             "if it doesn't exist, compute tiling and save it there. "
                             "Use across norm-sweep runs on the same PLY + grid shape.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full job matrix (cameras x schemes x budgets -> paths) and exit. "
                             "No GPU work or disk writes. Shows [x] for outputs that already exist.")
    parser.add_argument("--ml-model-dir", type=str, default=None,
                        help="Directory containing trained LightGBM models (model_raw.pkl, "
                             "model_resid.pkl, ols_coefs.json). Required when using ml_lgbm_* schemes.")
    parser.add_argument("--oracle-npz", type=str, default=None,
                        help="oracle_dq.npz from exp4_oracle_dq.py. Required when using oracle_* schemes.")
    args = parser.parse_args()

    # budget_list is resolved later once we know bytes_per_gaussian and N (for --budget-pct).
    # For --budgets-mb / --budget-mb we can resolve now.
    if args.budget_pct is not None:
        budget_list = None  # deferred; computed after GS load
    elif args.budgets_mb is not None:
        budget_list = sorted(args.budgets_mb)
    elif args.budget_mb is not None:
        budget_list = [args.budget_mb]
    else:
        raise ValueError("One of --budget-pct, --budgets-mb, --budget-mb must be provided")

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

    if any(s.startswith("ml_lgbm_") for s in scheme_list) and args.ml_model_dir is None:
        raise ValueError("--ml-model-dir is required when using ml_lgbm_* schemes")
    if any(s.startswith("oracle_") for s in scheme_list) and args.oracle_npz is None:
        raise ValueError("--oracle-npz is required when using oracle_* schemes")

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True)

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
    logger.info("grid_shape={} budgets_mb={} budget_pct={} num_lod={} schemes={}",
                args.grid_shape, budget_list, args.budget_pct, args.num_lod, scheme_list)
    logger.info("w_norm={} w_mode={} c_norm={} c_kind={} packing_mode={} weight_mode={}",
                args.w_norm, args.w_mode, args.c_norm, args.c_kind, args.packing_mode, args.weight_mode)

    # If --budget-pct, resolve to MB by loading PLY (cheap; needed for paths even in dry-run).
    if budget_list is None:
        _gs_for_budget = io_3dgs.GaussianModelV2(str(ply_path))
        _bpg = _bytes_per_gaussian(_gs_for_budget)
        _n = len(_gs_for_budget.data["x"]["data"])
        _full_bytes = _bpg * _n
        budget_list = sorted(
            (p / 100.0) * _full_bytes / (1024 * 1024) for p in args.budget_pct
        )
        logger.info("budget_pct={} resolved -> budgets_mb={} (full={} B, N={}, bpg={})",
                    args.budget_pct, [f"{b:.4f}" for b in budget_list], _full_bytes, _n, _bpg)
        del _gs_for_budget

    # ---- Dry-run: print job matrix and exit (no GPU work, no disk writes) ----
    if args.dry_run:
        dry_cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
            str(camera_trace), args.img_w, args.img_h
        )
        n_cams = len(dry_cam_infos)
        dry_indices = list(range(n_cams)) if args.camera_index < 0 else [args.camera_index]
        if args.camera_index >= 0 and args.camera_index >= n_cams:
            raise ValueError(f"--camera-index {args.camera_index} out of range (trace has {n_cams} cameras)")
        n_jobs = len(dry_indices) * len(scheme_list) * len(budget_list)
        tqdm.write(f"dry-run: {len(dry_indices)} cameras x {len(scheme_list)} schemes "
                   f"x {len(budget_list)} budgets = {n_jobs} PLY writes")
        for cam_idx in dry_indices:
            for scheme in scheme_list:
                for budget_mb in budget_list:
                    if args.output_root is not None:
                        out = (base_output_path / "ply"
                               / f"budget_{_budget_tag(budget_mb)}" / scheme
                               / f"camera_{cam_idx:03d}.ply")
                    else:
                        leg = Path(args.output)
                        if len(dry_indices) == 1 and len(budget_list) == 1 and len(scheme_list) == 1:
                            out = leg
                        else:
                            out = leg.with_name(
                                f"{leg.stem}_{scheme}_{_budget_tag(budget_mb)}"
                                f"_camera_{cam_idx:03d}{leg.suffix}"
                            )
                    mark = "[x]" if out.exists() else "[ ]"
                    tqdm.write(f"  {mark} cam={cam_idx:03d} scheme={scheme:<12} "
                               f"budget={budget_mb:>6}mb  ->  {out}")
        sys.exit(0)

    # ---- Run-metadata + timings setup ----
    output_root_for_meta = base_output_path if base_output_path.suffix == "" else base_output_path.parent
    _dump_run_params(output_root_for_meta, args, device)
    timings: list = []

    # ---- Shared preprocessing (once per run) ----
    with _timed("ply_load", timings):
        gs = io_3dgs.GaussianModelV2(str(ply_path))

    tiling_cache_path = Path(args.tiling_cache) if args.tiling_cache else None
    tile_aabbs = tile_indices = None  # only set when computing fresh (needed for save_tiles_to_npz)

    if tiling_cache_path is not None and tiling_cache_path.exists():
        with _timed("tiling_cache_load", timings):
            logger.info("tiling cache hit: {}", tiling_cache_path)
            _tc = np.load(str(tiling_cache_path))
            min_corners = _tc["min_corners"]
            max_corners = _tc["max_corners"]
            index_offsets = _tc["index_offsets"]
            flat_indices = _tc["flat_indices"]
    else:
        with _timed("tiling", timings):
            tile_aabbs, tile_indices, scene_min, scene_max = ggsp_tiling.tiling_uniform_layered_gs(
                [gs], grid_shape=tuple(args.grid_shape)
            )
            min_corners, max_corners, index_offsets, flat_indices, _sorted_tile_keys = _build_tile_arrays(
                tile_aabbs, tile_indices, layer_idx=0
            )
        if tiling_cache_path is not None:
            tiling_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(tiling_cache_path),
                     min_corners=min_corners, max_corners=max_corners,
                     index_offsets=index_offsets, flat_indices=flat_indices)
            logger.info("tiling cache saved: {}", tiling_cache_path)

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

    if args.camera_indices is not None:
        for ci in args.camera_indices:
            if ci >= len(cameras):
                raise ValueError(f"--camera-indices: index {ci} out of range")
        camera_indices = sorted(set(args.camera_indices))
    elif args.camera_index < 0:
        camera_indices = list(range(len(cameras)))
    else:
        if args.camera_index >= len(cameras):
            raise ValueError("camera-index out of range")
        camera_indices = [args.camera_index]

    logger.info("tiles={} cameras={} selected_camera_indices={} bytes_per_gaussian={}",
                len(index_offsets) - 1, len(cameras), camera_indices, bytes_per_gaussian)

    # ---- tiling.npz: write once per run (skip when loaded from shared cache) ----
    if tile_aabbs is not None:
        with _timed("tile_npz", timings):
            shared_tiling_npz = base_output_path / "tiling.npz"
            shared_tiling_npz.parent.mkdir(parents=True, exist_ok=True)
            ggsp_tiling.save_tiles_to_npz(
                tile_aabbs, tile_indices, str(shared_tiling_npz),
                grid_shape=tuple(args.grid_shape), scene_min=scene_min, scene_max=scene_max, layer_idx=0
            )
    else:
        shared_tiling_npz = tiling_cache_path

    # ---- Oracle data load (once per run) ----
    oracle_data = None
    if args.oracle_npz is not None:
        _od = np.load(args.oracle_npz, allow_pickle=False)
        _cam_ids = _od["camera_indices"].astype(np.int32)
        _mse_aoi = _od["mse_aoi"] if "mse_aoi" in _od else None
        oracle_data = {
            "mse_loo": _od["mse"].astype(np.float64),
            "mse_aoi": _mse_aoi.astype(np.float64) if _mse_aoi is not None else None,
            "cam_idx_to_row": {int(c): i for i, c in enumerate(_cam_ids)},
        }
        n_oracle_tiles = oracle_data["mse_loo"].shape[1]
        n_scene_tiles = len(index_offsets) - 1
        if n_oracle_tiles != n_scene_tiles:
            raise ValueError(
                f"oracle_dq.npz has {n_oracle_tiles} tiles but scene tiling has {n_scene_tiles}. "
                "Use the same --tiling-cache that was passed to exp4_oracle_dq.py."
            )
        logger.info("oracle NPZ loaded: {} cameras, {} tiles, aoi={}",
                    len(_cam_ids), n_oracle_tiles, _mse_aoi is not None)

    executor = ThreadPoolExecutor(max_workers=args.ply_workers)
    timings_path = output_root_for_meta / "timings.json"

    # ---- Per-camera loop ----
    cam_pbar = tqdm(camera_indices, desc="cameras", unit="cam")
    for camera_index in cam_pbar:
        cam = cameras[camera_index]
        cam_futures = []
        cam_pbar.set_postfix(idx=camera_index, stage="visibility")
        with _timed("visibility", timings, camera=camera_index):
            distances = uc.calculate_distances(tile_centers, cam.camera_center.to(device))
            visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
                min_corners_t, max_corners_t, cam, device=device
            )

        cam_pbar.set_postfix(idx=camera_index, stage="gaussian_weights")
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

        cam_pbar.set_postfix(idx=camera_index, stage="tile_weights")
        with _timed("tile_weights", timings, camera=camera_index,
                    w_norm=args.w_norm, c_norm=args.c_norm):
            W_k, N_k = uc.compute_tile_weights_and_counts(
                tile_index_offsets, tile_flat_indices, w_gi,
                w_norm=args.w_norm, c_norm=args.c_norm, w_mode=args.w_mode,
            )

        needs_c = any(s in ("vd_lod_c", "vd_lod_w_c") for s in scheme_list)
        if needs_c and args.c_kind != "count":
            with _timed("tile_complexity", timings, camera=camera_index, c_kind=args.c_kind):
                C_k = uc.compute_tile_complexity(
                    args.c_kind, tile_index_offsets, tile_flat_indices, w_gi, gs_xyz_t,
                    min_corners=min_corners_t, max_corners=max_corners_t,
                )
                C_k = uc.normalize_term(C_k, args.c_norm)
        else:
            C_k = N_k  # count (backward-compat) or not needed

        # vis.npz: write once per camera
        np_visibility_all = visibility.cpu().numpy() if hasattr(visibility, 'cpu') else np.asarray(visibility)
        np_distances = distances.cpu().numpy() if hasattr(distances, 'cpu') else np.asarray(distances)
        cam_w2v = cam.world_view_transform.cpu().numpy()
        cam_proj = cam.projection_matrix.cpu().numpy()
        cam_center_np = cam.camera_center.cpu().numpy()
        meta_positions = [i for i in range(len(index_offsets) - 1) if index_offsets[i + 1] - index_offsets[i] > 0]
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
        for scheme in tqdm(scheme_list, desc="schemes", leave=False):
            with _timed("utility", timings, camera=camera_index, scheme=scheme):
                if scheme.startswith("oracle_"):
                    n_tiles = len(index_offsets) - 1
                    utilities = _oracle_utilities(scheme, oracle_data, camera_index, n_tiles)
                elif scheme.startswith("ml_lgbm_"):
                    head = scheme[len("ml_lgbm_"):]
                    utilities = ml_predict.predict_utility(
                        args.ml_model_dir, head,
                        visibility_t=visibility,
                        distances_t=distances,
                        tile_index_offsets_t=tile_index_offsets,
                        W_k_t=W_k,
                    )
                else:
                    include_lod = scheme != "vd"
                    include_w = scheme in ("vd_lod_w", "vd_lod_w_c")
                    include_c = scheme in ("vd_lod_c", "vd_lod_w_c")
                    utilities = uc.calculate_utility_param(
                        visibility, distances,
                        num_of_level=args.num_lod,
                        weight_sum_tensor=W_k if include_w else None,
                        complexity_tensor=C_k if include_c else None,
                        include_lod=include_lod, include_w=include_w, include_c=include_c,
                    )

            with _timed("greedy", timings, camera=camera_index, scheme=scheme,
                        packing_mode=args.packing_mode):
                tile_cum_counts = None
                if args.packing_mode == "progressive":
                    all_ordered = _greedy_order_progressive(
                        visibility, tile_index_offsets, tile_flat_indices,
                        w_gi, bytes_per_gaussian, max_budget_bytes,
                    )
                elif args.packing_mode == "tile_strict":
                    all_ordered, tile_cum_counts = _greedy_order_tile_strict(
                        utilities, tile_index_offsets, tile_flat_indices, w_gi,
                    )
                else:  # tile_partial (default, our proposed method)
                    all_ordered = _greedy_order(
                        utilities, tile_index_offsets, tile_flat_indices,
                        w_gi, bytes_per_gaussian, max_budget_bytes,
                    )
                logger.info("camera={} scheme={} n_ordered={} n_tiles={}",
                            camera_index, scheme, len(all_ordered),
                            len(tile_cum_counts) if tile_cum_counts is not None else "-")

            # ---- Per-budget loop: submit PLY writes to thread pool ----
            for budget_mb in tqdm(budget_list, desc="budgets", leave=False, unit="MB", unit_scale=True):
                budget_bytes = int(budget_mb * 1024 * 1024)
                with _timed("select_at_budget", timings, camera=camera_index,
                            scheme=scheme, budget_mb=budget_mb):
                    selected_indices, used_bytes = _select_at_budget(
                        all_ordered, budget_bytes, bytes_per_gaussian,
                        tile_cum_counts=tile_cum_counts,
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

        cam_pbar.set_postfix(idx=camera_index, stage="draining_ply")
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

        # Flush timings once per camera — ~1 ms, recoverable if run crashes mid-sweep.
        timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")

    executor.shutdown(wait=True)
    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    logger.info("done; timings -> {}", timings_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("utility run failed")
        raise
