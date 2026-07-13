#!/usr/bin/env python3
"""Combined in-memory select + render + metrics pipeline.

Run in the gaussian_splatting conda environment (py3.7) — it can import both
the rasterizer (diff_gaussian_rasterization_lapisgs) and all selection deps.

Key difference from the test_utility.py + render_metrics.py pair:
  - No PLY files written (unless --save-ply).
  - GT rendered once at startup to gt_renders/ (float in RAM; PNG saved for inspection only).
  - Per (camera, scheme, budget): select → subset GaussianModel in memory
    → render → PSNR/SSIM → save PNG. Zero disk roundtrip.
  - Representative views (worst/median/best) picked automatically at the end.

Usage:
  conda run -n gaussian_splatting python test_utility_inmem.py \\
    --ply <full.ply> --gt-ply <full.ply> \\
    --camera-trace <trace.json> --grid-shape 8 8 8 \\
    --budget-pct 10 25 40 55 70 85 100 \\
    --schemes vd_lod oracle_loo --packing-mode progressive \\
    --weight-mode screen_area --num-lod 1 --camera-index -1 \\
    --output-root /path/to/output --scene chair
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent

# Selection deps
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GGSP"))
sys.path.insert(0, str(WORKSPACE / "GS-Interface"))
sys.path.insert(0, str(HERE / "experiments"))

import visibility_AABB_pytorch  # noqa: E402
import tiling as ggsp_tiling  # noqa: E402
import utility_calculation as uc  # noqa: E402
import io_3dgs  # noqa: E402  # pyright: ignore[reportMissingImports]
from ml import predict as ml_predict, features as ml_features  # noqa: E402
import selection_core as sc  # noqa: E402
from oracle_dq import ply_fingerprint  # noqa: E402

# Greedy/weight/render helpers live in selection_core.py (shared with time_selection.py, which
# had near-duplicate copies). Aliased under the old names so every call site below is
# unchanged -- only the definitions moved. See selection_core.py::greedy_order's own docstring
# for its current implementation and perf history -- don't restate it here, it'll go stale.
_bytes_per_gaussian       = sc.bytes_per_gaussian
_greedy_order_progressive = sc.greedy_order_progressive
_greedy_order_tile_strict = sc.greedy_order_tile_strict
_greedy_order             = sc.greedy_order
_build_greedy_order       = sc.build_greedy_order
_select_at_budget         = sc.select_at_budget
_oracle_scores            = sc.oracle_scores
_sort_tiles               = sc.sort_tiles
_compute_raw_scores       = sc.compute_raw_scores
_load_trace               = sc.load_trace
_compute_metrics          = sc.compute_metrics
_subset_gaussians         = sc.subset_gaussians
_render_gs                = sc.render_gs
_camera_weights_for_scope = sc.camera_weights_for_scope

# Renderer deps (gaussian_splatting env)
RENDERER_ROOT = WORKSPACE / "LapisGS-object-based-renderer"
sys.path.insert(0, str(RENDERER_ROOT))

_DUMMY = WORKSPACE / "exp-dataset" / "chair" / "predictions" / "color" / "test" / "r_0.png"
os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))

import torchvision  # noqa: E402
from gaussian_renderer_lapisgs import GaussianModel, render as gs_render  # noqa: E402  # type: ignore
from streaming_utils.camera_loader import load_camera_from_streaming_config  # noqa: E402  # type: ignore
from utils.image_utils import psnr as gs_psnr  # noqa: E402  # type: ignore
from utils.loss_utils import ssim as gs_ssim  # noqa: E402  # type: ignore

VALID_SCHEMES = ["vd", "vd_lod", "vd_lod_w", "v_lod_w", "w_lod", "vd_lod_c", "vd_lod_w_c",
                 "ml", "ml_blend",
                 "oracle_loo", "oracle_loo_ssim", "oracle_aoi", "oracle_combined"]
PLY_WORKERS = 4


class _FakePipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


PIPELINE = _FakePipe()


# ---------------------------------------------------------------------------
# Shared helpers (identical to test_utility.py)
# ---------------------------------------------------------------------------

@contextmanager
def _timed(name, store, **labels):
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


def _gpu_state() -> dict:
    """Snapshot system-level GPU state via nvidia-smi for the active CUDA device."""
    gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    try:
        def _query(fields):
            return subprocess.check_output(
                ["nvidia-smi", f"--id={gpu_idx}", f"--query-gpu={fields}",
                 "--format=csv,noheader,nounits"], text=True
            ).strip()
        free_mib, used_mib, util_pct = _query("memory.free,memory.used,utilization.gpu").split(",")
        procs_raw = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_idx}", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], text=True
        ).strip()
        procs = []
        for ln in procs_raw.splitlines():
            ln = ln.strip()
            if ln:
                pid_s, mem_s = ln.split(",", 1)
                procs.append({"pid": int(pid_s.strip()), "used_mib": mem_s.strip()})
        return {
            "gpu_idx": gpu_idx,
            "free_mib": int(free_mib.strip()),
            "used_mib": int(used_mib.strip()),
            "util_pct": util_pct.strip(),
            "other_procs": procs,
        }
    except Exception as exc:
        return {"gpu_idx": gpu_idx, "error": str(exc)}


def _yaml_safe(obj):
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
        return str(obj)
    return str(obj)


def _dump_run_params(output_root: Path, args: argparse.Namespace, device: str,
                     derived=None) -> None:
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
    if derived:
        payload["derived"] = derived
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
    flat_indices = (np.concatenate(flat_indices_list).astype(np.int64)
                    if flat_indices_list else np.empty((0,), dtype=np.int64))
    return min_corners, max_corners, index_offsets, flat_indices, sorted_tile_keys


def _budget_tag(budget_mb: float) -> str:
    return f"{budget_mb:g}".replace(".", "p") + "mb"


def _write_ply(gs, selected_indices, output_path, ascii_ply):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_gs = gs.extract_gaussians(selected_indices)
    selected_gs.export_gs_to_ply(str(output_path), ascii=ascii_ply)
    return output_path


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _infer_scene(gt_ply: Path) -> str:
    parts = gt_ply.resolve().parts
    if "checkpoint" in parts:
        return parts[parts.index("checkpoint") - 1]
    return gt_ply.resolve().parent.name


# ---------------------------------------------------------------------------
# Representative views (inline, matching pick_representative_views.py logic)
# ---------------------------------------------------------------------------

def _collect_rep_keys(summary_rows: list, group_by: str) -> set:
    """Return set of (camera_index, budget_mb, scheme) for worst/median/best per (scene,group,budget) cell."""
    by_cell: dict = {}
    for r in summary_rows:
        if r.get("psnr") in (None, ""):
            continue
        group_val = r.get(group_by, r.get("scheme", ""))
        if isinstance(group_val, list):
            group_val = tuple(group_val)
        key = (r.get("scene", ""), group_val, float(r["budget_mb"]))
        by_cell.setdefault(key, []).append(r)
    keys: set = set()
    for cell_rows in by_cell.values():
        cell_rows.sort(key=lambda r: float(r["psnr"]))
        for r in (cell_rows[0], cell_rows[(len(cell_rows) - 1) // 2], cell_rows[-1]):
            keys.add((int(r["camera_index"]), float(r["budget_mb"]), r["scheme"]))
    return keys


def _pick_representative_views(output_root: Path, summary_rows: list, group_by: str,
                               gt_dir: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    def _imshow_or_blank(ax, path, title):
        if path.exists():
            ax.imshow(mpimg.imread(str(path)))
        else:
            ax.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center",
                    fontsize=8, color="red", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    rep_root = output_root / "representative"
    rep_root.mkdir(parents=True, exist_ok=True)
    gt_dir = gt_dir if gt_dir is not None else output_root / "gt_renders"
    render_root = output_root / "renders"

    by_cell: dict = {}
    for r in summary_rows:
        if r.get("psnr") in (None, ""):
            continue
        group_val = r.get(group_by, r.get("scheme", ""))
        if isinstance(group_val, list):
            group_val = tuple(group_val)
        key = (r.get("scene", ""), group_val, float(r["budget_mb"]))
        by_cell.setdefault(key, []).append(r)

    distinct_groups = {k[1] for k in by_cell}
    use_group_subdir = len(distinct_groups) > 1
    index_rows = []
    for (scene, group_val, budget_mb), cell_rows in sorted(by_cell.items()):
        cell_rows.sort(key=lambda r: float(r["psnr"]))
        worst  = cell_rows[0]
        median = cell_rows[(len(cell_rows) - 1) // 2]
        best   = cell_rows[-1]

        budget_tag = f"budget_{_budget_tag(budget_mb)}"
        fig, axes = plt.subplots(3, 2, figsize=(8.0, 11.0))
        fig.suptitle(f"{scene}  {group_by}={group_val}  {budget_tag}", fontsize=12)

        for row_i, (rank, row) in enumerate((("worst", worst), ("median", median), ("best", best))):
            cam_idx = int(row["camera_index"])
            psnr_val = float(row["psnr"])
            ssim_val = float(row.get("ssim", "nan") or "nan")
            scheme = row["scheme"]
            subset_png = render_root / "ply" / budget_tag / scheme / f"camera_{cam_idx:03d}.png"
            gt_png = gt_dir / f"camera_{cam_idx:03d}.png"

            _imshow_or_blank(axes[row_i, 0], subset_png,
                             f"{rank} (cam {cam_idx})  PSNR={psnr_val:.2f}  SSIM={ssim_val:.3f}")
            _imshow_or_blank(axes[row_i, 1], gt_png, f"ground truth (cam {cam_idx})")

            index_rows.append({
                "scene": scene, "group_by": group_by, "group_value": group_val,
                "budget_mb": budget_mb, "rank": rank, "camera_index": cam_idx,
                "scheme": scheme, "psnr": psnr_val, "ssim": ssim_val,
                "subset_png": str(subset_png), "gt_png": str(gt_png),
            })

        out_dir = rep_root / str(group_val) if use_group_subdir else rep_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{budget_tag}.png"
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("[representative] {}", out_path)

    if index_rows:
        idx_csv = rep_root / "index.csv"
        with idx_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader(); w.writerows(index_rows)
        logger.info("[representative] {} picks across {} cells -> {}",
                    len(index_rows), len(by_cell), idx_csv)


# ---------------------------------------------------------------------------
# Selection helpers (camera-level)
# ---------------------------------------------------------------------------

def _rerender_rep_views(
        rep_keys, sel_cameras, rend_cameras, rend_gs_full,
        tile_index_offsets, tile_flat_indices, min_corners_t, max_corners_t, tile_centers,
        opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3, gs_xyz_t,
        bytes_per_gaussian, max_budget_bytes, budget_list, budget_bytes_list,
        oracle_data, base_output_path, device, args,
        ml_model=None, ml_static_feats=None, ml_feature_names=None,
        ml_blend_models=None) -> None:
    """Re-render and save PNGs for worst/median/best (camera, budget, scheme) combos."""
    import collections as _col
    _mb_to_bytes = dict(zip(budget_list, budget_bytes_list))
    rep_by_cam: dict = _col.defaultdict(list)
    for (ci, bm, sc) in rep_keys:
        rep_by_cam[ci].append((bm, sc))

    for camera_index in sorted(rep_by_cam):
        sel_cam = sel_cameras[camera_index]
        rend_cam = rend_cameras[camera_index]

        distances = uc.calculate_distances(tile_centers, sel_cam.camera_center.to(device))
        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, sel_cam, device=device)
        w_gi = _camera_weights_for_scope(
            args.gs_weight_scope, sel_cam, visibility, opacity, scale_0, scale_1, scale_2,
            rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
            args.weight_mode, args.img_w, args.img_h,
            tile_index_offsets, tile_flat_indices)
        W_k, N_k = uc.compute_tile_weights_and_counts(
            tile_index_offsets, tile_flat_indices, w_gi,
            w_norm=args.w_norm, c_norm=args.c_norm)

        needed_schemes = {sc for _, sc in rep_by_cam[camera_index]}
        needs_c = any(s in ("vd_lod_c", "vd_lod_w_c") for s in needed_schemes)
        if needs_c and args.c_kind != "count":
            C_k = uc.compute_tile_complexity(
                args.c_kind, tile_index_offsets, tile_flat_indices, w_gi, gs_xyz_t,
                min_corners=min_corners_t, max_corners=max_corners_t)
            C_k = uc.normalize_term(C_k, args.c_norm)
        else:
            C_k = N_k

        # ML camera features (only when an ml rep is needed for this camera)
        _index_offsets = (tile_index_offsets.cpu().numpy()
                          if hasattr(tile_index_offsets, "cpu") else np.asarray(tile_index_offsets))
        _n_gs_per_tile = (_index_offsets[1:] - _index_offsets[:-1]).astype(np.float32)
        ml_group_a = None
        if any(s in ("ml", "ml_blend") for s in needed_schemes):
            import math as _math
            np_visibility_all = (visibility.cpu().numpy()
                                 if hasattr(visibility, "cpu") else np.asarray(visibility))
            np_distances = (distances.cpu().numpy()
                            if hasattr(distances, "cpu") else np.asarray(distances))
            cam_w2v = sel_cam.world_view_transform.cpu().numpy()
            cam_center_np = sel_cam.camera_center.cpu().numpy()
            tile_centers_np = tile_centers.cpu().numpy()
            ml_group_a = ml_features.build_group_a(
                cam_center_np, cam_w2v[:3, 2],
                float(getattr(sel_cam, "FoVx", _math.pi / 2)),
                float(getattr(sel_cam, "FoVy", _math.pi / 2)),
                tile_centers_np, _n_gs_per_tile, np_distances, np_visibility_all,
            )

        ml_predict_kwargs = dict(
            model_dir=args.ml_model_dir, model_type=args.ml_model_type,
            static_features=ml_static_feats, group_a=ml_group_a,
            feature_names=ml_feature_names, model=ml_model,
            models=ml_blend_models,
        )

        for scheme in needed_schemes:
            n_tiles = len(tile_index_offsets) - 1
            raw_scores = _compute_raw_scores(
                scheme, oracle_data=oracle_data, camera_index=camera_index,
                n_tiles=n_tiles, visibility=visibility, distances=distances,
                num_lod=args.num_lod, W_k=W_k, C_k=C_k,
                ml_predict_kwargs=ml_predict_kwargs,
            )
            # LGBMRanker's raw score is a signed, arbitrary-scale lambdarank margin --
            # not a byte-divisible utility (positive or negative), so "marginal"
            # (score/bytes) division can invert relative order between two negative-
            # score tiles of different byte cost. Force direct-order ("utility") sort.
            _greedy_key = ("utility" if scheme == "ml"
                           and ml_predict_kwargs.get("model_type") == "lgbm_rank"
                           else args.greedy_key)
            utilities = _sort_tiles(raw_scores, _n_gs_per_tile, bytes_per_gaussian,
                                    _greedy_key, num_of_level=args.num_lod)

            all_ordered, tile_cum_counts = _build_greedy_order(
                args.packing_mode, scheme, utilities, visibility,
                tile_index_offsets, tile_flat_indices, w_gi,
                bytes_per_gaussian, max_budget_bytes, args.shuffle_visible_seed,
                gs_order=args.gs_order)

            needed_budgets = [bm for bm, sc in rep_by_cam[camera_index] if sc == scheme]
            for budget_mb in needed_budgets:
                budget_bytes = _mb_to_bytes.get(budget_mb, round(budget_mb * 1024 * 1024))
                selected_indices, _ = _select_at_budget(
                    all_ordered, budget_bytes, bytes_per_gaussian,
                    tile_cum_counts=tile_cum_counts)
                sub_gs = _subset_gaussians(rend_gs_full, selected_indices)
                rendered = _render_gs(sub_gs, rend_cam, args.white_bg)
                del sub_gs
                budget_tag = f"budget_{_budget_tag(budget_mb)}"
                render_png = (base_output_path / "renders" / "ply" / budget_tag
                              / scheme / f"camera_{camera_index:03d}.png")
                render_png.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(rendered, str(render_png))
                logger.info("[rep] cam={:03d} budget={} scheme={}", camera_index, budget_tag, scheme)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-memory 3DGS select + render + metrics (gaussian_splatting env)")

    # --- Selection args (identical to test_utility.py) ---
    parser.add_argument("--ply", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--camera-trace", required=True)
    parser.add_argument("--grid-shape", nargs=3, type=int, default=[8, 8, 8])
    parser.add_argument("--budget-mb", type=float, default=None)
    parser.add_argument("--budgets-mb", nargs="+", type=float, default=None)
    parser.add_argument("--budget-pct", nargs="+", type=float, default=None)
    parser.add_argument("--num-lod", type=int, default=1)
    parser.add_argument("--scheme", type=str, default=None, choices=VALID_SCHEMES)
    parser.add_argument("--schemes", nargs="+", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-indices", nargs="+", type=int, default=None)
    parser.add_argument("--img-w", type=int, default=1600)
    parser.add_argument("--img-h", type=int, default=1600)
    parser.add_argument("--w-norm", type=str, default="sum", choices=list(uc.NORM_MODES))
    parser.add_argument("--c-norm", type=str, default="sum", choices=list(uc.NORM_MODES),
                        help="C_k normalization. NOTE: C term unused by canonical schemes "
                             "(vd_lod_w/ml/oracle_loo); only active for vd_lod_c/vd_lod_w_c.")
    parser.add_argument("--c-kind", type=str, default="count",
                        choices=list(uc.COMPLEXITY_KINDS) + ["count"],
                        help="C_k complexity signal. NOTE: C term unused by canonical schemes "
                             "(vd_lod_w/ml/oracle_loo); only active for vd_lod_c/vd_lod_w_c.")
    parser.add_argument("--packing-mode", type=str, default="tile_partial",
                        choices=["tile_partial", "tile_strict", "progressive"])
    parser.add_argument("--gs-order", type=str, default="weight",
                        choices=["weight", "ply"],
                        help="Intra-tile Gaussian order when a tile is partially packed "
                             "(tile_partial). 'weight' = top-w(g_i) kept (ours); "
                             "'ply' = file/storage order, no per-GS signal (baseline).")
    parser.add_argument("--greedy-key", type=str, default="marginal",
                        choices=["utility", "marginal"],
                        help="Sort key: 'marginal' = score / tile_bytes (CANON, all tiled packing); "
                             "'utility' = raw score (legacy).")
    parser.add_argument("--weight-mode", type=str, default="screen_area",
                        choices=list(uc.WEIGHT_MODES))
    parser.add_argument("--gs-weight-scope", type=str, default="full",
                        choices=["full", "visible"],
                        help="Exp1/Exp2 (2026-07-13). 'full' (default) computes "
                             "gaussian_weights over every Gaussian in the scene, regardless "
                             "of tile visibility -- current/original behavior, unaffected "
                             "unless explicitly overridden. 'visible' only evaluates "
                             "visible-tile Gaussians; everything else gets epsilon=0.0 "
                             "(screen_area's FOV clamp doesn't naturally decay off-frustum "
                             "weight -- see selection_core.py::compute_camera_weights_culled). "
                             "Used by: --packing-mode progressive (Exp1) and --scheme w_lod "
                             "(Exp2, tile_partial -- w_lod's W_k is only sound when built "
                             "from culled weights).")
    parser.add_argument("--tiling-cache", type=str, default=None)

    parser.add_argument("--ml-model-dir", type=str, default=None)
    parser.add_argument("--ml-model-type", type=str, default="lgbm",
                        choices=["lgbm", "xgb", "rf", "lgbm_rank"])
    parser.add_argument("--ml-feature-cache", type=str, default="auto",
                        choices=["auto", "off"],
                        help="auto: use {ml-model-dir}/feature_cache.npz if present "
                             "& tiling matches; off: always rebuild static feats from PLY")
    parser.add_argument("--oracle-npz", type=str, default=None)

    # --- Renderer args ---
    parser.add_argument("--gt-ply", required=True,
                        help="Full-scene PLY for GT rendering")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-bg", action="store_true")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name tag in summary.csv. Inferred from --gt-ply if omitted.")
    parser.add_argument("--group-by", type=str, default="scheme",
                        help="Column used to group cells in representative view picker.")

    # --- Output control ---
    parser.add_argument("--shuffle-visible-seed", type=int, default=None,
                        help="Random control: shuffle Gaussians within visible/invisible partitions "
                             "instead of sorting by weight. Only used with --packing-mode progressive.")
    parser.add_argument("--save-ply", action="store_true",
                        help="Also write selected PLYs to disk (same layout as test_utility.py).")
    parser.add_argument("--ply-workers", type=int, default=PLY_WORKERS,
                        help="Thread pool size for PLY writes (only used with --save-ply).")
    parser.add_argument("--ascii-ply", action="store_true")
    parser.add_argument("--lpips", action="store_true",
                        help="Compute LPIPS alongside PSNR/SSIM (off by default; requires lpips pkg).")
    parser.add_argument("--png-workers", type=int, default=0,
                        help="Thread pool size for async PNG writes. 0 = synchronous (default).")
    parser.add_argument("--save-rep-only", action="store_true",
                        help="Skip per-camera PNG writes; re-render and save only worst/median/best "
                             "cameras per (budget, group) after the main loop.")

    args = parser.parse_args()

    # --- Resolve budget_list ---
    if args.budget_pct is not None:
        budget_list = None
    elif args.budgets_mb is not None:
        budget_list = sorted(args.budgets_mb)
    elif args.budget_mb is not None:
        budget_list = [args.budget_mb]
    else:
        raise ValueError("One of --budget-pct, --budgets-mb, --budget-mb must be provided")

    # --- Resolve scheme_list ---
    if args.schemes is not None:
        for s in args.schemes:
            if s not in VALID_SCHEMES:
                raise ValueError(f"Unknown scheme '{s}'. Valid: {VALID_SCHEMES}")
        scheme_list = args.schemes
    elif args.scheme is not None:
        scheme_list = [args.scheme]
    else:
        raise ValueError("Either --schemes or --scheme must be provided")

    if any(s in ("ml", "ml_blend") for s in scheme_list) and args.ml_model_dir is None:
        raise ValueError("--ml-model-dir required for ml/ml_blend scheme")
    if any(s.startswith("oracle_") for s in scheme_list) and args.oracle_npz is None:
        raise ValueError("--oracle-npz required for oracle_* schemes")

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ply_path = Path(args.ply)
    gt_ply_path = Path(args.gt_ply)
    camera_trace = Path(args.camera_trace)
    for p in (ply_path, gt_ply_path, camera_trace):
        if not p.exists():
            raise FileNotFoundError(p)

    base_output_path = Path(args.output_root)
    log_path = base_output_path / "utility.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="INFO")
    logger.info("device={} inmem=True save_ply={}", device, args.save_ply)
    logger.info("ply={} gt_ply={} trace={}", ply_path, gt_ply_path, camera_trace)

    # --- Resolve budget_pct → MB ---
    _budget_pct_pending = args.budget_pct if budget_list is None else None
    _budget_bytes_list: list[int] | None = None  # filled after sel_gs load if needed

    output_root_for_meta = base_output_path
    timings: list = []
    if torch.cuda.is_available():
        _gs = _gpu_state()
        _gs["stage"] = "gpu_state_startup"
        timings.append(_gs)
        (output_root_for_meta / "gpu_startup.json").write_text(json.dumps(_gs, indent=2))
        logger.info("GPU startup: free={}MiB used={}MiB other_procs={}",
                    _gs.get("free_mib"), _gs.get("used_mib"), _gs.get("other_procs"))

    # --- Load selection model (io_3dgs, numpy attrs) ---
    with _timed("ply_load_selection", timings):
        sel_gs = io_3dgs.GaussianModelV2(str(ply_path))

    # Load ML model at startup (fail fast, not at render time). load_model() also forces
    # n_jobs=1 for realtime predict -- see its docstring for why (real for RandomForest,
    # no-op for XGBoost) instead of restating that here. Moved here (after sel_gs load,
    # was before it) so expected_n_gs is available for the scene-identity check.
    _ml_model = None
    if "ml" in scheme_list:
        with _timed("ml_model_load", []):
            _ml_model = ml_predict.load_model(
                args.ml_model_dir, args.ml_model_type,
                expected_n_gs=len(sel_gs.data["x"]["data"]))

    _ml_blend_models = None
    if "ml_blend" in scheme_list:
        with _timed("ml_blend_models_load", []):
            _ml_blend_models = ml_predict.load_blend_models(
                args.ml_model_dir, expected_n_gs=len(sel_gs.data["x"]["data"]))

    # --- Load renderer model (GaussianModel, GPU tensors) ---
    with _timed("ply_load_renderer", timings):
        rend_gs_full = GaussianModel(args.sh_degree)
        rend_gs_full.load_ply(str(ply_path))

    # --- Tiling ---
    tiling_cache_path = Path(args.tiling_cache) if args.tiling_cache else None
    tile_aabbs = tile_indices = None

    if tiling_cache_path is not None and tiling_cache_path.exists():
        with _timed("tiling_cache_load", timings):
            _tc = np.load(str(tiling_cache_path), allow_pickle=True)
            _cached_grid = tuple(_tc["grid_shape"].tolist()) if "grid_shape" in _tc else None
            _req_grid = tuple(args.grid_shape)
            if _cached_grid is not None and _cached_grid != _req_grid:
                raise ValueError(
                    f"--tiling-cache was built with grid_shape={_cached_grid} "
                    f"but --grid-shape={_req_grid}. Use a different cache path."
                )
            if "n_gs" in _tc:
                _cached_n_gs, _cached_sha1 = int(_tc["n_gs"]), str(_tc["xyz_sha1"])
                _active_n_gs, _active_sha1 = ply_fingerprint(
                    sel_gs.data["x"]["data"], sel_gs.data["y"]["data"], sel_gs.data["z"]["data"])
                if _cached_n_gs != _active_n_gs:
                    raise ValueError(
                        f"--tiling-cache {tiling_cache_path} was built from a PLY with "
                        f"{_cached_n_gs} Gaussians, but --ply {ply_path} has {_active_n_gs}. "
                        "Regenerate the tiling cache or point --ply at the right scene."
                    )
                if _cached_sha1 != _active_sha1:
                    raise ValueError(
                        f"--tiling-cache {tiling_cache_path} was built from a different PLY "
                        f"(same Gaussian count, different xyz content) than --ply {ply_path}. "
                        "Regenerate the tiling cache."
                    )
            else:
                logger.warning(
                    "--tiling-cache {} has no PLY fingerprint (pre-fix cache) -- "
                    "skipping PLY/tiling identity check.", tiling_cache_path)
            min_corners = _tc["min_corners"]
            max_corners = _tc["max_corners"]
            index_offsets = _tc["index_offsets"]
            flat_indices = _tc["flat_indices"]
    else:
        with _timed("tiling", timings):
            tile_aabbs, tile_indices, scene_min, scene_max = ggsp_tiling.tiling_uniform_layered_gs(
                [sel_gs], grid_shape=tuple(args.grid_shape)
            )
            min_corners, max_corners, index_offsets, flat_indices, _sorted_tile_keys = \
                _build_tile_arrays(tile_aabbs, tile_indices, layer_idx=0)
        if tiling_cache_path is not None:
            tiling_cache_path.parent.mkdir(parents=True, exist_ok=True)
            _n_gs, _xyz_sha1 = ply_fingerprint(
                sel_gs.data["x"]["data"], sel_gs.data["y"]["data"], sel_gs.data["z"]["data"])
            np.savez(str(tiling_cache_path),
                     min_corners=min_corners, max_corners=max_corners,
                     index_offsets=index_offsets, flat_indices=flat_indices,
                     grid_shape=np.array(args.grid_shape, dtype=np.int32),
                     n_gs=np.int64(_n_gs), xyz_sha1=_xyz_sha1)

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers = (min_corners_t + max_corners_t) / 2.0
    # perf 2026-07-02: scene-static, was re-synced to CPU every camera in the main loop below.
    tile_centers_np = tile_centers.cpu().numpy()

    # --- Save tiling.npz (once per run, skip if loaded from shared cache) ---
    if tile_aabbs is not None:
        with _timed("tile_npz", timings):
            shared_tiling_npz = base_output_path / "tiling.npz"
            shared_tiling_npz.parent.mkdir(parents=True, exist_ok=True)
            ggsp_tiling.save_tiles_to_npz(
                tile_aabbs, tile_indices, str(shared_tiling_npz),
                grid_shape=tuple(args.grid_shape), scene_min=scene_min, scene_max=scene_max,
                layer_idx=0,
            )
    else:
        shared_tiling_npz = tiling_cache_path

    # --- Load cameras (two representations) ---
    with _timed("camera_load", timings):
        # MiniCam list for selection (visibility, distances)
        sel_cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(
            str(camera_trace), args.img_w, args.img_h
        )
        sel_cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(sel_cam_infos)

        # LapisGS cameras for rendering
        frames = _load_trace(camera_trace)
        rend_cameras = [
            load_camera_from_streaming_config(f, width=args.img_w, height=args.img_h)
            for f in frames
        ]
        for cam in rend_cameras:
            cam.original_image = None  # dummy placeholder, never read in inference

    assert len(sel_cameras) == len(rend_cameras), (
        f"Camera count mismatch: sel={len(sel_cameras)} rend={len(rend_cameras)}")

    # --- Pre-render GT (once, cached in RAM) ---
    gt_renders_dir = base_output_path / "gt_renders"
    gt_renders_dir.mkdir(parents=True, exist_ok=True)
    bg = [1, 1, 1] if args.white_bg else [0, 0, 0]

    with _timed("gt_ply_load", timings):
        gt_gaussians = GaussianModel(args.sh_degree)
        gt_gaussians.load_ply(str(gt_ply_path))
        gt_gs_res = torch.ones(len(gt_gaussians.get_xyz), device="cuda")

    gt_renders: list = []
    logger.info("Pre-rendering GT ({} cameras) ...", len(rend_cameras))
    with _timed("gt_render_all", timings, n=len(rend_cameras)):
        with torch.no_grad():
            for idx, cam in enumerate(tqdm(rend_cameras, desc="gt_render", leave=False)):
                bg_color = torch.tensor(bg, dtype=torch.float32, device="cuda").view(3, 1, 1)
                bg_color = bg_color.expand(3, cam.image_height, cam.image_width)
                bg_depth = torch.zeros(1, cam.image_height, cam.image_width, device="cuda")
                result = gs_render(cam, gt_gaussians, PIPELINE, bg_color, bg_depth,
                                   gs_res=gt_gs_res)
                frame_t = result["render"].clamp(0.0, 1.0)
                gt_renders.append(frame_t.cpu())
                torchvision.utils.save_image(frame_t, str(gt_renders_dir / f"camera_{idx:03d}.png"))
    logger.success("GT rendered: {} frames to {}", len(gt_renders), gt_renders_dir)
    del gt_gaussians
    torch.cuda.empty_cache()

    # --- Selection attrs (numpy, from sel_gs) ---
    with _timed("gs_attrs", timings):
        opacity = sel_gs.data["opacity"]["data"]
        scale_0 = sel_gs.data["scale_0"]["data"]
        scale_1 = sel_gs.data["scale_1"]["data"]
        scale_2 = sel_gs.data["scale_2"]["data"]
        rot_0 = sel_gs.data["rot_0"]["data"] if "rot_0" in sel_gs.data else None
        rot_1 = sel_gs.data["rot_1"]["data"] if "rot_1" in sel_gs.data else None
        rot_2 = sel_gs.data["rot_2"]["data"] if "rot_2" in sel_gs.data else None
        rot_3 = sel_gs.data["rot_3"]["data"] if "rot_3" in sel_gs.data else None
        gs_xyz = np.stack([sel_gs.data["x"]["data"], sel_gs.data["y"]["data"],
                           sel_gs.data["z"]["data"]], axis=1)
        gs_xyz_t = torch.tensor(gs_xyz, dtype=torch.float32, device=device)
        tile_index_offsets = torch.tensor(index_offsets, dtype=torch.long, device=device)
        tile_flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)

        _offsets_np = np.asarray(index_offsets, dtype=np.int64)
        _flat_np = np.asarray(flat_indices, dtype=np.int64)
        gs_to_tile = np.full(gs_xyz.shape[0], -1, dtype=np.int64)
        gs_to_tile[_flat_np] = np.repeat(
            np.arange(len(_offsets_np) - 1, dtype=np.int64),
            _offsets_np[1:] - _offsets_np[:-1],
        )

    bytes_per_gaussian = _bytes_per_gaussian(sel_gs)
    if _budget_pct_pending is not None:
        _full_n = len(sel_gs.data["x"]["data"])
        _full_bytes = _full_n * bytes_per_gaussian
        _budget_bytes_list = sorted(
            round(p / 100.0 * _full_bytes) for p in _budget_pct_pending
        )
        budget_list = sorted(b / (1024 * 1024) for b in _budget_bytes_list)
        logger.info("budget_pct={} resolved -> budget_bytes={} budget_mb={}",
                    _budget_pct_pending, _budget_bytes_list,
                    [f"{b:.4f}" for b in budget_list])
    else:
        _budget_bytes_list = [round(mb * 1024 * 1024) for mb in budget_list]
    max_budget_bytes = max(_budget_bytes_list)

    # --- ML: precompute static features (model already loaded at startup) ---
    ml_static_feats = ml_feature_names = None
    if any(s in ("ml", "ml_blend") for s in scheme_list):
        _ml_model_path = Path(args.ml_model_dir)
        ml_feature_names = json.loads((_ml_model_path / "feature_names.json").read_text())
        _cache_path = _ml_model_path / "feature_cache.npz"
        with _timed("ml_static_features", timings):
            ml_static_feats = None
            if args.ml_feature_cache == "auto" and _cache_path.exists():
                try:
                    ml_static_feats = ml_features.load_feature_cache(
                        _cache_path, index_offsets)
                    print(f"[ml] loaded static feature cache: {_cache_path}", flush=True)
                except ValueError as e:
                    print(f"[ml] feature_cache unusable ({e}) — rebuilding from PLY",
                          flush=True)
            if ml_static_feats is None:
                ml_static_feats = ml_features.build_static_features(
                    sel_gs.data, index_offsets, flat_indices
                )

    # --- Camera index list ---
    if args.camera_indices is not None:
        for ci in args.camera_indices:
            if ci >= len(sel_cameras):
                raise ValueError(f"--camera-indices: index {ci} out of range")
        camera_indices = sorted(set(args.camera_indices))
    elif args.camera_index < 0:
        camera_indices = list(range(len(sel_cameras)))
    else:
        if args.camera_index >= len(sel_cameras):
            raise ValueError("--camera-index out of range")
        camera_indices = [args.camera_index]

    _dump_run_params(output_root_for_meta, args, device, derived={
        "scheme_list": scheme_list,
        "budget_list_mb": budget_list,
        "budget_bytes_list": _budget_bytes_list,
        "camera_indices": camera_indices,
    })

    # --- Oracle data ---
    oracle_data = None
    if args.oracle_npz is not None:
        _od = np.load(args.oracle_npz, allow_pickle=False)
        _cam_ids = _od["camera_indices"].astype(np.int32)
        _mse_aoi = _od["mse_aoi"] if "mse_aoi" in _od else None
        _mse_blank = _od["mse_blank"] if "mse_blank" in _od else None
        oracle_data = {
            "mse_loo": _od["mse"].astype(np.float64),
            "ssim_loo": _od["ssim"].astype(np.float64),
            "mse_aoi": _mse_aoi.astype(np.float64) if _mse_aoi is not None else None,
            "mse_blank": _mse_blank.astype(np.float64) if _mse_blank is not None else None,
            "cam_idx_to_row": {int(c): i for i, c in enumerate(_cam_ids)},
        }
        n_oracle_tiles = oracle_data["mse_loo"].shape[1]
        n_scene_tiles = len(index_offsets) - 1
        if n_oracle_tiles != n_scene_tiles:
            raise ValueError(
                f"oracle_dq.npz has {n_oracle_tiles} tiles but scene has {n_scene_tiles}. "
                "Use the same --tiling-cache passed to exp4_oracle_dq.py."
            )
        _oracle_cam_set = set(oracle_data["cam_idx_to_row"].keys())
        _missing = [ci for ci in camera_indices if ci not in _oracle_cam_set]
        if _missing:
            logger.warning(
                "oracle NPZ missing {}/{} requested cameras; those will fall back to tile-index order. "
                "Missing: {}",
                len(_missing), len(camera_indices), _missing[:10],
            )

    logger.info("tiles={} cameras={} selected_indices={} bpg={}",
                len(index_offsets) - 1, len(sel_cameras), camera_indices, bytes_per_gaussian)

    # --- Metrics accumulation ---
    scene_tag = args.scene or _infer_scene(gt_ply_path)
    all_metric_rows: list[dict[str, Any]] = []
    metrics_dir = base_output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timings_path = output_root_for_meta / "timings.json"

    executor = ThreadPoolExecutor(max_workers=args.ply_workers) if args.save_ply else None
    png_executor = ThreadPoolExecutor(max_workers=args.png_workers) if args.png_workers > 0 else None
    png_futures: list = []

    # --- Per-camera loop ---
    cam_pbar = tqdm(camera_indices, desc="cameras", unit="cam")
    for camera_index in cam_pbar:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sel_cam = sel_cameras[camera_index]
        rend_cam = rend_cameras[camera_index]
        cam_futures = []
        _sel_peak_mib = 0
        _rend_peak_mib = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gt_render_gpu = gt_renders[camera_index].to(device)

        cam_pbar.set_postfix(idx=camera_index, stage="visibility")
        with _timed("visibility", timings, camera=camera_index):
            distances = uc.calculate_distances(tile_centers, sel_cam.camera_center.to(device))
            visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
                min_corners_t, max_corners_t, sel_cam, device=device
            )

        cam_pbar.set_postfix(idx=camera_index, stage="gaussian_weights")
        with _timed("gaussian_weights", timings, camera=camera_index,
                    weight_mode=args.weight_mode):
            w_gi = _camera_weights_for_scope(
                args.gs_weight_scope, sel_cam, visibility, opacity, scale_0, scale_1, scale_2,
                rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
                args.weight_mode, args.img_w, args.img_h,
                tile_index_offsets, tile_flat_indices)

        cam_pbar.set_postfix(idx=camera_index, stage="tile_weights")
        with _timed("tile_weights", timings, camera=camera_index):
            W_k, N_k = uc.compute_tile_weights_and_counts(
                tile_index_offsets, tile_flat_indices, w_gi,
                w_norm=args.w_norm, c_norm=args.c_norm,
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
            C_k = N_k

        # vis.npz
        np_visibility_all = (visibility.cpu().numpy()
                             if hasattr(visibility, "cpu") else np.asarray(visibility))
        np_distances = (distances.cpu().numpy()
                        if hasattr(distances, "cpu") else np.asarray(distances))
        cam_w2v = sel_cam.world_view_transform.cpu().numpy()
        cam_proj = sel_cam.projection_matrix.cpu().numpy()
        cam_center_np = sel_cam.camera_center.cpu().numpy()
        meta_positions = [i for i in range(len(index_offsets) - 1)
                          if index_offsets[i + 1] - index_offsets[i] > 0]
        visibility_meta = (np_visibility_all[meta_positions]
                           if len(meta_positions) > 0 else np.zeros((0,), dtype=bool))

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

        # ML camera features
        _n_gs_per_tile = (index_offsets[1:] - index_offsets[:-1]).astype(np.float32)
        ml_group_a = None
        if any(s in ("ml", "ml_blend") for s in scheme_list):
            with _timed("ml_camera_features", timings, camera=camera_index):
                import math as _math
                _cam_fwd = cam_w2v[:3, 2]
                ml_group_a = ml_features.build_group_a(
                    cam_center_np, _cam_fwd,
                    float(getattr(sel_cam, "FoVx", _math.pi / 2)),
                    float(getattr(sel_cam, "FoVy", _math.pi / 2)),
                    tile_centers_np, _n_gs_per_tile, np_distances, np_visibility_all,
                )

        ml_predict_kwargs = dict(
            model_dir=args.ml_model_dir, model_type=args.ml_model_type,
            static_features=ml_static_feats, group_a=ml_group_a,
            feature_names=ml_feature_names, model=_ml_model,
            models=_ml_blend_models,
        )

        # --- Per-scheme loop ---
        for scheme in tqdm(scheme_list, desc="schemes", leave=False):
            with _timed("utility", timings, camera=camera_index, scheme=scheme):
                n_tiles = len(index_offsets) - 1
                raw_scores = _compute_raw_scores(
                    scheme, oracle_data=oracle_data, camera_index=camera_index,
                    n_tiles=n_tiles, visibility=visibility, distances=distances,
                    num_lod=args.num_lod, W_k=W_k, C_k=C_k,
                    ml_predict_kwargs=ml_predict_kwargs,
                )
                # LGBMRanker's raw score is a signed, arbitrary-scale lambdarank margin --
                # not a byte-divisible utility (positive or negative), so "marginal"
                # (score/bytes) division can invert relative order between two negative-
                # score tiles of different byte cost. Force direct-order ("utility") sort.
                _greedy_key = ("utility" if scheme == "ml"
                               and ml_predict_kwargs.get("model_type") == "lgbm_rank"
                               else args.greedy_key)
                utilities = _sort_tiles(raw_scores, _n_gs_per_tile, bytes_per_gaussian,
                                        _greedy_key, num_of_level=args.num_lod)

            with _timed("greedy", timings, camera=camera_index, scheme=scheme,
                        packing_mode=args.packing_mode):
                all_ordered, tile_cum_counts = _build_greedy_order(
                    args.packing_mode, scheme, utilities, visibility,
                    tile_index_offsets, tile_flat_indices, w_gi,
                    bytes_per_gaussian, max_budget_bytes, args.shuffle_visible_seed,
                    gs_order=args.gs_order)

            if torch.cuda.is_available():
                _p = torch.cuda.max_memory_allocated() // (1024 * 1024)
                _sel_peak_mib = max(_sel_peak_mib, _p)
                torch.cuda.reset_peak_memory_stats()

            # --- Per-budget loop ---
            for budget_mb, budget_bytes in tqdm(
                    zip(budget_list, _budget_bytes_list), desc="budgets",
                    total=len(budget_list), leave=False, unit="MB"):
                with _timed("select_at_budget", timings, camera=camera_index,
                            scheme=scheme, budget_mb=budget_mb):
                    selected_indices, used_bytes = _select_at_budget(
                        all_ordered, budget_bytes, bytes_per_gaussian,
                        tile_cum_counts=tile_cum_counts,
                    )

                budget_tag = f"budget_{_budget_tag(budget_mb)}"
                render_out_dir = base_output_path / "renders" / "ply" / budget_tag / scheme
                render_out_dir.mkdir(parents=True, exist_ok=True)
                render_png = render_out_dir / f"camera_{camera_index:03d}.png"

                # --- In-memory render + metrics ---
                with _timed("render_selected", timings, camera=camera_index,
                            scheme=scheme, budget_mb=budget_mb):
                    sub_gs = _subset_gaussians(rend_gs_full, selected_indices)
                    rendered = _render_gs(sub_gs, rend_cam, args.white_bg)
                    del sub_gs

                if torch.cuda.is_available():
                    _p = torch.cuda.max_memory_allocated() // (1024 * 1024)
                    _rend_peak_mib = max(_rend_peak_mib, _p)
                    torch.cuda.reset_peak_memory_stats()

                with _timed("metrics", timings, camera=camera_index,
                            scheme=scheme, budget_mb=budget_mb):
                    m = _compute_metrics(rendered, gt_render_gpu, skip_lpips=not args.lpips)

                if not args.save_rep_only:
                    if png_executor:
                        png_futures.append(png_executor.submit(
                            torchvision.utils.save_image,
                            rendered.cpu().clone(), str(render_png)))
                    else:
                        with _timed("png_write", timings, camera=camera_index,
                                    scheme=scheme, budget_mb=budget_mb):
                            torchvision.utils.save_image(rendered, str(render_png))

                selected_tiles = (np.unique(gs_to_tile[selected_indices])
                                  if len(selected_indices) else np.empty(0, dtype=np.int64))
                selected_tiles = selected_tiles[selected_tiles >= 0]

                row: dict[str, Any] = {
                    "scene":              scene_tag,
                    "budget_mb":          budget_mb,
                    "scheme":             scheme,
                    "camera_index":       camera_index,
                    "psnr":               m["psnr"],
                    "ssim":               m["ssim"],
                    **( {"lpips": m["lpips"]} if "lpips" in m else {} ),
                    "used_bytes":         int(used_bytes),
                    "selected_gaussians": int(len(selected_indices)),
                    "n_selected_tiles":   int(selected_tiles.size),
                    "bytes_per_gaussian": bytes_per_gaussian,
                    "w_norm":             args.w_norm,
                    "c_norm":             args.c_norm,
                    "packing_mode":       args.packing_mode,
                    "weight_mode":        args.weight_mode,
                    "grid_shape":         list(args.grid_shape),
                    "num_lod":            args.num_lod,
                }
                all_metric_rows.append(row)

                per_json_dir = metrics_dir / budget_tag / scheme
                per_json_dir.mkdir(parents=True, exist_ok=True)
                (per_json_dir / f"camera_{camera_index:03d}.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )

                logger.success("[{}/{}/cam{:03d}]  PSNR={:.2f}  SSIM={:.4f}",
                               budget_tag, scheme, camera_index, m["psnr"], m["ssim"])

                # --- Optional PLY write ---
                if args.save_ply:
                    ply_dir = base_output_path / "ply" / budget_tag / scheme
                    output_ply = ply_dir / f"camera_{camera_index:03d}.ply"
                    fut = executor.submit(_write_ply, sel_gs, selected_indices,
                                         output_ply, args.ascii_ply)
                    cam_futures.append(fut)
                    # Write manifest alongside PLY
                    manifest = {
                        "scheme": scheme, "camera_index": camera_index,
                        "budget_mb": budget_mb, "budget_bytes": budget_bytes,
                        "used_bytes": used_bytes,
                        "selected_gaussians": len(selected_indices),
                        "n_selected_tiles": int(selected_tiles.size),
                        "selected_tiles": selected_tiles.tolist(),
                        "bytes_per_gaussian": bytes_per_gaussian,
                        "output_path": str(output_ply),
                        "tiling_metadata_npz": str(shared_tiling_npz),
                        "visibility_npz": str(shared_vis_npz),
                        "camera_trace": str(camera_trace),
                        "grid_shape": list(args.grid_shape),
                        "num_lod": args.num_lod,
                        "w_norm": args.w_norm, "c_norm": args.c_norm,
                        "packing_mode": args.packing_mode,
                        "weight_mode": args.weight_mode,
                    }
                    ply_dir.mkdir(parents=True, exist_ok=True)
                    output_ply.with_suffix(".json").write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )

        if executor and cam_futures:
            cam_pbar.set_postfix(idx=camera_index, stage="draining_ply")
            with _timed("ply_writes_drained", timings, camera=camera_index, n=len(cam_futures)):
                for fut in cam_futures:
                    fut.result()

        if torch.cuda.is_available():
            timings.append({"stage": "gpu_peak_selection", "camera": camera_index,
                            "peak_alloc_mib": _sel_peak_mib})
            timings.append({"stage": "gpu_peak_render", "camera": camera_index,
                            "peak_alloc_mib": _rend_peak_mib})

    if executor:
        executor.shutdown(wait=True)

    if png_executor:
        for fut in png_futures:
            fut.result(timeout=300)
        png_executor.shutdown(wait=True)

    # --- Write summary ---
    if all_metric_rows:
        all_metric_rows.sort(key=lambda r: (r["budget_mb"], r["scheme"], r["camera_index"]))
        csv_path = metrics_dir / "summary.csv"
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(all_metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_metric_rows)
        logger.info("Wrote {} rows -> {}", len(all_metric_rows), csv_path)
        (metrics_dir / "summary.json").write_text(
            json.dumps({"rows": all_metric_rows}, indent=2)
        )

    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    logger.info("done; timings -> {}", timings_path)

    # --- Rep-only second pass: re-render and save PNGs for worst/median/best cameras ---
    if args.save_rep_only and all_metric_rows:
        rep_keys = _collect_rep_keys(all_metric_rows, args.group_by)
        logger.info("save_rep_only: re-rendering {} (cam, budget, scheme) combos", len(rep_keys))
        _rerender_rep_views(
            rep_keys, sel_cameras, rend_cameras, rend_gs_full,
            tile_index_offsets, tile_flat_indices, min_corners_t, max_corners_t, tile_centers,
            opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3, gs_xyz_t,
            bytes_per_gaussian, max_budget_bytes, budget_list, _budget_bytes_list,
            oracle_data, base_output_path, device, args,
            ml_model=_ml_model, ml_static_feats=ml_static_feats,
            ml_feature_names=ml_feature_names, ml_blend_models=_ml_blend_models)

    # --- Representative views ---
    if all_metric_rows:
        logger.info("Picking representative views ...")
        _pick_representative_views(base_output_path, all_metric_rows, args.group_by,
                                   gt_dir=gt_renders_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("run failed")
        raise
