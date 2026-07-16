#!/usr/bin/env python3
"""Shared selection-pipeline core: greedy packing, per-camera weights, render/metric helpers.

Extracted 2026-07-02 from test_utility_inmem.py and experiments/0628/time_selection.py, which
had near-duplicate copies of every function below. Scoring math (utility, tile weights, per-GS
weight modes) lives in utility_calculation.py; ML inference lives in ml/. This module is the
third leg: tiling/greedy packing + render/metric helpers, i.e. the orchestration layer both the
canonical select+render+metrics pipeline and the selection-timing script need identically.

Each caller keeps its OWN instrumentation (per-stage timing wrappers), argparse, output
writers, and method-dispatch loop -- only the actual selection/render logic lives here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent

# utility_calculation -> GGSP.tiling -> io_3dgs; GS-Interface must be on sys.path first
# (utility_calculation.py itself only adds Frustum-for-3DGS/GGSP, matching the convention
# both calling scripts already use).
if str(WORKSPACE / "GS-Interface") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "GS-Interface"))

import utility_calculation as uc  # noqa: E402

RENDERER_ROOT = WORKSPACE / "LapisGS-object-based-renderer"
if str(RENDERER_ROOT) not in sys.path:
    sys.path.insert(0, str(RENDERER_ROOT))

_DUMMY = WORKSPACE / "exp-dataset" / "chair" / "predictions" / "color" / "test" / "r_0.png"
os.environ.setdefault("LAPISGS_DUMMY_IMAGE", str(_DUMMY))

from gaussian_renderer_lapisgs import GaussianModel, render as gs_render  # noqa: E402  # type: ignore
from utils.image_utils import psnr as gs_psnr  # noqa: E402  # type: ignore
from utils.loss_utils import ssim as gs_ssim  # noqa: E402  # type: ignore

try:
    import lpips as _lpips_mod
    _lpips_fn = None  # lazy-init on first use

    def _get_lpips_fn():
        global _lpips_fn
        if _lpips_fn is None:
            _lpips_fn = _lpips_mod.LPIPS(net="alex").cuda()
        return _lpips_fn
except ImportError:
    _get_lpips_fn = None  # type: ignore


class _FakePipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


PIPELINE = _FakePipe()


def bytes_per_gaussian(gs) -> int:
    return int(np.sum([np.dtype(v["val_dtype"]).itemsize for v in gs.data.values()]))


# ---------------------------------------------------------------------------
# Per-camera Gaussian weights
# ---------------------------------------------------------------------------

def compute_camera_weights(sel_cam, opacity, scale_0, scale_1, scale_2,
                            rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
                            weight_mode, img_w, img_h):
    """Compute per-Gaussian weights for one camera view."""
    cam_center = sel_cam.camera_center.to(device)
    kw = dict(opacity=opacity, scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
              xyz=gs_xyz_t, cam_center=cam_center)
    if weight_mode in ("screen_area", "screen_area_over_d2"):
        if any(r is None for r in (rot_0, rot_1, rot_2, rot_3)):
            raise RuntimeError(f"weight_mode={weight_mode} requires rot_0..rot_3 in PLY")
        kw.update(rot_0=rot_0, rot_1=rot_1, rot_2=rot_2, rot_3=rot_3,
                  world_view=sel_cam.world_view_transform,
                  proj=sel_cam.projection_matrix,
                  img_w=img_w, img_h=img_h,
                  fov_x=getattr(sel_cam, "FoVx", None),
                  fov_y=getattr(sel_cam, "FoVy", None))
    return uc.compute_gaussian_weights_v2(weight_mode, **kw).to(device)


def compute_camera_weights_culled(sel_cam, opacity, scale_0, scale_1, scale_2,
                                   rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
                                   weight_mode, img_w, img_h,
                                   visible_gs_indices, n_total_gs, epsilon=0.0):
    """Like compute_camera_weights, but only evaluates weight_mode's math for the
    visible-tile Gaussian subset (visible_gs_indices, global GS ids); every other index
    gets `epsilon`. Saves both the per-GS projection compute and the H2D transfer size,
    proportional to the culled fraction. Same dense length-n_total_gs output contract as
    compute_camera_weights, so every existing w_gi[...] consumer is unaffected.

    Exp1 (2026-07-13): screen_area's FOV clamp (project_covariance_2d, mirrors the real
    3DGS rasterizer's own clamp) bounds the off-axis *angle*, not the magnitude -- a
    Gaussian close to the near plane still gets an unbounded 1/tz^2-driven weight even if
    clamped off-axis. Measured: invisible-tile Gaussians can carry MORE weight than
    visible-tile ones (bicycle cam0: 59% of total weight mass sat in invisible tiles).
    This is why invisible-tile Gaussians get `epsilon`, not "left alone" -- the projection
    math does not naturally decay them.
    """
    w_gi = torch.full((n_total_gs,), float(epsilon), dtype=torch.float32, device=device)
    if visible_gs_indices.numel() == 0:
        return w_gi

    idx_np = visible_gs_indices.detach().cpu().numpy()
    kw = dict(opacity=opacity[idx_np], scale_0=scale_0[idx_np], scale_1=scale_1[idx_np],
              scale_2=scale_2[idx_np], xyz=gs_xyz_t[visible_gs_indices],
              cam_center=sel_cam.camera_center.to(device))
    if weight_mode in ("screen_area", "screen_area_over_d2"):
        if any(r is None for r in (rot_0, rot_1, rot_2, rot_3)):
            raise RuntimeError(f"weight_mode={weight_mode} requires rot_0..rot_3 in PLY")
        kw.update(rot_0=rot_0[idx_np], rot_1=rot_1[idx_np], rot_2=rot_2[idx_np], rot_3=rot_3[idx_np],
                  world_view=sel_cam.world_view_transform,
                  proj=sel_cam.projection_matrix,
                  img_w=img_w, img_h=img_h,
                  fov_x=getattr(sel_cam, "FoVx", None),
                  fov_y=getattr(sel_cam, "FoVy", None))
    sub_w = uc.compute_gaussian_weights_v2(weight_mode, **kw).to(device)
    w_gi[visible_gs_indices] = sub_w
    return w_gi


def camera_weights_for_scope(gs_weight_scope, sel_cam, visibility, opacity, scale_0, scale_1,
                              scale_2, rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
                              weight_mode, img_w, img_h,
                              tile_index_offsets, tile_flat_indices):
    """Shared full/visible GS-weight dispatch (Exp1/Exp2, 2026-07-13) -- single
    implementation for both callers (test_utility_inmem.py: progressive packing + w_lod
    scheme; time_selection.py: progressive_* methods), not duplicated per caller.
    'visible' derives visible_gs_indices from tile-level visibility and calls
    compute_camera_weights_culled; 'full' calls compute_camera_weights unchanged."""
    if gs_weight_scope == "visible":
        sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
        per_gs_vis = torch.repeat_interleave(visibility.to(device=device, dtype=torch.bool), sizes)
        visible_gs_indices = tile_flat_indices[per_gs_vis]
        return compute_camera_weights_culled(
            sel_cam, opacity, scale_0, scale_1, scale_2,
            rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
            weight_mode, img_w, img_h,
            visible_gs_indices, len(opacity))
    return compute_camera_weights(
        sel_cam, opacity, scale_0, scale_1, scale_2,
        rot_0, rot_1, rot_2, rot_3, gs_xyz_t, device,
        weight_mode, img_w, img_h)


# ---------------------------------------------------------------------------
# Greedy packing
# ---------------------------------------------------------------------------

# Schemes with no per-Gaussian weight signal in their own TILED (tile_partial) scoring
# formula -- these MUST use PLY/storage order for intra-tile Gaussian ordering there.
# Sorting them by a weight they never computed (e.g. screen_area W_k) silently gifts an
# unearned quality bump that grows with tile coarseness (grid 1^3, chair, budget 25%:
# weight-order 35.5dB vs ply-order 19.9dB, same GS count -- measured 2026-07-15). This is
# a property of the tiled scoring formula, so it's enforced once here, inside
# greedy_order() -- NOT left to each caller to remember. A caller's `gs_order` argument is
# only a preference for schemes that DO have a real weight signal; it is silently
# overridden for anything in this set. (Real incident: test_utility_inmem.py forwarded
# --gs-order uniformly to every scheme including vd_lod, corrupting Exp4's grid-size
# sweep; time_selection.py separately hardcoded gs_order="ply" at its own vd_lod call site
# and got it right -- two callers, two different answers, because the rule lived in
# neither of them. It belongs here.)
#
# Does NOT apply to greedy_order_progressive: progressive packing is a fundamentally
# per-Gaussian, visible-first-by-weight method (not a tile-scoring formula at all) --
# vd_lod run under progressive packing is a different algorithm from vd_lod run under
# tile_partial, and always uses real weight order there, same as every other scheme.
NO_WEIGHT_SCHEMES = frozenset({"vd_lod"})


def greedy_order_progressive(visibility_tile, tile_index_offsets, tile_flat_indices,
                              w_gi, bytes_per_gaussian, max_budget_bytes,
                              shuffle_seed=None):
    max_count = max_budget_bytes // bytes_per_gaussian
    device = w_gi.device
    vis = visibility_tile.to(device=device, dtype=torch.bool)
    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    per_gs_vis = torch.repeat_interleave(vis, sizes)
    if per_gs_vis.numel() == 0:
        return np.empty(0, dtype=np.int64)

    visible_gs = tile_flat_indices[per_gs_vis]
    invisible_gs = tile_flat_indices[~per_gs_vis]
    if shuffle_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(shuffle_seed)
        visible_sorted = visible_gs[torch.randperm(len(visible_gs), generator=gen, device=device)]
        gen.manual_seed(shuffle_seed + 1)
        invisible_sorted = invisible_gs[torch.randperm(len(invisible_gs), generator=gen, device=device)]
    else:
        # stable=True: under --gs-weight-scope visible (compute_camera_weights_culled),
        # every invisible-tile GS ties at exactly `epsilon` -- an unstable sort would make
        # this pool's fallback order nondeterministic run-to-run. Cheap either way.
        # torch.argsort(..., stable=) isn't accepted in this env's torch==1.12.1 -- use
        # torch.sort(..., stable=True).indices instead, which has supported it since 1.9.
        visible_sorted = visible_gs[torch.sort(w_gi[visible_gs], descending=True, stable=True).indices]
        invisible_sorted = invisible_gs[torch.sort(w_gi[invisible_gs], descending=True, stable=True).indices]

    if visible_sorted.numel() >= max_count:
        return visible_sorted[:max_count].detach().cpu().numpy().astype(np.int64, copy=False)

    remaining = max_count - int(visible_sorted.numel())
    ordered = torch.cat([visible_sorted, invisible_sorted[:remaining]])
    return ordered.detach().cpu().numpy().astype(np.int64, copy=False)


def greedy_order_tile_strict(order_pairs, tile_index_offsets, tile_flat_indices, w_gi):
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


def greedy_order(order_pairs, tile_index_offsets, tile_flat_indices, w_gi,
                  bytes_per_gaussian, max_budget_bytes, gs_order="weight", scheme=None):
    """Return flat Gaussian index array in scheduling order, truncated to budget.

    Two independent axes: tile-level rank (order_pairs, from whichever scoring scheme
    called this) and intra-tile Gaussian order (gs_order). gs_order="weight" applies the
    real per-GS screen_area weight as a secondary key -- keep the top-w(g_i) Gaussians
    within a partially-packed tile; this is the paper's two-level design (Thesis #1), not
    an incidental tie-break. gs_order="ply" is storage order, used when no per-GS weight
    was computed for this run (vd_lod always; ml/oracle_online when the caller chooses not
    to pay for gaussian_weights -- see time_selection.py --gs-order).

    `scheme` (optional): if it's in NO_WEIGHT_SCHEMES, gs_order is force-overridden to
    "ply" regardless of what the caller passed -- enforced here, once, so no caller can
    accidentally apply weight-order to a scheme that never computed a real w_gi (this is
    what corrupted Exp4's grid sweep -- see NO_WEIGHT_SCHEMES docstring above). Callers
    that don't pass `scheme` keep the old caller-trusts-gs_order behavior.

    perf: this used to be a per-tile Python loop, then a CPU np.lexsort over the full scene
    (both regressed back into this codebase repeatedly with no recorded verdict -- if
    replacing the code below, re-measure and update this comment, don't do it silently).
    Current: two-pass GPU stable sort (sort by secondary key, then stable-sort that by
    tile_rank) -- exact lexicographic order, 7.9-38x faster than the lexsort version,
    identity-checked against it on real chair+bicycle data.

    Tried combining (tile_rank, -w_gi) into one float64 key, single argsort: didn't work,
    wrong on bicycle/weight (precision collapse compressing distinct float32 weights into
    the same float64 after normalization) -- don't retry.

    Tried skipping intra-tile order for fully-included/excluded tiles (only sort the
    budget-boundary tile): didn't work, wrong -- QUIC streams a tile's own Gaussians in
    priority order, so even a fully-included tile's internal order matters for progressive
    delivery. Every Gaussian needs a defined position, not just budget-adjacent ones.
    """
    if scheme in NO_WEIGHT_SCHEMES:
        gs_order = "ply"
    max_count = max_budget_bytes // bytes_per_gaussian
    n_tiles = int(tile_index_offsets.numel()) - 1
    device = w_gi.device

    tile_prio = torch.zeros(n_tiles, dtype=torch.long, device=device)
    tile_order_t = torch.as_tensor(order_pairs[:, 0], dtype=torch.long, device=device)
    tile_prio[tile_order_t] = torch.arange(len(order_pairs), dtype=torch.long, device=device)

    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    tile_ranks_flat = torch.repeat_interleave(tile_prio, sizes)
    n_gs = tile_flat_indices.numel()

    if gs_order == "weight":
        # "ours": within a partially-packed tile, keep the top-w(g_i) Gaussians.
        secondary = w_gi[tile_flat_indices]
        _, ord1 = torch.sort(secondary, descending=True, stable=True)
    elif gs_order == "ply":
        # file/storage order, no per-Gaussian weight signal (baseline).
        idx = torch.arange(n_gs, device=device)
        _, ord1 = torch.sort(idx, stable=True)
    else:
        raise ValueError(f"unknown gs_order '{gs_order}'")

    ranks_by_ord1 = tile_ranks_flat[ord1]
    _, ord2 = torch.sort(ranks_by_ord1, stable=True)
    final_order = ord1[ord2]
    return tile_flat_indices[final_order][:max_count].detach().cpu().numpy().astype(np.int64)


def prune_tile_gs(tile_index_offsets, tile_flat_indices, w_gi, keep_frac, scheme,
                   max_gs_per_tile=None):
    """Online per-tile Gaussian pruning (2026-07-15, Workstream C of the scene-size-
    reduction plan): cap each tile's OWN Gaussian set to its top-`keep_frac` (by
    descending w_gi) before the flat priority interleave in greedy_order() ever runs --
    a deliberate cap, not the accidental byte-budget truncation select_at_budget() already
    does. A tile "finishes" after fewer bytes, so the schedule reaches more tiles per
    cadence window at coarser per-tile fidelity.

    Schemes in NO_WEIGHT_SCHEMES (no real per-GS weight -- currently just vd_lod) pass
    through UNMODIFIED: pruning by a weight that was never computed is meaningless for
    them, same rule this codebase already enforces for gs_order inside greedy_order()
    (see its docstring / NO_WEIGHT_SCHEMES above). keep_frac >= 1.0 is also a no-op.

    `max_gs_per_tile` (optional) is the starvation backstop: an absolute cap applied
    AFTER the proportional keep_frac cut, so one oversized tile (bicycle has a 1.37M-GS
    outlier against a scene median of 850) can't alone consume an entire cadence window's
    budget regardless of its keep_frac share -- same fix shape as the size-aware/
    load-balanced track-assignment candidate already flagged for the n_tracks SSIM-
    regression root cause (PLAN.md).

    Returns (new_tile_index_offsets, new_tile_flat_indices), same dtype/device as the
    inputs, still tile-contiguous (offsets mark per-tile block boundaries) -- ready to
    feed into build_greedy_order()/greedy_order() unchanged. Order within a surviving
    tile's block does NOT need to be weight-sorted: greedy_order() re-derives its own
    gs_order from scratch on whatever it's given, it doesn't trust caller order.
    """
    if scheme in NO_WEIGHT_SCHEMES:
        return tile_index_offsets, tile_flat_indices
    if keep_frac >= 1.0 and max_gs_per_tile is None:
        return tile_index_offsets, tile_flat_indices

    device = tile_flat_indices.device
    offsets_np = tile_index_offsets.cpu().numpy()
    flat_np = tile_flat_indices.cpu().numpy()
    n_tiles = len(offsets_np) - 1

    kept_chunks = []
    new_offsets = [0]
    for t in range(n_tiles):
        tile_idx = flat_np[offsets_np[t]:offsets_np[t + 1]]
        if len(tile_idx) == 0:
            new_offsets.append(new_offsets[-1])
            continue
        n_keep = max(1, int(round(len(tile_idx) * keep_frac)))
        if max_gs_per_tile is not None:
            n_keep = min(n_keep, max_gs_per_tile)
        n_keep = min(n_keep, len(tile_idx))
        tile_idx_t = torch.as_tensor(tile_idx, dtype=torch.long, device=device)
        tile_w = w_gi[tile_idx_t]
        _, order = torch.sort(tile_w, descending=True, stable=True)
        survivors = tile_idx_t[order[:n_keep]].cpu().numpy()
        kept_chunks.append(survivors)
        new_offsets.append(new_offsets[-1] + n_keep)

    new_flat_np = (np.concatenate(kept_chunks) if kept_chunks
                   else np.empty((0,), dtype=np.int64))
    new_tile_index_offsets = torch.tensor(new_offsets, dtype=tile_index_offsets.dtype, device=device)
    new_tile_flat_indices = torch.tensor(new_flat_np, dtype=tile_flat_indices.dtype, device=device)
    return new_tile_index_offsets, new_tile_flat_indices


def build_greedy_order(packing_mode, scheme, utilities, visibility,
                        tile_index_offsets, tile_flat_indices, w_gi,
                        bytes_per_gaussian, max_budget_bytes, shuffle_visible_seed=None,
                        gs_order="weight"):
    """Return (all_ordered, tile_cum_counts) for one (camera, scheme) pair."""
    tile_cum_counts = None
    if packing_mode == "progressive":
        all_ordered = greedy_order_progressive(
            visibility, tile_index_offsets, tile_flat_indices,
            w_gi, bytes_per_gaussian, max_budget_bytes,
            shuffle_seed=shuffle_visible_seed)
    elif packing_mode == "tile_strict":
        all_ordered, tile_cum_counts = greedy_order_tile_strict(
            utilities, tile_index_offsets, tile_flat_indices, w_gi)
    elif packing_mode == "tile_partial":
        all_ordered = greedy_order(
            utilities, tile_index_offsets, tile_flat_indices,
            w_gi, bytes_per_gaussian, max_budget_bytes, gs_order=gs_order, scheme=scheme)
    else:
        raise ValueError(f"Unknown packing_mode '{packing_mode}'.")
    return all_ordered, tile_cum_counts


def select_at_budget(all_ordered, budget_bytes, bytes_per_gaussian, tile_cum_counts=None):
    count = budget_bytes // bytes_per_gaussian
    if tile_cum_counts is not None and len(tile_cum_counts) > 0:
        fit = int(np.searchsorted(tile_cum_counts, count, side="right"))
        count = int(tile_cum_counts[fit - 1]) if fit > 0 else 0
    selected = all_ordered[:count]
    selected = np.sort(selected)
    return selected, len(selected) * bytes_per_gaussian


# ---------------------------------------------------------------------------
# Tile scoring / sort
# ---------------------------------------------------------------------------

def oracle_scores(scheme, oracle_data, camera_index, n_tiles):
    """Return raw score array (N_tiles,) or None if camera not in oracle."""
    cam_idx_to_row = oracle_data["cam_idx_to_row"]
    if camera_index not in cam_idx_to_row:
        return None

    row = cam_idx_to_row[camera_index]
    mse_loo = oracle_data["mse_loo"][row]

    # All branches return a POSITIVE, LINEAR per-tile importance I_k (invisible/
    # non-finite tiles get -inf so they sort last). This lets sort_tiles divide by
    # byte cost for the per-byte ("marginal") sort without sign/scale corruption.
    if scheme == "oracle_loo":
        # I_k = MSE rise when tile k is dropped (already a positive ΔMSE).
        return np.where(np.isfinite(mse_loo), mse_loo, -np.inf)

    elif scheme == "oracle_loo_ssim":
        # I_k = ΔSSIM = 1 - SSIM(R_-k, R_full): structural-similarity drop when
        # tile k is dropped. Larger = more important. Metric-matched to SSIM eval.
        ssim_loo = oracle_data["ssim_loo"][row]
        return np.where(np.isfinite(ssim_loo), 1.0 - ssim_loo, -np.inf)

    elif scheme == "oracle_aoi":
        mse_aoi = oracle_data["mse_aoi"]
        mse_blank = oracle_data.get("mse_blank")
        if mse_aoi is None or mse_blank is None:
            raise ValueError("oracle_aoi requires mse_aoi and mse_blank in oracle NPZ")
        aoi_row = mse_aoi[row]
        blank_c = float(mse_blank[row])
        # I_k = MSE this tile removes from a blank image (add-one-in ΔMSE).
        # Clamp at 0: a tile alone worse than blank contributes no value.
        imp = np.maximum(blank_c - aoi_row, 0.0)
        return np.where(np.isfinite(aoi_row), imp, -np.inf)

    else:  # oracle_combined
        mse_aoi = oracle_data["mse_aoi"]
        mse_blank = oracle_data.get("mse_blank")
        if mse_aoi is None or mse_blank is None:
            raise ValueError("oracle_combined requires mse_aoi and mse_blank in oracle NPZ")
        aoi_row = mse_aoi[row]
        blank_c = float(mse_blank[row])
        finite = np.isfinite(mse_loo) & np.isfinite(aoi_row)
        # Combine the two LINEAR importances (not their rankings): each L1-normalized
        # per camera so loo and aoi contribute on a common scale, then summed.
        I_loo = np.where(finite, np.maximum(mse_loo, 0.0), 0.0)
        I_aoi = np.where(finite, np.maximum(blank_c - aoi_row, 0.0), 0.0)
        s_loo, s_aoi = I_loo.sum(), I_aoi.sum()
        I_loo_n = I_loo / s_loo if s_loo > 0 else I_loo
        I_aoi_n = I_aoi / s_aoi if s_aoi > 0 else I_aoi
        combined = I_loo_n + I_aoi_n
        return np.where(finite, combined, -np.inf)


def sort_tiles(raw_scores, n_gs_per_tile, bytes_per_gaussian, greedy_key,
               num_of_level=1, beta=10.0):
    """Sort (tile, lod) pairs by greedy_key. raw_scores=None -> tile-index order.

    Owns LOD expansion: when num_of_level > 1, applies log(β·(ℓ+1)) weighting
    and returns (N*num_of_level, 2) pairs. For marginal+multi-LOD, per-tile bytes
    are used (approximation; per-LOD counts can be threaded in later if needed).
    """
    if raw_scores is None:
        n = len(n_gs_per_tile)
        return np.stack([np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)], axis=1)

    if greedy_key == "marginal":
        tile_bytes = np.where(n_gs_per_tile > 0, n_gs_per_tile * bytes_per_gaussian, 1.0)
        scores = raw_scores / tile_bytes
    else:
        scores = raw_scores

    if num_of_level <= 1:
        order = np.argsort(-scores, kind="stable").astype(np.int64)
        return np.stack([order, np.zeros(len(order), dtype=np.int64)], axis=1)

    levels = np.arange(num_of_level, dtype=np.float64)
    log_weights = np.log(beta * (levels + 1.0))
    utility_matrix = scores[:, None] * log_weights[None, :]   # (N, num_of_level)
    flat = np.argsort(-utility_matrix.ravel(), kind="stable").astype(np.int64)
    return np.stack([flat // num_of_level, flat % num_of_level], axis=1)


def compute_raw_scores(scheme, *, oracle_data, camera_index, n_tiles,
                        visibility, distances, num_lod, W_k, C_k,
                        ml_predict_kwargs=None):
    """Return raw score array (N_tiles,) or None (oracle camera-missing fallback)."""
    if scheme.startswith("oracle_"):
        return oracle_scores(scheme, oracle_data, camera_index, n_tiles)
    elif scheme == "ml":
        from ml import predict as ml_predict
        ml_kwargs = {k: v for k, v in ml_predict_kwargs.items()
                     if k in ("model_dir", "model_type", "static_features",
                              "group_a", "feature_names", "model")}
        raw = ml_predict.predict_utility(**ml_kwargs)
        if ml_kwargs.get("model_type") == "lgbm_rank":
            # LGBMRanker's predict() is an arbitrary-scale lambdarank margin, never
            # log(mse_loo) -- exp()-ing it distorts marginal (score/bytes) ratios between
            # tiles of different byte cost, unlike the regression models below where exp()
            # genuinely recovers a linear ΔMSE. Use the raw margin directly.
            return raw
        # Regression models (rf/lgbm/xgb via ml/train.py) predict log(mse_loo); exp()
        # recovers the positive linear ΔMSE so the per-byte sort divides a real importance.
        # exp_clipped guards against a single OOD prediction dominating the marginal sort
        # (see ml/predict.py::LOG_MSE_CLIP).
        return ml_predict.exp_clipped(raw)
    elif scheme == "ml_blend":
        from ml import predict as ml_predict
        blend_kwargs = {k: v for k, v in ml_predict_kwargs.items()
                        if k in ("model_dir", "static_features", "group_a",
                                 "feature_names", "models")}
        return ml_predict.predict_utility_blend(**blend_kwargs)
    else:
        include_lod = scheme != "vd"
        # w_lod / d_lod_w (Exp2, 2026-07-13/14): drop the explicit v term -- only sound when
        # W_k itself was built from --gs-weight-scope visible (culled) weights, otherwise an
        # invisible tile's own W_k isn't naturally near-zero (screen_area's FOV clamp
        # doesn't decay off-frustum weight -- see compute_camera_weights_culled). Caller
        # is responsible for that; this dispatcher just wires the flags. d_lod_w keeps the
        # explicit /d term (w_lod doesn't) -- the 4th of 4 W_k-inclusive formula variants
        # (vd_lod_w, v_lod_w, w_lod, d_lod_w = all 4 {v,d} subsets paired with w).
        include_v = scheme not in ("w_lod", "d_lod_w")
        include_d = scheme.startswith("vd") or scheme == "d_lod_w"
        include_w = scheme in ("vd_lod_w", "vd_lod_w_c", "v_lod_w", "w_lod", "d_lod_w")
        include_c = scheme in ("vd_lod_c", "vd_lod_w_c")
        return uc.calculate_utility_param(
            visibility, distances, num_of_level=num_lod,
            weight_sum_tensor=W_k if include_w else None,
            complexity_tensor=C_k if include_c else None,
            include_lod=include_lod, include_v=include_v, include_d=include_d,
            include_w=include_w, include_c=include_c,
        )


# ---------------------------------------------------------------------------
# Render / metrics / IO helpers
# ---------------------------------------------------------------------------

def load_trace(trace_path) -> list:
    with open(trace_path) as f:
        data = json.load(f)
    angle_x = data["camera_angle_x"]
    for frame in data["frames"]:
        frame["camera_angle_x"] = angle_x
    return data["frames"]


def subset_gaussians(full_gs, selected_indices: np.ndarray):
    """Return a shallow-copied GaussianModel subsetted to selected_indices (no disk IO)."""
    import copy
    sub = copy.copy(full_gs)
    device = full_gs._xyz.device
    idx = torch.tensor(selected_indices, dtype=torch.long, device=device)
    sub._xyz           = full_gs._xyz[idx]
    sub._features_dc   = full_gs._features_dc[idx]
    sub._features_rest = full_gs._features_rest[idx]
    sub._scaling       = full_gs._scaling[idx]
    sub._rotation      = full_gs._rotation[idx]
    sub._opacity       = full_gs._opacity[idx]
    return sub


def render_gs(gaussians, camera, white_bg: bool) -> torch.Tensor:
    bg = [1, 1, 1] if white_bg else [0, 0, 0]
    bg_color = torch.tensor(bg, dtype=torch.float32, device="cuda").view(3, 1, 1)
    bg_color = bg_color.expand(3, camera.image_height, camera.image_width)
    bg_depth = torch.zeros(1, camera.image_height, camera.image_width, device="cuda")
    gs_res = torch.ones(len(gaussians.get_xyz), device="cuda")
    with torch.no_grad():
        result = gs_render(camera, gaussians, PIPELINE, bg_color, bg_depth, gs_res=gs_res)
    return result["render"].clamp(0.0, 1.0)


def compute_metrics(rendered: torch.Tensor, gt: torch.Tensor, skip_lpips: bool = True) -> dict:
    r, g = rendered.unsqueeze(0), gt.unsqueeze(0)
    m = {
        "psnr": float(gs_psnr(r, g).mean().item()),
        "ssim": float(gs_ssim(r, g).item()),
    }
    if not skip_lpips and _get_lpips_fn is not None:
        with torch.no_grad():
            m["lpips"] = float(_get_lpips_fn()(r * 2 - 1, g * 2 - 1).item())
    return m
