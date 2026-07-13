"""ML feature builder — Groups A/C.

Feature groups:
  A (15)  : viewport + tile geometry + cos(camera-fwd, camera→tile) — view-dependent
  C (118) : GS aggregate stats (mean+std of all 59 PLY attrs after transforms) — view-independent

C is view-independent; call its builder once at startup. A is view-dependent;
call its builder per camera.

(Groups B/D/F and the once-separate G were dropped 2026-07-05 after the
pooled-domain-ACG vs pooled-all10-ACG evaluation: B added no selection-PSNR
lift, D/F never showed lift in any session, and G — cos_cam_tile — is now
computed directly inside build_group_a instead of as a separate appended
block. Column selection in ml/predict.py is by feature NAME, so this is
backward-compatible with already-trained "ACG" model pickles.)

Training entry point:
    df, names = build_feature_matrix(oracle_npz_path)
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent.parent
WORKSPACE = HERE.parent
for _p in (
    WORKSPACE / "Frustum-for-3DGS",
    WORKSPACE / "GGSP",
    WORKSPACE / "GS-Interface",
    WORKSPACE / "LapisGS-object-based-renderer",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import visibility_AABB_pytorch  # noqa: E402
import io_3dgs  # noqa: E402
import utility_calculation as uc  # noqa: E402

# ─── canonical 59 PLY attribute names (excluding nx/ny/nz which are all-zero) ─

_XYZ = ["x", "y", "z"]
_F_DC = ["f_dc_0", "f_dc_1", "f_dc_2"]
_F_REST = [f"f_rest_{i}" for i in range(45)]
_OPACITY = ["opacity"]
_SCALE = ["scale_0", "scale_1", "scale_2"]
_ROT = ["rot_0", "rot_1", "rot_2", "rot_3"]

ATTR_NAMES: list = _XYZ + _F_DC + _F_REST + _OPACITY + _SCALE + _ROT
assert len(ATTR_NAMES) == 59, len(ATTR_NAMES)

_OP_IDX = ATTR_NAMES.index("opacity")
_SC_IDX = [ATTR_NAMES.index(a) for a in ("scale_0", "scale_1", "scale_2")]

GROUP_A_NAMES: list = [
    "cam_pos_x", "cam_pos_y", "cam_pos_z",
    "cam_fwd_x", "cam_fwd_y", "cam_fwd_z",
    "fov_x", "fov_y",
    "tile_cx", "tile_cy", "tile_cz",
    "N_k", "d_k", "v_k",
    "cos_cam_tile",  # cos angle between camera forward dir and cam->tile-center vector
]

GROUP_C_NAMES: list = (
    [f"{a}_mean" for a in ATTR_NAMES]
    + [f"{a}_std" for a in ATTR_NAMES]
)  # 118

ALL_FEATURE_NAMES: list = GROUP_A_NAMES + GROUP_C_NAMES  # 15 + 118 = 133

_GROUP_MAP: dict = {
    "A": GROUP_A_NAMES,
    "C": GROUP_C_NAMES,
}
_GROUP_ORDER = "AC"

ABLATION_NAMES = ("AC",)


def feature_names_for_ablation(ablation: str) -> list:
    """Return ordered feature names for an ablation (subset of 'AC')."""
    unknown = set(ablation) - set(_GROUP_MAP)
    if unknown:
        raise ValueError(f"unknown group(s) {sorted(unknown)} in ablation '{ablation}'")
    return [name for g in _GROUP_ORDER if g in ablation for name in _GROUP_MAP[g]]


# ─── view-independent features (Groups C+D) ───────────────────────────────────

def build_static_features(gs_data: dict, index_offsets: np.ndarray,
                           flat_indices: np.ndarray) -> np.ndarray:
    """Precompute Group C for all tiles — call once at startup.

    Args:
        gs_data:       gs.data dict from GaussianModelV2
        index_offsets: (N_tiles+1,) int array
        flat_indices:  (N_total_gs,) int array

    Returns:
        float32 array (N_tiles, 118)  [Group C]
    """
    N_tiles = len(index_offsets) - 1

    # ── load + transform all 59 attributes ──────────────────────────────────
    cols = []
    for a in ATTR_NAMES:
        if a in gs_data:
            cols.append(gs_data[a]["data"].astype(np.float32))
        else:
            # missing attribute — fill with zeros
            n = len(gs_data["x"]["data"])
            cols.append(np.zeros(n, dtype=np.float32))
    raw = np.stack(cols, axis=1)  # (N_gs, 59)

    # Transforms: sigmoid(opacity), exp(scale_*)
    raw[:, _OP_IDX] = 1.0 / (1.0 + np.exp(-raw[:, _OP_IDX].clip(-30, 30)))
    for idx in _SC_IDX:
        raw[:, idx] = np.exp(raw[:, idx].clip(-20, 20))

    # ── Group C: mean + std (vectorized) ────────────────────────────────────
    group_c = np.zeros((N_tiles, 118), dtype=np.float32)

    for t in range(N_tiles):
        s, e = int(index_offsets[t]), int(index_offsets[t + 1])
        if e <= s:
            continue
        tile_gs = raw[flat_indices[s:e]]  # (n_gs, 59)
        group_c[t, :59] = tile_gs.mean(axis=0)
        group_c[t, 59:] = tile_gs.std(axis=0, ddof=0)

    return group_c  # (N_tiles, 118)


# ─── static feature cache (Group C, shipped with the model) ───────────────────
#
# The (N_tiles, 118) static block is camera-invariant — same for every frame of a
# scene. Caching it as a model artifact means a deployed selector never touches the
# PLY at inference: per frame = Group A (camera scalars) + model.predict + greedy.
# Tiling-match is validated by N_tiles + a checksum of index_offsets (the tiling is
# deterministic from PLY+grid, so a matching checksum == identical tile ordering).

CACHE_FEATURE_NAMES: list = GROUP_C_NAMES  # 118


def _index_offsets_sha(index_offsets: np.ndarray) -> str:
    arr = np.ascontiguousarray(index_offsets, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def save_feature_cache(path, static_feats: np.ndarray,
                       index_offsets: np.ndarray) -> None:
    """Write the (N_tiles, 134) static block + tiling fingerprint to path."""
    np.savez(
        str(path),
        static=static_feats.astype(np.float32),
        feature_names=np.array(CACHE_FEATURE_NAMES),
        n_tiles=np.int64(static_feats.shape[0]),
        index_offsets_sha=_index_offsets_sha(index_offsets),
    )


def load_feature_cache(path, index_offsets: np.ndarray) -> np.ndarray:
    """Load a static cache, validating it matches the active tiling.

    Raises ValueError on tiling mismatch (caller falls back to a live build).
    Returns float32 (N_tiles, 134).
    """
    z = np.load(str(path), allow_pickle=True)
    n_expected = len(index_offsets) - 1
    n_cached = int(z["n_tiles"])
    if n_cached != n_expected:
        raise ValueError(
            f"feature_cache n_tiles={n_cached} != active tiling {n_expected}")
    sha_cached = str(z["index_offsets_sha"])
    sha_active = _index_offsets_sha(index_offsets)
    if sha_cached != sha_active:
        raise ValueError("feature_cache tiling checksum mismatch")
    return z["static"].astype(np.float32)


def static_cache_from_oracle(oracle_npz_path):
    """Build the (N_tiles, 134) static block from an oracle_dq.npz.

    Identical to the selection-time path: same build_static_features call on the
    same PLY + tiling. Returns (static_feats, index_offsets).
    """
    npz = np.load(str(oracle_npz_path), allow_pickle=True)
    meta = json.loads(str(npz["gen_meta"].item()))
    index_offsets = npz["index_offsets"]
    flat_indices = npz["flat_indices"]
    gs = io_3dgs.GaussianModelV2(meta["scene_ply"])
    static_feats = build_static_features(gs.data, index_offsets, flat_indices)
    return static_feats, index_offsets


# ─── view-dependent features (Group A) ────────────────────────────────────────

def build_group_a(cam_pos_np: np.ndarray, cam_fwd_np: np.ndarray,
                  fov_x: float, fov_y: float,
                  tile_centroid_np: np.ndarray,
                  n_gs_per_tile: np.ndarray,
                  distances_np: np.ndarray,
                  visibility_np: np.ndarray) -> np.ndarray:
    """Build Group A features for one camera.

    Args:
        cam_pos_np:      (3,) camera world position
        cam_fwd_np:      (3,) camera forward direction in world space
        fov_x, fov_y:    horizontal/vertical FoV in radians
        tile_centroid_np: (N_tiles, 3)
        n_gs_per_tile:   (N_tiles,) float32 GS count per tile
        distances_np:    (N_tiles,) camera-to-tile distances
        visibility_np:   (N_tiles,) float (1.0 = visible, 0.0 = not)

    Returns:
        float32 array (N_tiles, 15)
    """
    N = len(tile_centroid_np)

    # cos angle between camera forward dir and camera->tile-center vector
    to_tile = tile_centroid_np.astype(np.float64) - cam_pos_np[None, :].astype(np.float64)
    norm = np.linalg.norm(to_tile, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    to_tile_unit = to_tile / norm
    fwd = cam_fwd_np.astype(np.float64)
    fwd_unit = fwd / (np.linalg.norm(fwd) + 1e-12)
    cos_cam_tile = (to_tile_unit @ fwd_unit).astype(np.float32)[:, None]

    return np.column_stack([
        np.broadcast_to(cam_pos_np, (N, 3)).copy().astype(np.float32),
        np.broadcast_to(cam_fwd_np, (N, 3)).copy().astype(np.float32),
        np.full((N, 1), float(fov_x), dtype=np.float32),
        np.full((N, 1), float(fov_y), dtype=np.float32),
        tile_centroid_np.astype(np.float32),
        n_gs_per_tile.astype(np.float32)[:, None],
        distances_np.astype(np.float32)[:, None],
        visibility_np.astype(np.float32)[:, None],
        cos_cam_tile,
    ])  # (N_tiles, 15)


# ─── training entry point ─────────────────────────────────────────────────────

def build_feature_matrix(oracle_npz_path: str) -> "tuple[pd.DataFrame, list]":
    """Build features from oracle_dq.npz for training.

    Returns:
        df:    DataFrame, one row per (camera, tile). Columns = ALL_FEATURE_NAMES
               + ["camera", "tile", "log_mse_loo"].
        names: ALL_FEATURE_NAMES (length 133).
    """
    oracle_npz_path = Path(oracle_npz_path)
    npz = np.load(str(oracle_npz_path), allow_pickle=True)

    meta = json.loads(str(npz["gen_meta"].item()))
    img_w, img_h = int(meta["width"]), int(meta["height"])
    assert img_w == 1600 and img_h == 1600, \
        f"oracle_dq.npz resolution {img_w}×{img_h} ≠ 1600×1600 — re-run exp4 with 1600×1600"

    ply_path   = meta["scene_ply"]
    trace_path = meta["trace"]

    min_corners  = npz["min_corners"]    # (N_tiles, 3)
    max_corners  = npz["max_corners"]    # (N_tiles, 3)
    index_offsets = npz["index_offsets"] # (N_tiles+1,)
    flat_indices  = npz["flat_indices"]  # (N_total_gs,)
    n_gs_per_tile = (index_offsets[1:] - index_offsets[:-1]).astype(np.float32)
    camera_indices = npz["camera_indices"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gs = io_3dgs.GaussianModelV2(ply_path)

    # ── static features (C): one-time ─────────────────────────────────────────
    static_feats = build_static_features(gs.data, index_offsets, flat_indices)

    # ── per-camera features (A) ────────────────────────────────────────────────
    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers_np = ((min_corners + max_corners) / 2.0).astype(np.float32)
    tile_centers_t  = torch.tensor(tile_centers_np, dtype=torch.float32, device=device)

    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(trace_path, img_w, img_h)
    cameras   = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    cam_to_row = {int(c): i for i, c in enumerate(camera_indices)}

    mse_loo = npz["mse"].astype(np.float64)  # (N_cams, N_tiles)

    N_tiles = len(n_gs_per_tile)
    tile_col = np.arange(N_tiles, dtype=np.int32)

    rows = []
    for cam_idx in camera_indices:
        cam_idx_int = int(cam_idx)
        cam = cameras[cam_to_row[cam_idx_int]]

        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device
        )
        distances = uc.calculate_distances(tile_centers_t, cam.camera_center.to(device))

        wv = cam.world_view_transform
        cam_pos_np  = cam.camera_center.cpu().numpy()
        cam_fwd_np  = wv.cpu().numpy()[:3, 2]  # camera forward = 3rd column of R_wv
        fov_x = float(getattr(cam, "FoVx", math.pi / 2))
        fov_y = float(getattr(cam, "FoVy", math.pi / 2))

        vis_np  = visibility.float().cpu().numpy()
        dist_np = distances.cpu().numpy()

        group_a = build_group_a(
            cam_pos_np, cam_fwd_np, fov_x, fov_y,
            tile_centers_np, n_gs_per_tile, dist_np, vis_np,
        )

        X_cam = np.hstack([group_a, static_feats])  # (N_tiles, 133)

        row_idx = cam_to_row[cam_idx_int]
        mse_row = mse_loo[row_idx]   # (N_tiles,)
        # mse_loo <= 0 (bit-identical LOO render, or floating-point noise making it
        # marginally closer to GT than the baseline) means "negligible importance", not
        # "unmeasurable" -- floor at MSE_EPS instead of dropping to NaN, so the model
        # still trains on these as strongly-negative-log examples. Real missing/NaN
        # entries in mse_row stay NaN (np.maximum propagates NaN) and are filtered
        # downstream by log_mse_loo.notna(). Was silently NaN-dropping ~90%+ of tiny-tile
        # (cam,tile) rows on real scenes at fine grids, starving the model of the exact
        # "this genuinely doesn't matter" signal it needs (2026-07-09 grid16 investigation).
        # MSE_EPS must sit below every genuinely-measured positive mse_loo in the dataset,
        # or floored (<=0) rows would rank as MORE important than real tiny-but-nonzero
        # survivors -- checked across all grids/scenes/train splits, global min positive
        # mse_loo is ~7.2e-24 (log~-53.3); 1e-30 (log~-69.1) stays ~16 orders below that,
        # comfortably clear of float32 underflow (~1.2e-38 normal, ~1e-45 subnormal).
        MSE_EPS = 1e-30
        log_mse = np.log(np.maximum(mse_row, MSE_EPS))

        cam_df = pd.DataFrame(
            X_cam, columns=ALL_FEATURE_NAMES, dtype=np.float32
        )
        cam_df["camera"] = cam_idx_int
        cam_df["tile"]   = tile_col
        cam_df["log_mse_loo"] = log_mse.astype(np.float32)
        rows.append(cam_df)

    df = pd.concat(rows, ignore_index=True)
    return df, ALL_FEATURE_NAMES
