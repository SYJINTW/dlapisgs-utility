"""ML feature builder — Groups A/B/C/D/E/F.

Feature groups:
  A (14)  : viewport + tile geometry — view-dependent
  B (12)  : handcrafted tile signals — view-dependent
  C (118) : GS aggregate stats (mean+std of all 59 PLY attrs after transforms) — view-independent
  D (16)  : higher-order moments for opacity+scale_0/1/2 — view-independent
  E (50)  : PCA of GS attrs (top-25 PCs) → per-tile mean+std — view-independent
  F (6)   : SH-evaluated color (opacity-weighted mean+std RGB per tile) — view-dependent

Groups C/D/E are view-independent; call their builders once at startup.
Groups A/B/F are view-dependent; call their builders per camera.

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
from scipy import stats as scipy_stats

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
from utils.sh_utils import eval_sh  # noqa: E402

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

# Group D: 4 attrs × 4 moments (skew, kurt, median, max)
GROUP_D_ATTRS = _OPACITY + _SCALE   # ["opacity", "scale_0", "scale_1", "scale_2"]
GROUP_D_ATTR_IDX: list = [ATTR_NAMES.index(a) for a in GROUP_D_ATTRS]

GROUP_A_NAMES: list = [
    "cam_pos_x", "cam_pos_y", "cam_pos_z",
    "cam_fwd_x", "cam_fwd_y", "cam_fwd_z",
    "fov_x", "fov_y",
    "tile_cx", "tile_cy", "tile_cz",
    "N_k", "d_k", "v_k",
]

GROUP_B_NAMES: list = [
    "wbar_k_screen", "W_k_screen",
    "wbar_k_vol_d2", "W_k_vol_d2",
    "C_eigenentropy", "C_omnivariance", "C_voxel_entropy", "C_spectral_energy",
    "C_eigenentropy_pos", "C_omnivariance_pos", "C_voxel_entropy_pos", "C_spectral_energy_pos",
]

GROUP_C_NAMES: list = (
    [f"{a}_mean" for a in ATTR_NAMES]
    + [f"{a}_std" for a in ATTR_NAMES]
)  # 118

GROUP_D_NAMES: list = [
    f"{a}_{m}"
    for a in GROUP_D_ATTRS
    for m in ("skew", "kurt", "median", "max")
]  # 16

GROUP_F_NAMES: list = [
    "sh_mean_r", "sh_mean_g", "sh_mean_b",
    "sh_std_r",  "sh_std_g",  "sh_std_b",
]  # 6

ALL_FEATURE_NAMES: list = GROUP_A_NAMES + GROUP_B_NAMES + GROUP_C_NAMES + GROUP_D_NAMES + GROUP_F_NAMES
# 14 + 12 + 118 + 16 + 6 = 166

_GROUP_MAP: dict = {
    "A": GROUP_A_NAMES,
    "B": GROUP_B_NAMES,
    "C": GROUP_C_NAMES,
    "D": GROUP_D_NAMES,
    "F": GROUP_F_NAMES,
}
_GROUP_ORDER = "ABCDF"

# Canonical example ablations (any subset of ABCDF is valid).
ABLATION_NAMES = ("ABCDF", "ABCD", "ACDF", "ACD", "ACF", "AC", "AF", "ABC", "AB", "A")
_GROUP_B_SET = set(GROUP_B_NAMES)


def feature_names_for_ablation(ablation: str) -> list:
    """Return ordered feature names for an ablation.

    Ablation = subset of 'ABCDF' in canonical order (A < B < C < D < F).
    Any combination is valid, e.g. 'AF', 'ACDF', 'ABCDF'.
    """
    unknown = set(ablation) - set(_GROUP_MAP)
    if unknown:
        raise ValueError(f"unknown group(s) {sorted(unknown)} in ablation '{ablation}'")
    return [name for g in _GROUP_ORDER if g in ablation for name in _GROUP_MAP[g]]


# ─── view-independent features (Groups C+D) ───────────────────────────────────

def build_static_features(gs_data: dict, index_offsets: np.ndarray,
                           flat_indices: np.ndarray) -> np.ndarray:
    """Precompute Groups C+D for all tiles — call once at startup.

    Args:
        gs_data:       gs.data dict from GaussianModelV2
        index_offsets: (N_tiles+1,) int array
        flat_indices:  (N_total_gs,) int array

    Returns:
        float32 array (N_tiles, 134)  [118 Group C + 16 Group D]
    """
    N_tiles = len(index_offsets) - 1
    sizes = index_offsets[1:] - index_offsets[:-1]  # (N_tiles,)

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

    # ── Group D: higher-order moments ────────────────────────────────────────
    group_d = np.zeros((N_tiles, 16), dtype=np.float32)

    for t in range(N_tiles):
        n_gs = int(sizes[t])
        if n_gs < uc.COMPLEXITY_SPARSE_GUARD:
            continue
        s, e = int(index_offsets[t]), int(index_offsets[t + 1])
        tile_gs = raw[flat_indices[s:e]]
        for j, attr_idx in enumerate(GROUP_D_ATTR_IDX):
            col = tile_gs[:, attr_idx]
            group_d[t, j * 4 + 0] = float(scipy_stats.skew(col))
            group_d[t, j * 4 + 1] = float(scipy_stats.kurtosis(col))
            group_d[t, j * 4 + 2] = float(np.median(col))
            group_d[t, j * 4 + 3] = float(col.max())

    return np.hstack([group_c, group_d])  # (N_tiles, 134)


# ─── static feature cache (Group C+D, shipped with the model) ─────────────────
#
# The (N_tiles, 134) static block is camera-invariant — same for every frame of a
# scene. Caching it as a model artifact means a deployed selector never touches the
# PLY at inference: per frame = Group A (camera scalars) + model.predict + greedy.
# Tiling-match is validated by N_tiles + a checksum of index_offsets (the tiling is
# deterministic from PLY+grid, so a matching checksum == identical tile ordering).

CACHE_FEATURE_NAMES: list = GROUP_C_NAMES + GROUP_D_NAMES  # 134


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


# ─── view-dependent features (Groups A + B) ───────────────────────────────────

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
        float32 array (N_tiles, 14)
    """
    N = len(tile_centroid_np)
    return np.column_stack([
        np.broadcast_to(cam_pos_np, (N, 3)).copy().astype(np.float32),
        np.broadcast_to(cam_fwd_np, (N, 3)).copy().astype(np.float32),
        np.full((N, 1), float(fov_x), dtype=np.float32),
        np.full((N, 1), float(fov_y), dtype=np.float32),
        tile_centroid_np.astype(np.float32),
        n_gs_per_tile.astype(np.float32)[:, None],
        distances_np.astype(np.float32)[:, None],
        visibility_np.astype(np.float32)[:, None],
    ])  # (N_tiles, 14)


@torch.no_grad()
def build_group_b(tile_index_offsets_t: torch.Tensor,
                  tile_flat_indices_t: torch.Tensor,
                  min_corners_t: torch.Tensor,
                  max_corners_t: torch.Tensor,
                  gs_xyz_t: torch.Tensor,
                  gs_opacity, gs_scale_0, gs_scale_1, gs_scale_2,
                  gs_rot_0, gs_rot_1, gs_rot_2, gs_rot_3,
                  cam_center_t: torch.Tensor,
                  world_view_t: torch.Tensor,
                  proj_t: torch.Tensor,
                  img_w: int, img_h: int,
                  n_gs_per_tile: np.ndarray,
                  device: str,
                  fov_x=None, fov_y=None) -> np.ndarray:
    """Build Group B features for one camera (12 handcrafted signals).

    Computes both screen-area and volume/d² weight modes, and complexity
    descriptors with both screen-area weights (context-dependent) and
    uniform weights (positional structure only).

    Returns:
        float32 array (N_tiles, 12)
    """
    # ── screen_area weights ─────────────────────────────────────────────────
    w_screen = uc.compute_gaussian_weights_v2(
        "screen_area",
        opacity=gs_opacity, scale_0=gs_scale_0, scale_1=gs_scale_1, scale_2=gs_scale_2,
        xyz=gs_xyz_t, cam_center=cam_center_t,
        rot_0=gs_rot_0, rot_1=gs_rot_1, rot_2=gs_rot_2, rot_3=gs_rot_3,
        world_view=world_view_t, proj=proj_t,
        img_w=img_w, img_h=img_h,
        fov_x=fov_x, fov_y=fov_y,
    ).to(device)

    # ── volume/d² weights ───────────────────────────────────────────────────
    w_vol_d2 = uc.compute_gaussian_weights_v2(
        "volume_over_d2",
        opacity=gs_opacity, scale_0=gs_scale_0, scale_1=gs_scale_1, scale_2=gs_scale_2,
        xyz=gs_xyz_t, cam_center=cam_center_t,
    ).to(device)

    # ── W_k and wbar_k ───────────────────────────────────────────────────────
    N_k_t = (tile_index_offsets_t[1:] - tile_index_offsets_t[:-1]).float()
    safe_n = N_k_t.clamp(min=1.0)

    W_k_screen_t, _ = uc.compute_tile_weights_and_counts(
        tile_index_offsets_t, tile_flat_indices_t, w_screen,
        w_norm="none", c_norm="none", w_mode="sum",
    )
    W_k_vol_d2_t, _ = uc.compute_tile_weights_and_counts(
        tile_index_offsets_t, tile_flat_indices_t, w_vol_d2,
        w_norm="none", c_norm="none", w_mode="sum",
    )

    wbar_k_screen = (W_k_screen_t / safe_n).cpu().numpy()
    W_k_screen    = W_k_screen_t.cpu().numpy()
    wbar_k_vol_d2 = (W_k_vol_d2_t / safe_n).cpu().numpy()
    W_k_vol_d2    = W_k_vol_d2_t.cpu().numpy()

    # ── complexity descriptors (screen-area weighted + positional) ───────────
    # Uniform (positional) weights: all ones per GS
    w_ones = torch.ones(gs_xyz_t.shape[0], dtype=torch.float32, device=device)

    complexity = {}
    for kind in uc.COMPLEXITY_KINDS:
        complexity[kind] = uc.compute_tile_complexity(
            kind, tile_index_offsets_t, tile_flat_indices_t, w_screen, gs_xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        ).cpu().numpy()
        complexity[f"{kind}_pos"] = uc.compute_tile_complexity(
            kind, tile_index_offsets_t, tile_flat_indices_t, w_ones, gs_xyz_t,
            min_corners=min_corners_t, max_corners=max_corners_t,
        ).cpu().numpy()

    return np.column_stack([
        wbar_k_screen, W_k_screen,
        wbar_k_vol_d2, W_k_vol_d2,
        complexity["eigenentropy"],
        complexity["omnivariance"],
        complexity["voxel_entropy"],
        complexity["spectral_energy"],
        complexity["eigenentropy_pos"],
        complexity["omnivariance_pos"],
        complexity["voxel_entropy_pos"],
        complexity["spectral_energy_pos"],
    ]).astype(np.float32)  # (N_tiles, 12)


def build_group_f(
    gs_data: dict,
    flat_indices: np.ndarray,
    index_offsets: np.ndarray,
    gs_xyz: np.ndarray,
    cam_pos_np: np.ndarray,
    sh_deg: int,
) -> np.ndarray:
    """Build Group F: SH-evaluated color per tile, per camera.

    Direction convention matches 3DGS renderer:
        dir = normalize(gs_xyz - cam_pos)  [Gaussian → camera, then negated inside SH]

    Args:
        gs_data:       gs.data dict from GaussianModelV2
        flat_indices:  (N_total_gs,) tile membership
        index_offsets: (N_tiles+1,) CSR offsets
        gs_xyz:        (N_gs, 3) float32
        cam_pos_np:    (3,) camera world position
        sh_deg:        SH degree of the model (0-3)

    Returns:
        float32 array (N_tiles, 6)
        columns: sh_mean_r/g/b, sh_std_r/g/b  (opacity-weighted)
    """
    N_gs   = len(gs_xyz)
    N_tiles = len(index_offsets) - 1
    n_coeff = (sh_deg + 1) ** 2

    # Build SH tensor (N_gs, 3, n_coeff).
    # PLY layout (standard 3DGS): f_rest_{(i-1)*3 + c} = i-th higher-order coeff, channel c.
    shs = np.zeros((N_gs, 3, n_coeff), dtype=np.float32)
    for c, dc_key in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
        shs[:, c, 0] = gs_data[dc_key]["data"].astype(np.float32)
    for i in range(1, n_coeff):
        for c in range(3):
            rest_key = f"f_rest_{(i - 1) * 3 + c}"
            if rest_key in gs_data:
                shs[:, c, i] = gs_data[rest_key]["data"].astype(np.float32)

    # Viewing directions: gs → cam (matches 3DGS renderer sign convention).
    dirs = gs_xyz - cam_pos_np[None, :]          # (N_gs, 3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-6)
    dirs = (dirs / norms).astype(np.float32)      # (N_gs, 3)

    # Evaluate SH → RGB, then apply +0.5 offset and clamp (matches 3DGS renderer).
    rgb = eval_sh(sh_deg, shs, dirs)              # (N_gs, 3)
    rgb = np.clip(rgb + 0.5, 0.0, 1.0).astype(np.float32)

    # Opacity weights: sigmoid(raw opacity).
    raw_op = gs_data["opacity"]["data"].astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-raw_op.clip(-30, 30)))).astype(np.float32)

    # Per-tile opacity-weighted mean and std of RGB.
    out = np.zeros((N_tiles, 6), dtype=np.float32)
    for t in range(N_tiles):
        s, e = int(index_offsets[t]), int(index_offsets[t + 1])
        if e <= s:
            continue
        idx  = flat_indices[s:e]
        w    = opacity[idx]                        # (n,)
        c    = rgb[idx]                            # (n, 3)
        w_sum = float(w.sum()) + 1e-8
        mean_rgb  = (w[:, None] * c).sum(axis=0) / w_sum
        diff      = c - mean_rgb[None, :]
        std_rgb   = np.sqrt((w[:, None] * diff ** 2).sum(axis=0) / w_sum)
        out[t, :3] = mean_rgb
        out[t, 3:] = std_rgb

    return out  # (N_tiles, 6)


# ─── training entry point ─────────────────────────────────────────────────────

def build_feature_matrix(oracle_npz_path: str) -> "tuple[pd.DataFrame, list]":
    """Build all features (A+B+C+D) from oracle_dq.npz for training.

    Returns:
        df:    DataFrame, one row per (camera, tile). Columns = ALL_FEATURE_NAMES
               + ["camera", "tile", "log_mse_loo"].
        names: ALL_FEATURE_NAMES (length 160).
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

    # ── static features (C+D): one-time ──────────────────────────────────────
    static_feats = build_static_features(gs.data, index_offsets, flat_indices)

    # ── per-camera features (A+B) ─────────────────────────────────────────────
    opacity   = gs.data["opacity"]["data"]
    scale_0   = gs.data["scale_0"]["data"]
    scale_1   = gs.data["scale_1"]["data"]
    scale_2   = gs.data["scale_2"]["data"]
    rot_0     = gs.data.get("rot_0", {}).get("data")
    rot_1     = gs.data.get("rot_1", {}).get("data")
    rot_2     = gs.data.get("rot_2", {}).get("data")
    rot_3     = gs.data.get("rot_3", {}).get("data")
    gs_xyz    = np.stack([gs.data["x"]["data"], gs.data["y"]["data"],
                          gs.data["z"]["data"]], axis=1)
    gs_xyz_t  = torch.tensor(gs_xyz, dtype=torch.float32, device=device)

    min_corners_t = torch.tensor(min_corners, dtype=torch.float32, device=device)
    max_corners_t = torch.tensor(max_corners, dtype=torch.float32, device=device)
    tile_centers_np = ((min_corners + max_corners) / 2.0).astype(np.float32)
    tile_centers_t  = torch.tensor(tile_centers_np, dtype=torch.float32, device=device)
    tile_index_offsets_t = torch.tensor(index_offsets, dtype=torch.long, device=device)
    tile_flat_indices_t  = torch.tensor(flat_indices,  dtype=torch.long, device=device)

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
        cam_center_t = cam.camera_center.to(device)

        visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
            min_corners_t, max_corners_t, cam, device=device
        )
        distances = uc.calculate_distances(tile_centers_t, cam_center_t)

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

        if rot_0 is None:
            raise RuntimeError("PLY missing rot_0..rot_3 — required for Group B (screen_area)")
        group_b = build_group_b(
            tile_index_offsets_t, tile_flat_indices_t,
            min_corners_t, max_corners_t,
            gs_xyz_t, opacity, scale_0, scale_1, scale_2,
            rot_0, rot_1, rot_2, rot_3,
            cam_center_t, wv, cam.projection_matrix,
            img_w, img_h, n_gs_per_tile, device,
            fov_x=getattr(cam, "FoVx", None), fov_y=getattr(cam, "FoVy", None),
        )

        group_f = build_group_f(
            gs.data, flat_indices, index_offsets,
            gs_xyz, cam_pos_np, gs.sh_deg,
        )

        X_cam = np.hstack([group_a, group_b, static_feats, group_f])  # (N_tiles, 166)

        row_idx = cam_to_row[cam_idx_int]
        mse_row = mse_loo[row_idx]   # (N_tiles,)
        # Avoid log(0) warning: replace non-positive values with 1.0 before log,
        # then mask them out to nan in the final array.
        _safe = np.where(mse_row > 0, mse_row, 1.0)
        log_mse = np.where(mse_row > 0, np.log(_safe), np.nan)

        cam_df = pd.DataFrame(
            X_cam, columns=ALL_FEATURE_NAMES, dtype=np.float32
        )
        cam_df["camera"] = cam_idx_int
        cam_df["tile"]   = tile_col
        cam_df["log_mse_loo"] = log_mse.astype(np.float32)
        rows.append(cam_df)

    df = pd.concat(rows, ignore_index=True)
    return df, ALL_FEATURE_NAMES
