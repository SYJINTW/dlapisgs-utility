"""Tile-utility scoring for view-conditioned 3DGS streaming.

Computes U(k, l) = log(β·(l+1)) · (v_k / d_k) [· W_k] [· C_k] for each
(tile, LOD-level) pair and returns them sorted by descending priority.

Sub-components:
  - Gaussian weights  : per-Gaussian saliency (volume, screen area, …)
  - Tile aggregation  : W_k (weight sum/mean) and C_k (complexity) per tile
  - Utility scoring   : ranked (tile, lod) output consumed by the packing step
"""

import math
import torch
import numpy as np
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "Frustum-for-3DGS"))
import visibility_AABB_pytorch
sys.path.insert(0, str(WORKSPACE_ROOT / "GGSP"))
import tiling


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INVISIBLE_PRIORITY_EPS = 1e-2   # >0 to prevent invisible-tile starvation
DISTANCE_EPS = 1e-3

W_MODES = ("sum", "mean")
WEIGHT_MODES = ("volume", "volume_over_d2", "screen_area", "random")
NORM_MODES = ("none", "max", "minmax", "log1p", "sum")

COMPLEXITY_KINDS = ("eigenentropy", "omnivariance", "voxel_entropy", "spectral_energy")
COMPLEXITY_SPARSE_GUARD = 20    # min GS for reliable covariance / voxel-entropy estimate
COMPLEXITY_SPECTRAL_FC  = 2.0   # frequency cutoff (cycles); N=8 → Nyquist ≈ 4√3 ≈ 6.9


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _t(x, device):
    """Cast x to float32 tensor on device (no-copy if already correct)."""
    return torch.as_tensor(x, dtype=torch.float32).to(device)


def _make_seg_ids(num_tiles, sizes, device):
    """Return (M_total,) tile-index tensor: tile k repeated sizes[k] times."""
    return torch.repeat_interleave(torch.arange(num_tiles, device=device), sizes)


def _voxelize(pts_n, w, seg_ids, num_tiles, voxel_n, device):
    """Scatter weighted GS positions into a (num_tiles, voxel_n³) occupancy grid."""
    N3 = voxel_n ** 3
    vox_ijk = (pts_n.clamp(0.0, 1.0 - 1e-6) * voxel_n).long()
    vox_flat = (vox_ijk[:, 0] * voxel_n * voxel_n
                + vox_ijk[:, 1] * voxel_n
                + vox_ijk[:, 2])
    occ = torch.zeros(num_tiles * N3, dtype=torch.float32, device=device)
    occ.scatter_add_(0, seg_ids * N3 + vox_flat, w)
    return occ.view(num_tiles, N3)


def _compute_base_scores(visibility_mask_tensor, tile_distances_tensor):
    """Per-tile score v_k / d_k; invisible tiles get INVISIBLE_PRIORITY_EPS instead of 1."""
    vis_factor = torch.where(visibility_mask_tensor, 1.0, INVISIBLE_PRIORITY_EPS)
    return vis_factor / (tile_distances_tensor + DISTANCE_EPS)


def normalize_term(x, mode="none", eps=1e-12):
    """Normalize a 1-D tensor of nonneg scores. Modes: none | max | minmax | log1p | sum."""
    if mode == "none" or x is None:
        return x
    if mode not in NORM_MODES:
        raise ValueError(f"unknown norm mode '{mode}'; valid: {NORM_MODES}")
    if mode == "max":
        m = x.max()
        return x / m if m > 0 else x
    if mode == "minmax":
        lo, hi = x.min(), x.max()
        span = hi - lo
        return (x - lo) / span if span > eps else torch.zeros_like(x)
    if mode == "log1p":
        lx = torch.log1p(x)
        m = lx.max()
        return lx / m if m > 0 else lx
    if mode == "sum":
        s = x.sum()
        return x / s if s > 0 else x
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Gaussian weights
# ---------------------------------------------------------------------------


def project_covariance_2d(xyz, scale_0, scale_1, scale_2,
                          rot_0, rot_1, rot_2, rot_3,
                          world_view, proj, img_w, img_h,
                          fov_x=None, fov_y=None, near_plane: float = 0.2):
    """Project 3D Gaussian covariances to screen space: Σ_2D = J·W·Σ_3D·W^T·J^T.

    Mirrors EWA splatting math in 3DGS forward.cu. Returns (det(Σ_2D), in_front)
    where in_front is True for Gaussians in front of the near plane.
    """
    device = xyz.device
    dtype = xyz.dtype
    N = xyz.shape[0]

    s0 = torch.exp(scale_0).to(device=device, dtype=dtype)
    s1 = torch.exp(scale_1).to(device=device, dtype=dtype)
    s2 = torch.exp(scale_2).to(device=device, dtype=dtype)
    q = torch.stack([rot_0, rot_1, rot_2, rot_3], dim=1).to(device=device, dtype=dtype)
    q = q / (q.norm(dim=1, keepdim=True).clamp(min=1e-8))
    r, x, y, z = q.unbind(dim=1)
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - r*z),     2*(x*z + r*y)],     dim=1),
        torch.stack([2*(x*y + r*z),     1 - 2*(x*x + z*z), 2*(y*z - r*x)],     dim=1),
        torch.stack([2*(x*z - r*y),     2*(y*z + r*x),     1 - 2*(x*x + y*y)], dim=1),
    ], dim=1)
    S = torch.zeros((N, 3, 3), device=device, dtype=dtype)
    S[:, 0, 0] = s0; S[:, 1, 1] = s1; S[:, 2, 2] = s2
    cov3d = (R @ S) @ (R @ S).transpose(1, 2)

    p_view = torch.cat([xyz, torch.ones((N, 1), device=device, dtype=dtype)], dim=1) @ world_view
    # near_plane must be consistent: in_front and tz clamp must agree.
    # Clamping tz to near_plane while marking p_view[:,2] > near_plane as in_front
    # prevents the Jacobian blowup that occurs if tz → 0.
    in_front = p_view[:, 2] > near_plane
    tz = p_view[:, 2].clamp(min=near_plane)
    tx = p_view[:, 0]
    ty = p_view[:, 1]

    if fov_x is not None and fov_y is not None:
        focal_x = 0.5 * img_w / math.tan(0.5 * fov_x)
        focal_y = 0.5 * img_h / math.tan(0.5 * fov_y)
        tan_fovx = math.tan(0.5 * fov_x)
        tan_fovy = math.tan(0.5 * fov_y)
    else:
        focal_x = 0.5 * img_w * float(proj[0, 0].item())
        focal_y = 0.5 * img_h * float(proj[1, 1].item())
        tan_fovx = 1.0 / float(proj[0, 0].item())
        tan_fovy = 1.0 / float(proj[1, 1].item())

    # Clamp off-axis angle before computing J — mirrors forward.cu:82-87.
    tx = (tx / tz).clamp(-1.3 * tan_fovx, 1.3 * tan_fovx) * tz
    ty = (ty / tz).clamp(-1.3 * tan_fovy, 1.3 * tan_fovy) * tz

    J = torch.zeros((N, 2, 3), device=device, dtype=dtype)
    J[:, 0, 0] = focal_x / tz;       J[:, 0, 2] = -focal_x * tx / (tz * tz)
    J[:, 1, 1] = focal_y / tz;       J[:, 1, 2] = -focal_y * ty / (tz * tz)

    JW = J @ world_view[:3, :3].to(device=device, dtype=dtype).unsqueeze(0)
    cov2d = JW @ cov3d @ JW.transpose(1, 2)
    # 3DGS adds 0.3 px isotropic blur to avoid singular covariance.
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]).clamp(min=0.0)
    return det, in_front


def compute_gaussian_weights_v2(weight_mode, *, opacity, scale_0, scale_1, scale_2,
                                xyz=None, cam_center=None,
                                rot_0=None, rot_1=None, rot_2=None, rot_3=None,
                                world_view=None, proj=None, img_w=None, img_h=None,
                                fov_x=None, fov_y=None, random_seed=42):
    """Per-Gaussian weight under one of the named WEIGHT_MODES.

    volume: sigmoid(o)·det(Σ)^0.5  |  volume_over_d2: volume/d²
    screen_area: sigmoid(o)·π·√det(Σ_2D)  |  random: U(0,1) seeded baseline
    """
    if not isinstance(opacity, torch.Tensor):
        opacity = torch.as_tensor(opacity, dtype=torch.float32)
        scale_0 = torch.as_tensor(scale_0, dtype=torch.float32)
        scale_1 = torch.as_tensor(scale_1, dtype=torch.float32)
        scale_2 = torch.as_tensor(scale_2, dtype=torch.float32)
    o_i = torch.sigmoid(opacity)

    if weight_mode == "random":
        gen = torch.Generator(device=o_i.device)
        gen.manual_seed(random_seed)
        return torch.rand(len(o_i), generator=gen, device=o_i.device)

    # det(Σ_3D)^0.5 = exp(scale_0 + scale_1 + scale_2)
    vol = torch.exp(scale_0 + scale_1 + scale_2)

    if weight_mode == "volume":
        return o_i * vol

    if weight_mode == "volume_over_d2":
        if xyz is None or cam_center is None:
            raise ValueError("volume_over_d2 requires xyz and cam_center")
        d2 = ((_t(xyz, o_i.device) - _t(cam_center, o_i.device).unsqueeze(0)) ** 2).sum(dim=1).clamp(min=1e-6)
        return o_i * vol / d2

    if weight_mode == "screen_area":
        required = [rot_0, rot_1, rot_2, rot_3, xyz, world_view, proj, img_w, img_h]
        if any(r is None for r in required):
            raise ValueError("screen_area requires rot_0..rot_3, xyz, world_view, proj, img_w, img_h")
        dev = o_i.device
        det2d, in_front = project_covariance_2d(
            _t(xyz, dev), scale_0, scale_1, scale_2,
            _t(rot_0, dev), _t(rot_1, dev), _t(rot_2, dev), _t(rot_3, dev),
            _t(world_view, dev), _t(proj, dev), img_w, img_h, fov_x=fov_x, fov_y=fov_y,
        )
        area = math.pi * torch.sqrt(det2d.clamp(min=0.0))
        area = torch.where(in_front, area, torch.zeros_like(area))
        return o_i * area.clamp(max=float(img_w * img_h))

    raise ValueError(f"unknown weight_mode '{weight_mode}'; valid: {WEIGHT_MODES}")


# ---------------------------------------------------------------------------
# Tile aggregation
# ---------------------------------------------------------------------------

def compute_tile_weights_and_counts(tile_index_offsets, tile_flat_indices, w_gi,
                                    w_norm="none", c_norm="max", w_mode="sum"):
    """Compute W_k (aggregate Gaussian weight) and C_k (GS count) for all tiles.

    w_mode 'sum': W_k = Σw (scales with tile density).
    w_mode 'mean': W_k = Σw/N (size-invariant per-Gaussian quality).
    Both outputs are normalized by w_norm / c_norm before return.
    """
    num_tiles = len(tile_index_offsets) - 1
    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    C_k = sizes.float()

    W_k = torch.zeros(num_tiles, dtype=torch.float32, device=w_gi.device)
    if tile_flat_indices.numel() > 0:
        seg_ids = _make_seg_ids(num_tiles, sizes, w_gi.device)
        W_k.scatter_add_(0, seg_ids, w_gi[tile_flat_indices])

    if w_mode == "mean":
        W_k = W_k / sizes.float().clamp(min=1).to(W_k.device)

    return normalize_term(W_k, w_norm), normalize_term(C_k, c_norm)


def compute_tile_complexity(c_kind, tile_index_offsets, tile_flat_indices,
                            w_gi, xyz, *, min_corners, max_corners,
                            sparse_guard=COMPLEXITY_SPARSE_GUARD, voxel_n=8):
    """Per-tile structural complexity descriptor (unnormalized; caller applies normalize_term).

    c_kind options:
      eigenentropy   : Shannon entropy of weighted-covariance eigenvalue distribution
      omnivariance   : geometric mean of eigenvalues = det(cov)^(1/3)
      voxel_entropy  : entropy of weighted voxel occupancy (voxel_n³ grid)
      spectral_energy: fraction of voxel-FFT power above COMPLEXITY_SPECTRAL_FC cycles
    Tiles with fewer than sparse_guard Gaussians get 0 (statistics unreliable).
    """
    if c_kind not in COMPLEXITY_KINDS:
        raise ValueError(f"unknown c_kind '{c_kind}'; valid: {COMPLEXITY_KINDS}")

    num_tiles = len(tile_index_offsets) - 1
    device = w_gi.device
    sizes = tile_index_offsets[1:] - tile_index_offsets[:-1]
    out = torch.zeros(num_tiles, dtype=torch.float32, device=device)

    if tile_flat_indices.numel() == 0:
        return out

    seg_ids = _make_seg_ids(num_tiles, sizes, device)  # (M_total,)
    pts = xyz[tile_flat_indices]   # (M_total, 3)
    w   = w_gi[tile_flat_indices]  # (M_total,)

    lo = min_corners[seg_ids]  # (M_total, 3)
    hi = max_corners[seg_ids]  # (M_total, 3)
    pts_n = (pts - lo) / (hi - lo).clamp(min=1e-6)  # (M_total, 3) in [0,1]^3

    if c_kind in ("eigenentropy", "omnivariance"):
        wsum = torch.zeros(num_tiles, dtype=torch.float32, device=device)
        wsum.scatter_add_(0, seg_ids, w)

        mu = torch.zeros(num_tiles, 3, dtype=torch.float32, device=device)
        for d in range(3):
            mu[:, d].scatter_add_(0, seg_ids, w * pts_n[:, d])
        mu = mu / wsum.unsqueeze(1).clamp(min=1e-12)

        diff = pts_n - mu[seg_ids]
        M_total = tile_flat_indices.numel()
        w_outer = w.unsqueeze(1).unsqueeze(2) * diff.unsqueeze(2) * diff.unsqueeze(1)
        M_cov = torch.zeros(num_tiles, 9, dtype=torch.float32, device=device)
        M_cov.scatter_add_(0, seg_ids.unsqueeze(1).expand(-1, 9), w_outer.view(M_total, 9))
        M_cov = M_cov.view(num_tiles, 3, 3) / wsum.view(num_tiles, 1, 1).clamp(1e-12)

        lam = torch.linalg.eigvalsh(M_cov).clamp(min=0.0)  # (N, 3) ascending
        lsum = lam.sum(dim=1)
        valid = (sizes >= sparse_guard) & (lsum > 1e-12)

        if c_kind == "eigenentropy":
            lam_n = (lam / lsum.unsqueeze(1).clamp(1e-12)).clamp(min=1e-12)
            out[valid] = -(lam_n * lam_n.log()).sum(dim=1)[valid]
        else:
            out[valid] = lam.prod(dim=1).clamp(min=0.0)[valid] ** (1.0 / 3.0)

    if c_kind in ("voxel_entropy", "spectral_energy"):
        occ = _voxelize(pts_n, w, seg_ids, num_tiles, voxel_n, device)  # (num_tiles, N³)
        dense = sizes >= sparse_guard

        if c_kind == "voxel_entropy":
            p = occ / occ.sum(dim=1, keepdim=True).clamp(min=1e-12)
            log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
            out[dense] = -(p * log_p).sum(dim=1)[dense]

        else:
            F = torch.fft.fftn(occ.view(num_tiles, voxel_n, voxel_n, voxel_n),
                               dim=(-3, -2, -1))                           # complex
            power = F.real ** 2 + F.imag ** 2                             # |F|²

            freq = torch.fft.fftfreq(voxel_n, device=device) * voxel_n   # (N,)
            fx, fy, fz = torch.meshgrid(freq, freq, freq, indexing="ij")  # (N, N, N)
            high_mask = ((fx**2 + fy**2 + fz**2).sqrt() > COMPLEXITY_SPECTRAL_FC).unsqueeze(0)

            total = power.sum(dim=(-3, -2, -1))                           # (num_tiles,)
            out[dense] = (power * high_mask).sum(dim=(-3, -2, -1))[dense] / total[dense].clamp(1e-12)

    return out


# ---------------------------------------------------------------------------
# Utility scoring
# ---------------------------------------------------------------------------

def calculate_distances(tile_centers_tensor, cam_center_tensor):
    """Euclidean distance from cam_center to each tile center."""
    return torch.norm(tile_centers_tensor - cam_center_tensor.unsqueeze(0), dim=1)


def calculate_utility_param(
    visibility_mask_tensor,
    tile_distances_tensor,
    num_of_level,
    weight_sum_tensor=None,
    complexity_tensor=None,
    include_lod=True,
    include_w=False,
    include_c=False,
    beta=10.0,
):
    """Rank (tile, lod) pairs by U(k,l) = log(β·(l+1)) · (v_k/d_k) [·W_k] [·C_k].

    Returns ndarray shape (N*num_of_level, 2) of (tile_index, lod_level) sorted
    by descending utility. With include_lod=False or num_of_level=1, all lod=0.
    """
    if include_w and weight_sum_tensor is None:
        raise ValueError("weight_sum_tensor required when include_w=True")
    if include_c and complexity_tensor is None:
        raise ValueError("complexity_tensor required when include_c=True")

    scores = _compute_base_scores(visibility_mask_tensor, tile_distances_tensor)
    if include_w:
        scores = scores * weight_sum_tensor
    if include_c:
        scores = scores * complexity_tensor

    if not include_lod or num_of_level <= 1:
        idx = torch.argsort(scores, descending=True)
        return torch.stack((idx, torch.zeros_like(idx)), dim=1).cpu().numpy()

    levels = torch.arange(num_of_level, device=scores.device, dtype=torch.float32)
    log_weights = torch.log(beta * (levels + 1.0))          # (num_of_level,)
    utility_matrix = scores.unsqueeze(1) * log_weights.unsqueeze(0)  # (N, num_of_level)

    flat = torch.argsort(utility_matrix.view(-1), descending=True)
    return torch.stack((flat // num_of_level, flat % num_of_level), dim=1).cpu().numpy()


def calculate_utility_baseline(visibility_mask_tensor, tile_distances_tensor):
    """Rank tiles by visibility/distance only (no LOD, weights, or complexity)."""
    return calculate_utility_param(
        visibility_mask_tensor, tile_distances_tensor,
        num_of_level=1, include_lod=False,
    )
