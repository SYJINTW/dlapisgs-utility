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


# Numerical constants
INVISIBLE_PRIORITY_EPS = 1e-2   # priority weight for invisible tiles (kept >0 to avoid starvation)
DISTANCE_EPS = 1e-3             # added to distance to prevent division by zero

NORM_MODES = ("none", "max", "minmax", "log1p", "sum")
# Per-Gaussian weight formulas. Each name describes the math literally:
#   det_gamma_over_d2 : sigmoid(o) * det(Σ)^gamma / d^2     (current default; gamma tunable)
#   volume            : sigmoid(o) * det(Σ)^0.5             (i.e. s_x·s_y·s_z, the 1-σ ellipsoid volume proxy)
#   volume_over_d2    : sigmoid(o) * det(Σ)^0.5 / d^2
#   screen_area       : sigmoid(o) * π · sqrt(det(Σ_2D))    (true projected footprint via EWA Jacobian)
WEIGHT_MODES = ("det_gamma_over_d2", "volume", "volume_over_d2", "screen_area")


def normalize_term(x, mode="none", eps=1e-12):
    """Normalize a 1-D tensor of nonnegative tile/Gaussian scores.

    Supported modes:
      - "none":   identity
      - "max":    x / max(x)
      - "minmax": (x - min) / (max - min)
      - "log1p":  log(1 + x)
      - "sum":    x / sum(x)
    """
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
        return torch.log1p(x)
    if mode == "sum":
        s = x.sum()
        return x / s if s > 0 else x
    raise AssertionError("unreachable")


def _compute_base_scores(visibility_mask_tensor, tile_distances_tensor):
    vis_factor = torch.where(visibility_mask_tensor, 1.0, INVISIBLE_PRIORITY_EPS)
    dist_factor = 1.0 / (tile_distances_tensor + DISTANCE_EPS)
    return vis_factor * dist_factor


def calculate_utility_baseline(visibility_mask_tensor, tile_distances_tensor):
    """
    Compute the baseline utility using visibility and distance only.

    Args:
        visibility_mask_tensor: length-N tensor; True if the tile is visible.
        tile_distances_tensor:  length-N tensor; Euclidean distance from camera to each tile center.

    Returns:
        np.array: shape (N, 2). Pairs of (tile_index, lod_level=0), sorted by score.
    """
    base_scores = _compute_base_scores(visibility_mask_tensor, tile_distances_tensor)
    sorted_indices = torch.argsort(base_scores, descending=True)
    lod_levels = torch.zeros_like(sorted_indices, dtype=torch.long)
    result_tensor = torch.stack((sorted_indices, lod_levels), dim=1)
    return result_tensor.cpu().numpy()

# Our method with configurable factors and LOD consideration
def calculate_utility_param(
    visibility_mask_tensor, # [TODO] change to probability in [0,1] later
    tile_distances_tensor,
    num_of_level,
    weight_sum_tensor=None,
    complexity_tensor=None,
    include_lod=True,
    include_w=False,
    include_c=False,
    beta=10.0,
):
    """
    U(k, l) = log(β·(l+1)) · (v_k / d_k) [· W_k] [· C_k]

    Args:
        visibility_mask_tensor: length-N tensor; True if the tile is visible.
        tile_distances_tensor:  length-N tensor; Euclidean distance from camera.
        num_of_level (int):     Total number of LOD layers (1 = plain 3DGS, no LOD effect).
        weight_sum_tensor:      length-N tensor; W_k (required when include_w=True).
        complexity_tensor:      length-N tensor; C_k (required when include_c=True).
        include_lod (bool):     If True, weight (tile, lod) pairs by log(β·(l+1)).
        include_w (bool):       Multiply scores by W_k.
        include_c (bool):       Multiply scores by C_k.
        beta (float):           Controls diminishing-returns curve shape in the log term.

    Returns:
        np.ndarray: shape (N, 2) or (N*num_of_level, 2) — (tile_index, lod_level) sorted
                    by descending utility.
    """
    if include_w and weight_sum_tensor is None:
        raise ValueError("weight_sum_tensor is required when include_w=True")
    if include_c and complexity_tensor is None:
        raise ValueError("complexity_tensor is required when include_c=True")

    base_scores = _compute_base_scores(visibility_mask_tensor, tile_distances_tensor)

    if include_w:
        base_scores = base_scores * weight_sum_tensor
    if include_c:
        base_scores = base_scores * complexity_tensor

    if not include_lod or num_of_level <= 1:
        sorted_indices = torch.argsort(base_scores, descending=True)
        lod_levels = torch.zeros_like(sorted_indices, dtype=torch.long)
        result_tensor = torch.stack((sorted_indices, lod_levels), dim=1)
        return result_tensor.cpu().numpy()

    # log(β·(l+1)) for l in {0, 1, ..., num_of_level-1}
    levels = torch.arange(num_of_level, device=base_scores.device, dtype=torch.float32)
    log_weights = torch.log(beta * (levels + 1.0))          # shape (num_of_level,)
    utility_matrix = base_scores.unsqueeze(1) * log_weights.unsqueeze(0)  # (N, num_of_level)

    flattened_scores = utility_matrix.view(-1)
    sorted_1d_indices = torch.argsort(flattened_scores, descending=True)
    tile_indices = sorted_1d_indices // num_of_level
    lod_levels = sorted_1d_indices % num_of_level
    result_tensor = torch.stack((tile_indices, lod_levels), dim=1)
    return result_tensor.cpu().numpy()


def compute_gaussian_weights(opacity, scale_0, scale_1, scale_2, gamma=1.0,
                             xyz=None, cam_center=None):
    """
    Compute individual Gaussian weights.

    View-independent (xyz/cam_center not given):
        w(g_i) = sigmoid(opacity) * det(Sigma)^gamma

    View-dependent (xyz and cam_center given):
        w(g_i) = sigmoid(opacity) * det(Sigma)^gamma / d(g_i, cam)^2

    The 1/d^2 term approximates projected pixel footprint: a Gaussian's 2D
    covariance scales as det(Sigma_3D)/d^2, so distant large Gaussians are
    correctly down-weighted relative to nearby ones.

    Args:
        opacity, scale_0/1/2: raw PLY attributes (Tensor or ndarray)
        gamma: volume exponent
        xyz: (N, 3) Gaussian world positions (Tensor or ndarray), optional
        cam_center: (3,) camera world position (Tensor or ndarray), optional
    """
    if not isinstance(opacity, torch.Tensor):
        opacity = torch.tensor(opacity, dtype=torch.float32)
        scale_0 = torch.tensor(scale_0, dtype=torch.float32)
        scale_1 = torch.tensor(scale_1, dtype=torch.float32)
        scale_2 = torch.tensor(scale_2, dtype=torch.float32)

    o_i = torch.sigmoid(opacity)
    det_sigma = torch.exp(2.0 * (scale_0 + scale_1 + scale_2))
    w_gi = o_i * (det_sigma ** gamma)

    if xyz is not None and cam_center is not None:
        if not isinstance(xyz, torch.Tensor):
            xyz = torch.tensor(xyz, dtype=torch.float32)
        if not isinstance(cam_center, torch.Tensor):
            cam_center = torch.tensor(cam_center, dtype=torch.float32)
        xyz = xyz.to(w_gi.device)
        cam_center = cam_center.to(w_gi.device)
        d2 = ((xyz - cam_center.unsqueeze(0)) ** 2).sum(dim=1).clamp(min=1e-6)
        w_gi = w_gi / d2

    return w_gi


def compute_tile_weights_and_counts(tile_index_offsets, tile_flat_indices, w_gi,
                                    w_norm="none", c_norm="max"):
    """
    Computes W_k (aggregate weight) and C_k (count) for all tiles.

    Both W_k and C_k are normalized according to `w_norm` and `c_norm`. Defaults
    preserve historical behavior: W_k unnormalized, C_k max-normalized.

    Returns:
        W_k (torch.Tensor): length-N
        C_k (torch.Tensor): length-N
    """
    num_tiles = len(tile_index_offsets) - 1
    W_k = torch.zeros(num_tiles, dtype=torch.float32, device=w_gi.device)
    C_k = torch.zeros(num_tiles, dtype=torch.float32, device=w_gi.device)

    for i in range(num_tiles):
        start = tile_index_offsets[i]
        end = tile_index_offsets[i+1]
        indices_for_tile = tile_flat_indices[start:end]
        if len(indices_for_tile) > 0:
            C_k[i] = len(indices_for_tile)
            W_k[i] = w_gi[indices_for_tile].sum()

    W_k = normalize_term(W_k, w_norm)
    C_k = normalize_term(C_k, c_norm)
    return W_k, C_k


def project_covariance_2d(xyz, scale_0, scale_1, scale_2,
                          rot_0, rot_1, rot_2, rot_3,
                          world_view, proj, img_w, img_h,
                          fov_x=None, fov_y=None):
    """Project 3D Gaussian covariances to 2D screen-space covariances.

    Mirrors the math in 3DGS's forward.cu / EWA splatting:
        Σ_2D = J · W · Σ_3D · W^T · J^T
    where W is the upper-left 3x3 of world_view, and J is the Jacobian of the
    perspective projection at each Gaussian's view-space position.

    Inputs are torch tensors, all on the same device. world_view and proj follow
    the 3DGS convention (pre-transposed, row-vector left-multiplication).

    Returns:
        cov2d_det (N,) tensor — determinant of the 2D covariance (units: px^2 if
        fov_x/fov_y are supplied; otherwise unit-less NDC).
        in_front (N,) bool tensor — True if the Gaussian is in front of the camera.
    """
    device = xyz.device
    dtype = xyz.dtype
    N = xyz.shape[0]

    # Build per-Gaussian Σ_3D from scale + rotation (quaternion).
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
    ], dim=1)  # (N, 3, 3)
    S = torch.zeros((N, 3, 3), device=device, dtype=dtype)
    S[:, 0, 0] = s0
    S[:, 1, 1] = s1
    S[:, 2, 2] = s2
    M = R @ S
    cov3d = M @ M.transpose(1, 2)  # (N, 3, 3)

    # View-space positions p_v = [xyz, 1] @ world_view; take first 3 components.
    ones = torch.ones((N, 1), device=device, dtype=dtype)
    p_h = torch.cat([xyz, ones], dim=1)
    p_view = p_h @ world_view  # (N, 4)
    tz = p_view[:, 2].clamp(min=1e-6)
    tx = p_view[:, 0]
    ty = p_view[:, 1]
    in_front = p_view[:, 2] > 0

    # Focal lengths in pixels. If fov_x/fov_y given, derive; else assume
    # the projection matrix follows the 3DGS convention where focal_x =
    # img_w / (2 tan(fov_x/2)). We accept either path.
    if fov_x is not None and fov_y is not None:
        focal_x = 0.5 * img_w / math.tan(0.5 * fov_x)
        focal_y = 0.5 * img_h / math.tan(0.5 * fov_y)
    else:
        # proj is row-major, pre-transposed; the (0,0) entry encodes 1/tan(fov_x/2).
        focal_x = 0.5 * img_w * float(proj[0, 0].item())
        focal_y = 0.5 * img_h * float(proj[1, 1].item())

    # Jacobian J of the projection at (tx, ty, tz). 2x3.
    J = torch.zeros((N, 2, 3), device=device, dtype=dtype)
    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -focal_x * tx / (tz * tz)
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -focal_y * ty / (tz * tz)

    # W: upper-left 3x3 of the view transform. world_view is row-major
    # pre-transposed (clip = p @ wv @ proj), so the rotation block is
    # world_view[:3, :3] when read as row-vector left-multiplication.
    W = world_view[:3, :3].to(device=device, dtype=dtype)
    # JW = J · W, shape (N, 2, 3)
    JW = J @ W.unsqueeze(0)

    cov2d = JW @ cov3d @ JW.transpose(1, 2)  # (N, 2, 2)
    # A low-pass filter (3DGS adds 0.3 px isotropic blur to avoid singular cov).
    cov2d[:, 0, 0] = cov2d[:, 0, 0] + 0.3
    cov2d[:, 1, 1] = cov2d[:, 1, 1] + 0.3
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det = det.clamp(min=0.0)
    return det, in_front


def compute_gaussian_weights_v2(weight_mode, *, opacity, scale_0, scale_1, scale_2,
                                xyz=None, cam_center=None,
                                rot_0=None, rot_1=None, rot_2=None, rot_3=None,
                                world_view=None, proj=None, img_w=None, img_h=None,
                                fov_x=None, fov_y=None):
    """Per-Gaussian weight under one of three named modes.

    Modes:
      - "volume":         sigmoid(o) * det(Σ_3D)^0.5  (≈ Gaussian "volume")
      - "volume_over_d2": sigmoid(o) * det(Σ_3D)^0.5 / d^2  (view-dependent)
      - "screen_area":    sigmoid(o) * π · √det(Σ_2D)  (projected footprint area)
    """
    if not isinstance(opacity, torch.Tensor):
        opacity = torch.as_tensor(opacity, dtype=torch.float32)
        scale_0 = torch.as_tensor(scale_0, dtype=torch.float32)
        scale_1 = torch.as_tensor(scale_1, dtype=torch.float32)
        scale_2 = torch.as_tensor(scale_2, dtype=torch.float32)
    o_i = torch.sigmoid(opacity)
    # det(Σ_3D)^0.5 = exp(scale_0 + scale_1 + scale_2)  (since Σ = R S^2 R^T → det = exp(2Σs))
    vol = torch.exp(scale_0 + scale_1 + scale_2)

    if weight_mode == "volume":
        return o_i * vol

    if weight_mode == "volume_over_d2":
        if xyz is None or cam_center is None:
            raise ValueError("volume_over_d2 requires xyz and cam_center")
        xyz_t = torch.as_tensor(xyz, dtype=torch.float32).to(o_i.device)
        cc_t = torch.as_tensor(cam_center, dtype=torch.float32).to(o_i.device)
        d2 = ((xyz_t - cc_t.unsqueeze(0)) ** 2).sum(dim=1).clamp(min=1e-6)
        return o_i * vol / d2

    if weight_mode == "screen_area":
        required = [rot_0, rot_1, rot_2, rot_3, xyz, world_view, proj, img_w, img_h]
        if any(r is None for r in required):
            raise ValueError("screen_area requires rot_0..rot_3, xyz, world_view, proj, img_w, img_h")
        xyz_t = torch.as_tensor(xyz, dtype=torch.float32).to(o_i.device)
        wv = torch.as_tensor(world_view, dtype=torch.float32).to(o_i.device)
        pj = torch.as_tensor(proj, dtype=torch.float32).to(o_i.device)
        r0 = torch.as_tensor(rot_0, dtype=torch.float32).to(o_i.device)
        r1 = torch.as_tensor(rot_1, dtype=torch.float32).to(o_i.device)
        r2 = torch.as_tensor(rot_2, dtype=torch.float32).to(o_i.device)
        r3 = torch.as_tensor(rot_3, dtype=torch.float32).to(o_i.device)
        det2d, in_front = project_covariance_2d(
            xyz_t, scale_0, scale_1, scale_2, r0, r1, r2, r3,
            wv, pj, img_w, img_h, fov_x=fov_x, fov_y=fov_y,
        )
        # Area of 1-sigma footprint = π · √det(Σ_2D). Hide back-of-camera Gaussians.
        area = math.pi * torch.sqrt(det2d.clamp(min=0.0))
        area = torch.where(in_front, area, torch.zeros_like(area))
        return o_i * area

    raise ValueError(f"unknown weight_mode '{weight_mode}'; valid: {WEIGHT_MODES}")

def calculate_distances(tile_centers_tensor, cam_center_tensor):
    return torch.norm(tile_centers_tensor - cam_center_tensor.unsqueeze(0), dim=1)


# main func for unit test
def main():
    base_dir = Path("/home/syjintw/Desktop/NUS/dlapisgs-output/longdress/opacity")
    res1_frame0_path = base_dir / "longdress_res1" / "dynamic_1051" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    user_traj_path = Path("/home/syjintw/Desktop/NUS/AABB_frustum/sample_data/camera_trace/trace1.json")
    tile_metadata_path = WORKSPACE_ROOT / "GGSP" / "tiles_metadata_allFrames.npz"
    
    # Load tiles from NPZ
    tiles_info = tiling.load_tiles_from_npz(tile_metadata_path)
    
    # Change tiles to pyTorch tensors for utility calculation
    min_corners_tensor = torch.tensor(tiles_info["min_corners"], dtype=torch.float32, device="cuda") # [N, 3]
    max_corners_tensor = torch.tensor(tiles_info["max_corners"], dtype=torch.float32, device="cuda") # [N, 3]
    
    # AABB centers for utility calculation
    tile_centers_tensor = (min_corners_tensor + max_corners_tensor) / 2.0 # [N, 3]

    # Load camera trajectory
    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(user_traj_path, 800, 800)
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    
    # For each camera, calculate visible tiles and their utilities
    for cam in cameras:
        cam_center_tensor = cam.camera_center.to("cuda")
        tile_distances_tensor = calculate_distances(tile_centers_tensor, cam_center_tensor)
        print(f"Distances from camera {cam.uid} to tile centers:", tile_distances_tensor.cpu().numpy())
        
        visibility_mask_tensor = visibility_AABB_pytorch.batched_check_tiles_visible(min_corners_tensor, max_corners_tensor, cam, device="cuda")    
        visible_indices = torch.where(visibility_mask_tensor)[0].cpu().tolist()
        visible_tile_idxs = [tiles_info["tile_idxs"][i] for i in visible_indices]
        print(f"Camera {cam.uid} can see tiles: {visible_tile_idxs}")
        # print(visible_mask)
        
        tile_utilities = calculate_utility_param(
            visibility_mask_tensor,
            tile_distances_tensor,
            num_of_level=3,
            include_lod=True,
            include_w=False,
            include_c=False,
        )
        print(f"Camera {cam.uid} utility scores (tile_index, lod_level):")
        print(tile_utilities[:])
        break
    
if __name__ == "__main__":
    main()