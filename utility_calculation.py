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


def compute_tile_weights_and_counts(tile_index_offsets, tile_flat_indices, w_gi):
    """
    Computes W_k (aggregate weight) and C_k (normalized count) for all tiles.
    
    Returns:
        W_k (torch.Tensor): length-N
        C_k (torch.Tensor): length-N, normalized to [0, 1] across tiles
    """
    num_tiles = len(tile_index_offsets) - 1
    W_k = torch.zeros(num_tiles, dtype=torch.float32, device=w_gi.device)
    C_k = torch.zeros(num_tiles, dtype=torch.float32, device=w_gi.device)
    
    for i in range(num_tiles):
        start = tile_index_offsets[i]
        end = tile_index_offsets[i+1]
        
        # Get the global Gaussian indices for this tile
        indices_for_tile = tile_flat_indices[start:end]
        
        if len(indices_for_tile) > 0:
            C_k[i] = len(indices_for_tile)
            W_k[i] = w_gi[indices_for_tile].sum()
            
    # Normalize C_k to be between 0 and 1 or dividing by max
    
    # TODO other normalization strategies? e.g. log(1+C_k) or C_k / (C_k + alpha) to prevent outliers dominating
    if C_k.max() > 0:
        C_k = C_k / C_k.max()
    
    return W_k, C_k

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