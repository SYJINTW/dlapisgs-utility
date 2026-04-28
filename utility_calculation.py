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
LAYER_WEIGHT_BASE = 10.0        # base of the per-layer geometric weighting


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
):
    """
    full version:
    U(k) = log(beta * (l_k + 1)) * (v_k / d_k) * C_k * W_k

    Compute a configurable utility score with optional LOD, weight, and complexity factors.
    
    sent l_k in {0, 1, 2,..., L-1}
    l_k = -1 means not sent 

    Args:
        visibility_mask_tensor: length-N tensor; True if the tile is visible. 
        tile_distances_tensor:  length-N tensor; Euclidean distance from camera.
        num_of_level (int):     total number of LOD layers (1 means plain 3DGS). 
        weight_sum_tensor:      length-N tensor; Aggregate Gaussian weight per tile (W_k).
        complexity_tensor:      length-N tensor; Normalized count or entropy per tile (C_k).
        include_lod (bool):     If True, rank (tile, lod) pairs using per-layer weights.
        include_w (bool):       If True, multiply scores by W_k.
        include_c (bool):       If True, multiply scores by C_k.

    Returns:
        np.array: shape (N, 2) if LOD disabled or num_of_level==1, otherwise (N*num_of_level, 2).
                  The second column is the target lod_level.
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

    levels = torch.arange(num_of_level, device=base_scores.device, dtype=torch.float32)
    layer_weights = LAYER_WEIGHT_BASE ** ((num_of_level - 1) - levels)
    utility_matrix = base_scores.unsqueeze(1) * layer_weights.unsqueeze(0)

    flattened_scores = utility_matrix.view(-1)
    sorted_1d_indices = torch.argsort(flattened_scores, descending=True)
    tile_indices = sorted_1d_indices // num_of_level
    lod_levels = sorted_1d_indices % num_of_level
    result_tensor = torch.stack((tile_indices, lod_levels), dim=1)
    return result_tensor.cpu().numpy()


def compute_gaussian_weights(opacity, scale_0, scale_1, scale_2, gamma=1.0):
    """
    Compute individual Gaussian weights: w(g_i) = o_i * det(Sigma_i)^gamma
    det(Sigma_i) relates to the 3D volume = exp(2 * (scale_0 + scale_1 + scale_2))
    Opacity o_i = 1 / (1 + exp(-opacity))
    
    refer to original 3DGS implementation if anything here seems unclear.
    
    Args:
        opacity (torch.Tensor or np.ndarray): raw opacity values from PLY
        scale_0, scale_1, scale_2: raw scale values from PLY
        gamma (float): empirical exponent
    """
    if not isinstance(opacity, torch.Tensor):
        opacity = torch.tensor(opacity, dtype=torch.float32) 
        scale_0 = torch.tensor(scale_0, dtype=torch.float32)
        scale_1 = torch.tensor(scale_1, dtype=torch.float32)
        scale_2 = torch.tensor(scale_2, dtype=torch.float32)

    o_i = torch.sigmoid(opacity) # activate opacity to [0, 1]
    det_sigma = torch.exp(2.0 * (scale_0 + scale_1 + scale_2)) # calculate det covariance (volume) from log scales of std
    w_gi = o_i * (det_sigma ** gamma)
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
    if C_k.max() > 0:
        C_k = C_k / C_k.max()
        
    return W_k, C_k

def calculate_distances(tile_centers_tensor, cam_center_tensor):
    return torch.norm(tile_centers_tensor - cam_center_tensor.unsqueeze(0), dim=1)


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