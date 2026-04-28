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


def calculate_utility_basic(visibility_mask_tensor, tile_distances_tensor, num_of_level):
    """
    Compute the utility score matrix for 3DGS tiles.
    (transmission-priority)

    Args:
        visibility_mask_tensor: length-N tensor; True if the tile is visible.
        tile_distances_tensor:  length-N tensor; Euclidean distance from camera to each tile center.
        num_of_level (int):     total number of LOD layers (e.g. 3 means LOD 0, 1, 2).

    Returns:
        np.array: shape (N * num_of_level, 2). Higher score = higher transmission priority.
    """
    device = visibility_mask_tensor.device
    N = visibility_mask_tensor.shape[0]

    # --- 1. Per-factor weights ---
    # Visibility: 1.0 if visible, INVISIBLE_PRIORITY_EPS otherwise (avoids hard-zeroing the score).
    vis_factor = torch.where(visibility_mask_tensor, 1.0, INVISIBLE_PRIORITY_EPS)
    print("Visibility factor:", vis_factor.cpu().numpy())
    # Distance: closer tiles score higher; DISTANCE_EPS guards against div-by-zero.
    dist_factor = 1.0 / (tile_distances_tensor + DISTANCE_EPS)
    print("Distance factor:", dist_factor.cpu().numpy())
    # Layer: prefer base layer (for num_of_level=3, weights are [100.0, 10.0, 1.0]).
    levels = torch.arange(num_of_level, device=device, dtype=torch.float32)
    layer_weights = LAYER_WEIGHT_BASE ** ((num_of_level - 1) - levels)

    # --- 2. Build the utility matrix ---
    base_scores = vis_factor * dist_factor   # shape: (N,)
    print("Base scores (visibility * distance):", base_scores.cpu().numpy())

    # Broadcast to (N, num_of_level)
    utility_matrix = base_scores.unsqueeze(1) * layer_weights.unsqueeze(0)

    # --- 3. Flatten and sort ---
    flattened_scores = utility_matrix.view(-1)
    sorted_1d_indices = torch.argsort(flattened_scores, descending=True)

    # --- 4. Recover (tile_index, lod_level) from the flat index ---
    tile_indices = sorted_1d_indices // num_of_level
    lod_levels = sorted_1d_indices % num_of_level

    # --- 5. Stack and return ---
    result_tensor = torch.stack((tile_indices, lod_levels), dim=1)
    return result_tensor.cpu().numpy()


def calculate_two_level_utility(
    visibility_mask_tensor,
    tile_distances_tensor,
    current_tile_lods_tensor,
    num_of_level,
    complexity_tensor,
    weight_sum_tensor,
    alpha=1.0,
    beta=2.0
):
    """
    Compute the utility score matrix U(k) for 3DGS tiles according to the new rate-utility model.

    U(k) = alpha * log(beta * (l_k + 1)) * (v_k / d_k) * C_k * W_k

    Args:
        visibility_mask_tensor: length-N tensor; True if the tile is visible.
        tile_distances_tensor:  length-N tensor; Euclidean distance from camera.
        current_tile_lods_tensor: length-N tensor; Current LOD level of each tile l_k.
        num_of_level (int):     total number of LOD layers.
        complexity_tensor:      length-N tensor; Normalized Gaussian count or entropy.
        weight_sum_tensor:      length-N tensor; Aggregate Gaussian weight per tile.
        alpha (float):          Alpha parameter for log curve.
        beta (float):           Beta parameter for log curve.

    Returns:
        np.array: shape (N, 2). Pairs of (tile_index, target_lod_level), sorted by U(k).
                  This function evaluates U(k) at target_lod = current_lod + 1.
    """
    device = visibility_mask_tensor.device
    N = visibility_mask_tensor.shape[0]

    # Evaluate at next LOD level: l_next = l_k + 1
    # Note: Using torch.clamp to ensure we don't exceed max LOD layer
    l_next = current_tile_lods_tensor + 1
    
    # Base factors
    vis_factor = torch.where(visibility_mask_tensor, 1.0, INVISIBLE_PRIORITY_EPS)
    dist_factor = 1.0 / (tile_distances_tensor + DISTANCE_EPS)
    
    # Utility log term: log(beta * (l_next + 1)) -> Note: if formula is log(beta*(l_k+1)), 
    # we evaluate for the *new* state which means we plug in l_next for l_k in the formula, 
    # so we do log(beta * (l_next + 1))? The prompt says "Evaluate U(k) at LOD = ℓ_k + 1".
    # Wait, the prompt formula is U(k) = log(beta*(l_k+1)). So evaluated at l_next, it's log(beta*(l_next)).
    utility_val = alpha * torch.log(beta * l_next) * vis_factor * dist_factor * complexity_tensor * weight_sum_tensor

    # Sort the single utility scores for the next layer promotion
    flattened_scores = utility_val.view(-1)
    
    # Only consider tiles that haven't reached the max LOD layer
    valid_mask = current_tile_lods_tensor < num_of_level
    # Make invalid ones have -inf priority
    flattened_scores = torch.where(valid_mask, flattened_scores, -torch.inf)

    sorted_indices = torch.argsort(flattened_scores, descending=True)
    
    # Filter out the invalid ones (-inf)
    valid_sorted_indices = sorted_indices[valid_mask[sorted_indices]]

    # Target LODs are just the current LODs + 1 for each tile
    target_lods = (current_tile_lods_tensor[valid_sorted_indices] + 1)
    
    result_tensor = torch.stack((valid_sorted_indices, target_lods.to(torch.long)), dim=1)
    return result_tensor.cpu().numpy()


def compute_gaussian_weights(opacity, scale_0, scale_1, scale_2, gamma=1.0):
    """
    Compute individual Gaussian weights: w(g_i) = o_i * det(Sigma_i)^gamma
    det(Sigma_i) relates to the 3D volume = exp(2 * (scale_0 + scale_1 + scale_2))
    Opacity o_i = 1 / (1 + exp(-opacity))
    
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

    o_i = torch.sigmoid(opacity)
    det_sigma = torch.exp(2.0 * (scale_0 + scale_1 + scale_2))
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
        
        tile_utilities = calculate_utility_basic(visibility_mask_tensor, tile_distances_tensor, 3)
        print(f"Camera {cam.uid} utility scores (tile_index, lod_level):")
        print(tile_utilities[:])
        break
    
if __name__ == "__main__":
    main()