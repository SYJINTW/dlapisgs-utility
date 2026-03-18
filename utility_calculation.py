import sys
import os
import torch
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath("../AABB_frustum"))
import visibility_AABB_pytorch
sys.path.append(os.path.abspath("../dlapisgs-tiling"))
import tiling_dynamic_frame


def calculate_utility(visibility_mask_tensor, tile_distances_tensor, num_of_level):
    """
    計算 3DGS Tiles 的傳輸效能分數矩陣。
    
    Args:
        visibility_mask_tensor: 長度為 N，True 表示可見，False 表示不可見。
        tile_distances_tensor: 長度為 N，各 Tile 中心到相機的距離。
        num_of_level (int): LOD 的總層數 (例如 3 代表有 LOD 0, 1, 2)。
        
    Returns:
        np.array: 形狀為 (N * num_of_level, 2) 的效能矩陣。分數越高越優先。
    """
    # 取得當前的設備 (CPU 或 GPU)，確保運算都在同一個設備上進行
    device = visibility_mask_tensor.device
    N = visibility_mask_tensor.shape[0]
    
    # --- 1. 計算各項權重 ---
    # 可見度：看見為 1.0，看不見為 0.01 (保留極低優先權而非 0，避免完全卡死)
    vis_factor = torch.where(visibility_mask_tensor, 1.0, 0.01)
    print("Visibility factor:", vis_factor.cpu().numpy())
    # 距離：距離越近分數越高 (加 0.001 避免除以零)
    dist_factor = 1.0 / (tile_distances_tensor + 0.001)
    print("Distance factor:", dist_factor.cpu().numpy())
    # 層級：Base layer 優先 (例如 num_of_level=3 時，權重為 [100.0, 10.0, 1.0])
    levels = torch.arange(num_of_level, device=device, dtype=torch.float32)
    layer_weights = 10.0 ** ((num_of_level - 1) - levels)
    
    # --- 2. 組合效能矩陣 ---
    # base_scores 形狀: (N,)
    base_scores = vis_factor * dist_factor
    print("Base scores (visibility * distance):", base_scores.cpu().numpy())
    
    # utility_matrix 形狀: (N, num_of_level)
    # 利用 unsqueeze 增加維度來進行廣播 (Broadcasting) 相乘
    utility_matrix = base_scores.unsqueeze(1) * layer_weights.unsqueeze(0)
    
    # --- 3. 攤平並排序 ---
    # 將 (N, L) 矩陣攤平成 (N * L,) 的一維張量
    flattened_scores = utility_matrix.view(-1)
    
    # 取得從大到小排序的 1D 索引 (Indices)
    sorted_1d_indices = torch.argsort(flattened_scores, descending=True)
    
    # --- 4. 轉換回 (tile_index, lod_level) ---
    # 利用商數與餘數，將 1D 索引還原成 2D 座標
    tile_indices = sorted_1d_indices // num_of_level
    lod_levels = sorted_1d_indices % num_of_level
    
    # --- 5. 組合與輸出 ---
    # 將兩個 1D 張量合併成 (N * L, 2) 的二維張量
    result_tensor = torch.stack((tile_indices, lod_levels), dim=1)
    
    # 轉回 CPU 並輸出為 numpy array 供後續串流邏輯使用
    return result_tensor.cpu().numpy()


def calculate_distances(tile_centers_tensor, cam_center_tensor):
    return torch.norm(tile_centers_tensor - cam_center_tensor.unsqueeze(0), dim=1)


def main():
    base_dir = Path("/home/syjintw/Desktop/NUS/dlapisgs-output/longdress/opacity")
    res1_frame0_path = base_dir / "longdress_res1" / "dynamic_1051" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    user_traj_path = Path("/home/syjintw/Desktop/NUS/AABB_frustum/sample_data/camera_trace/trace1.json")
    tile_metadata_path = Path("/home/syjintw/Desktop/NUS/dlapisgs-tiling/tiles_metadata_allFrames.npz")
    
    # Load tiles from NPZ
    tiles_info = tiling_dynamic_frame.load_tiles_from_npz(tile_metadata_path)
    
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
        
        tile_utilities = calculate_utility(visibility_mask_tensor, tile_distances_tensor, 3)
        print(f"Camera {cam.uid} utility scores (tile_index, lod_level):")
        print(tile_utilities[:])
        break
    
if __name__ == "__main__":
    main()