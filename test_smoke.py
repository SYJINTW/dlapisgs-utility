"""
Smoke test for calculate_utility().

Runs the existing tile-utility pipeline end-to-end against the local sample
data (no GPU, no .ply files needed) and prints the top-K (tile, lod) pairs
for the first camera in the trace.
"""
import os
import sys
from pathlib import Path

import torch

# ---- Path wiring ----
# The cloned sibling repos live under the workspace root. utility_calculation.py
# imports `visibility_AABB_pytorch` from Frustum-for-3DGS and `tiling` from GGSP,
# so we make them importable BEFORE importing it.
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(WORKSPACE / "Frustum-for-3DGS"))
sys.path.insert(0, str(WORKSPACE / "GGSP"))

import visibility_AABB_pytorch  # noqa: E402
import tiling  # noqa: E402
import utility_calculation as uc  # noqa: E402


# ---- Config ----
TILE_NPZ = WORKSPACE / "GGSP" / "tiles_metadata_allFrames.npz"
TRACE_JSON = WORKSPACE / "Frustum-for-3DGS" / "sample_data" / "camera_trace" / "trace1.json"
NUM_LOD = 3
TOP_K = 20
IMG_W, IMG_H = 800, 800


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")
    print(f"[smoke] tile npz   = {TILE_NPZ}")
    print(f"[smoke] trace json = {TRACE_JSON}")

    assert TILE_NPZ.exists(), f"missing {TILE_NPZ}"
    assert TRACE_JSON.exists(), f"missing {TRACE_JSON}"

    tiles_info = tiling.load_tiles_from_npz(TILE_NPZ)
    min_corners = torch.tensor(tiles_info["min_corners"], dtype=torch.float32, device=device)
    max_corners = torch.tensor(tiles_info["max_corners"], dtype=torch.float32, device=device)
    tile_centers = (min_corners + max_corners) / 2.0
    print(f"[smoke] loaded {min_corners.shape[0]} tiles")

    cam_infos = visibility_AABB_pytorch.readCamerasFromTransforms(TRACE_JSON, IMG_W, IMG_H)
    cameras = visibility_AABB_pytorch.camera_infos_to_MiniCam_list(cam_infos)
    print(f"[smoke] loaded {len(cameras)} cameras; using camera 0")

    cam = cameras[0]
    cam_center = cam.camera_center.to(device)

    distances = uc.calculate_distances(tile_centers, cam_center)
    visibility = visibility_AABB_pytorch.batched_check_tiles_visible(
        min_corners, max_corners, cam, device=device
    )
    n_visible = int(visibility.sum().item())
    print(f"[smoke] visible tiles: {n_visible} / {visibility.numel()}")

    utilities = uc.calculate_utility_basic(visibility, distances, NUM_LOD)
    print(f"[smoke] top-{TOP_K} (tile_index, lod_level):")
    print(utilities[:TOP_K])


if __name__ == "__main__":
    main()
