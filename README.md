# dlapisgs-utility

This module provides the core scoring logic for the **View-Conditioned Rate-Utility Model** used in 3D Gaussian Splatting (3DGS) tiled streaming.

It evaluates which 3DGS tiles to transmit first (and at which Level-of-Detail) based on visibility, distance, tile complexity, and individual Gaussian rendering impact.

## Key Logic (`utility_calculation.py`)

- **`calculate_utility_basic(...)`**: The original baseline heuristic scoring. Ranks tiles based strictly on binary visibility, inverse distance, and static layer weights.
- **`calculate_utility_proposed(...)`**: The proposed rate-utility mathematical model. Computes the utility gain if tile `k` is promoted to its next LOD layer.
- **`compute_gaussian_weights(...)`**: Calculates the importance weight $w(g_i)$ for individual Gaussians using their spatial volume (determinant of covariance) and opacity.
- **`compute_tile_weights_and_counts(...)`**: Aggregates the per-Gaussian weights within a tile to produce the aggregate weight $W_k$ and extracts the normalized Gaussian count $C_k$.

## Pipeline Integration

1. Extract global 3DGS Gaussian attributes using `GSInterface`.
2. Load tile boundary structures (`.npz`) from `dlapisgs-tiling`.
3. Check visibility mask and calculate exact spatial distance using `Frustum-for-3DGS`.
4. Compute the utility scores.
5. Provide the greedily sorted `(tile_index, target_lods)` array to the streaming server scheduler to transmit highest-value data first.

See `test_smoke.py` for a minimal execution pipeline reference.
