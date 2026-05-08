# Research Plan

## Current focus: pipeline runtime

### dlapisgs-utility (ours)
- [x] test_utility.py: multi-budget/scheme in one pass, prefix-slice greedy
- [x] Vectorize `_greedy_order` — replace 3M-element Python loop with numpy concat (350× speedup, 355s → ~1s)
- [ ] Vectorize `compute_tile_weights_and_counts` (utility_calculation.py) — replace per-tile loop with GPU `scatter_add_` + `bincount`
- [ ] Thread pool for PLY writes + dedup tile_npz (test_utility_fast.py prototype exists)
- [ ] Save experiment params as YAML alongside utility.log in output dir

### upstream (do later)
- [ ] `tiling_uniform_layered_gs` (GGSP/tiling.py) — triple-nested loop does 64 passes over 6M Gaussians; fix with single-pass floor-division tile assignment + argsort grouping (~172s → seconds)
- [ ] `export_gs_to_ply` (GS-Interface/io_3dgs.py) — `list(map(tuple, ...))` allocates a Python tuple per Gaussian; fix with numpy structured array field assignment

### render + metrics
- [ ] Fix render + metrics pipeline bottlenecks
- [ ] Explore in-memory render pipeline: skip PLY writes entirely, render selected Gaussians directly in-memory → save PNG + metrics only. HDD is the bottleneck (ROTA=1 spinning disk, ~50MB/s effective with parallel writes). In-memory approach: after greedy selection, pass indices directly to renderer, capture image tensor, compute PSNR/SSIM, write PNG (tiny). Eliminates ~2050MB/camera of PLY I/O per scheme.

## Up next
1. Utility function design iteration (scoring schemes, weighting factors)
2. Algorithm design experiments (scheduler variants, LOD strategies)
3. Multi-scene testing (beyond bicycle)
