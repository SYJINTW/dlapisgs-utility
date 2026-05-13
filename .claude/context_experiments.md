# Experiments & Modules — Session Context (2026-05-13)

## Active experiment: Setup 1 norm sweep

**Script**: `experiments/0513/run_setup1_norm_sweep.sh`
**Goal**: Find the best (w_norm, c_norm) normalization strategy for W_k and C_k.
**Fixed knobs**: `weight_mode=volume_over_d2`, `packing_mode=tile_strict`, `scheme=vd_lod_w_c`, `num_lod=1`

### Norm pairs tested

| Tag | w_norm | c_norm | Notes |
|---|---|---|---|
| w-none_c-max | none | max | **0508 baseline** — W unbounded, C in [0,1] |
| w-none_c-none | none | none | Unbounded reference |
| w-max_c-max | max | max | Both in [0,1] |
| w-minmax_c-minmax | minmax | minmax | Both in [0,1], zero-shifted |
| w-log1p_c-log1p | log1p | log1p | Both in [0,1], log-compressed |
| w-sum_c-sum | sum | sum | Tile probability mass, sums to 1 |

Asymmetric pairs dropped (no principled rationale when both terms multiply in U).

### Why `tile_strict` for this sweep

`tile_partial` partially fills the last tile — the norm affects which tile is "last"
and how much it spills, confounding attribution. `tile_strict` gives clean all-or-nothing
tile selection, so metric differences are purely due to normalization ranking.

### Why `volume_over_d2` as weight

`det_gamma_over_d2` with gamma=1.0 uses `det(Σ)^1 = exp(2*(sx+sy+sz))` — the square of
volume. `volume_over_d2` uses `det(Σ)^0.5 = exp(sx+sy+sz)` — actual 1-σ ellipsoid volume.
More physically meaningful; not dominated by a few huge Gaussians.

### Smoke run status

Single camera (index 0), budgets 60+200 MB — **completed clean (exit 0)**.
All 6 pairs selected 253,687 GS @ 60MB and 845,625 @ 200MB (budget-capped, expected).
Differences in *which* tiles are selected — needs render+metrics to quantify.

Full sweep (`CAMERA_INDEX=-1`, 5 budgets: 20/60/100/200/500 MB) not yet run.

## Key code changes this session

### `utility_calculation.py`

- `normalize_term` `log1p` mode: was `log(1+x)` (unbounded), now `log1p(x)/log1p(max(x))` → [0,1].
- No other math changes.

### `test_utility.py`

- `--tiling-cache <path>`: shared tiling cache across sweep runs on same PLY+grid.
  First invocation computes and saves; subsequent ones load in ~0.06s instead of ~150s.
  Cache stores: `min_corners`, `max_corners`, `index_offsets`, `flat_indices` (numpy arrays).
  `scene_min`/`scene_max` excluded (3D arrays, not needed after tiling skip).
- `meta_positions` recomputed from `index_offsets` (no longer needs `tile_indices` dict).
- `--weight-mode`, `--packing-mode`, `--w-norm`, `--c-norm` flags (all added previously).

### `experiments/0513/run_setup1_norm_sweep.sh`

- Changed from `tile_partial + det_gamma_over_d2` to `tile_strict + volume_over_d2`.
- Passes `--tiling-cache "$OUT_BASE/shared_tiling_cache.npz"` — tiling runs once.
- Pairs trimmed to 6 (symmetric + 2 reference baselines).

## Utility formula

```
U(k) = log(β·(ℓ+1)) · (v_k / d_k) · W_k · C_k
```

With `num_lod=1`, the LOD log-factor is constant → effectively `U = (v_k/d_k) · W_k · C_k`.

`w(g_i) = sigmoid(opacity) · exp(sx+sy+sz) / d(g_i, cam)²`   ← `volume_over_d2`

W_k = sum of w(g_i) over tile k, then normalized by w_norm.
C_k = Gaussian count in tile k, then normalized by c_norm.

## NORM_MODES

Defined in `utility_calculation.py`:
- `none`: identity (unbounded)
- `max`: x / max(x) → [0,1]
- `minmax`: (x−min)/(max−min) → [0,1]
- `log1p`: log1p(x)/log1p(max(x)) → [0,1]  ← bug fixed this session
- `sum`: x / sum(x) → [0,1], sums to 1 across tiles

## Next steps

1. Run full sweep: `bash experiments/0513/run_setup1_norm_sweep.sh`
2. Run renders + metrics on outputs
3. Plot PSNR vs budget per norm pair → pick winner
4. Lock winner as default; update PLAN.md
