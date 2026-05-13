# Research Plan

## Current focus: instrumentation, run metadata, debug viewer, two new experiments (2026-05-13)

**Why:** PSNR results from the 0507/0508 sweeps disappointed; the utility/weight design is the suspected bottleneck (W_k unnormalized, GS weight formulation may be wrong). Before more sweeps we need (a) trustworthy per-stage timings, (b) reproducible run metadata, (c) a debug viewer rich enough to inspect W_k / C_k / per-GS weight distributions, and (d) two cleanly-scoped experiment modes.

### Deliverable 1 — Per-stage timing instrumentation
- [ ] Add `_timed(name, store, **labels)` context manager in `test_utility.py`
- [ ] Wrap missing spans: `compute_tile_weights_and_counts`, `_select_at_budget` (per scheme/budget), `_write_ply` aggregate (mean / max / total)
- [ ] Dump `timings.json` to `<output_root>/`
- [ ] Mirror in `experiments/render_metrics.py`: GT PLY load, GT render-all-cameras, per-manifest render, PSNR+SSIM compute, PNG write → `render_timings.json`

### Deliverable 2 — `params.yaml` run-metadata dump
- [ ] `_dump_run_params(output_root, args)` helper
- [ ] Captured fields: all argparse args + timestamp + hostname + device + python version
- [ ] PyYAML `safe_dump(sort_keys=False)`
- [ ] Same shape for `render_metrics.py` → `render_params.yaml`

### Deliverable 3 — Headless debug viewer (`experiments/debug_viewport.py`)
- [ ] Per-tile #GS heatmap panel (color = raw `C_k`)
- [ ] Per-Gaussian density via ~1% subsample, colored by `w(g_i)`, seed-controlled
- [ ] W_k normalization comparison row: `none / max / minmax / log1p / sum` side-by-side
- [ ] Camera-pose scrubbing: `--cameras i:j --mp4 path` via `matplotlib.animation.FFMpegWriter`
- [ ] `--ply` stays optional; per-GS / W_k panels degrade gracefully

### Deliverable 4a — Setup 1: tile-level + W/C normalization sweep
- [ ] `normalize_term(x, mode)` helper in `utility_calculation.py` supporting `{none, max, minmax, log1p, sum}`
- [ ] CLI flags `--w-norm` (default `none`, legacy) and `--c-norm` (default `max`, legacy)
- [ ] No change to `_greedy_order` / `_select_at_budget`
- [ ] `experiments/0513/run_setup1_norm_sweep.sh` sweep wrapper
- [ ] New manifest fields: `w_norm`, `c_norm`

### Deliverable 4b — Setup 2: per-Gaussian progressive packing
- [ ] `compute_gaussian_weights_v2(weight_mode=...)` in `utility_calculation.py`:
    - `volume`: `sigmoid(o) * det(Σ)^0.5`
    - `volume_over_d2`: `sigmoid(o) * det(Σ)^0.5 / d²`
    - `screen_area`: `sigmoid(o) * π · √det(Σ_2D)` via Jacobian-of-projection (pure torch, GPU)
- [ ] Helper `project_covariance_2d(xyz, scales, rots, world_view, proj, img_w, img_h)` (mirrors `forward.cu`)
- [ ] CLI flags `--packing-mode {tile, progressive}` (default `tile`) and `--weight-mode`
- [ ] `_greedy_order_progressive(...)`: visible-tile mask → flatten GS pool → sort by w_gi → trim to budget
- [ ] `experiments/0513/run_setup2_progressive.sh`
- [ ] New manifest fields: `packing_mode`, `weight_mode`

### Verification
- [ ] Smoke test: legacy flags (`--w-norm none --c-norm max --packing-mode tile`) reproduces baseline 0508 PSNR/SSIM bit-stable
- [ ] `timings.json` and `params.yaml` present and well-formed
- [ ] Debug viewer renders 4 new panels and produces a 30-frame mp4
- [ ] Setup 1 sweep produces one output tree per (w_norm, c_norm) combo
- [ ] Setup 2 sweep produces one output tree per weight_mode
- [ ] Screen-area sanity: Spearman ρ > 0.9 vs real-render radii on one camera

---

## Backlog (was previously "current focus", deferred)

### Perf / runtime
- [x] test_utility.py: multi-budget/scheme in one pass, prefix-slice greedy
- [x] Vectorize `_greedy_order` — replace 3M-element Python loop with numpy concat (350× speedup, 355s → ~1s)
- [ ] Vectorize `compute_tile_weights_and_counts` (utility_calculation.py) — replace per-tile loop with GPU `scatter_add_` + `bincount`. **Note:** Deliverable 1's new timer will tell us if this is still the bottleneck under the new modes; revisit after.
- [ ] Thread pool for PLY writes + dedup tile_npz (test_utility_fast.py prototype exists)

### upstream
- [ ] `tiling_uniform_layered_gs` (GGSP/tiling.py) — triple-nested loop does 64 passes over 6M Gaussians; fix with single-pass floor-division tile assignment + argsort grouping (~172s → seconds)
- [ ] `export_gs_to_ply` (GS-Interface/io_3dgs.py) — `list(map(tuple, ...))` allocates a Python tuple per Gaussian; fix with numpy structured array field assignment

### render + metrics
- [ ] Fix render + metrics pipeline bottlenecks
- [ ] Explore in-memory render pipeline: skip PLY writes entirely, render selected Gaussians directly in-memory → save PNG + metrics only. HDD is the bottleneck (ROTA=1 spinning disk, ~50MB/s effective with parallel writes).

## Up next (after this push lands)
1. Pick the winning normalization and weight-mode from the sweeps; lock as default.
2. Algorithm design experiments (scheduler variants, LOD strategies).
3. Multi-scene testing (beyond bicycle).
