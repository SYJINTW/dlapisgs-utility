# Research Plan

## Status: 2026-05-14 — 0514 sweep scaffolded (Exp 1 GS-weight × 3 scenes; Exp 2 tile-utility × 2 weight modes)

New artifacts (implemented this session):

- `experiments/gen_sparse_views.py` — generates 100-view Blender-format trace alongside each PLY, with frustum-based void rejection (min subsampled-Gaussian visibility threshold).
- `experiments/0514/run_exp1_gs_weight.sh` — Exp 1: progressive packing × {volume, volume_over_d2, screen_area} on bicycle/hotdog/ship at 10/25/40/55/70/85/100 % budget tiers (default `CUDA_VISIBLE_DEVICES=2`).
- `experiments/0514/run_exp2_tile_utility.sh` — Exp 2: bicycle only, `tile_strict` packing, two sub-sweeps (weight-mode ∈ {volume_over_d2, screen_area}) × 4 schemes × 7 budgets (default `CUDA_VISIBLE_DEVICES=3`).
- `experiments/0514/pick_representative_views.py` — symlinks worst / median / best PSNR cameras per cell (+ GT counterpart) into `<output_root>/representative/`.
- `experiments/aggregate_timings.py` — turns the per-stage `timings.json`/`render_timings.json` into a readable `timings_summary.csv` with mean / std / 95 % CI / p50 / p95 / total and a `_per_camera_e2e` row.
- `experiments/render_metrics.py` edits: emits a persistent `<output_root>/gt_renders/camera_NNN.png` (idempotent, reused on rerun) and adds a `scene` column to `summary.csv`.

Run order:

```bash
# Phase B — generate per-scene traces (CPU only, ~minutes)
python experiments/gen_sparse_views.py --ply exp-dataset/bicycle/point_cloud.ply \
    --scene-type mipnerf360 --n-views 100 \
    --out exp-dataset/bicycle/sparse_views_100.json
python experiments/gen_sparse_views.py --ply exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/point_cloud.ply \
    --scene-type synthetic --n-views 100 \
    --out exp-dataset/hotdog/checkpoint/point_cloud/iteration_30000/sparse_views_100.json
python experiments/gen_sparse_views.py --ply exp-dataset/ship/checkpoint/point_cloud/iteration_30000/point_cloud.ply \
    --scene-type synthetic --n-views 100 \
    --out exp-dataset/ship/checkpoint/point_cloud/iteration_30000/sparse_views_100.json

# Phase B' — dry-run to verify dir layout
DRY_RUN=1 bash experiments/0514/run_exp1_gs_weight.sh
DRY_RUN=1 bash experiments/0514/run_exp2_tile_utility.sh

# Phase C — smoke (one camera, smallest budget)
CAMERA_INDEX=0 BUDGET_PCTS=10 bash experiments/0514/run_exp1_gs_weight.sh
CAMERA_INDEX=0 BUDGET_PCTS=10 bash experiments/0514/run_exp2_tile_utility.sh

# Phase D — full sweep, parallel
CUDA_VISIBLE_DEVICES=2 bash experiments/0514/run_exp1_gs_weight.sh
CUDA_VISIBLE_DEVICES=3 bash experiments/0514/run_exp2_tile_utility.sh
```

---

## Status: 2026-05-13 — instrumentation + Setup 1/2 plumbing landed; DX pass (tqdm, vectorized tile weights)

The 0513 push wired up per-stage timing, machine-readable run metadata, a Plotly Dash debug viewer, three packing modes, four weight modes, and W/C normalization knobs. Smoke-tested across all combinations. Ready for the bigger sweeps and the math-design iteration that prompted this work.

---

## Completed

### Deliverable 1 — Per-stage timing instrumentation

- [x] `_timed(name, store, **labels)` context manager in `test_utility.py` and `experiments/render_metrics.py`
- [x] Wrapped new spans: `compute_tile_weights_and_counts`, `_select_at_budget` (per scheme/budget), per-PLY-write summary stats (mean/max/total)
- [x] Dumps `<output_root>/timings.json` and `<output_root>/render_timings.json`

### Deliverable 2 — `params.yaml` run-metadata dump

- [x] `_dump_run_params` helper in both runners
- [x] Captures CLI args + timestamp + hostname + device + cuda + torch + python versions
- [x] `_yaml_safe` recursive coercion (handles torch's str subclasses and Paths)
- [x] Written as `<output_root>/params.yaml` and `<output_root>/render_params.yaml`

### Deliverable 3 — Plotly Dash interactive debug viewer

- [x] New script `experiments/debug_viewer_app.py`
- [x] Widgets: camera-index slider, weight_mode dropdown, w_norm / c_norm dropdowns, GS-subsample slider, overlay checklist
- [x] Panels: tile utility, per-tile #GS, per-tile W_k, per-Gaussian density (subsample), tile visibility, status text
- [x] Reuses `_project_points` and the shared normalize/weight helpers from `utility_calculation.py`

### Deliverable 4a — Setup 1: W/C normalization sweep

- [x] `normalize_term(x, mode)` helper supporting `{none, max, minmax, log1p, sum}`
- [x] CLI flags `--w-norm` (default `none`) and `--c-norm` (default `max`) — matches pre-0513 behavior when both are at default
- [x] `experiments/0513/run_setup1_norm_sweep.sh`
- [x] Manifest fields `w_norm`, `c_norm` added

### Deliverable 4b — Setup 2: progressive + tile_strict

- [x] `compute_gaussian_weights_v2(weight_mode=...)` with `{volume, volume_over_d2, screen_area}`
- [x] `project_covariance_2d` helper (EWA Jacobian, pure torch, GPU)
- [x] CLI flag `--packing-mode {tile_partial, tile_strict, progressive}` (default `tile_partial`)
- [x] CLI flag `--weight-mode {det_gamma_over_d2, volume, volume_over_d2, screen_area}` (default `det_gamma_over_d2`)
- [x] `_greedy_order_tile_strict` and `_greedy_order_progressive`
- [x] `experiments/0513/run_setup2_progressive.sh`
- [x] Manifest fields `packing_mode`, `weight_mode`, `gamma` added

### Smoke tests

- [x] `tile_partial` × `det_gamma_over_d2` (legacy-equivalent path) — produces same selected GS count as before
- [x] `tile_strict` — drops the overflowing tile (252 840 GS vs 253 687 for partial @ 60 MB)
- [x] `progressive` × `volume` — sorts globally, same byte count
- [x] `progressive` × `screen_area` — Jacobian path runs (~1.2 s for 6 M Gaussians on the bicycle scene)
- [x] `params.yaml` and `timings.json` (15 spans) verified on a minimal run

### Naming hygiene

- [x] Renamed `--weight-mode legacy` → `det_gamma_over_d2` (descriptive, not historical)
- [x] Confirmed `tile_partial` is **our proposed method**, not "legacy"

---

## How to use the new pieces

### Runner — `test_utility.py`

New flags (all backwards-compatible; defaults reproduce pre-0513 behavior):

```bash
conda run -n gsquic python test_utility.py \
    --ply <model.ply> --output-root <dir> --camera-trace <trace.json> \
    --grid-shape 8 8 8 --budgets-mb 60 100 200 --schemes vd_lod_w_c \
    --camera-index -1 \
    --w-norm    {none,max,minmax,log1p,sum}      # default none
    --c-norm    {none,max,minmax,log1p,sum}      # default max
    --packing-mode {tile_partial,tile_strict,progressive}   # default tile_partial
    --weight-mode  {det_gamma_over_d2,volume,volume_over_d2,screen_area}  # default det_gamma_over_d2
```

Outputs (under `<output_root>`):

- `params.yaml` — run config + execution context
- `timings.json` — per-stage wall times (one row per span; load into pandas)
- `tiling.npz`, `camera_viz/*.npz`, `ply/budget_*/<scheme>/camera_*.ply + .json`
- `utility.log` — same loguru lines as before

### Render + metrics — `experiments/render_metrics.py`

Same call signature as before. New outputs:

- `<output_root>/render_params.yaml`
- `<output_root>/render_timings.json`
- `metrics/summary.csv` now includes `w_norm`, `c_norm`, `packing_mode`, `weight_mode` columns

### Plotly Dash debug viewer — `experiments/debug_viewer_app.py`

On the server:

```bash
conda run -n gsquic python experiments/debug_viewer_app.py \
    --ply <model.ply> \
    --camera-trace <trace.json> \
    --grid-shape 4 4 4 \
    --port 8050
```

On your laptop:

```bash
ssh -L 8050:localhost:8050 <server>
# then open http://localhost:8050
```

Slide the camera index, toggle overlays, change weight_mode / w_norm / c_norm live — no re-run needed. PLY load + tiling happens once at startup.

### Sweep wrappers — `experiments/0513/`

```bash
# W/C normalization sweep (7 pairs by default)
bash experiments/0513/run_setup1_norm_sweep.sh
# defaults: bicycle, 8×8×8 grid, budgets 20/60/100/200/500 MB, all 50 cameras

# Packing-mode × weight-mode sweep (5 combos by default)
bash experiments/0513/run_setup2_progressive.sh
```

Override via env vars: `OUTPUT_ROOT`, `BUDGET_LIST`, `GRID_SHAPE`, `CAMERA_INDEX`, `PACKING_MODE`, etc.

---

## Next session

### Math design — the real reason this push exists

1. (DONE) **Run the Dash viewer first** on the bicycle scene, scrub camera index, and compare what `det_gamma_over_d2` vs `volume` vs `volume_over_d2` vs `screen_area` do to `W_k` distributions across tiles. The hypothesis (per the memory at `project_gamma_default_history.md`) is that γ=1 in the default mode weights by volume² and is the root cause of the "GS weight still sucks" symptom from 0510.
2. (RUNNING) **Setup 2 sweep:** run `run_setup2_progressive.sh`. Tells us whether progressive packing or any of the three new weight modes beats `tile_partial × det_gamma_over_d2`.
3. **Setup 1 sweep:** run `run_setup1_norm_sweep.sh`, aggregate metrics, plot PSNR vs budget for each (w_norm, c_norm) pair. The "none × max" cell is the pre-0513 baseline.

### Verification tasks not yet done

- [ x ] **Screen-area sanity check** — compare `screen_area` weights against the radii returned by a real render pass on one camera. Expect Spearman ρ > 0.9. If not, the Jacobian implementation in `project_covariance_2d` is wrong.
- [ ] **Baseline regression** — re-run 0508 budget sweep with `--w-norm none --c-norm max --packing-mode tile_partial --weight-mode det_gamma_over_d2 --gamma 1.0` and confirm PSNR/SSIM are bit-stable vs the pre-0513 run.

### Backlog (still deferred)

**DX**

- [x] add `tqdm` progress bars (cameras / schemes / budgets) with per-stage postfix so you can see which sub-stage is running
- [x] loguru routed through `tqdm.write` — no torn output
- [x] incremental `timings.json` flush once per camera (recoverable on crash; ~1 ms overhead, not per-span)
- [ ] **Resume/skip-existing** (`--resume` flag): check if `camera_{NNN}.ply` already exists for the first budget/scheme combo, skip that camera. Saves hours on reruns after a mid-sweep crash.
- [x] **`--dry-run`**: prints full (camera × scheme × budget → path) matrix, marks `[x]` for outputs that already exist, exits; no GPU work, no disk writes.
- [ ] **GPU memory logging**: append `torch.cuda.max_memory_allocated()` (MB) to each camera's timing rows; zero-cost, catches OOM risk before it strikes.
- [ ] **`_greedy_order_progressive` scheme-consistency**: it ignores `scheme` entirely and always sorts by `w_gi`. Running it under `vd` vs `vd_lod_w_c` produces identical PLYs — add a `logger.warning` or raise if `scheme != "vd"` when `packing_mode=progressive`.

**Perf / runtime**

- [x] Vectorize `compute_tile_weights_and_counts` — Python loop replaced by `scatter_add_` + `repeat_interleave` (single GPU kernel).
- [ ] Dedup tile_npz writes (test_utility_fast.py prototype exists)

**Upstream**

- [ ] `tiling_uniform_layered_gs` (GGSP/tiling.py) — triple-nested loop, 64 passes over 6M Gaussians, ~26 s on the smoke test. Fix with single-pass floor-division tile assignment + argsort grouping (~172s → seconds).
- [ ] `export_gs_to_ply` (GS-Interface/io_3dgs.py) — `list(map(tuple, ...))` allocates a Python tuple per Gaussian.

**Render + metrics**

- [ ] In-memory render pipeline: skip PLY writes entirely, render selected Gaussians directly in-memory → save PNG + metrics only. HDD is the bottleneck (~50 MB/s spinning disk).

**Research follow-ups**

1. Pick the winning (w_norm, c_norm, weight_mode, packing_mode) from the sweeps; lock as project default.
2. Algorithm design experiments (scheduler variants, LOD strategies once we re-enable num_lod > 1).
3. Multi-scene testing (beyond bicycle).
