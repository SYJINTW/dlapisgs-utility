# CLAUDE.md — dlapisgs-utility

Scope: this file is the local guide for the `dlapisgs-utility` repo (utility scoring engine, primary development target). For cross-repo workspace, full pipeline, and upstream/downstream repo summaries, see `../.claude/CLAUDE.md`.

## Environment

```bash
conda env create -f environment.yml
conda activate gsquic
# render env (separate; Python 3.7 + diff_gaussian_rasterization_lapisgs):
#   conda activate gaussian_splatting   # used only by experiments/render_metrics.py
```

## Run the offline selection

```bash
python test_utility.py \
  --ply /path/to/model.ply \
  --output-root /path/to/output/ \
  --camera-trace /path/to/trace.json \
  --grid-shape 8 8 8 \
  --budget-pct 10 25 40 55 70 85 100   \   # absolute budget = pct × N × bytes_per_gaussian
  --schemes vd_lod_w_c \
  --num-lod 1 \
  --camera-index -1   \                    # -1 = all cameras
  --packing-mode {tile_partial,tile_strict,progressive} \
  --weight-mode  {det_gamma_over_d2,volume,volume_over_d2,screen_area} \
  --w-norm sum --c-norm sum \
  --tiling-cache /path/.tiling_cache.npz
```

Legacy alternatives: `--budgets-mb 60 100 200` or `--budget-mb 100` still work; `--budget-pct` is preferred because 100 % gives exact identity at saturation.

## Render + metrics (separate env)

```bash
conda run -n gaussian_splatting python experiments/render_metrics.py \
  --output-root <selection_output> \
  --gt-ply <full.ply> --trace <trace.json> \
  --scene <name> --render-dir <out>/renders \
  --delete-ply         # unlink each selected.ply right after metrics; drops disk by ~99 %
```

## Inspect one (camera, budget) quickly

Re-run the selection on a single cell with a warm tiling cache (~1–5 s):

```bash
conda run -n gsquic python test_utility.py \
  --ply <full.ply> --output-root /tmp/inspect \
  --camera-trace <trace.json> --grid-shape 8 8 8 \
  --camera-index N --budget-pct B \
  --schemes vd_lod_w_c --packing-mode progressive \
  --weight-mode screen_area --w-norm sum --c-norm sum \
  --tiling-cache <existing>/.tiling_cache.npz
```

Then render that single PLY via `experiments/render_metrics.py` pointed at the new `/tmp/inspect`.

## Interactive debug viewer

```bash
conda run -n gsquic python experiments/debug_viewer_app.py \
  --ply <full.ply> --camera-trace <trace.json> \
  --grid-shape 8 8 8 --port 8050 --debug
# laptop:  ssh -L 8050:localhost:8050 <server>  → open http://localhost:8050
```

Live widgets for camera index, weight mode, w/c normalization, GS subsample, overlays. PLY load + tiling happens once at startup.

## Source map

- `utility_calculation.py` — scoring math: `calculate_utility_param`, `compute_gaussian_weights`, `compute_gaussian_weights_v2`, `compute_tile_weights_and_counts`, `project_covariance_2d`. `INVISIBLE_PRIORITY_EPS` is the tile-level soft floor for invisible tiles inside `_compute_base_scores` (used by `tile_partial`/`tile_strict`).
- `test_utility.py` — offline integration runner (CLI). Packing modes: `tile_partial` (default, our proposed), `tile_strict`, `progressive` (two-pass: visible-tile GS first, then invisible-tile GS — guarantees identity at byte_budget ≥ scene_size).
- `experiments/` — batch wrappers and tooling:
  - `0514/run_exp1_gs_weight.sh`, `0514/run_exp2_tile_utility.sh` — current sweep wrappers (use `--budget-pct`).
  - `render_metrics.py` — render selection PLYs → PSNR/SSIM CSV. `--delete-ply` unlinks PLYs after metrics; reusable `gt_renders/camera_NNN.png` cache.
  - `plot_metrics.py` — PSNR/SSIM vs budget with **95 % CI** error bars (ddof=1, ±1.96·σ/√n). PSNR plots draw a dotted 60 dB saturation line.
  - `aggregate_timings.py` — turns per-stage `timings.json`/`render_timings.json` into CSVs.
  - `concat_summaries.py` — globs many `summary.csv` files into a single sweep-level CSV.
  - `gen_sparse_views.py` — generates per-scene 100-view trace JSON (frustum-based void rejection).
  - `debug_viewer_app.py` — Plotly Dash debug viewer.
  - `0514/pick_representative_views.py` — symlinks worst/median/best PSNR cameras per cell into `representative/`.

## Output layout (test_utility.py `--output-root`)

```
output_root/
├── tiling.npz              # written once; or use --tiling-cache <shared.npz>
├── params.yaml             # CLI args + execution context
├── timings.json            # per-stage wall times
├── utility.log
├── camera_viz/{NNN}.npz    # per-camera visibility/distance/pose snapshot
└── ply/budget_{B}mb/{scheme}/camera_{NNN}.ply + .json
```

`render_metrics.py` adds:

```
output_root/
├── render_params.yaml
├── render_timings.json
├── gt_renders/camera_{NNN}.png       # persistent; reused on rerun
├── renders/budget_{B}mb/{scheme}/camera_{NNN}.png
└── metrics/
    ├── summary.csv / summary.json
    └── budget_{B}mb/{scheme}/camera_{NNN}.json
```

## Saturation rule of thumb

A (view, budget) cell is **saturated** when PSNR ≥ 60 dB (MSE < 10⁻⁶ in [0,1]²; visually identical even before PNG quantization). SSIM saturates earlier than PSNR and is not a reliable identity check.

## Utility scoring model

```
U(k, ℓ) = log(β·(ℓ+1)) · (v_k / d_k) · W_k · C_k       (when include_lod/w/c are on)
```

`v_k` ∈ {1, ε=`INVISIBLE_PRIORITY_EPS`} from frustum-AABB test; `d_k` camera→tile-center distance; `W_k` = Σ w(g_i) over tile k (then normalized by `--w-norm`); `C_k` = #GS in tile (then `--c-norm`). Per-Gaussian weight `w(g_i)` is one of:

| weight_mode | formula |
|---|---|
| `det_gamma_over_d2` | sigmoid(o) · det(Σ)^γ / d² |
| `volume`            | sigmoid(o) · det(Σ)^0.5 (≈ s_x·s_y·s_z) |
| `volume_over_d2`    | volume / d² |
| `screen_area`       | sigmoid(o) · π · √det(Σ_2D) (EWA-projected footprint) |

`screen_area` is the current best (see `.claude/PLAN.md` for sweep results).
