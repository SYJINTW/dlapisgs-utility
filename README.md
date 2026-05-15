# dlapisgs-utility

View-conditioned Rate-Utility scoring for tiled 3D Gaussian Splatting (3DGS) streaming. Given a full-scene 3DGS PLY, a tile partition, and a camera trace, this module scores and ranks Gaussians under a byte budget so a streaming server can transmit highest-value content first.

Companion docs (under `.claude/`):
- `CLAUDE.md` — agent / contributor guide; full CLI surface, source map, output layout.
- `PLAN.md` — current research roadmap and sweep state.
- `0514_run_commands.md` — current sweep commands, ETAs, and disk footprint.
- `context_experiments.md`, `context_debug_viewer.md` — narrower context notes.

## Quick start

```bash
conda activate gsquic

# Selection (offline; all cameras, 7 budget tiers, screen-area weighting):
python test_utility.py \
  --ply <full.ply> --output-root <out>/ \
  --camera-trace <trace.json> --grid-shape 8 8 8 \
  --budget-pct 10 25 40 55 70 85 100 \
  --schemes vd_lod_w_c --num-lod 1 --camera-index -1 \
  --packing-mode progressive --weight-mode screen_area \
  --w-norm sum --c-norm sum \
  --tiling-cache <out>/.tiling_cache.npz

# Render + metrics (separate env; deletes each PLY after its metric row):
conda run -n gaussian_splatting python experiments/render_metrics.py \
  --output-root <out>/ --gt-ply <full.ply> --trace <trace.json> \
  --scene <name> --render-dir <out>/renders --delete-ply
```

For the 0514 sweep wrappers (Exp 1 GS-weight × 3 scenes, Exp 2 tile-utility × bicycle), see `.claude/0514_run_commands.md`.

## Scoring model

```
U(k, ℓ) = log(β·(ℓ+1)) · (v_k / d_k) · W_k · C_k
```

- `v_k` ∈ {1, ε=1e-2} from frustum-AABB test (`Frustum-for-3DGS`).
- `d_k` Euclidean camera→tile-center distance.
- `W_k` = Σ_g w(g_i) over tile k, then `--w-norm`-normalized.
- `C_k` = Gaussian count in tile, then `--c-norm`-normalized.
- `ℓ` LOD level (currently fixed at 1; `--num-lod 1` disables the log factor).

Per-Gaussian weight `w(g_i)` modes:

| `--weight-mode` | formula |
|---|---|
| `det_gamma_over_d2` | sigmoid(o) · det(Σ)^γ / d² |
| `volume` | sigmoid(o) · det(Σ)^0.5 |
| `volume_over_d2` | volume / d² |
| `screen_area` | sigmoid(o) · π · √det(Σ_2D) — EWA-projected footprint |

`screen_area` is the current best across hotdog and ship (5/14 sweep, partial results); bicycle pending.

## Schemes

| `--schemes` | active factors in U |
|---|---|
| `vd` | V + D only (baseline, no LOD) |
| `vd_lod` | + LOD |
| `vd_lod_w` | + W_k |
| `vd_lod_c` | + C_k |
| `vd_lod_w_c` | all (full model) |

## Packing modes

| `--packing-mode` | behavior |
|---|---|
| `tile_partial` (default, proposed) | greedy tile-by-utility; partial-fill the overflowing last tile |
| `tile_strict` | greedy by tile-utility; drop any tile that won't fit whole |
| `progressive` | two-pass: visible-tile GS by w_gi, then invisible-tile GS by w_gi; identity at byte_budget ≥ scene_size |

## Pipeline placement

```
GGSP (tile metadata) ─┐
GS-Interface (PLY)  ──┼─→ dlapisgs-utility ─→ subset PLY ─→ QUIC-for-3DGS ─→ renderer
Frustum-for-3DGS  ────┘
```

See `../.claude/CLAUDE.md` for cross-repo workspace info.

## Source map

- `utility_calculation.py` — scoring math (`calculate_utility_param`, `compute_gaussian_weights*`, `compute_tile_weights_and_counts`, `project_covariance_2d`).
- `test_utility.py` — offline integration runner (CLI).
- `experiments/` — sweep wrappers, render/metrics, plotter, debug viewer, view generator, timing aggregation.

## Date / status

2026-05-15 — 0514 sweep restarted on a clean output dir after three correctness fixes (two-pass progressive packer, `--budget-pct` flag, `--delete-ply` default-on). See `.claude/PLAN.md` for the 0515 status block.
