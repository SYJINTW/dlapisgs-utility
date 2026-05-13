# Debug Viewer — Session Context (2026-05-13)

## What it is

`experiments/debug_viewer_app.py` — interactive Plotly Dash app for inspecting the
tile-utility pipeline per camera. Runs on the server, accessed via VSCode Simple Browser
(no SSH tunnel needed when using VSCode remote).

## How to run

```bash
conda run -n gsquic python experiments/debug_viewer_app.py \
    --ply /mnt/data1/samk/gs-quic/cs5262_tile_quic/exp-dataset/bicycle/point_cloud.ply \
    --camera-trace /mnt/data1/samk/gs-quic/cs5262_tile_quic/Frustum-for-3DGS/sample_data/camera_trace/trace1.json \
    --grid-shape 4 4 4 \
    --port 8050 --debug
```

`--debug` enables Dash hot-reload — edits to the file reflect in the browser on save,
no restart needed.

## Layout (2×3 subplot grid + table)

| Panel | Content |
|---|---|
| (1,1) | Tile utility U projected to NDC — Viridis colormap |
| (1,2) | Per-tile raw #GS count projected to NDC — Cividis |
| (1,3) | Per-tile W_k projected to NDC — Magma |
| (2,1) | Per-Gaussian weight density (subsampled) — Plasma |
| (2,2) | **Bird's-eye world XZ** — tile centers (red=visible, gray=culled), gold star = camera |
| (2,3) | Status: camera index, visible tiles, U/W_k/C_k ranges, #GS total |
| Below | **Per-tile DataTable** sorted by U descending: rank / tile / visible / dist / #GS / W_k / C_k / U |

## Widgets

- Camera slider (0 → N-1)
- weight_mode dropdown: `{det_gamma_over_d2, volume, volume_over_d2, screen_area}`
- w_norm / c_norm dropdowns: `{none, max, minmax, log1p, sum}`
- GS subsample fraction slider (for density panel)
- Overlay checklist — toggle individual panels

## Bugs fixed this session

- `scaleanchor="x"` on `update_yaxes` broke all subplots below row 1 (blank panels).
  Fixed: removed `scaleanchor`; NDC range applied per-panel via `row=`/`col=` kwargs.
- Loguru format string mixed automatic `{}` and manual `{0}` field numbering → crash.
  Fixed: use named kwargs (`host=`, `port=`).
- `log1p` norm was unbounded (`log(1+x)` only) — fixed to `log1p(x)/log1p(max(x))` → [0,1].
- Status panel showed hardcoded "Camera: index in trace" — fixed to show actual cam_idx.
- `u` was computed inside the overlay guard; moved to `_compute_per_camera` so status
  panel and DataTable always have it.

## Axis convention

- NDC panels: x/y in [-1, 1] where (0,0) is screen center; anything outside is off-screen.
- Bird's-eye panel (2,2): world XZ coordinates, auto-ranged, grid on.

## Deleted

`experiments/debug_viewport.py` — static matplotlib script, superseded by this viewer.
