# 0428 Experiments: Complete Workflow

Three-step pipeline for tiling, visualization, and rendering evaluation:

## Quick Start

**Run entire pipeline at once:**

```bash
cd dlapisgs-utility/experiments
bash run_full_pipeline.sh
```

**Or run individual steps:**

### Step 1: Tiling + Utility Selection

```bash
bash run_0428.sh
```

**Output:**

- `bicycle_vd.ply` — Gaussians selected by vd scheme
- `bicycle_vd.npz` — Tile metadata for visualization
- `bicycle_vd.log` — Execution log with metrics

Generates 3 schemes for comparison:

- `vd` — baseline (visibility + distance)
- `vd_lod` — with LOD-aware ranking
- `vd_lod_w_c` — full model (weights + complexity)

### Step 2: Visualize Tiling

```bash
bash visualize_0428.sh
```

**Output:** 3D plots of tile AABBs comparing schemes

Shows side-by-side visualization of how each scheme partitions the scene.

### Step 3: Render for Evaluation

```bash
bash render_0428.sh
```

**Optional parameters:**

```bash
IMG_WIDTH=1024 IMG_HEIGHT=1024 bash render_0428.sh
SH_DEGREE=4 bash render_0428.sh
WHITE_BG=1 bash render_0428.sh
```

**Output:**

- `0428/renders/bicycle_vd/renders/*.png`
- `0428/renders/bicycle_vd_lod/renders/*.png`
- `0428/renders/bicycle_vd_lod_w_c/renders/*.png`

## Computing Metrics

After rendering, evaluate PSNR/SSIM/VMAF:

```bash
cd LapisGS-object-based-renderer

python metrics.py \
  --model-paths "$ROOT/dlapisgs-utility/experiments/0428/renders" \
  --gt-dir <path_to_ground_truth_renders> \
  --renders-dir <path_to_renders>
```

## File Structure

```
experiments/
├── run_0428.sh                 ← Tiling + Utility
├── visualize_0428.sh           ← Tile visualization
├── render_0428.sh              ← Rendering
├── run_full_pipeline.sh        ← Complete pipeline
└── 0428/
    ├── bicycle_vd.ply          ← Output PLY
    ├── bicycle_vd.npz          ← Tile metadata
    ├── bicycle_vd.log          ← Logs
    ├── bicycle_vd_lod.ply
    ├── bicycle_vd_lod.npz
    ├── bicycle_vd_lod.log
    ├── bicycle_vd_lod_w_c.ply
    ├── bicycle_vd_lod_w_c.npz
    ├── bicycle_vd_lod_w_c.log
    └── renders/
        ├── bicycle_vd/renders/*.png
        ├── bicycle_vd_lod/renders/*.png
        └── bicycle_vd_lod_w_c/renders/*.png
```

## Troubleshooting

**"Output directory does not exist"**
→ Run `bash run_0428.sh` first

**"NPZ files not found"**
→ Using old test_utility.py; update to latest version with NPZ export

**Renderer script not found**
→ Check LapisGS-object-based-renderer path

## Performance Tips

- Use smaller image size for fast iteration: `IMG_WIDTH=512 IMG_HEIGHT=512`
- Use SH degree 1-2 for faster rendering: `SH_DEGREE=1`
- Run visualization first (no GPU needed) before launching renders
