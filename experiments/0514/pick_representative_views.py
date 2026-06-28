#!/usr/bin/env python3
"""Pick representative (worst / median / best PSNR) cameras per cell and emit
a side-by-side comparison figure: 3 rows × 2 columns (rank × {subset, gt}).

Assumes the cell directory layout produced by render_metrics.py with
--render-dir <output_root>/renders:

    <output_root>/
      metrics/summary.csv
      gt_renders/camera_NNN.png                                    (full-scene GT)
      renders/ply/budget_<MB>mb/<scheme>/camera_NNN.png            (selected subset)

For each (scene, group_key, budget_mb) tuple in summary.csv:
    <output_root>/representative/<group_key>/budget_<MB>mb.png

Also dumps representative/index.csv with one row per pick.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _budget_tag(budget_mb: float) -> str:
    """Matches test_utility.py's on-disk budget directory naming."""
    return f"{budget_mb:g}".replace(".", "p") + "mb"


def _imshow_or_blank(ax, path: Path, title: str) -> None:
    if path.exists():
        ax.imshow(mpimg.imread(str(path)))
    else:
        ax.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center",
                fontsize=8, color="red", transform=ax.transAxes)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-csv", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--group-by", default="scheme",
                    help="Which CSV column distinguishes the cells (e.g. weight_mode for Exp1, scheme for Exp2).")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.summary_csv.open()))
    if not rows:
        print(f"[pick] no rows in {args.summary_csv}"); return

    rep_root = args.output_root / "representative"
    rep_root.mkdir(parents=True, exist_ok=True)
    gt_dir = args.output_root / "gt_renders"
    render_root = args.output_root / "renders"

    by_cell: dict[tuple, list] = {}
    for r in rows:
        if r.get("psnr") in (None, ""):
            continue
        key = (r.get("scene", ""), r[args.group_by], float(r["budget_mb"]))
        by_cell.setdefault(key, []).append(r)

    distinct_groups = {k[1] for k in by_cell}
    use_group_subdir = len(distinct_groups) > 1
    index_rows = []
    for (scene, group_val, budget_mb), cell_rows in sorted(by_cell.items()):
        cell_rows.sort(key=lambda r: float(r["psnr"]))
        worst  = cell_rows[0]
        median = cell_rows[len(cell_rows) // 2]
        best   = cell_rows[-1]

        budget_tag = f"budget_{_budget_tag(budget_mb)}"
        meta = cell_rows[0]
        pmode = meta.get("packing_mode", "")
        wmode = meta.get("weight_mode", "")
        grid  = meta.get("grid_shape", "")
        fig, axes = plt.subplots(3, 2, figsize=(8.0, 11.0))
        fig.suptitle(
            f"{scene}  {args.group_by}={group_val}  packing={pmode}  weight={wmode}  grid={grid}  {budget_tag}",
            fontsize=10,
        )

        for row_i, (rank, row) in enumerate((("worst", worst), ("median", median), ("best", best))):
            cam_idx = int(row["camera_index"])
            psnr_val = float(row["psnr"])
            ssim_val = float(row.get("ssim", "nan") or "nan")
            scheme = row["scheme"]
            subset_png = render_root / "ply" / budget_tag / scheme / f"camera_{cam_idx:03d}.png"
            gt_png = gt_dir / f"camera_{cam_idx:03d}.png"

            _imshow_or_blank(axes[row_i, 0], subset_png,
                             f"{rank} (cam {cam_idx})  PSNR={psnr_val:.2f}  SSIM={ssim_val:.3f}")
            _imshow_or_blank(axes[row_i, 1], gt_png, f"ground truth (cam {cam_idx})")

            index_rows.append({
                "scene": scene, "group_by": args.group_by, "group_value": group_val,
                "budget_mb": budget_mb, "rank": rank, "camera_index": cam_idx,
                "scheme": scheme, "psnr": psnr_val, "ssim": ssim_val,
                "subset_png": str(subset_png), "gt_png": str(gt_png),
            })

        out_dir = rep_root / str(group_val) if use_group_subdir else rep_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{budget_tag}.png"
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[pick] {out_path}")

    if index_rows:
        idx_csv = rep_root / "index.csv"
        with idx_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader(); w.writerows(index_rows)
        print(f"[pick] wrote {len(index_rows)} picks across {len(by_cell)} cells -> {idx_csv}")


if __name__ == "__main__":
    main()
