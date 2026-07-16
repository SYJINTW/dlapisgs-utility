#!/usr/bin/env python3
"""Plot measure_downsample_degradation.py's summary.csv: PSNR/SSIM/LPIPS vs keep_frac,
per_tile vs per_scene as two lines, one subplot per scene (Workstream A of the
scene-size-reduction plan, 2026-07-15).

Usage:
  python experiments/plot_downsample_degradation.py \
      --glob "output/0715/downsample_degradation/*/*/summary.csv" \
      --out-dir output/0715/downsample_degradation/agg_plots
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csfont = {"family": "serif", "serif": ["Times New Roman", "Times"], "size": 23}
plt.rc("text", usetex=False)
plt.rc("font", **csfont)
DPI = 300
PSNR_SATURATION_DB = 60.0

_MODE_STYLE = {"per_tile": {"color": "#2b6cb0", "marker": "o"},
               "per_scene": {"color": "#c53030", "marker": "s"}}
_METRIC_YLABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS"}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(pooled: dict, metric: str, modes: list[str], out_path: Path, scene: str):
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    for mode in modes:
        if mode not in pooled:
            continue
        kf_to_vals = pooled[mode]
        xs = sorted(kf_to_vals)
        means, los, his = [], [], []
        for x in xs:
            vals = np.asarray(kf_to_vals[x])
            if metric == "psnr":
                vals = np.clip(vals, None, PSNR_SATURATION_DB)
            m = vals.mean()
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            means.append(m)
            los.append(m - ci)
            his.append(m + ci)
        style = _MODE_STYLE.get(mode, {})
        ax.plot(xs, means, linewidth=2.0, label=mode, zorder=2, **style)
        ax.fill_between(xs, los, his, color=style.get("color"), alpha=0.15, linewidth=0, zorder=1)
    ax.set_title(f"{scene}", fontsize=17)
    ax.set_xlabel("Keep fraction", fontsize=15)
    ax.set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=15)
    ax.tick_params(labelsize=12, direction="out", which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(loc="best", framealpha=0.9, fontsize=11)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="glob matching summary.csv files")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matched: {args.glob}")
    print(f"pooling {len(paths)} file(s)")

    all_rows = []
    for path in paths:
        all_rows.extend(_read_rows(Path(path)))

    scenes = sorted({r["scene"] for r in all_rows})
    modes = sorted({r["mode"] for r in all_rows})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        rows_scene = [r for r in all_rows if r["scene"] == scene]
        for metric in ("psnr", "ssim", "lpips"):
            if not any(r.get(metric) for r in rows_scene):
                continue
            pooled: dict = defaultdict(lambda: defaultdict(list))
            for r in rows_scene:
                if not r.get(metric):
                    continue
                pooled[r["mode"]][round(float(r["keep_frac"]), 6)].append(float(r[metric]))
            out_path = out_dir / f"{metric}_vs_keepfrac_{scene}.png"
            plot_metric(pooled, metric, modes, out_path, scene)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
