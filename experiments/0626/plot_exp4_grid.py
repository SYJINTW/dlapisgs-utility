#!/usr/bin/env python
"""exp4 tile-granularity plots. Scene-level CI (cam->scene-mean->aggregate over 10 scenes).

Two figure families, emitted per packing mode:
  (a) PSNR/SSIM vs budget%, lines=grid, one panel per scheme.
  (b) PSNR/SSIM vs grid{1,2,4,8}, lines=scheme, one panel per budget (25/55/85%).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PSNR_SAT = 60.0
B_PANELS = [25, 55, 85]                       # budgets for fig (b) panels
GRIDS = [1, 2, 4, 8]
SCHEME_ORDER = ["vd_lod", "vd_lod_w", "ml", "oracle_loo"]
SCHEME_LABEL = {"vd_lod": "vd_lod (baseline)", "vd_lod_w": "vd_lod_w (heuristic)",
                "ml": "ml (learned)", "oracle_loo": "oracle_loo (UB)"}
GRID_COLOR = {1: "#d62728", 2: "#ff7f0e", 4: "#2ca02c", 8: "#1f77b4"}


def load(csv):
    df = pd.read_csv(csv)
    df["psnr"] = df["psnr"].clip(upper=PSNR_SAT)
    df["grid"] = df["grid_shape"].str.extract(r"(\d+)").astype(int)
    # budget% = budget_mb / scene-max * 100
    mx = df.groupby("scene")["budget_mb"].transform("max")
    df["bpct"] = (df["budget_mb"] / mx * 100).round().astype(int)
    return df


def scene_ci(sub, metric):
    """Two-stage: per-scene mean over cams, then mean +-95%CI over scenes."""
    per_scene = sub.groupby("scene")[metric].mean()
    n = len(per_scene)
    mean = per_scene.mean()
    ci = 1.96 * per_scene.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, ci, n


def fig_a(df, packing, metric, out):
    sub = df[df.packing_mode == packing]
    schemes = [s for s in SCHEME_ORDER if s in sub.scheme.unique()]
    fig, axes = plt.subplots(1, len(schemes), figsize=(5 * len(schemes), 4.2),
                             squeeze=False, sharey=True)
    for ax, sch in zip(axes[0], schemes):
        s2 = sub[sub.scheme == sch]
        for g in GRIDS:
            s3 = s2[s2.grid == g]
            xs = sorted(s3.bpct.unique())
            ys, es = [], []
            for b in xs:
                m, c, _ = scene_ci(s3[s3.bpct == b], metric)
                ys.append(m); es.append(c)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3,
                        color=GRID_COLOR[g], label=f"{g}³")
        ax.set_title(SCHEME_LABEL.get(sch, sch))
        ax.set_xlabel("budget %"); ax.grid(alpha=.3)
        if metric == "psnr":
            ax.axhline(PSNR_SAT, ls=":", c="gray", lw=.8)
    axes[0][0].set_ylabel(metric.upper())
    axes[0][0].legend(title="grid")
    fig.suptitle(f"exp4 (a) {metric.upper()} vs budget | packing={packing} "
                 f"| lines=grid | scene-level 95% CI (n=10)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_b(df, packing, metric, out):
    sub = df[df.packing_mode == packing]
    schemes = [s for s in SCHEME_ORDER if s in sub.scheme.unique()]
    fig, axes = plt.subplots(1, len(B_PANELS), figsize=(5 * len(B_PANELS), 4.2),
                             squeeze=False, sharey=True)
    xpos = np.arange(len(GRIDS))
    nsch = len(schemes)
    width = 0.8 / nsch
    for ax, bp in zip(axes[0], B_PANELS):
        for i, sch in enumerate(schemes):
            s2 = sub[(sub.scheme == sch) & (sub.bpct == bp)]
            ys, es = [], []
            for g in GRIDS:
                m, c, _ = scene_ci(s2[s2.grid == g], metric)
                ys.append(m); es.append(c)
            off = (i - (nsch - 1) / 2) * width
            ax.bar(xpos + off, ys, width, yerr=es, capsize=3,
                   label=SCHEME_LABEL.get(sch, sch))
        ax.set_title(f"budget = {bp}%")
        ax.set_xticks(xpos); ax.set_xticklabels([f"{g}³" for g in GRIDS])
        ax.set_xlabel("grid (tile count)"); ax.grid(alpha=.3, axis="y")
        if metric == "psnr":
            ax.axhline(PSNR_SAT, ls=":", c="gray", lw=.8)
    axes[0][0].set_ylabel(metric.upper())
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"exp4 (b) {metric.upper()} vs grid | packing={packing} "
                 f"| bars=scheme | scene-level 95% CI (n=10)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="output/0618/exp4_grid/summary_all.csv")
    ap.add_argument("--out-dir", default="output/quickplots/0626/exp4")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load(a.csv)
    for pk in df.packing_mode.unique():
        for metric in ("psnr", "ssim"):
            fig_a(df, pk, metric, out / f"a_{pk}_{metric}.png")
            # fig_b (grid-on-x bars) dropped: strict info-subset of fig_a, harder to read.


if __name__ == "__main__":
    main()
