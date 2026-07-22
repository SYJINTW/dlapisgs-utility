#!/usr/bin/env python3
"""Standalone helper for paper_plot_metrics.ipynb's quality/latency tradeoff figures
(psnr|ssim|lpips_vs_selection_time.png/.eps). MUST be run with a Python whose matplotlib
is >=3.10, NOT the `gsquic` conda env's Python (matplotlib 3.9.2) -- 2026-07-22 root
cause, confirmed by direct A/B: matplotlib 3.9.2's EPS bbox_inches="tight" + LaTeX +
Ghostscript-crop pipeline silently produces a structurally-valid but visually blank EPS
for this exact plot shape (errorbar with BOTH xerr and yerr on a log-x axis, plus the
"Better" annotate arrow) -- reproducible every time under 3.9.2, in-process or in a fresh
subprocess, with hand-typed data or the real (verified finite, no inf/nan) pipeline data;
matplotlib 3.10.6 (the box's base anaconda env) renders the identical figure correctly
every time. None of this notebook's other ~40 figures hit it because none combine
log-scale + dual-error errorbar + annotate the way this one does. The caller
(paper_plot_metrics.ipynb) invokes this via `subprocess.run(["python3", ...])`, not
`sys.executable`, specifically to escape the gsquic env's matplotlib -- don't change that
back to sys.executable, it reintroduces the blank EPS.

Usage: python3 plotting/_plot_tradeoff_subprocess.py <input_json> <out_dir>
Input JSON keys: timing_pooled_agg, quality_tradeoff_agg (per metric), tradeoff_ylabels
(per metric: [ylabel_base, lower_better]), cond_order, cond_labels, budget_pct.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csfont = {"family": "serif", "serif": ["Times New Roman", "Times"], "size": 23}
try:
    plt.rc("text", usetex=True)
    plt.rc("font", **csfont)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}"
    fig = plt.figure()
    fig.text(0.5, 0.5, r"test $x^2$")
    fig.canvas.draw()
    plt.close(fig)
except Exception:
    plt.rc("text", usetex=False)
    plt.rc("font", **csfont)

DPI = 300
color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
err_lw, err_capsize, err_capthick = 1.5, 4, 1.5


def plot_metric_vs_time(bar_agg, metric_agg, cond_order, cond_labels, ylabel_base,
                         lower_better, budget_pct, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    markers = ["o", "s", "^", "D", "P", "*"]
    for i, cond in enumerate(cond_order):
        if cond not in bar_agg or cond not in metric_agg:
            continue
        x, xerr = bar_agg[cond]["mean"], bar_agg[cond]["ci95"]
        y, yerr = metric_agg[cond]["mean"], metric_agg[cond]["ci95"]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    marker=markers[i % len(markers)], markersize=13,
                    color=color_palette[i % len(color_palette)],
                    capsize=err_capsize, elinewidth=err_lw, capthick=err_capthick,
                    label=cond_labels[cond].replace("\n", " "))

    ax.set_xscale("log")
    ax.set_xlabel(r"Selection Latency (ms/frame)", fontsize=18)
    _bpct_tex = budget_pct.replace("%", r"\%")
    _suffix = " (Lower Is Better)" if lower_better else ""
    ax.set_ylabel(f"{ylabel_base} @ {_bpct_tex} Budget{_suffix}", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15, framealpha=0.9, loc="best")
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(direction="out", which="both", top=False, right=False)

    corner_y = 0.06 if lower_better else 0.94
    tail_y = 0.32 if lower_better else 0.68
    ax.annotate("", xy=(0.03, corner_y), xytext=(0.24, tail_y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.6, color="dimgray"))
    ax.text(0.26, tail_y, "Better", transform=ax.transAxes, fontsize=15,
            color="dimgray", ha="left", va="center", style="italic")

    fig.set_constrained_layout(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".eps"), format="eps", bbox_inches="tight")
    print(f"Wrote {out_path}")
    plt.close(fig)


def main():
    in_json, out_dir = sys.argv[1], Path(sys.argv[2])
    data = json.loads(Path(in_json).read_text())
    for metric, (ylabel_base, lower_better) in data["tradeoff_ylabels"].items():
        plot_metric_vs_time(
            data["timing_pooled_agg"], data["quality_tradeoff_agg"][metric],
            data["cond_order"], data["cond_labels"], ylabel_base, lower_better,
            data["budget_pct"], out_dir / f"{metric}_vs_selection_time.png")
    print("Done.")


if __name__ == "__main__":
    main()
