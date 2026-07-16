#!/usr/bin/env python3
"""Plot PSNR/SSIM/VMAF vs time from a streaming_sim.py run's summary.csv.

One figure per metric, 1xN subplots faceted by bandwidth, one line per method (no
markers -- dense time series, not a sparse budget curve), x-axis = time (s). Paper
style, matching plotting/paper_plot_metrics.ipynb's settings (Times New Roman + LaTeX,
same KEY_CONFIG colors/labels as the rest of the paper) and
plotting/paper/plot_format_ref/streaming/*.png's line-only look.

Usage:
  python experiments/plot_streaming_sim.py --summary-csv <output_root>/metrics/summary.csv \
      --out-dir <output_root>/plots
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paper style (mirrors plotting/paper_plot_metrics.ipynb cell 2) ────────────
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

# Same colors/labels as plotting/paper_plot_metrics.ipynb's KEY_CONFIG, for
# cross-figure consistency in the paper.
_METHOD_CONFIG = {
    "vd_lod":  {"label": "Heuristic (baseline)", "color": "#1f77b4"},
    "v_lod_w": {"label": "Heuristic (ours)",      "color": "#8c564b"},
    "ml":      {"label": "Learned (ours)",        "color": "#2ca02c"},
}
_METRIC_YLABEL = {"psnr": "Quality in PSNR (dB)", "ssim": "Quality in SSIM", "vmaf": "Quality in VMAF"}
PSNR_SATURATION_DB = 60.0  # matches experiments/plot_metrics.py's saturation convention


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _smooth(ys: tuple, window: int) -> np.ndarray:
    """Centered moving average, window in samples. window<=1 is a no-op. Blurs real
    fast dynamics (motion-induced quality wiggles) along with sample noise, and rounds
    off the sharp step-jumps at cadence reorders into ramps -- opt-in only, not a
    substitute for denser --render-interval-sec sampling."""
    if window <= 1:
        return np.asarray(ys)
    kernel = np.ones(window) / window
    return np.convolve(ys, kernel, mode="same")


def _draw_panel(ax, rows: list[dict], metric: str, methods: list[str], bw: str,
                 smooth_window: int, title: str):
    for method in methods:
        cfg = _METHOD_CONFIG.get(method, {})
        pts = sorted(
            ((float(r["t_sec"]), float(r[metric])) for r in rows
             if r["bandwidth_mbps"] == bw and r["method"] == method and r[metric]),
            key=lambda p: p[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        if metric == "psnr":
            # A pixel-perfect (MSE=0) frame gives PSNR=inf, which poisons the moving-
            # average window (and blows up autoscale even unsmoothed) -- clip at the
            # project's existing saturation convention before either.
            ys = np.clip(np.asarray(ys), None, PSNR_SATURATION_DB)
        mean_y = float(np.mean(ys))
        avg_str = f"{mean_y:.2f}" if metric == "ssim" else f"{mean_y:.1f}"
        ax.plot(xs, _smooth(ys, smooth_window), color=cfg.get("color"), linewidth=2.0,
                 label=f"{cfg.get('label', method)} (avg {avg_str})", zorder=2)
        ax.plot([xs[0], xs[-1]], [mean_y, mean_y], color=cfg.get("color"),
                 linestyle=(0, (1, 1)), linewidth=3.0, zorder=1)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Time (sec)", fontsize=15)
    ax.tick_params(labelsize=13, direction="out", which="both", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(loc="best", framealpha=0.9, fontsize=11)


def plot_metric(rows: list[dict], metric: str, bandwidths: list[str],
                 methods: list[str], out_path: Path, smooth_window: int = 1):
    fig, axes = plt.subplots(1, len(bandwidths), figsize=(5.5 * len(bandwidths), 4.8),
                              sharey=True, constrained_layout=True)
    if len(bandwidths) == 1:
        axes = [axes]
    for ax, bw in zip(axes, bandwidths):
        _draw_panel(ax, rows, metric, methods, bw, smooth_window, title=f"{bw} Mbps")
    axes[0].set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=17)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)


def plot_representative(root: Path, traces: list[str], labels: list[str], metric: str,
                         bw: str, out_path: Path):
    fig, axes = plt.subplots(1, len(traces), figsize=(5.5 * len(traces), 4.8),
                              sharey=True, constrained_layout=True)
    if len(traces) == 1:
        axes = [axes]
    for ax, trace, label in zip(axes, traces, labels):
        rows = _read_rows(root / trace / "metrics" / "summary.csv")
        methods = [m for m in _METHOD_CONFIG if m in {r["method"] for r in rows}]
        _draw_panel(ax, rows, metric, methods, bw, smooth_window=1,
                    title=f"{label} ({trace}, {bw} Mbps)")
    axes[0].set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=17)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".eps")), format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv")
    p.add_argument("--out-dir")
    p.add_argument("--smooth-window", type=int, default=5,
                    help="Centered moving-average window, in samples, for the smoothed twin "
                         "set (always written alongside the raw plots, to <out-dir>_smoothed -- "
                         "no separate invocation needed). Cosmetic only -- prefer denser "
                         "--render-interval-sec in streaming_sim.py for a genuinely smoother "
                         "curve; this blurs real signal (motion wiggles, cadence step-jumps) "
                         "along with noise.")
    p.add_argument("--rep-root",
                    help="Representative-trace mode: dir containing <trace>/metrics/summary.csv "
                         "per trace (e.g. .../greedy_n4). Combine with --rep-traces/--rep-labels/"
                         "--rep-bandwidth/--rep-metric/--rep-out instead of --summary-csv.")
    p.add_argument("--rep-traces", nargs="+")
    p.add_argument("--rep-labels", nargs="+")
    p.add_argument("--rep-bandwidth")
    p.add_argument("--rep-metric", choices=["psnr", "ssim", "vmaf"])
    p.add_argument("--rep-out")
    args = p.parse_args()

    if args.rep_root:
        assert len(args.rep_traces) == len(args.rep_labels), "--rep-traces/--rep-labels length mismatch"
        plot_representative(Path(args.rep_root), args.rep_traces, args.rep_labels,
                             args.rep_metric, args.rep_bandwidth, Path(args.rep_out))
        return

    rows = _read_rows(Path(args.summary_csv))
    bandwidths = sorted({r["bandwidth_mbps"] for r in rows}, key=float)
    methods = [m for m in _METHOD_CONFIG if m in {r["method"] for r in rows}]

    out_dir = Path(args.out_dir)
    smoothed_dir = out_dir.parent / f"{out_dir.name}_smoothed"
    out_dir.mkdir(parents=True, exist_ok=True)
    smoothed_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("psnr", "ssim", "vmaf"):
        if any(r.get(metric) for r in rows):
            plot_metric(rows, metric, bandwidths, methods, out_dir / f"{metric}_vs_time.png",
                        smooth_window=1)
            print(f"wrote {out_dir / f'{metric}_vs_time.png'}")
            plot_metric(rows, metric, bandwidths, methods, smoothed_dir / f"{metric}_vs_time.png",
                        smooth_window=args.smooth_window)
            print(f"wrote {smoothed_dir / f'{metric}_vs_time.png'}")


if __name__ == "__main__":
    main()
