#!/usr/bin/env python3
"""Plot PSNR/SSIM/VMAF vs time from a streaming_sim.py run's summary.csv.

One figure per metric, 1xN subplots faceted by bandwidth, one line per method,
x-axis = interval time (seconds). Styled on experiments/plot_metrics.py's
conventions (Agg backend, marker/color palette) -- not a copy, just reuse.

Usage:
  python experiments/plot_streaming_sim.py --summary-csv <output_root>/metrics/summary.csv \
      --out-dir <output_root>/plots
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_METHOD_LABELS = {
    "vd_lod": "VD+LOD (baseline)",
    "v_lod_w": "V+LOD+W (ours)",
    "ml": "ML (ours)",
}
MARKERS = ["s", "^", "o", "D", "v", "P"]
COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#FF8C00", "#00CED1"]


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(rows: list[dict], metric: str, bandwidths: list[str],
                 methods: list[str], out_path: Path):
    fig, axes = plt.subplots(1, len(bandwidths), figsize=(5 * len(bandwidths), 4),
                              sharey=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bandwidths):
        for i, method in enumerate(methods):
            pts = sorted(
                ((float(r["t_sec"]), float(r[metric])) for r in rows
                 if r["bandwidth_mbps"] == bw and r["method"] == method and r[metric]),
                key=lambda p: p[0])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                     label=_METHOD_LABELS.get(method, method))
        ax.set_title(f"{bw} Mbps")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric.upper())
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    rows = _read_rows(Path(args.summary_csv))
    bandwidths = sorted({r["bandwidth_mbps"] for r in rows}, key=float)
    methods = sorted({r["method"] for r in rows})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("psnr", "ssim", "vmaf"):
        if any(r.get(metric) for r in rows):
            plot_metric(rows, metric, bandwidths, methods, out_dir / f"{metric}_vs_time.png")
            print(f"wrote {out_dir / f'{metric}_vs_time.png'}")


if __name__ == "__main__":
    main()
