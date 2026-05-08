#!/usr/bin/env python3
"""Plot PSNR and SSIM vs budget from a render+metrics sweep.

Reads metrics/summary.csv produced by render_metrics.py and generates:
  - psnr_vs_budget.png
  - ssim_vs_budget.png
Each has one line per scheme with error bars (std across cameras).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEME_LABELS = {
    "vd_lod":     "VD+LOD (baseline)",
    "vd_lod_w":   "VD+LOD+W",
    "vd_lod_c":   "VD+LOD+C",
    "vd_lod_w_c": "VD+LOD+W+C (proposed)",
}
SCHEME_ORDER = ["vd_lod", "vd_lod_w", "vd_lod_c", "vd_lod_w_c"]
MARKERS = ["s", "^", "D", "o"]
COLORS  = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict]) -> dict[str, dict[float, dict]]:
    buckets: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        buckets[r["scheme"]][float(r["budget_mb"])].append(r)

    result: dict[str, dict[float, dict]] = {}
    for scheme, budgets in buckets.items():
        result[scheme] = {}
        for budget_mb, entries in budgets.items():
            psnr_vals = np.array([float(e["psnr"]) for e in entries])
            ssim_vals = np.array([float(e["ssim"]) for e in entries])
            result[scheme][budget_mb] = {
                "psnr_mean": float(psnr_vals.mean()),
                "psnr_std":  float(psnr_vals.std()),
                "ssim_mean": float(ssim_vals.mean()),
                "ssim_std":  float(ssim_vals.std()),
                "n":         len(entries),
            }
    return result


def _plot(agg: dict[str, dict[float, dict]], metric: str, ylabel: str,
          title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for scheme, marker, color in zip(SCHEME_ORDER, MARKERS, COLORS):
        if scheme not in agg:
            continue
        pts = sorted(agg[scheme].items())
        x   = [p[0] for p in pts]
        y   = [p[1][f"{metric}_mean"] for p in pts]
        err = [p[1][f"{metric}_std"]  for p in pts]
        ax.errorbar(x, y, yerr=err, marker=marker, color=color,
                    linewidth=1.8, markersize=6, capsize=3,
                    label=SCHEME_LABELS.get(scheme, scheme))
    ax.set_xlabel("Budget (MiB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PSNR/SSIM vs budget")
    parser.add_argument("--summary-csv", required=True,
                        help="Path to metrics/summary.csv")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write PNG plots into")
    parser.add_argument("--title-suffix", default="",
                        help="Appended to plot titles (e.g. '8x8x8 grid')")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    rows = _read_csv(summary_csv)
    print(f"Loaded {len(rows)} rows from {summary_csv}")
    agg = _aggregate(rows)

    n_views = max(v["n"] for s in agg.values() for v in s.values())
    suffix  = f" — {args.title_suffix}" if args.title_suffix else ""

    _plot(agg, "psnr", "PSNR (dB)",
          f"PSNR vs Budget — 4 Schemes (mean ± std, {n_views} views){suffix}",
          out_dir / "psnr_vs_budget.png")

    _plot(agg, "ssim", "SSIM",
          f"SSIM vs Budget — 4 Schemes (mean ± std, {n_views} views){suffix}",
          out_dir / "ssim_vs_budget.png")


if __name__ == "__main__":
    main()
