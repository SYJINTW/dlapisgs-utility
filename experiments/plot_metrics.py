#!/usr/bin/env python3
"""Plot PSNR and SSIM vs budget from a render+metrics sweep.

Reads metrics/summary.csv produced by render_metrics.py and generates:
  - psnr_vs_budget.png
  - ssim_vs_budget.png
Each has one line per group (controlled by --group-by) with error bars (std across cameras).
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

# Stable label/order/style for the default scheme grouping.
_SCHEME_LABELS = {
    "vd_lod":     "VD+LOD (baseline)",
    "vd_lod_w":   "VD+LOD+W",
    "vd_lod_c":   "VD+LOD+C",
    "vd_lod_w_c": "VD+LOD+W+C (proposed)",
}
_SCHEME_ORDER = ["vd_lod", "vd_lod_w", "vd_lod_c", "vd_lod_w_c"]

# For weight_mode grouping — worst-to-best order matches the hypothesis.
_WEIGHT_MODE_ORDER = ["det_gamma_over_d2", "volume", "volume_over_d2", "screen_area"]
_WEIGHT_MODE_LABELS = {
    "det_gamma_over_d2": "det_γ/d² (original, vol²/d²)",
    "volume":            "volume (vol, view-indep.)",
    "volume_over_d2":    "volume/d² (vol/d², view-dep.)",
    "screen_area":       "screen_area (projected footprint)",
}

MARKERS = ["s", "^", "D", "o", "v", "P"]
COLORS  = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#FF8C00", "#00CED1"]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict], group_by: str) -> dict[str, dict[float, dict]]:
    buckets: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r.get(group_by) or f"missing_{group_by}"
        buckets[key][float(r["budget_mb"])].append(r)

    result: dict[str, dict[float, dict]] = {}
    for key, budgets in buckets.items():
        result[key] = {}
        for budget_mb, entries in budgets.items():
            psnr_vals = np.array([float(e["psnr"]) for e in entries])
            ssim_vals = np.array([float(e["ssim"]) for e in entries])
            n = len(entries)
            # 95% CI on the mean: ±1.96·σ/√n. Tight when n is large; this is the
            # right bar for "is method A's mean above method B's mean?"
            # std would answer "how variable is a single camera?" — not what we want.
            ci = 1.96 / max(np.sqrt(n), 1.0)
            result[key][budget_mb] = {
                "psnr_mean": float(psnr_vals.mean()),
                "psnr_ci95": float(psnr_vals.std(ddof=1) * ci) if n > 1 else 0.0,
                "ssim_mean": float(ssim_vals.mean()),
                "ssim_ci95": float(ssim_vals.std(ddof=1) * ci) if n > 1 else 0.0,
                "n":         n,
            }
    return result


def _resolve_order_and_labels(agg: dict, group_by: str) -> tuple[list[str], dict[str, str]]:
    if group_by == "scheme":
        order = _SCHEME_ORDER
        labels = _SCHEME_LABELS
    elif group_by == "weight_mode":
        order = _WEIGHT_MODE_ORDER
        labels = _WEIGHT_MODE_LABELS
    else:
        order = sorted(agg.keys())
        labels = {}
    # include any keys not in the preset order (future-proofing)
    extras = [k for k in sorted(agg.keys()) if k not in order]
    return order + extras, labels


def _plot(agg: dict[str, dict[float, dict]], order: list[str], labels: dict[str, str],
          metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key, marker, color in zip(order, MARKERS, COLORS):
        if key not in agg:
            continue
        pts = sorted(agg[key].items())
        x   = [p[0] for p in pts]
        y   = [p[1][f"{metric}_mean"] for p in pts]
        err = [p[1][f"{metric}_ci95"] for p in pts]
        ax.errorbar(x, y, yerr=err, marker=marker, color=color,
                    linewidth=1.8, markersize=6, capsize=3,
                    label=labels.get(key, key))
    ax.set_xlabel("Budget (MiB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)
    # Saturation guide for PSNR plots only: 60 dB ⇒ MSE < 10⁻⁶ ⇒ visually identical
    # even before 8-bit PNG quantization (which itself sits around 48 dB).
    if metric == "psnr":
        ax.axhline(60.0, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(ax.get_xlim()[1], 60.0, " saturated (≥60 dB)",
                fontsize=9, color="gray", va="bottom", ha="right")
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
    parser.add_argument("--group-by", default="scheme",
                        help="CSV column to group lines by (default: scheme). "
                             "Use 'weight_mode' for setup2, 'w_norm' for setup1, etc.")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    rows = _read_csv(summary_csv)
    print(f"Loaded {len(rows)} rows from {summary_csv}, grouping by '{args.group_by}'")
    agg = _aggregate(rows, args.group_by)
    order, labels = _resolve_order_and_labels(agg, args.group_by)

    n_views = max(v["n"] for s in agg.values() for v in s.values())
    suffix  = f" — {args.title_suffix}" if args.title_suffix else ""
    n_groups = len([k for k in order if k in agg])

    _plot(agg, order, labels, "psnr", "PSNR (dB)",
          f"PSNR vs Budget — {n_groups} {args.group_by}s (mean ± 95% CI, {n_views} views){suffix}",
          out_dir / "psnr_vs_budget.png")

    _plot(agg, order, labels, "ssim", "SSIM",
          f"SSIM vs Budget — {n_groups} {args.group_by}s (mean ± 95% CI, {n_views} views){suffix}",
          out_dir / "ssim_vs_budget.png")


if __name__ == "__main__":
    main()
