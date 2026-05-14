#!/usr/bin/env python3
"""Paper-ready PSNR/SSIM vs budget plots.

Reads metrics/summary.csv and writes publication-quality figures
following the Times New Roman / usetex style used in the project's
reference plotter notebook.

Usage:
    python experiments/paper_plot_metrics.py \
        --summary-csv output/0513_setup2_progressive/metrics/summary.csv \
        --out-dir     output/0513_setup2_progressive/metrics/paper \
        --group-by    weight_mode
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

# ---------------------------------------------------------------------------
# Global style (mirrors plotter.ipynb / adhoc_plotter_to_refactor.ipynb)
# ---------------------------------------------------------------------------
_CSFONT = {"family": "Times New Roman", "serif": "Times", "size": 18}
plt.rc("text", usetex=True)
plt.rc("font", **_CSFONT)

_FIGSIZE    = (6.4, 4.8)
_ERR_LW     = 1.5
_ERR_CAP    = 4
_ERR_CAPTHK = 1.5
_MARKER_SZ  = 7
_LINE_W     = 1.8

# ---------------------------------------------------------------------------
# Weight-mode ordering and labels
# ---------------------------------------------------------------------------
_WEIGHT_MODE_ORDER = [
    "det_gamma_over_d2",
    "volume",
    "volume_over_d2",
    "screen_area",
]
_WEIGHT_MODE_LABELS = {
    "det_gamma_over_d2": r"Vol$^2$/d$^2$ (original)",
    "volume":            r"Volume (view-indep.)",
    "volume_over_d2":    r"Vol/d$^2$ (view-dep.)",
    "screen_area":       r"Screen Area",
}
_WEIGHT_MODE_MARKERS = {
    "det_gamma_over_d2": "s",
    "volume":            "^",
    "volume_over_d2":    "D",
    "screen_area":       "o",
}
# Colour palette matches plotter.ipynb (matplotlib default cycle)
_WEIGHT_MODE_COLORS = {
    "det_gamma_over_d2": "#1f77b4",
    "volume":            "#2ca02c",
    "volume_over_d2":    "#d62728",
    "screen_area":       "#9467bd",
}

# Scheme ordering and labels
_SCHEME_ORDER = ["vd_lod", "vd_lod_w", "vd_lod_c", "vd_lod_w_c"]
_SCHEME_LABELS = {
    "vd_lod":     r"VD+LOD (baseline)",
    "vd_lod_w":   r"VD+LOD+W",
    "vd_lod_c":   r"VD+LOD+C",
    "vd_lod_w_c": r"VD+LOD+W+C (proposed)",
}
_SCHEME_MARKERS = {k: m for k, m in zip(_SCHEME_ORDER, ["s", "^", "D", "o"])}
_SCHEME_COLORS  = {
    "vd_lod":     "#1f77b4",
    "vd_lod_w":   "#ff7f0e",
    "vd_lod_c":   "#2ca02c",
    "vd_lod_w_c": "#d62728",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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
            result[key][budget_mb] = {
                "psnr_mean": float(psnr_vals.mean()),
                "psnr_std":  float(psnr_vals.std()),
                "ssim_mean": float(ssim_vals.mean()),
                "ssim_std":  float(ssim_vals.std()),
                "n":         len(entries),
            }
    return result


def _resolve_style(group_by: str) -> tuple[list[str], dict, dict, dict]:
    """Return (order, labels, markers, colors) for the chosen grouping."""
    if group_by == "weight_mode":
        return (_WEIGHT_MODE_ORDER, _WEIGHT_MODE_LABELS,
                _WEIGHT_MODE_MARKERS, _WEIGHT_MODE_COLORS)
    if group_by == "scheme":
        return (_SCHEME_ORDER, _SCHEME_LABELS,
                _SCHEME_MARKERS, _SCHEME_COLORS)
    # generic fallback — use matplotlib defaults
    fallback_colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fallback_markers = ["s", "^", "D", "o", "v", "P"]
    keys = sorted({})  # filled at call site
    return ([], {}, {}, {})

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_metric(
    agg: dict[str, dict[float, dict]],
    order: list[str],
    labels: dict[str, str],
    markers: dict[str, str],
    colors: dict[str, str],
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    present = [k for k in order if k in agg]
    # include any keys not in preset order
    extras = [k for k in sorted(agg.keys()) if k not in order]
    fallback_colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fallback_markers = ["s", "^", "D", "o", "v", "P"]

    for i, key in enumerate(present + extras):
        if key not in agg:
            continue
        pts = sorted(agg[key].items())
        x   = np.array([p[0] for p in pts])
        y   = np.array([p[1][f"{metric}_mean"] for p in pts])
        err = np.array([p[1][f"{metric}_std"]  for p in pts])

        marker = markers.get(key, fallback_markers[i % len(fallback_markers)])
        color  = colors.get(key,  fallback_colors[i % len(fallback_colors)])
        label  = labels.get(key,  key)

        ax.errorbar(
            x, y, yerr=err,
            marker=marker, color=color,
            linewidth=_LINE_W, markersize=_MARKER_SZ,
            capsize=_ERR_CAP, elinewidth=_ERR_LW, capthick=_ERR_CAPTHK,
            label=label,
        )

    ax.set_xlabel(r"Budget (MiB)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13, framealpha=0.9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-quality PSNR/SSIM vs budget plots")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out-dir",     required=True)
    parser.add_argument("--group-by",    default="weight_mode",
                        help="CSV column to group lines by (default: weight_mode)")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    rows = _read_csv(summary_csv)
    print(f"Loaded {len(rows)} rows, grouping by '{args.group_by}'")

    agg = _aggregate(rows, args.group_by)
    order, labels, markers, colors = _resolve_style(args.group_by)

    # fill generic fallback order if needed
    if not order:
        order = sorted(agg.keys())

    _plot_metric(agg, order, labels, markers, colors,
                 "psnr", r"PSNR (dB)", out_dir / "psnr_vs_budget.pdf")
    _plot_metric(agg, order, labels, markers, colors,
                 "psnr", r"PSNR (dB)", out_dir / "psnr_vs_budget.png")

    _plot_metric(agg, order, labels, markers, colors,
                 "ssim", r"SSIM", out_dir / "ssim_vs_budget.pdf")
    _plot_metric(agg, order, labels, markers, colors,
                 "ssim", r"SSIM", out_dir / "ssim_vs_budget.png")


if __name__ == "__main__":
    main()
