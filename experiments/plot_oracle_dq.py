#!/usr/bin/env python3
"""Plots over the LOO oracle ΔQ artifact (Exp 4).

Reads `output/0519/exp4_oracle_dq/<scene>/oracle_dq.npz` and writes:

  plots/
    hist_mean_<metric>.png       per-tile distribution of camera-averaged ΔQ
    heatmap_<metric>.png         (camera × tile) heatmap, tiles sorted by mean
    scatter_<metric>_vs_ngs.png  per-tile mean ΔQ vs Gaussians-in-tile
    top_tiles_<metric>.png       top-20 tile ranking bar (mean ± 95% CI over cams)

`metric` ∈ {mse, psnr_delta, ssim_delta}. PSNR and SSIM are reported as the
*drop* from the saturated reference (ΔPSNR = PSNR_ref − PSNR_-k, ΔSSIM =
1 − SSIM_-k) so "bigger = matters more", matching the MSE convention.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oracle_dq import OracleDQResult

# PSNR(R_full,R_full) = +inf; clamp to 100 dB so ΔPSNR is finite.
PSNR_REF_CLAMP_DB = 100.0


def _build_metric_views(r: OracleDQResult) -> dict[str, np.ndarray]:
    """Return {name: (N_cams, N_tiles) array} where bigger = matters more."""
    psnr_clamped = np.minimum(r.psnr, PSNR_REF_CLAMP_DB)
    return {
        "mse":         r.mse.astype(np.float64),
        "psnr_delta":  (PSNR_REF_CLAMP_DB - psnr_clamped).astype(np.float64),
        "ssim_delta":  (1.0 - r.ssim).astype(np.float64),
    }


_METRIC_LABEL = {
    "mse":        "ΔMSE",
    "psnr_delta": "ΔPSNR (dB, clamped ref=100)",
    "ssim_delta": "1 − SSIM",
}


def _plot_hist(per_tile_mean: np.ndarray, metric: str, scene: str, out: Path) -> None:
    """Log-x histogram. All metrics here are heavy-tailed (most tiles ~0, a few large).
    Linear-x bins collapse the visible part into the leftmost bin; log-x bins
    spread the tail so the shape of "few important tiles" is readable.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    nz = per_tile_mean[per_tile_mean > 0]
    n_zero = int((per_tile_mean <= 0).sum())

    if nz.size:
        floor = max(nz.min(), 1e-12)
        bins = np.logspace(np.log10(floor), np.log10(nz.max()), 40)
    else:
        bins = 40
    ax.hist(nz, bins=bins, color="#4878CF", edgecolor="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(f"per-tile mean {_METRIC_LABEL[metric]} (avg over cameras, log scale)",
                  fontsize=11)
    ax.set_ylabel("# tiles", fontsize=11)
    extra = f" ({n_zero} tiles ≤0 excluded)" if n_zero else ""
    ax.set_title(
        f"{scene}: tile importance distribution ({_METRIC_LABEL[metric]}){extra}\n"
        f"N_tiles={len(per_tile_mean)}, "
        f"min={per_tile_mean.min():.3e}, median={np.median(per_tile_mean):.3e}, "
        f"max={per_tile_mean.max():.3e}",
        fontsize=12,
    )
    ax.grid(alpha=0.25, which="both")
    med = np.median(per_tile_mean)
    if med > 0:
        ax.axvline(med, color="red", linestyle="--",
                   linewidth=1.2, label=f"median ({med:.2e})")
    if nz.size:
        ax.axvline(nz.mean(), color="orange", linestyle=":",
                   linewidth=1.2, label=f"mean ({nz.mean():.2e})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def _plot_heatmap(arr: np.ndarray, metric: str, scene: str, out: Path) -> None:
    """(cams × tiles) with tiles sorted by descending mean. log-scale colors."""
    tile_order = np.argsort(arr.mean(axis=0))[::-1]
    sorted_ = arr[:, tile_order]
    # log color norm; clip tiny noise floor
    floor = max(sorted_[sorted_ > 0].min(initial=1e-12), 1e-12) if (sorted_ > 0).any() else 1e-12
    plot_arr = np.clip(sorted_, floor, None)

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    im = ax.imshow(plot_arr, aspect="auto", origin="lower",
                   norm=matplotlib.colors.LogNorm(vmin=floor, vmax=plot_arr.max()),
                   cmap="viridis")
    ax.set_xlabel(f"tile (sorted by mean {_METRIC_LABEL[metric]}, ↓)", fontsize=11)
    ax.set_ylabel("camera index", fontsize=11)
    ax.set_title(f"{scene}: {_METRIC_LABEL[metric]} heatmap (cam × tile, log color)",
                 fontsize=12)
    fig.colorbar(im, ax=ax, label=_METRIC_LABEL[metric])
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def _plot_scatter_vs_ngs(per_tile_mean: np.ndarray, n_gs: np.ndarray,
                         metric: str, scene: str, out: Path) -> None:
    """Log-log scatter with a least-squares fit in log-space.

    The fit answers "does ΔQ scale as a power of tile size?" — i.e.
    ΔQ ≈ a · n_gs^b. We report both:
      - Pearson R² of the log-log linear fit (power-law strength).
      - Spearman ρ on raw values (monotone-rank strength, robust to outliers).
    Spearman is the one to trust for "does utility rank well by n_gs alone?"
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    valid = (per_tile_mean > 0) & (n_gs > 0)
    n_valid = int(valid.sum())

    sc = ax.scatter(n_gs, per_tile_mean, c=per_tile_mean,
                    cmap="viridis", s=22, alpha=0.85,
                    norm=matplotlib.colors.LogNorm(
                        vmin=max(per_tile_mean[per_tile_mean > 0].min(initial=1e-12), 1e-12),
                        vmax=per_tile_mean.max()) if (per_tile_mean > 0).any() else None,
                    edgecolors="black", linewidths=0.3)

    annot = ""
    if n_valid >= 3:
        lx = np.log10(n_gs[valid].astype(np.float64))
        ly = np.log10(per_tile_mean[valid].astype(np.float64))
        slope, intercept = np.polyfit(lx, ly, 1)
        ly_hat = slope * lx + intercept
        ss_res = float(((ly - ly_hat) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Spearman without scipy — rank-correlate raw values.
        rx = np.argsort(np.argsort(n_gs[valid])).astype(np.float64)
        ry = np.argsort(np.argsort(per_tile_mean[valid])).astype(np.float64)
        rho = float(np.corrcoef(rx, ry)[0, 1]) if rx.size > 1 else float("nan")

        x_line = np.array([n_gs[valid].min(), n_gs[valid].max()], dtype=np.float64)
        y_line = 10.0 ** (slope * np.log10(x_line) + intercept)
        ax.plot(x_line, y_line, color="red", linewidth=1.8, linestyle="--",
                label=f"R² = {r2:.3f}\nSpearman ρ = {rho:.3f}")
        ax.legend(fontsize=10, loc="lower right", framealpha=0.85)
        annot = f"  (R²={r2:.3f}, ρ={rho:.3f})"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("# Gaussians in tile", fontsize=11)
    ax.set_ylabel(f"per-tile mean {_METRIC_LABEL[metric]}", fontsize=11)
    ax.set_title(f"{scene}: tile importance vs tile size{annot}", fontsize=12)
    ax.grid(alpha=0.25, which="both")
    fig.colorbar(sc, ax=ax, label=_METRIC_LABEL[metric])
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def _plot_top_tiles(arr: np.ndarray, n_gs: np.ndarray, metric: str,
                    scene: str, out: Path, top_n: int = 20) -> None:
    mean_per_tile = arr.mean(axis=0)
    n_cams = arr.shape[0]
    ci95 = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(n_cams)
    order = np.argsort(mean_per_tile)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    y = np.arange(top_n)
    ax.barh(y, mean_per_tile[order], xerr=ci95[order],
            color="#D65F5F", edgecolor="black", alpha=0.85,
            error_kw={"capsize": 3, "elinewidth": 0.8})
    labels = [f"tile {int(k)}  (n_gs={int(n_gs[k])})" for k in order]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean {_METRIC_LABEL[metric]} (± 95% CI over cameras)", fontsize=11)
    ax.set_title(f"{scene}: top-{top_n} tiles by {_METRIC_LABEL[metric]}", fontsize=12)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_one_scene(npz_path: Path, out_dir: Path, scene: str) -> None:
    print(f"[{scene}] loading {npz_path}")
    r = OracleDQResult.load_npz(npz_path)
    print(f"  shape = {r.mse.shape}   (N_cams, N_tiles)")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _build_metric_views(r)
    n_gs = r.n_gs_per_tile

    for name, arr in metrics.items():
        per_tile_mean = arr.mean(axis=0)
        _plot_hist(per_tile_mean, name, scene, out_dir / f"hist_mean_{name}.png")
        _plot_heatmap(arr, name, scene, out_dir / f"heatmap_{name}.png")
        _plot_scatter_vs_ngs(per_tile_mean, n_gs, name, scene,
                             out_dir / f"scatter_{name}_vs_ngs.png")
        _plot_top_tiles(arr, n_gs, name, scene, out_dir / f"top_tiles_{name}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="output/0519/exp4_oracle_dq",
                   help="Root holding <scene>/oracle_dq.npz subdirs")
    p.add_argument("--scenes", nargs="+", default=["bicycle", "ship"])
    p.add_argument("--out-subdir", default="plots",
                   help="Written under <root>/<scene>/<out-subdir>/")
    args = p.parse_args()

    root = Path(args.root)
    for scene in args.scenes:
        npz = root / scene / "oracle_dq.npz"
        if not npz.exists():
            print(f"[{scene}] missing {npz} — skipped")
            continue
        plot_one_scene(npz, root / scene / args.out_subdir, scene)


if __name__ == "__main__":
    main()
