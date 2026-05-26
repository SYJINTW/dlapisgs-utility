#!/usr/bin/env python3
"""Human-readable Exp 4 summary.

Prints a markdown table to stdout and writes ONE plot:
  <root>/concentration.png  — cumulative fraction of total error captured
                              by top-K% tiles (per scene, one figure).

The concentration curve is the packer-relevant question: if I keep the
top X% of tiles, what fraction of total quality loss have I avoided?
A steeper curve means a few tiles dominate (easy packer problem); a
shallower curve means error is spread (hard packer problem).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PSNR_REF_CLAMP_DB = 100.0
SCENES_DEFAULT = ["chair", "drums", "ficus", "hotdog", "materials", "mic", "ship", "bicycle"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(np.float64)
    rb = np.argsort(np.argsort(b[m])).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _scene_stats(npz: Path) -> dict:
    d = np.load(npz, allow_pickle=True)
    mse = d["mse"].astype(np.float64)              # (cams, tiles), bigger=worse-when-removed
    n_gs = d["n_gs_per_tile"].astype(np.int64)

    per_tile_mse = mse.mean(axis=0)                # camera-averaged tile importance
    total = per_tile_mse.sum()
    if total <= 0:
        return None

    order = np.argsort(per_tile_mse)[::-1]
    sorted_ = per_tile_mse[order]
    n_gs_sorted = n_gs[order]
    cum = np.cumsum(sorted_) / total
    cum_ngs = np.cumsum(n_gs_sorted) / max(n_gs.sum(), 1)

    # "How many tiles to capture 90% of error?"
    k_for = lambda frac: int(np.searchsorted(cum, frac) + 1)
    # "Fraction of GS that those tiles hold" — packer cost
    ngs_at = lambda frac: float(cum_ngs[min(k_for(frac) - 1, len(cum_ngs) - 1)])

    # Across-camera consistency: how often is a tile in its scene's top-10 across cams?
    n_cams, n_tiles = mse.shape
    top10_per_cam = np.argsort(mse, axis=1)[:, -10:]  # (cams, 10)
    counts = np.zeros(n_tiles, dtype=np.int64)
    for row in top10_per_cam:
        counts[row] += 1
    consistency = float((counts.max() / n_cams)) if n_cams else 0.0  # frac of cams that share #1 top tile

    return dict(
        n_cams=n_cams,
        n_tiles=n_tiles,
        total_gs=int(n_gs.sum()),
        k_p50=k_for(0.50), ngs_p50=ngs_at(0.50),
        k_p90=k_for(0.90), ngs_p90=ngs_at(0.90),
        k_p99=k_for(0.99), ngs_p99=ngs_at(0.99),
        rho_dmse_ngs=_spearman(per_tile_mse, n_gs.astype(np.float64)),
        consistency=consistency,
        cum=cum,
        x_frac=np.arange(1, len(cum) + 1) / len(cum),
        x_ngs_frac=cum_ngs,
    )


def _print_md_table(rows: list[tuple[str, dict]]) -> None:
    # Columns chosen to answer: how concentrated is error? do tiles consistent across cams? does AOI confine error?
    headers = [
        "scene", "K", "total_GS",
        "k50 (ngs%)", "k90 (ngs%)", "k99 (ngs%)",
        "ρ(ΔMSE,ngs)", "consist",
    ]

    def fmt(r):
        return [
            r["_name"],
            f"{r['n_tiles']}",
            f"{r['total_gs']/1e6:.2f}M",
            f"{r['k_p50']} ({r['ngs_p50']*100:.0f}%)",
            f"{r['k_p90']} ({r['ngs_p90']*100:.0f}%)",
            f"{r['k_p99']} ({r['ngs_p99']*100:.0f}%)",
            f"{r['rho_dmse_ngs']:+.2f}",
            f"{r['consistency']*100:.0f}%",
        ]

    rendered = [headers] + [fmt({**r[1], "_name": r[0]}) for r in rows]
    widths = [max(len(c) for c in col) for col in zip(*rendered)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print(line(rendered[0]))
    print(sep)
    for row in rendered[1:]:
        print(line(row))

    print()
    print("Legend:")
    print("  K            — # non-empty tiles")
    print("  k50/k90/k99  — # top tiles needed to cover 50/90/99% of total ΔMSE")
    print("                 (ngs%) = fraction of all Gaussians those tiles hold = packer cost")
    print("  ρ(ΔMSE,ngs)  — Spearman ρ of per-tile mean ΔMSE vs tile size")
    print("                 high ρ → tile count alone ranks tiles well; low → other signal needed")
    print("  consist      — fraction of cameras agreeing on the single most-important tile")


def _plot_concentration(rows: list[tuple[str, dict]], out: Path) -> None:
    """Two-panel rate-distortion view under perfect oracle tile ranking.

    Left:  cum ΔMSE captured vs cum % of TILES kept (importance concentration).
    Right: cum ΔMSE captured vs cum % of GAUSSIANS kept (packer ceiling — the
           one that actually matters for byte budget).
    A flat-then-steep curve on the right means "important tiles are also huge"
    → byte budget gives you little leverage. Steep-then-flat means
    "important tiles are small" → huge gains possible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (name, r) in enumerate(rows):
        c = cmap(i % 10)
        axes[0].plot(r["x_frac"] * 100, r["cum"] * 100,
                     label=name, color=c, linewidth=1.8)
        axes[1].plot(r["x_ngs_frac"] * 100, r["cum"] * 100,
                     label=name, color=c, linewidth=1.8)

    for ax, xlabel, title in [
        (axes[0], "top-K% of tiles (sorted by mean ΔMSE)",
         "Tile concentration — how many tiles matter?"),
        (axes[1], "cumulative % of Gaussians kept (those tiles' GS sum)",
         "Packer ceiling — rate-distortion under perfect oracle"),
    ]:
        ax.axhline(90, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(99, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.plot([0, 100], [0, 100], color="black", linestyle=":",
                linewidth=0.6, alpha=0.5, label="y=x (no leverage)")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("cumulative % of total ΔMSE captured", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="output/0522/exp4_oracle_dq")
    p.add_argument("--scenes", nargs="+", default=SCENES_DEFAULT)
    p.add_argument("--plot", default=None,
                   help="Concentration-curve plot path; default <root>/concentration.png")
    args = p.parse_args()

    root = Path(args.root)
    rows: list[tuple[str, dict]] = []
    for scene in args.scenes:
        npz = root / scene / "oracle_dq.npz"
        if not npz.exists():
            print(f"[{scene}] missing {npz} — skipped")
            continue
        s = _scene_stats(npz)
        if s is None:
            print(f"[{scene}] zero total ΔMSE — skipped")
            continue
        rows.append((scene, s))

    if not rows:
        print("No data.")
        return

    _print_md_table(rows)
    _plot_concentration(rows, Path(args.plot) if args.plot else root / "concentration.png")


if __name__ == "__main__":
    main()
