"""Plot oracle GT vs model prediction score distributions.

For each ablation: one figure with KDE of oracle GT + LGBM + XGB + RF
on held-out test cameras.

Usage:
    conda run -n gsquic python experiments/0527/plot_ml_distributions.py \\
        --model-root ml/models/ship \\
        --oracle-npz output/0522/exp4_oracle_dq/ship/oracle_dq.npz \\
        --out-dir output/quickplots/0527/ml_distributions/ship
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from ml.features import (  # noqa: E402
    build_feature_matrix, feature_names_for_ablation, ABLATION_NAMES,
)

try:
    from scipy.stats import gaussian_kde
    _HAS_KDE = True
except ImportError:
    _HAS_KDE = False


def _plot_kde(ax, data, label, color, linestyle="-"):
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return
    if _HAS_KDE:
        kde = gaussian_kde(data, bw_method=0.3)
        xs = np.linspace(data.min(), data.max(), 500)
        ax.plot(xs, kde(xs), label=label, color=color, linestyle=linestyle, linewidth=1.5)
    else:
        ax.hist(data, bins=60, density=True, label=label, color=color,
                alpha=0.4, histtype="step", linewidth=1.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True,
                        help="Root dir with {ABCD,ACD,AC,ABC}/ ablation subdirs")
    parser.add_argument("--oracle-npz", required=True,
                        help="oracle_dq.npz used for training")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root)

    print("Building feature matrix ...", flush=True)
    df, _ = build_feature_matrix(args.oracle_npz)
    df = df[df["log_mse_loo"].notna()].copy()

    # ── one figure per ablation ──────────────────────────────────────────────
    for ablation in ABLATION_NAMES:
        ablation_dir = model_root / ablation
        if not ablation_dir.exists():
            print(f"Skip {ablation} (not trained yet)", flush=True)
            continue

        # Load test camera split from metrics.json
        metrics_path = ablation_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Skip {ablation} (metrics.json missing)", flush=True)
            continue
        metrics = json.loads(metrics_path.read_text())
        test_cams = set(metrics["test_cams"])

        feat_cols = feature_names_for_ablation(ablation)
        te = df[df["camera"].isin(test_cams)]
        if len(te) == 0:
            print(f"Skip {ablation} (no test rows)", flush=True)
            continue

        X_te = te[feat_cols].values.astype(np.float32)
        y_te = te["log_mse_loo"].values

        print(f"{ablation}: {len(te):,} test rows, {len(test_cams)} cameras", flush=True)

        # ── collect predictions ──────────────────────────────────────────────
        model_preds = {}
        model_rhos  = {}
        colors = {
            "lgbm":  "#1f77b4", "xgb":   "#ff7f0e", "rf":    "#2ca02c",
            "ols":   "#9467bd", "ridge":  "#8c564b", "lasso": "#e377c2",
        }
        from scipy.stats import spearmanr

        for mtype in ("lgbm", "xgb", "rf", "ols", "ridge", "lasso"):
            pkl = ablation_dir / f"{mtype}.pkl"
            if not pkl.exists():
                continue
            model = joblib.load(pkl)
            preds = model.predict(X_te).astype(np.float64)
            model_preds[mtype] = preds
            rho, _ = spearmanr(preds, y_te)
            model_rhos[mtype] = float(rho)
            r2 = metrics.get(mtype, {}).get("r2_test", float("nan"))
            print(f"  {mtype} ρ_test = {rho:.4f}  R²_test = {r2:.4f}", flush=True)

        # ── plot ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))

        _plot_kde(ax, y_te, "Oracle GT (log MSE)", "#333333", linestyle="-")
        for mtype, preds in model_preds.items():
            rho = model_rhos[mtype]
            _plot_kde(ax, preds, f"{mtype.upper()} (ρ={rho:.3f})",
                      colors[mtype], linestyle="--")

        ax.set_xlabel("log(MSE LOO)")
        ax.set_ylabel("Density")
        ax.set_title(f"Score distribution — {ablation} ({Path(args.model_root).name})\n"
                     f"test: {len(test_cams)} cameras, {len(te):,} tiles")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"{ablation}.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
