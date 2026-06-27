"""Out-of-fold (OOF) evaluation of the per-scene RF utility model.

Reproduces the 5-fold camera-blocked CV from ml/train.py to collect out-of-fold
predictions, then reports the metrics that training did NOT log — RMSE, MAE — next
to R² and Spearman ρ, plus distribution/agreement plots.

Why both R² and ρ: label log(mse_loo) is heavy-tailed (a few large-ΔMSE tiles, a
dense mass of near-zero tiles). R² (variance of the value) is dominated by the few
big tiles → high; ρ (rank of every pair) is dragged down by mis-ranking the tied
small mass. RMSE/MAE quantify the value error directly.

Usage (gaussian_splatting env — needs ml.features / torch / PLY):
    conda run -n gaussian_splatting python experiments/0612/eval_ml_oof.py \
        --oracle-npz output/0606/exp4_oracle_dq/train/ship/oracle_dq.npz \
        --scene ship --ablation AC --cv-folds 5 --seed 0 \
        --out-dir output/quickplots/ml_oof/ship \
        --metrics-csv output/quickplots/ml_oof/_oof_metrics.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib  # noqa: F401  (parity with training env)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
from ml.features import build_feature_matrix, feature_names_for_ablation  # noqa: E402


def oof_predict(df, feat_cols, n_folds, seed):
    """Camera-blocked K-fold; return (y_true, y_oof) over all rows."""
    cams = np.array(sorted(df["camera"].unique()))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true = df["log_mse_loo"].values.astype(np.float64)
    y_oof = np.full(len(df), np.nan, dtype=np.float64)
    cam_col = df["camera"].values
    X = df[feat_cols].values.astype(np.float32)
    for fold, (tr_c, te_c) in enumerate(kf.split(cams)):
        tr_cams, te_cams = set(cams[tr_c]), set(cams[te_c])
        tr = np.isin(cam_col, list(tr_cams))
        te = np.isin(cam_col, list(te_cams))
        rf = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y_true[tr])
        y_oof[te] = rf.predict(X[te])
        print(f"  fold {fold}: train {tr.sum()} / test {te.sum()}", flush=True)
    return y_true, y_oof


def metrics(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    resid = p - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = float(spearmanr(p, y).correlation)
    return dict(rmse=rmse, mae=mae, r2=r2, rho=rho, n=int(len(y)))


def kde_line(ax, data, label, color, ls="-"):
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return
    kde = gaussian_kde(data, bw_method=0.3)
    xs = np.linspace(data.min(), data.max(), 500)
    ax.plot(xs, kde(xs), label=label, color=color, linestyle=ls, linewidth=1.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-npz", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ablation", default="AC")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics-csv", default=None, help="append per-scene metrics row")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.scene}] building features ...", flush=True)
    feat_cols = feature_names_for_ablation(args.ablation)
    from ml.features import GROUP_B_NAMES, GROUP_F_NAMES
    need_b = bool(set(feat_cols) & set(GROUP_B_NAMES))
    need_f = bool(set(feat_cols) & set(GROUP_F_NAMES))
    df, _ = build_feature_matrix(args.oracle_npz, need_b=need_b, need_f=need_f)
    df = df[df["log_mse_loo"].notna()].copy()
    print(f"[{args.scene}] {len(df):,} rows, {df['camera'].nunique()} cameras, "
          f"{len(feat_cols)} feats", flush=True)

    y, p = oof_predict(df, feat_cols, args.cv_folds, args.seed)
    mt = metrics(y, p)
    print(f"[{args.scene}] OOF  RMSE={mt['rmse']:.3f}  MAE={mt['mae']:.3f}  "
          f"R2={mt['r2']:.3f}  rho={mt['rho']:.3f}", flush=True)
    (out_dir / "oof_metrics.json").write_text(json.dumps({**mt, "scene": args.scene}, indent=2))

    if args.metrics_csv:
        mc = Path(args.metrics_csv)
        mc.parent.mkdir(parents=True, exist_ok=True)
        new = not mc.exists()
        with mc.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["scene", "rmse", "mae", "r2", "rho", "n"])
            w.writerow([args.scene, mt["rmse"], mt["mae"], mt["r2"], mt["rho"], mt["n"]])

    # ── plots ────────────────────────────────────────────────────────────────
    mfin = np.isfinite(y) & np.isfinite(p)
    yy, pp = y[mfin], p[mfin]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) pred vs true hexbin + y=x
    ax = axes[0]
    lo, hi = min(yy.min(), pp.min()), max(yy.max(), pp.max())
    ax.hexbin(yy, pp, gridsize=50, cmap="viridis", bins="log", mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
    ax.set_xlabel("True log(MSE LOO)"); ax.set_ylabel("OOF prediction")
    ax.set_title(f"Pred vs true — {args.scene}")
    ax.text(0.04, 0.96,
            f"RMSE={mt['rmse']:.2f}\nMAE={mt['mae']:.2f}\n$R^2$={mt['r2']:.3f}\n"
            rf"$\rho$={mt['rho']:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="lower right", fontsize=9)

    # (b) residual distribution
    ax = axes[1]
    resid = pp - yy
    ax.hist(resid, bins=80, density=True, color="#4878CF", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(resid.mean(), color="r", ls="--", lw=1.2, label=f"mean={resid.mean():.2f}")
    ax.set_xlabel("Residual  (pred − true)"); ax.set_ylabel("Density")
    ax.set_title(f"Residuals — bias={resid.mean():.2f}, sd={resid.std():.2f}")
    ax.legend(fontsize=9)

    # (c) score distribution (pred vs GT), like ml_distributions/AC.png
    ax = axes[2]
    kde_line(ax, yy, "Oracle GT (log MSE)", "#333333", "-")
    kde_line(ax, pp, rf"RF OOF ($\rho$={mt['rho']:.3f})", "#2ca02c", "--")
    ax.set_xlabel("log(MSE LOO)"); ax.set_ylabel("Density")
    ax.set_title(f"Score distribution — {args.scene}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "oof_eval.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"[{args.scene}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
