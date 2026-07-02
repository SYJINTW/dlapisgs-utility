"""Compare RF vs LightGBM vs XGBoost on the per-scene tile-utility task.

Camera-blocked 5-fold CV. For each model reports TRAIN (in-sample, on the fold's
train cameras) vs TEST (out-of-fold) RMSE/MAE/R²/Spearman-ρ — the train→test gap
exposes overfitting. Hyperparameters mirror ml/train.py (RF 500 trees; LGBM/XGB
1000 estimators with early stopping on a camera-blocked val split).

Usage (gaussian_splatting env; pick a free GPU):
    CUDA_VISIBLE_DEVICES=2 conda run -n gaussian_splatting python \
        experiments/0612/compare_models_oof.py \
        --oracle-npz output/0606/exp4_oracle_dq/train/ship/oracle_dq.npz \
        --scene ship --ablation AC --cv-folds 5 --seed 0 \
        --out-dir output/quickplots/ml_compare/ship \
        --metrics-csv output/quickplots/ml_compare/_compare_metrics.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import lightgbm as lgb
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
from ml.features import build_feature_matrix, feature_names_for_ablation  # noqa: E402

MODELS = ["rf", "lgbm", "xgb"]
COLORS = {"rf": "#2ca02c", "lgbm": "#1f77b4", "xgb": "#ff7f0e"}


def _metrics(y, p):
    resid = p - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = float(spearmanr(p, y).correlation)
    return dict(rmse=rmse, mae=mae, r2=r2, rho=rho)


def _fit_predict(model, Xtr, ytr, Xval, yval, Xte, seed):
    if model == "rf":
        m = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
        m.fit(Xtr, ytr)
    elif model == "lgbm":
        m = lgb.LGBMRegressor(n_estimators=1000, random_state=seed, n_jobs=-1, verbosity=-1)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    elif model == "xgb":
        m = xgb.XGBRegressor(n_estimators=1000, random_state=seed, n_jobs=-1,
                             verbosity=0, eval_metric="rmse", early_stopping_rounds=150)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return m.predict(Xtr), m.predict(Xte)


def run_cv(df, feat_cols, n_folds, seed):
    cams = np.array(sorted(df["camera"].unique()))
    cam_col = df["camera"].values
    X = df[feat_cols].values.astype(np.float32)
    y = df["log_mse_loo"].values.astype(np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # accumulate train metrics per fold; pool OOF test preds
    train_acc = {m: [] for m in MODELS}
    oof = {m: np.full(len(df), np.nan) for m in MODELS}

    for fold, (tr_idx, te_idx) in enumerate(kf.split(cams)):
        te_cams = set(cams[te_idx])
        tr_all = cams[tr_idx]
        # camera-blocked val split (20% of train cams) for GBM early stopping
        rng = np.random.RandomState(seed + fold)
        perm = rng.permutation(len(tr_all))
        n_val = max(1, len(tr_all) // 5)
        val_cams = set(tr_all[perm[:n_val]])
        fit_cams = set(tr_all[perm[n_val:]])

        m_te = np.isin(cam_col, list(te_cams))
        m_fit = np.isin(cam_col, list(fit_cams))
        m_val = np.isin(cam_col, list(val_cams))

        for model in MODELS:
            if model == "xgb" and not _HAS_XGB:
                continue
            ptr, pte = _fit_predict(model, X[m_fit], y[m_fit], X[m_val], y[m_val], X[m_te], seed)
            train_acc[model].append(_metrics(y[m_fit], ptr))
            oof[model][m_te] = pte
        print(f"  fold {fold}: fit {m_fit.sum()} val {m_val.sum()} test {m_te.sum()}", flush=True)

    out = {}
    for model in MODELS:
        if not train_acc[model]:
            continue
        mask = np.isfinite(oof[model])
        test_m = _metrics(y[mask], oof[model][mask])
        train_m = {k: float(np.mean([d[k] for d in train_acc[model]])) for k in ("rmse", "mae", "r2", "rho")}
        out[model] = {"train": train_m, "test": test_m}
    return out, y, oof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-npz", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ablation", default="AC")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics-csv", default=None)
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
    print(f"[{args.scene}] {len(df):,} rows, {df['camera'].nunique()} cameras", flush=True)

    res, y, oof = run_cv(df, feat_cols, args.cv_folds, args.seed)
    (out_dir / "compare_metrics.json").write_text(json.dumps({"scene": args.scene, **res}, indent=2))

    print(f"\n[{args.scene}] model   |  TRAIN rho/R2/RMSE/MAE  |  TEST rho/R2/RMSE/MAE", flush=True)
    for m, d in res.items():
        tr, te = d["train"], d["test"]
        print(f"  {m:5s} | {tr['rho']:.3f} {tr['r2']:.3f} {tr['rmse']:.2f} {tr['mae']:.2f}"
              f"  |  {te['rho']:.3f} {te['r2']:.3f} {te['rmse']:.2f} {te['mae']:.2f}", flush=True)

    if args.metrics_csv:
        mc = Path(args.metrics_csv)
        mc.parent.mkdir(parents=True, exist_ok=True)
        new = not mc.exists()
        with mc.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["scene", "model", "split", "rmse", "mae", "r2", "rho"])
            for m, d in res.items():
                for split in ("train", "test"):
                    s = d[split]
                    w.writerow([args.scene, m, split, s["rmse"], s["mae"], s["r2"], s["rho"]])

    # save OOF predictions for later diagnostic use
    import numpy as _np2
    _oof_save = {"y_true": y}
    for m in MODELS:
        if m in oof:
            _oof_save[f"{m}_oof"] = oof[m]
    _np2.savez(str(out_dir / "oof_preds.npz"), **_oof_save)

    # ── diagnostic: pred-vs-actual scatter + residual distribution ───────────
    fig2, axes2 = plt.subplots(2, len(res), figsize=(5 * len(res), 9))
    if len(res) == 1:
        axes2 = axes2[:, None]
    for ci, model in enumerate(res):
        mask = np.isfinite(oof[model])
        y_true = y[mask]
        y_pred = oof[model][mask]
        rho_val = res[model]["test"]["rho"]
        r2_val = res[model]["test"]["r2"]
        col = COLORS[model]

        # top: pred vs actual
        ax = axes2[0, ci]
        ax.scatter(y_true, y_pred, s=4, alpha=0.25, color=col, rasterized=True)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlabel("true log_mse_loo"); ax.set_ylabel("predicted (OOF)")
        ax.set_title(f"{model.upper()}  ρ={rho_val:.3f}  R²={r2_val:.3f}")
        ax.grid(alpha=0.2)

        # bottom: residual histogram
        ax = axes2[1, ci]
        resid = y_pred - y_true
        ax.hist(resid, bins=60, color=col, alpha=0.7, density=True)
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.axvline(resid.mean(), color="red", lw=1, ls=":", label=f"mean={resid.mean():.2f}")
        ax.set_xlabel("residual (pred − true)"); ax.set_ylabel("density")
        ax.set_title(f"{model.upper()}  RMSE={res[model]['test']['rmse']:.3f}  MAE={res[model]['test']['mae']:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    fig2.suptitle(f"OOF diagnostics (test) — {args.scene}", fontsize=13)
    fig2.tight_layout()
    diag_out = out_dir / "oof_diagnostics.png"
    fig2.savefig(str(diag_out), dpi=150)
    plt.close(fig2)
    print(f"[{args.scene}] saved {diag_out}", flush=True)

    # ── plot: train vs test bars for rho, R2, RMSE ──────────────────────────
    metr = [("rho", "Spearman ρ", True), ("r2", "R²", True), ("rmse", "RMSE", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    models = list(res.keys())
    x = np.arange(len(models))
    w = 0.38
    for ax, (key, lab, hib) in zip(axes, metr):
        tr = [res[m]["train"][key] for m in models]
        te = [res[m]["test"][key] for m in models]
        ax.bar(x - w / 2, tr, w, label="train", color="#bbbbbb")
        ax.bar(x + w / 2, te, w, label="test (OOF)", color=[COLORS[m] for m in models])
        for xi, (a, b) in enumerate(zip(tr, te)):
            ax.text(xi - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(f"{lab}  ({'higher' if hib else 'lower'} = better)")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"RF vs LGBM vs XGB — train vs test (OOF) — {args.scene}", fontsize=13)
    fig.tight_layout()
    out = out_dir / "model_compare.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"[{args.scene}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
