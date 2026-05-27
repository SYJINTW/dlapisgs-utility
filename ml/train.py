"""Train LightGBM + XGBoost + RandomForest tile utility predictors.

Label: log(mse_loo) from oracle_dq.npz["mse"] — higher = more important tile.
Evaluation: 5-fold camera-blocked CV (default). Split by tile when --label-mode cam_avg.

Usage:
    conda run -n gsquic python ml/train.py \\
        --oracle-npz output/0522/exp4_oracle_dq/ship/oracle_dq.npz \\
        --output-dir ml/models/ship \\
        [--ablations A B C D E F ABCDEF] \\
        [--models rf lgbm xgb] \\
        [--cv-folds 5] \\
        [--label-mode raw|cam_avg] \\
        [--seed 0]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ml.features import build_feature_matrix, feature_names_for_ablation, ABLATION_NAMES  # noqa: E402

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGB model", flush=True)


def _spearman(preds, targets):
    if len(preds) < 3:
        return 0.0, 1.0
    rho, pval = spearmanr(preds, targets)
    return float(rho), float(pval)


_ALL_MODELS = ("rf", "lgbm", "xgb")


def _apply_label_mode(df, label_mode, feat_cols):
    """Return (df_out, split_key).

    raw:     original per-(cam,tile) rows; split by camera.
    cam_avg: one row per tile (mean of features + label across cameras); split by tile.
    """
    if label_mode == "raw":
        return df, "camera"
    if label_mode == "cam_avg":
        agg_cols = [c for c in feat_cols if c in df.columns] + ["log_mse_loo"]
        df_out = df.groupby("tile")[agg_cols].mean().reset_index()
        return df_out, "tile"
    raise ValueError(f"unknown label_mode '{label_mode}'; choices: raw, cam_avg")


_RF_SWEEP_LEAF    = (1, 2, 5, 10, 20, 50)
_RF_SWEEP_FEATS   = (0.3, 0.5, 0.7, 1.0)


def _run_cv(df, feat_cols, split_key, cv_folds, seed, models_to_run, rf_kwargs=None):
    """5-fold (or n-fold) cross-validation.

    Within each fold, 20% of train units are carved as val for LGBM/XGB early stopping.

    Returns:
        cv_stats: dict  model -> {cv_rho_mean, cv_rho_std, cv_r2_mean, cv_r2_std}
        best_iters: dict  model -> list of best_iteration per fold (for final refit)
        X_all, y_all: full arrays for final refit
    """
    y_col = "log_mse_loo"
    all_units = np.sort(df[split_key].unique())
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_rho  = {m: [] for m in _ALL_MODELS}
    fold_r2   = {m: [] for m in _ALL_MODELS}
    best_iters = {"lgbm": [], "xgb": []}

    X_all = df[feat_cols].values.astype(np.float32)
    y_all = df[y_col].values.astype(np.float32)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(all_units)):
        tr_units = all_units[tr_idx]
        te_units = set(all_units[te_idx].tolist())

        # carve 20% of tr_units as within-fold val for early stopping
        rng_fold = np.random.default_rng(seed + fold)
        n_val = max(1, int(0.20 * len(tr_units)))
        val_mask = np.zeros(len(tr_units), dtype=bool)
        val_mask[rng_fold.choice(len(tr_units), n_val, replace=False)] = True
        val_units = set(tr_units[val_mask].tolist())
        fit_units  = set(tr_units[~val_mask].tolist())

        mask_fit = df[split_key].isin(fit_units)
        mask_val = df[split_key].isin(val_units)
        mask_te  = df[split_key].isin(te_units)

        X_tr = df.loc[mask_fit, feat_cols].values.astype(np.float32)
        X_va = df.loc[mask_val, feat_cols].values.astype(np.float32)
        X_te = df.loc[mask_te,  feat_cols].values.astype(np.float32)
        y_tr = df.loc[mask_fit, y_col].values.astype(np.float32)
        y_va = df.loc[mask_val, y_col].values.astype(np.float32)
        y_te = df.loc[mask_te,  y_col].values.astype(np.float32)

        if len(y_te) == 0:
            continue

        # ── LightGBM ────────────────────────────────────────────────────────
        if "lgbm" in models_to_run:
            m = lgb.LGBMRegressor(n_estimators=1000, random_state=seed, n_jobs=-1, verbosity=-1)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            preds = m.predict(X_te)
            fold_rho["lgbm"].append(_spearman(preds, y_te)[0])
            fold_r2["lgbm"].append(float(r2_score(y_te, preds)))
            best_iters["lgbm"].append(int(m.best_iteration_))

        # ── XGBoost ─────────────────────────────────────────────────────────
        if "xgb" in models_to_run and _HAS_XGB:
            m = xgb.XGBRegressor(n_estimators=1000, random_state=seed, n_jobs=-1,
                                  verbosity=0, eval_metric="rmse", early_stopping_rounds=150)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = m.predict(X_te)
            fold_rho["xgb"].append(_spearman(preds, y_te)[0])
            fold_r2["xgb"].append(float(r2_score(y_te, preds)))
            best_iters["xgb"].append(int(m.best_iteration))

        # ── RandomForest ─────────────────────────────────────────────────────
        if "rf" in models_to_run:
            kw = rf_kwargs or {}
            m = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1, **kw)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            fold_rho["rf"].append(_spearman(preds, y_te)[0])
            fold_r2["rf"].append(float(r2_score(y_te, preds)))

    cv_stats = {}
    for model in models_to_run:
        rhos = fold_rho[model]
        r2s  = fold_r2[model]
        if not rhos:
            continue
        cv_stats[model] = {
            "cv_rho_mean": float(np.mean(rhos)),
            "cv_rho_std":  float(np.std(rhos, ddof=1) if len(rhos) > 1 else 0.0),
            "cv_r2_mean":  float(np.mean(r2s)),
            "cv_r2_std":   float(np.std(r2s, ddof=1) if len(r2s) > 1 else 0.0),
            "cv_folds":    len(rhos),
        }

    return cv_stats, best_iters, X_all, y_all


def sweep_rf_hparams(df, feat_cols, split_key, cv_folds, seed,
                     leaf_grid=_RF_SWEEP_LEAF, feat_grid=_RF_SWEEP_FEATS):
    """Grid search min_samples_leaf × max_features for RF via CV.

    Returns list of dicts sorted by cv_rho_mean desc.
    """
    results = []
    for msl in leaf_grid:
        for mf in feat_grid:
            kw = {"min_samples_leaf": msl, "max_features": mf}
            cv_stats, _, _, _ = _run_cv(
                df, feat_cols, split_key, cv_folds, seed,
                models_to_run={"rf"}, rf_kwargs=kw,
            )
            rf = cv_stats.get("rf", {})
            results.append({
                "min_samples_leaf": msl,
                "max_features": mf,
                "cv_rho_mean": rf.get("cv_rho_mean", float("nan")),
                "cv_rho_std":  rf.get("cv_rho_std",  float("nan")),
                "cv_r2_mean":  rf.get("cv_r2_mean",  float("nan")),
                "cv_r2_std":   rf.get("cv_r2_std",   float("nan")),
            })
            print(f"  msl={msl:3d}  mf={mf:.1f}  "
                  f"ρ={rf.get('cv_rho_mean',float('nan')):.4f}"
                  f"±{rf.get('cv_rho_std',float('nan')):.4f}  "
                  f"R²={rf.get('cv_r2_mean',float('nan')):.4f}", flush=True)
    results.sort(key=lambda r: r["cv_rho_mean"], reverse=True)
    return results


def _refit_final(X_all, y_all, seed, models_to_run, best_iters, rf_kwargs=None):
    """Refit on all data for deployment. Returns dict model -> fitted model."""
    finals = {}

    if "lgbm" in models_to_run:
        iters = best_iters["lgbm"]
        n = int(np.mean(iters) * 1.1) if iters else 500
        m = lgb.LGBMRegressor(n_estimators=n, random_state=seed, n_jobs=-1, verbosity=-1)
        m.fit(X_all, y_all)
        finals["lgbm"] = m

    if "xgb" in models_to_run and _HAS_XGB:
        iters = best_iters["xgb"]
        n = int(np.mean(iters) * 1.1) if iters else 500
        m = xgb.XGBRegressor(n_estimators=n, random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(X_all, y_all)
        finals["xgb"] = m

    if "rf" in models_to_run:
        kw = rf_kwargs or {}
        m = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1, **kw)
        m.fit(X_all, y_all)
        finals["rf"] = m

    return finals


def train_scene(oracle_npz_path: str, output_dir: str, seed: int = 0,
                ablations=None, models=None, cv_folds: int = 5,
                label_mode: str = "raw", rf_kwargs=None,
                rf_sweep: bool = False):
    """Train all ablations for one scene. Saves models to output_dir/{ablation}/."""
    out_root = Path(output_dir)
    print(f"\n{'='*60}", flush=True)
    print(f"Building feature matrix from: {oracle_npz_path}", flush=True)
    df, all_names = build_feature_matrix(oracle_npz_path)
    print(f"  rows={len(df):,}  tiles={df['tile'].nunique()}  cameras={df['camera'].nunique()}", flush=True)

    df = df[df["log_mse_loo"].notna()].copy()
    print(f"  After nan filter: {len(df):,} rows", flush=True)

    models_to_run = set(models) if models is not None else set(_ALL_MODELS)
    names_to_train = ablations if ablations is not None else list(ABLATION_NAMES)

    for ablation in names_to_train:
        feat_cols = feature_names_for_ablation(ablation)
        ablation_dir = out_root / ablation
        ablation_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Ablation: {ablation} ({len(feat_cols)} features) "
              f"label={label_mode} cv={cv_folds} ---", flush=True)

        df_abl, split_key = _apply_label_mode(df, label_mode, feat_cols)
        n_units = df_abl[split_key].nunique()
        print(f"  {len(df_abl):,} rows  {n_units} {split_key}s  "
              f"(split_key={split_key})", flush=True)

        if rf_sweep:
            print("  RF hparam sweep:", flush=True)
            sweep_results = sweep_rf_hparams(
                df_abl, feat_cols, split_key, cv_folds, seed,
            )
            best = sweep_results[0]
            print(f"  Best: msl={best['min_samples_leaf']}  "
                  f"mf={best['max_features']}  "
                  f"ρ={best['cv_rho_mean']:.4f}", flush=True)
            (ablation_dir / "rf_sweep.json").write_text(
                json.dumps(sweep_results, indent=2)
            )

        cv_stats, best_iters, X_all, y_all = _run_cv(
            df_abl, feat_cols, split_key, cv_folds, seed, models_to_run,
            rf_kwargs=rf_kwargs,
        )

        # Print CV summary
        for model, s in cv_stats.items():
            print(f"    {model.upper():5s}  ρ={s['cv_rho_mean']:.4f}±{s['cv_rho_std']:.4f}  "
                  f"R²={s['cv_r2_mean']:.4f}±{s['cv_r2_std']:.4f}", flush=True)

        # Refit on all data and save
        finals = _refit_final(X_all, y_all, seed, models_to_run, best_iters,
                              rf_kwargs=rf_kwargs)
        rho_train_full = {}
        for model, fitted in finals.items():
            joblib.dump(fitted, ablation_dir / f"{model}.pkl")
            rho_full, _ = _spearman(fitted.predict(X_all), y_all)
            rho_train_full[model] = float(rho_full)
            print(f"    {model.upper():5s}  ρ_train_full={rho_full:.4f}  "
                  f"(saved {model}.pkl)", flush=True)

        include_b = any(c in feat_cols for c in ("wbar_k_screen", "W_k_screen"))
        include_d = any("_skew" in c for c in feat_cols)

        metrics = {
            "ablation":       ablation,
            "n_features":     len(feat_cols),
            "label_mode":     label_mode,
            "cv_folds":       cv_folds,
            "n_rows":         int(len(df_abl)),
            "n_split_units":  int(n_units),
            "split_key":      split_key,
            "seed":           seed,
            "include_group_b": include_b,
            "include_group_d": include_d,
        }
        for model, s in cv_stats.items():
            metrics[model] = {**s, "rho_train_full": rho_train_full.get(model)}

        (ablation_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2))
        (ablation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  Saved to {ablation_dir}/", flush=True)

    print(f"\nDone: {out_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Any subset of ABCDEF. E.g. --ablations A E ABCDEF")
    parser.add_argument("--models", nargs="+", default=None, choices=list(_ALL_MODELS))
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="CV folds (default 5). Use 1 for single 80/20 split.")
    parser.add_argument("--label-mode", default="raw", choices=["raw", "cam_avg"],
                        help="raw: per-(cam,tile) rows. cam_avg: one row per tile.")
    parser.add_argument("--rf-sweep", action="store_true",
                        help="Grid search min_samples_leaf × max_features for RF.")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1,
                        help="RF min_samples_leaf (default 1).")
    parser.add_argument("--rf-max-features", type=float, default=1.0,
                        help="RF max_features fraction (default 1.0).")
    args = parser.parse_args()

    rf_kwargs = {}
    if args.rf_min_samples_leaf != 1:
        rf_kwargs["min_samples_leaf"] = args.rf_min_samples_leaf
    if args.rf_max_features != 1.0:
        rf_kwargs["max_features"] = args.rf_max_features

    train_scene(args.oracle_npz, args.output_dir, seed=args.seed,
                ablations=args.ablations, models=args.models,
                cv_folds=args.cv_folds, label_mode=args.label_mode,
                rf_kwargs=rf_kwargs or None, rf_sweep=args.rf_sweep)


if __name__ == "__main__":
    main()
