"""Train LightGBM + XGBoost + RandomForest tile utility predictors.

Trains all 4 ablations (ABCD, ACD, AC, ABC) and 3 model types per ablation.
Label: log(mse_loo) from oracle_dq.npz["mse"] — higher = more important tile.
Camera-split 70/15/15; split by camera index to avoid tile leakage.

Usage:
    conda run -n gsquic python ml/train.py \\
        --oracle-npz output/0522/exp4_oracle_dq/ship/oracle_dq.npz \\
        --output-dir ml/models/ship \\
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


def train_scene(oracle_npz_path: str, output_dir: str, seed: int = 0):
    """Train all ablations for one scene. Saves models to output_dir/{ablation}/."""
    out_root = Path(output_dir)
    print(f"\n{'='*60}", flush=True)
    print(f"Building feature matrix from: {oracle_npz_path}", flush=True)
    df, all_names = build_feature_matrix(oracle_npz_path)
    print(f"  rows={len(df):,}  tiles={df['tile'].nunique()}  cameras={df['camera'].nunique()}", flush=True)

    # Filter to rows with valid label
    df = df[df["log_mse_loo"].notna()].copy()
    print(f"  After nan filter: {len(df):,} rows", flush=True)

    # 70/15/15 camera split (random permutation, deterministic via seed)
    rng = np.random.default_rng(seed)
    all_cams = np.sort(df["camera"].unique())
    perm = rng.permutation(len(all_cams))
    n_train = int(0.70 * len(all_cams))
    n_val   = int(0.15 * len(all_cams))
    train_cams = set(int(c) for c in all_cams[perm[:n_train]])
    val_cams   = set(int(c) for c in all_cams[perm[n_train:n_train + n_val]])
    test_cams  = set(int(c) for c in all_cams[perm[n_train + n_val:]])

    tr = df[df["camera"].isin(train_cams)]
    va = df[df["camera"].isin(val_cams)]
    te = df[df["camera"].isin(test_cams)]
    print(f"  Split: {len(tr):,} train / {len(va):,} val / {len(te):,} test  "
          f"({len(train_cams)}/{len(val_cams)}/{len(test_cams)} cameras)", flush=True)

    split_info = {
        "n_train_rows": int(len(tr)),
        "n_val_rows":   int(len(va)),
        "n_test_rows":  int(len(te)),
        "train_cams":   sorted(train_cams),
        "val_cams":     sorted(val_cams),
        "test_cams":    sorted(test_cams),
        "seed":         seed,
    }

    # Train all 4 ablations
    for ablation in ABLATION_NAMES:
        feat_cols = feature_names_for_ablation(ablation)
        ablation_dir = out_root / ablation
        ablation_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Ablation: {ablation} ({len(feat_cols)} features) ---", flush=True)

        X_tr = tr[feat_cols].values.astype(np.float32)
        X_va = va[feat_cols].values.astype(np.float32)
        X_te = te[feat_cols].values.astype(np.float32)
        y_tr = tr["log_mse_loo"].values.astype(np.float32)
        y_va = va["log_mse_loo"].values.astype(np.float32)
        y_te = te["log_mse_loo"].values.astype(np.float32)

        include_b = any(c in feat_cols for c in ("wbar_k_screen", "W_k_screen"))
        include_d = any("_skew" in c for c in feat_cols)

        metrics = {**split_info,
                   "ablation": ablation,
                   "n_features": len(feat_cols),
                   "include_group_b": include_b,
                   "include_group_d": include_d}

        # ── LightGBM ────────────────────────────────────────────────────────
        print("  Training LightGBM ...", flush=True)
        model_lgbm = lgb.LGBMRegressor(
            n_estimators=1000, random_state=seed, n_jobs=-1,
            verbosity=-1,
        )
        model_lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds_lgbm = model_lgbm.predict(X_te)
        rho_lgbm_tr, _ = _spearman(model_lgbm.predict(X_tr), y_tr)
        rho_lgbm_te, _ = _spearman(preds_lgbm, y_te)
        print(f"    LGBM  ρ_train={rho_lgbm_tr:.4f}  ρ_test={rho_lgbm_te:.4f}  "
              f"best_iter={model_lgbm.best_iteration_}", flush=True)
        joblib.dump(model_lgbm, ablation_dir / "lgbm.pkl")
        metrics["lgbm"] = {
            "rho_train": rho_lgbm_tr, "rho_test": rho_lgbm_te,
            "best_iter": int(model_lgbm.best_iteration_),
        }

        # ── XGBoost ─────────────────────────────────────────────────────────
        if _HAS_XGB:
            print("  Training XGBoost ...", flush=True)
            model_xgb = xgb.XGBRegressor(
                n_estimators=1000, random_state=seed, n_jobs=-1,
                verbosity=0, eval_metric="rmse",
                early_stopping_rounds=50,
            )
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds_xgb = model_xgb.predict(X_te)
            rho_xgb_tr, _ = _spearman(model_xgb.predict(X_tr), y_tr)
            rho_xgb_te, _ = _spearman(preds_xgb, y_te)
            print(f"    XGB   ρ_train={rho_xgb_tr:.4f}  ρ_test={rho_xgb_te:.4f}  "
                  f"best_iter={model_xgb.best_iteration}", flush=True)
            joblib.dump(model_xgb, ablation_dir / "xgb.pkl")
            metrics["xgb"] = {
                "rho_train": rho_xgb_tr, "rho_test": rho_xgb_te,
                "best_iter": int(model_xgb.best_iteration),
            }

        # ── RandomForest ─────────────────────────────────────────────────────
        print("  Training RandomForest ...", flush=True)
        model_rf = RandomForestRegressor(
            n_estimators=500, random_state=seed, n_jobs=-1,
        )
        model_rf.fit(X_tr, y_tr)
        preds_rf = model_rf.predict(X_te)
        rho_rf_tr, _ = _spearman(model_rf.predict(X_tr), y_tr)
        rho_rf_te, _ = _spearman(preds_rf, y_te)
        print(f"    RF    ρ_train={rho_rf_tr:.4f}  ρ_test={rho_rf_te:.4f}", flush=True)
        joblib.dump(model_rf, ablation_dir / "rf.pkl")
        metrics["rf"] = {
            "rho_train": rho_rf_tr, "rho_test": rho_rf_te,
        }

        # ── Save feature names and metrics ───────────────────────────────────
        (ablation_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2))
        (ablation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  Saved to {ablation_dir}/", flush=True)

    print(f"\nDone: {out_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-npz", required=True,
                        help="Path to oracle_dq.npz (PLY/trace extracted from gen_meta)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir; ablation subdirs created inside (e.g. ml/models/ship)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_scene(args.oracle_npz, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
