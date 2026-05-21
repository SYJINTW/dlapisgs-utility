"""Train LightGBM utility regressors from oracle_dq.npz labels.

Usage:
    conda run -n gsquic python ml/train_lgbm.py \\
        --oracle-npz output/0519/exp4_oracle_dq/ship/oracle_dq.npz \\
        --output-dir ml/models/ship/ \\
        --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ml.features import build_feature_matrix  # noqa: E402

FEAT_COLS = ["lod", "v_k", "d_k", "C_k", "W_k"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-npz", required=True,
                        help="Path to oracle_dq.npz (PLY/trace extracted from gen_meta)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save models and metadata")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building feature matrix from {args.oracle_npz} ...")
    feat_df = build_feature_matrix(args.oracle_npz)

    npz = np.load(args.oracle_npz, allow_pickle=True)
    mse_matrix = npz["mse"]         # (N_cams, N_tiles) float32
    camera_indices = npz["camera_indices"]  # (N_cams,)

    # Join mse label: camera_indices[i] -> row i in mse_matrix
    cam_to_row = {int(c): i for i, c in enumerate(camera_indices)}
    row_idx = np.array([cam_to_row[c] for c in feat_df["camera"].values])
    feat_df["mse"] = mse_matrix[row_idx, feat_df["tile"].values]

    df = feat_df[feat_df["mse"] > 0].copy()
    print(f"Rows after mse>0 filter: {len(df):,} / {len(feat_df):,} "
          f"({100 * len(df) / len(feat_df):.1f}% non-zero)")

    # 70/15/15 camera split
    rng = np.random.default_rng(args.seed)
    all_cams = np.sort(df["camera"].unique())
    perm = rng.permutation(len(all_cams))
    n_train = int(0.70 * len(all_cams))
    n_val = int(0.15 * len(all_cams))
    train_cams = set(int(c) for c in all_cams[perm[:n_train]])
    val_cams = set(int(c) for c in all_cams[perm[n_train:n_train + n_val]])
    test_cams = set(int(c) for c in all_cams[perm[n_train + n_val:]])

    tr = df[df["camera"].isin(train_cams)]
    va = df[df["camera"].isin(val_cams)]
    te = df[df["camera"].isin(test_cams)]
    print(f"Split: {len(tr):,} train / {len(va):,} val / {len(te):,} test rows | "
          f"cams {len(train_cams)}/{len(val_cams)}/{len(test_cams)}")

    # OLS: log(mse) ~ β₀ + β₁·log(C_k) on train rows (for residual head)
    log_mse_tr = np.log(tr["mse"].values)
    log_c_tr = np.log(tr["C_k"].values.clip(1))
    A = np.column_stack([np.ones_like(log_c_tr), log_c_tr])
    coefs, _, _, _ = np.linalg.lstsq(A, log_mse_tr, rcond=None)
    beta0, beta1 = float(coefs[0]), float(coefs[1])
    print(f"OLS coefs: β₀={beta0:.4f}  β₁={beta1:.4f}")
    (out_dir / "ols_coefs.json").write_text(json.dumps({"beta0": beta0, "beta1": beta1}))

    def _make_y(split_df, head):
        log_mse = np.log(split_df["mse"].values)
        if head == "raw":
            return log_mse
        log_c = np.log(split_df["C_k"].values.clip(1))
        return log_mse - (beta0 + beta1 * log_c)

    metrics: dict = {
        "n_train_rows": int(len(tr)),
        "n_val_rows": int(len(va)),
        "n_test_rows": int(len(te)),
        "train_cams": sorted(train_cams),
        "val_cams": sorted(val_cams),
        "test_cams": sorted(test_cams),
        "ols_beta0": beta0,
        "ols_beta1": beta1,
    }

    for head in ("raw", "resid"):
        print(f"\n--- head={head} ---")
        X_tr = tr[FEAT_COLS].values
        X_va = va[FEAT_COLS].values
        X_te = te[FEAT_COLS].values
        y_tr, y_va, y_te = _make_y(tr, head), _make_y(va, head), _make_y(te, head)

        model = lgb.LGBMRegressor(n_estimators=1000, random_state=args.seed, n_jobs=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        preds_te = model.predict(X_te)
        rho, pval = spearmanr(preds_te, y_te)
        print(f"Spearman ρ={rho:.4f}  p={pval:.2e}  n={len(X_te)}")
        metrics[f"rho_{head}"] = float(rho)
        metrics[f"best_iter_{head}"] = int(model.best_iteration_)

        imp = dict(zip(FEAT_COLS, model.feature_importances_))
        print(f"Feature importance (gain): {imp}")
        metrics[f"feature_importance_{head}"] = {k: int(v) for k, v in imp.items()}

        model_path = out_dir / f"model_{head}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {model_path}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone. ρ_raw={metrics['rho_raw']:.4f}  ρ_resid={metrics['rho_resid']:.4f}")


if __name__ == "__main__":
    main()
