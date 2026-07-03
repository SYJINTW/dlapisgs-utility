"""Train a LightGBM ranker (objective=lambdarank) for per-camera tile priority.

Candidate 1 of the 2026-07-02 ML-improvement track (PLAN.md "Proposed ML-improvement
candidates", item 1). The deployed rf/lgbm/xgb models (ml/train.py) fit log(mse_loo) via
plain regression loss; greedy selection only cares about rank within one camera's tile
set, not point-accuracy on the label. lambdarank optimizes that directly via a per-camera
`group` array (LightGBM's native ranking API).

Label: the true marginal greedy key, mse_loo / N_k (byte_per_gaussian is a per-scene
constant, so it cancels out of within-camera rank order -- no PLY load needed here),
recovered as exp(log_mse_loo) / N_k from the same feature matrix ml/train.py uses.
lambdarank's default gain table is 2^label - 1, which needs small non-negative integer
relevance grades, not a continuous value -- the marginal key is binned into per-camera
percentile buckets (0..n_bins-1, default 31 -- LightGBM's default label_gain table covers
exactly labels 0..30). Binning is done PER CAMERA (not globally) since one camera's
marginal-key scale could otherwise dominate the global bin edges.

Saves lgbm_rank.pkl (a plain LGBMRanker) + feature_names.json + metrics.json into
output_dir/<ablation>/ -- loadable at inference time via ml/predict.py exactly like a
regressor (.predict() returns a raw score; sort_tiles argsorts it the same way either way,
grade discreteness only matters at train time).

Usage:
    conda run -n gsquic python ml/train_lgbm_rank.py \\
        --oracle-npz output/oracle/8/eval/<scene>/oracle_dq.npz \\
        --output-dir output/ml_models_experimental/rank/<scene> \\
        [--ablation AC] [--n-bins 32] [--cv-folds 5] [--seed 0]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ml.features import build_feature_matrix, feature_names_for_ablation  # noqa: E402


def _rank_label(marginal_key: "np.ndarray", n_bins: int) -> np.ndarray:
    """Percentile-bin one camera's marginal-key values into 0..n_bins-1 integer grades."""
    order = marginal_key.argsort(kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(marginal_key))
    pct = ranks / max(len(marginal_key) - 1, 1)
    return np.clip((pct * n_bins).astype(np.int32), 0, n_bins - 1)


def _add_rank_label(df, n_bins: int):
    df = df.copy()
    df["marginal_key"] = np.exp(df["log_mse_loo"]) / df["N_k"].clip(lower=1.0)
    df["rank_label"] = df.groupby("camera")["marginal_key"].transform(
        lambda s: _rank_label(s.to_numpy(), n_bins))
    return df


def _fold_arrays(df, cams, feat_cols):
    """Sort by camera (stable) so rows are contiguous per group; return X, y, group."""
    sub = df[df["camera"].isin(cams)].sort_values("camera", kind="stable")
    X = sub[feat_cols].values.astype(np.float32)
    y = sub["rank_label"].values.astype(np.int32)
    group = sub.groupby("camera", sort=False).size().to_numpy()
    marginal = sub["marginal_key"].values
    cam_ids = sub["camera"].values
    return X, y, group, marginal, cam_ids


def _per_camera_rho(preds, marginal, cam_ids):
    rhos = []
    for cam in np.unique(cam_ids):
        mask = cam_ids == cam
        if mask.sum() < 3:
            continue
        rho, _ = spearmanr(preds[mask], marginal[mask])
        if np.isfinite(rho):
            rhos.append(rho)
    return rhos


def train_rank(oracle_npz_path: str, output_dir: str, ablation: str = "AC",
               n_bins: int = 31, cv_folds: int = 5, seed: int = 0,
               lgbm_kwargs=None):
    out_dir = Path(output_dir) / ablation
    out_dir.mkdir(parents=True, exist_ok=True)
    lgbm_kwargs = lgbm_kwargs or {}

    print(f"Building feature matrix from: {oracle_npz_path}", flush=True)
    df, _ = build_feature_matrix(oracle_npz_path, need_b=False, need_f=False)
    df = df[df["log_mse_loo"].notna()].copy()
    print(f"  {len(df):,} rows  {df['tile'].nunique()} tiles  {df['camera'].nunique()} cameras",
          flush=True)

    feat_cols = feature_names_for_ablation(ablation)
    df = _add_rank_label(df, n_bins)

    all_cams = np.sort(df["camera"].unique())
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_rho = []
    best_iters = []
    for fold, (tr_i, te_i) in enumerate(kf.split(all_cams)):
        tr_cams, te_cams = all_cams[tr_i], all_cams[te_i]
        rng = np.random.default_rng(seed + fold)
        n_val = max(1, int(0.20 * len(tr_cams)))
        val_mask = np.zeros(len(tr_cams), dtype=bool)
        val_mask[rng.choice(len(tr_cams), n_val, replace=False)] = True
        fit_cams, val_cams = tr_cams[~val_mask], tr_cams[val_mask]

        X_tr, y_tr, g_tr, _, _ = _fold_arrays(df, fit_cams, feat_cols)
        X_va, y_va, g_va, _, _ = _fold_arrays(df, val_cams, feat_cols)
        X_te, _, _, marg_te, cam_te = _fold_arrays(df, te_cams, feat_cols)

        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=1000,
                            random_state=seed, n_jobs=-1, verbosity=-1, **lgbm_kwargs)
        m.fit(X_tr, y_tr, group=g_tr,
              eval_set=[(X_va, y_va)], eval_group=[g_va],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        best_iters.append(int(m.best_iteration_))

        preds = m.predict(X_te)
        rhos = _per_camera_rho(preds, marg_te, cam_te)
        fold_rho.extend(rhos)
        print(f"  fold {fold}: median_rho={np.median(rhos):.4f}  n_cams={len(rhos)}",
              flush=True)

    cv_rho_mean = float(np.mean(fold_rho))
    cv_rho_std = float(np.std(fold_rho, ddof=1)) if len(fold_rho) > 1 else 0.0
    print(f"CV per-camera marginal-key rho: {cv_rho_mean:.4f} +/- {cv_rho_std:.4f} "
          f"(n_cams={len(fold_rho)})", flush=True)

    # Refit on all data for deployment.
    X_all, y_all, g_all, _, _ = _fold_arrays(df, all_cams, feat_cols)
    n_est = int(np.mean(best_iters) * 1.1) if best_iters else 500
    final = lgb.LGBMRanker(objective="lambdarank", n_estimators=n_est,
                           random_state=seed, n_jobs=-1, verbosity=-1, **lgbm_kwargs)
    final.fit(X_all, y_all, group=g_all)
    joblib.dump(final, out_dir / "lgbm_rank.pkl")

    _oracle_meta = json.loads(str(np.load(str(oracle_npz_path), allow_pickle=True)
                                  ["gen_meta"].item()))
    metrics = {
        "ablation": ablation,
        "n_features": len(feat_cols),
        "n_bins": n_bins,
        "cv_folds": cv_folds,
        "seed": seed,
        "cv_rho_mean": cv_rho_mean,
        "cv_rho_std": cv_rho_std,
        "n_test_cams": len(fold_rho),
        "n_estimators_final": n_est,
        "source_ply": _oracle_meta.get("scene_ply"),
        "source_ply_n_gs": _oracle_meta.get("n_total_gs"),
    }
    (out_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {out_dir}/", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation", default="AC")
    parser.add_argument("--n-bins", type=int, default=31,
                        help="Per-camera relevance-grade bucket count (default 31 -- "
                             "LightGBM's default label_gain table covers exactly labels "
                             "0..30; raising this requires passing --label-gain explicitly, "
                             "not currently exposed).")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-child-samples", type=int, default=None)
    args = parser.parse_args()

    lgbm_kwargs = {}
    if args.num_leaves is not None:
        lgbm_kwargs["num_leaves"] = args.num_leaves
    if args.min_child_samples is not None:
        lgbm_kwargs["min_child_samples"] = args.min_child_samples

    train_rank(args.oracle_npz, args.output_dir, ablation=args.ablation,
               n_bins=args.n_bins, cv_folds=args.cv_folds, seed=args.seed,
               lgbm_kwargs=lgbm_kwargs or None)


if __name__ == "__main__":
    main()
