"""Pooled cross-scene training: does more same-domain data beat per-scene training?

Different question from loso_rf.py (which excludes the held-out scene from its own
training pool, to test generalization to an unseen scene). Here the target scene's
own train split IS in the pool -- testing whether adding neighboring scenes' data as
extra training signal improves on today's single-scene-only training, not whether the
model generalizes to a totally unseen scene.

Label is z-scored per scene before pooling (loso_rf.py's approach) so no scene's raw
MSE magnitude dominates the fit. Ranking-only inference (selection_core.sort_tiles
only ever compares tiles within one (scene, camera) call) means this affine per-scene
label transform does not affect deployment-time ranking.

Saved models have no single source scene, so metrics.json's source_ply_n_gs is
null -- ml/predict.py::load_model's scene-identity guard warning-skips on null
(predict.py:54-57) instead of raising, since a pooled model has no one PLY to check
against. Deploy with --ml-feature-cache off (test_utility_inmem.py) so static
features are rebuilt from the target scene's own PLY rather than looking for a
scene-specific feature_cache.npz inside this pooled model directory (none is written).

Usage:
    conda run -n gsquic python ml/cross_scene/train_pooled.py \\
        --scene-set real3 --ablation AC --models rf lgbm xgb \\
        --out output/ml_models_experimental/pooled_real3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
from ml.features import build_feature_matrix, feature_names_for_ablation  # noqa: E402

REAL3 = ("bicycle", "garden", "stump")
SYNTH7 = ("chair", "drums", "ficus", "hotdog", "materials", "mic", "ship")
ALL10 = REAL3 + SYNTH7
SCENE_SETS = {"real3": REAL3, "synth7": SYNTH7, "all10": ALL10}

LABEL = "log_mse_loo"


def zscore(y):
    mu, sd = float(y.mean()), float(y.std())
    return (y - mu) / (sd if sd > 0 else 1.0)


def load_split(root, split, scene, feat_cols):
    npz = Path(root) / split / scene / "oracle_dq.npz"
    df, _ = build_feature_matrix(str(npz))
    df = df[df[LABEL].notna()].copy()
    X = df[feat_cols].values.astype(np.float32)
    y = df[LABEL].values.astype(np.float32)
    return X, y


def fit_model(model_type, X, y, seed):
    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    elif model_type == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=500, random_state=seed, n_jobs=-1, verbosity=-1)
    elif model_type == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=500, random_state=seed, n_jobs=-1, verbosity=0)
    else:
        raise ValueError(f"unknown model_type '{model_type}'")
    m.fit(X, y)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-set", choices=list(SCENE_SETS), default="real3")
    ap.add_argument("--ablation", default="AC")
    ap.add_argument("--models", nargs="+", default=["rf", "lgbm", "xgb"],
                    choices=["rf", "lgbm", "xgb"])
    ap.add_argument("--root", default="output/oracle/8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = SCENE_SETS[args.scene_set]
    feat_cols = feature_names_for_ablation(args.ablation)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{time.time()-t0:6.1f}s] scene-set={args.scene_set} ({len(scenes)} scenes) "
          f"ablation={args.ablation} ({len(feat_cols)} feats)",
          flush=True)

    tr, ev = {}, {}
    for s in scenes:
        tr[s] = load_split(args.root, "train", s, feat_cols)
        ev[s] = load_split(args.root, "eval", s, feat_cols)
        print(f"[{time.time()-t0:6.1f}s]   {s:10s} train={tr[s][0].shape} eval={ev[s][0].shape}",
              flush=True)

    # ── pooled training set: every scene's train split, label z-scored per scene ──
    Xs = [tr[s][0] for s in scenes]
    ys = [zscore(tr[s][1]) for s in scenes]
    X_pool = np.vstack(Xs)
    y_pool = np.concatenate(ys)
    print(f"[{time.time()-t0:6.1f}s] pooled train set: {X_pool.shape}", flush=True)

    pooled_models = {}
    for model_type in args.models:
        print(f"[{time.time()-t0:6.1f}s] fitting pooled {model_type}...", flush=True)
        pooled_models[model_type] = fit_model(model_type, X_pool, y_pool, args.seed)

    # ── per-scene comparison table: pooled model vs a same-config per-scene-only
    #    baseline, evaluated on each scene's own EVAL split (identical test rows) ──
    rows = []
    for model_type in args.models:
        for s in scenes:
            Xte, yte = ev[s]

            Xtr_ps, ytr_ps = tr[s]
            m_ps = fit_model(model_type, Xtr_ps, zscore(ytr_ps), args.seed)
            rho_ps = float(spearmanr(m_ps.predict(Xte), yte).correlation)

            rho_pooled = float(spearmanr(pooled_models[model_type].predict(Xte), yte).correlation)

            rows.append(dict(model=model_type, scene=s,
                             per_scene_rho=rho_ps, pooled_rho=rho_pooled,
                             delta=rho_pooled - rho_ps))
            print(f"[{time.time()-t0:6.1f}s] {model_type:5s} {s:10s} "
                  f"per-scene rho={rho_ps:.3f}  pooled rho={rho_pooled:.3f}  "
                  f"Δ={rho_pooled-rho_ps:+.3f}", flush=True)

    # ── save pooled models (new path only -- never overwrites output/ml_models/) ──
    for model_type, m in pooled_models.items():
        joblib.dump(m, out_dir / f"{model_type}.pkl")
    (out_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2))
    metrics = {
        "ablation": args.ablation,
        "scene_set": args.scene_set,
        "scenes": list(scenes),
        "n_rows_pooled": int(len(y_pool)),
        "seed": args.seed,
        "models": list(pooled_models),
        "source_ply_n_gs": None,  # pooled across scenes -- no single PLY; ml/predict.py
                                  # load_model's scene-identity guard warning-skips on null
        "comparison_table": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 60)
    print(f"{'model':6s} {'scene':10s} {'per-scene':>10s} {'pooled':>8s} {'delta':>8s}")
    for r in rows:
        print(f"{r['model']:6s} {r['scene']:10s} {r['per_scene_rho']:10.3f} "
              f"{r['pooled_rho']:8.3f} {r['delta']:+8.3f}")
    print(f"\n[{time.time()-t0:6.1f}s] saved -> {out_dir}/{{model}}.pkl + metrics.json")


if __name__ == "__main__":
    main()
