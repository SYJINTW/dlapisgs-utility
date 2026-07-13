"""Pooled cross-scene LightGBM ranker: candidate 1 (lambdarank) x candidate 5 (pooling).

Combines ml/train_lgbm_rank.py's ranking objective with ml/cross_scene/train_pooled.py's
multi-scene pooling. Neither alone has been tried at cross-scene scale before this.

Unlike train_pooled.py's regressor label (z-scored log_mse_loo, needs cross-scene rescaling
because the model fits absolute label magnitude), the rank label here is a per-(scene,camera)
percentile bin (0..n_bins-1) -- already scale-free by construction, so no z-scoring is needed
when pooling. What DOES need care is the LightGBM ranker `group` array: camera indices repeat
0..149 in every scene, so grouping by "camera" alone (as train_lgbm_rank.py does for its
single-scene case) would silently merge tiles from two different scenes' cameras into one
bogus query group and corrupt the lambdarank loss. Every group boundary here is keyed by the
compound (scene, camera) pair instead.

Evaluation mirrors train_pooled.py's design (pool TRAIN splits, score against each scene's own
EVAL split, compare against a fresh per-scene-only baseline fit in the same run) rather than
train_lgbm_rank.py's internal k-fold-over-cameras CV -- the pooled setting already has a
genuine held-out eval set per scene, so no CV is needed to estimate generalization.

LGBM-only (no --models flag) -- RF/XGB de-prioritized for all new candidate combinations this
session.

Usage:
    conda run -n gsquic python ml/cross_scene/train_pooled_rank.py \\
        --scene-set {real3,synth7,all10} --ablation ACG \\
        --out output/ml_models_experimental/pooled_rank_<scene-set>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
from ml.features import build_feature_matrix, feature_names_for_ablation  # noqa: E402
from ml.train_lgbm_rank import (_rank_label, _per_camera_rho,  # noqa: E402
                                fit_single_scene_ranker)
from ml.cross_scene.train_pooled import SCENE_SETS  # noqa: E402
from ml.predict import exp_clipped  # noqa: E402

LABEL = "log_mse_loo"


def _load_scene_df(root, split, scene):
    """Wrap build_feature_matrix, drop rows with no label, tag with the scene name.

    Returns the full DataFrame (not bare (X,y) arrays like train_pooled.py's load_split) --
    this script needs "camera" (and the "scene" tag added here) intact for (scene,camera)
    group construction.
    """
    npz = Path(root) / split / scene / "oracle_dq.npz"
    df, _ = build_feature_matrix(str(npz))
    df = df[df[LABEL].notna()].copy()
    df["scene"] = scene
    return df


def _add_rank_label_pooled(df, n_bins):
    """Same percentile-bin transform as train_lgbm_rank.py's _add_rank_label, but grouped
    by (scene, camera) -- camera indices repeat across scenes, "camera" alone would collide.
    """
    df = df.copy()
    df["marginal_key"] = exp_clipped(df[LABEL].to_numpy()) / df["N_k"].clip(lower=1.0)
    df["rank_label"] = (
        df.groupby(["scene", "camera"])["marginal_key"]
          .transform(lambda s: _rank_label(s.to_numpy(), n_bins))
    )
    return df


def _pooled_fold_arrays(df, feat_cols):
    """Sort by (scene, camera) so rows are contiguous per group; group array respects
    scene boundaries -- lambdarank never compares tiles across scenes.
    """
    sub = df.sort_values(["scene", "camera"], kind="stable")
    X = sub[feat_cols].values.astype(np.float32)
    y = sub["rank_label"].values.astype(np.int32)
    group = sub.groupby(["scene", "camera"], sort=False).size().to_numpy()
    return X, y, group


def _scene_eval_arrays(df, scene, feat_cols):
    sub = df[df["scene"] == scene].sort_values("camera", kind="stable")
    X = sub[feat_cols].values.astype(np.float32)
    marginal = sub["marginal_key"].values
    cam_ids = sub["camera"].values
    return X, marginal, cam_ids


def _val_split(df, scenes, seed, frac=0.20):
    """Per-scene camera holdout for early stopping -- every scene contributes to the
    validation slice, not just one (avoids one scene's tile-count dominating the split).
    """
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(df), dtype=bool)
    for s in scenes:
        cams = np.sort(df.loc[df["scene"] == s, "camera"].unique())
        n_val = max(1, int(frac * len(cams)))
        val_cams = rng.choice(cams, n_val, replace=False)
        val_mask |= (df["scene"] == s).to_numpy() & df["camera"].isin(val_cams).to_numpy()
    return val_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-set", choices=list(SCENE_SETS), default="real3")
    ap.add_argument("--ablation", default="ACG")
    ap.add_argument("--root", default="output/oracle/8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bins", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = SCENE_SETS[args.scene_set]
    feat_cols = feature_names_for_ablation(args.ablation)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{time.time()-t0:6.1f}s] scene-set={args.scene_set} ({len(scenes)} scenes) "
          f"ablation={args.ablation} ({len(feat_cols)} feats)", flush=True)

    tr_dfs, ev_dfs = {}, {}
    for s in scenes:
        tr_dfs[s] = _load_scene_df(args.root, "train", s)
        ev_dfs[s] = _load_scene_df(args.root, "eval", s)
        print(f"[{time.time()-t0:6.1f}s]   {s:10s} train_rows={len(tr_dfs[s])} "
              f"eval_rows={len(ev_dfs[s])}", flush=True)

    df_pool = pd.concat([tr_dfs[s] for s in scenes], ignore_index=True)
    df_pool = _add_rank_label_pooled(df_pool, args.n_bins)
    print(f"[{time.time()-t0:6.1f}s] pooled train set: {len(df_pool):,} rows", flush=True)

    # ── fit the pooled ranker, early-stopped on a per-scene camera holdout ──────────────
    val_mask = _val_split(df_pool, scenes, args.seed)
    df_fit, df_val = df_pool[~val_mask], df_pool[val_mask]
    X_fit, y_fit, g_fit = _pooled_fold_arrays(df_fit, feat_cols)
    X_val, y_val, g_val = _pooled_fold_arrays(df_val, feat_cols)

    print(f"[{time.time()-t0:6.1f}s] fitting pooled ranker "
          f"(fit={len(df_fit):,} rows, val={len(df_val):,} rows)...", flush=True)
    pooled_model = fit_single_scene_ranker(
        X_fit, y_fit, g_fit, args.seed, n_estimators=2000,
        eval_set=[(X_val, y_val)], eval_group=[g_val], early_stopping_rounds=50,
    )

    # ── per-scene comparison: pooled ranker vs a fresh per-scene-only ranker baseline,
    #    both evaluated on that scene's own EVAL split ──────────────────────────────────
    rows = []
    for s in scenes:
        df_s = _add_rank_label_pooled(tr_dfs[s].copy().assign(scene=s), args.n_bins)
        val_mask_s = _val_split(df_s, [s], args.seed)
        Xs_fit, ys_fit, gs_fit = _pooled_fold_arrays(df_s[~val_mask_s], feat_cols)
        Xs_val, ys_val, gs_val = _pooled_fold_arrays(df_s[val_mask_s], feat_cols)
        m_ps = fit_single_scene_ranker(
            Xs_fit, ys_fit, gs_fit, args.seed, n_estimators=2000,
            eval_set=[(Xs_val, ys_val)], eval_group=[gs_val], early_stopping_rounds=50,
        )

        df_ev_s = _add_rank_label_pooled(ev_dfs[s].copy().assign(scene=s), args.n_bins)
        X_ev, marg_ev, cam_ev = _scene_eval_arrays(df_ev_s, s, feat_cols)

        rho_ps = np.median(_per_camera_rho(m_ps.predict(X_ev), marg_ev, cam_ev))
        rho_pooled = np.median(_per_camera_rho(pooled_model.predict(X_ev), marg_ev, cam_ev))

        rows.append(dict(scene=s, per_scene_rho=float(rho_ps), pooled_rho=float(rho_pooled),
                         delta=float(rho_pooled - rho_ps), n_cams=int(len(np.unique(cam_ev)))))
        print(f"[{time.time()-t0:6.1f}s] {s:10s} per-scene rho={rho_ps:.3f}  "
              f"pooled rho={rho_pooled:.3f}  delta={rho_pooled-rho_ps:+.3f}", flush=True)

    # ── save ──────────────────────────────────────────────────────────────────────────
    joblib.dump(pooled_model, out_dir / "lgbm_rank.pkl")
    (out_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2))
    metrics = {
        "ablation": args.ablation,
        "scene_set": args.scene_set,
        "scenes": list(scenes),
        "n_bins": args.n_bins,
        "seed": args.seed,
        "n_rows_pooled": int(len(df_pool)),
        "comparison_table": rows,
        "source_ply_n_gs": None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 60)
    print(f"{'scene':10s} {'per-scene':>10s} {'pooled':>8s} {'delta':>8s}")
    for r in rows:
        print(f"{r['scene']:10s} {r['per_scene_rho']:10.3f} {r['pooled_rho']:8.3f} "
              f"{r['delta']:+8.3f}")
    print(f"\n[{time.time()-t0:6.1f}s] saved -> {out_dir}/lgbm_rank.pkl + metrics.json")


if __name__ == "__main__":
    main()
