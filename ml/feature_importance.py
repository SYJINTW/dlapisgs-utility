"""RF feature-importance: MDI vs permutation, per-scene + cross-scene.

Loads each scene's fitted RF (models_clean/{scene}/{ablation}/rf.pkl), rebuilds
the held-out EVAL feature matrix, and computes two importance measures:

  - MDI  : rf.feature_importances_  (impurity, train-biased — for reference)
  - perm : sklearn permutation_importance on the held-out eval set
           (model-agnostic, fixes MDI's train-set + cardinality bias)

Reports per-feature CSV, group-level aggregation (A / C-mean / C-std), the
MDI-vs-perm agreement (Spearman) per scene, and cross-scene perm consistency.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n gsquic python ml/feature_importance.py \\
        --eval-root output/0527_clean/exp4_oracle_dq/eval \\
        --models-root ml/models_clean \\
        --ablation AC --n-repeats 5 \\
        --out ml/feature_importance/AC
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from ml.features import build_feature_matrix, feature_names_for_ablation  # noqa: E402

SCENES = ("bicycle", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship")


def _group_of(name: str) -> str:
    """Semantic group for AC features."""
    if name in ("v_k", "d_k", "N_k") or name.startswith(("cam_", "fov", "tile_")):
        return "A:geom/viewport"
    if name.endswith("_std"):
        return "C:attr_std"
    return "C:attr_mean"


def run_scene(scene, eval_root, models_root, ablation, n_repeats, seed, out_dir):
    npz = Path(eval_root) / scene / "oracle_dq.npz"
    rf_path = Path(models_root) / scene / ablation / "rf.pkl"
    print(f"\n=== {scene} ===", flush=True)
    print(f"  building eval matrix: {npz}", flush=True)
    df, _ = build_feature_matrix(str(npz))
    df = df[df["log_mse_loo"].notna()].copy()
    feat_cols = feature_names_for_ablation(ablation)
    X = df[feat_cols].values.astype(np.float32)
    y = df["log_mse_loo"].values.astype(np.float32)
    print(f"  rows={len(X)} feats={len(feat_cols)}  loading {rf_path}", flush=True)

    rf = joblib.load(rf_path)
    mdi = rf.feature_importances_
    base_r2 = rf.score(X, y)
    print(f"  eval R²={base_r2:.4f}  running permutation_importance "
          f"(n_repeats={n_repeats})…", flush=True)
    pi = permutation_importance(rf, X, y, n_repeats=n_repeats,
                                random_state=seed, n_jobs=-1)

    rows = pd.DataFrame({
        "feature":   feat_cols,
        "group":     [_group_of(f) for f in feat_cols],
        "mdi":       mdi,
        "perm_mean": pi.importances_mean,
        "perm_std":  pi.importances_std,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / f"{scene}.csv", index=False)

    rho_agree = float(spearmanr(mdi, pi.importances_mean).correlation)
    print(f"  saved {scene}.csv  | Spearman(MDI, perm) = {rho_agree:.3f}", flush=True)
    return {
        "scene": scene,
        "eval_r2": float(base_r2),
        "mdi_vs_perm_spearman": rho_agree,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--models-root", default="ml/models_clean")
    ap.add_argument("--ablation", default="AC")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    results = [
        run_scene(s, args.eval_root, args.models_root, args.ablation,
                  args.n_repeats, args.seed, out_dir)
        for s in args.scenes
    ]

    feat_cols = list(results[0]["rows"]["feature"])
    groups = [_group_of(f) for f in feat_cols]
    G = sorted(set(groups))
    scenes = [r["scene"] for r in results]
    MDI  = np.vstack([r["rows"]["mdi"].values       for r in results])
    PERM = np.vstack([r["rows"]["perm_mean"].values for r in results])
    # perm can be slightly negative (noise); clip for group-share readout only
    PERM_share = np.clip(PERM, 0, None)

    summary = {"ablation": args.ablation, "scenes": scenes,
               "n_repeats": args.n_repeats,
               "eval_r2": {r["scene"]: r["eval_r2"] for r in results},
               "mdi_vs_perm_spearman": {r["scene"]: r["mdi_vs_perm_spearman"]
                                        for r in results}}

    print("\n" + "=" * 64)
    print("MDI vs permutation agreement (Spearman over 132 feats), per scene")
    for r in results:
        print(f"  {r['scene']:10s} eval R²={r['eval_r2']:.3f}  "
              f"ρ(MDI,perm)={r['mdi_vs_perm_spearman']:.3f}")
    print(f"  {'MEAN':10s}            "
          f"ρ(MDI,perm)={np.mean([r['mdi_vs_perm_spearman'] for r in results]):.3f}")

    print("\nGroup-level importance share (% of positive total)")
    print("  measure  " + "".join(g.ljust(18) for g in G))
    grp_tbl = {}
    for tag, Mat in (("MDI", MDI), ("perm", PERM_share)):
        shares = {}
        for g in G:
            mask = np.array([gg == g for gg in groups])
            frac = (Mat[:, mask].sum(1) / Mat.sum(1)).mean() * 100
            shares[g] = float(frac)
        grp_tbl[tag] = shares
        print(f"  {tag:7s}  " + "".join(f"{shares[g]:6.1f}".ljust(18) for g in G))
    summary["group_share_pct"] = grp_tbl

    print("\nTop-12 features by cross-scene MEAN permutation importance")
    perm_mean = PERM.mean(0); perm_sd = PERM.std(0)
    top_perm = []
    for i in np.argsort(perm_mean)[::-1][:12]:
        print(f"  {perm_mean[i]:+.4f} ±{perm_sd[i]:.4f}  {feat_cols[i]}")
        top_perm.append({"feature": feat_cols[i],
                         "perm_mean_xscene": float(perm_mean[i]),
                         "perm_std_xscene": float(perm_sd[i])})
    summary["top_perm_xscene"] = top_perm

    # cross-scene consistency of permutation rankings
    pair = [spearmanr(PERM[a], PERM[b]).correlation
            for a in range(len(scenes)) for b in range(a + 1, len(scenes))]
    summary["perm_xscene_pairwise_spearman_mean"] = float(np.mean(pair))
    summary["perm_xscene_pairwise_spearman_min"]  = float(np.min(pair))
    summary["perm_xscene_pairwise_spearman_max"]  = float(np.max(pair))
    print(f"\nCross-scene perm consistency: mean pairwise ρ={np.mean(pair):.3f} "
          f"(min {np.min(pair):.3f}, max {np.max(pair):.3f})")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_dir}/summary.json + per-scene CSVs")


if __name__ == "__main__":
    main()
