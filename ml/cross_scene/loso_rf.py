"""Cross-scene RF generalization: LOSO + synthetic->real.

Question: does ONE RF trained across scenes predict oracle (LOO proxy) tile
importance on a held-out scene as well as a per-scene specialist?

Protocol (SPEC.md Deliverable 1):
  - Leave-one-scene-out: train on the other 7 scenes, test on held-out scene's
    EVAL split. Rotate. Headline = LOSO rho vs per-scene rho, same test rows.
  - synthetic->real: train on 7 synthetic, test on each real scene's eval split.
  - Per-scene baseline: RF trained on scene's TRAIN split, tested on its EVAL
    split (identical test rows as LOSO -> clean head-to-head).
  - Label z-scored WITHIN each scene before pooling (so scenes contribute
    comparably to the regression). Eval metric = Spearman rho within held-out
    scene; z-score is rank-preserving so it does not affect rho.

All RFs use the deployed config: n_estimators=500, msl=2, mf=0.3.

Usage:
    conda run -n gsquic python ml/cross_scene/loso_rf.py \\
        --root output/0527_clean/exp4_oracle_dq \\
        --ablation AC --out ml/cross_scene
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
from ml.features import build_feature_matrix, feature_names_for_ablation  # noqa: E402

SCENES = ("bicycle", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship")
REAL = ("bicycle",)  # garden/stump not yet onboarded
SYNTH = tuple(s for s in SCENES if s not in REAL)

RF_KW = dict(n_estimators=500, min_samples_leaf=2, max_features=0.3,
             random_state=0, n_jobs=-1)
LABEL = "log_mse_loo"


def load_split(root, split, scene, feat_cols):
    """Return (X, y) for one scene's split, visible/labelled rows only."""
    npz = Path(root) / split / scene / "oracle_dq.npz"
    df, _ = build_feature_matrix(str(npz))
    df = df[df[LABEL].notna()].copy()
    X = df[feat_cols].values.astype(np.float32)
    y = df[LABEL].values.astype(np.float32)
    return X, y


def zscore(y):
    mu, sd = float(y.mean()), float(y.std())
    return (y - mu) / (sd if sd > 0 else 1.0)


def fit_predict_rho(X_tr, y_tr, X_te, y_te):
    rf = RandomForestRegressor(**RF_KW)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    rho = float(spearmanr(pred, y_te).correlation)
    return rho


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="output/0527_clean/exp4_oracle_dq")
    ap.add_argument("--ablation", default="AC")
    ap.add_argument("--out", default="output/cross_scene")
    args = ap.parse_args()

    feat_cols = feature_names_for_ablation(args.ablation)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Cache all train/eval matrices once.
    print(f"[{time.time()-t0:6.1f}s] loading {len(SCENES)} scenes x 2 splits, "
          f"ablation={args.ablation} ({len(feat_cols)} feats)…", flush=True)
    tr, ev = {}, {}
    for s in SCENES:
        tr[s] = load_split(args.root, "train", s, feat_cols)
        ev[s] = load_split(args.root, "eval", s, feat_cols)
        print(f"[{time.time()-t0:6.1f}s]   {s:10s} train={tr[s][0].shape} "
              f"eval={ev[s][0].shape}", flush=True)

    rows = []
    # --- per-scene baseline + LOSO, same eval rows ---
    for held in SCENES:
        Xte, yte = ev[held]

        # per-scene: train on held's own train split
        Xtr_ps, ytr_ps = tr[held]
        rho_ps = fit_predict_rho(Xtr_ps, ytr_ps, Xte, yte)

        # LOSO: pool the OTHER 7 scenes' train+eval, z-score label per scene
        Xs, ys = [], []
        for s in SCENES:
            if s == held:
                continue
            for X, y in (tr[s], ev[s]):
                Xs.append(X)
                ys.append(zscore(y))
        Xtr_lo = np.vstack(Xs)
        ytr_lo = np.concatenate(ys)
        rho_lo = fit_predict_rho(Xtr_lo, ytr_lo, Xte, yte)

        rows.append(dict(held_scene=held, kind="loso",
                         per_scene_rho=rho_ps, loso_rho=rho_lo,
                         delta=rho_lo - rho_ps, n_train=len(ytr_lo),
                         n_test=len(yte)))
        print(f"[{time.time()-t0:6.1f}s] LOSO {held:10s} "
              f"per-scene rho={rho_ps:.3f}  loso rho={rho_lo:.3f}  "
              f"Δ={rho_lo-rho_ps:+.3f}", flush=True)

    # --- synthetic -> real ---
    Xs = [X for s in SYNTH for X, _ in (tr[s], ev[s])]
    ys = [zscore(y) for s in SYNTH for _, y in (tr[s], ev[s])]
    Xtr_sr, ytr_sr = np.vstack(Xs), np.concatenate(ys)
    s2r = []
    for held in REAL:
        Xte, yte = ev[held]
        rho_sr = fit_predict_rho(Xtr_sr, ytr_sr, Xte, yte)
        rho_ps = next(r["per_scene_rho"] for r in rows if r["held_scene"] == held)
        s2r.append(dict(held_scene=held, kind="synth2real",
                        per_scene_rho=rho_ps, synth2real_rho=rho_sr,
                        delta=rho_sr - rho_ps, n_train=len(ytr_sr),
                        n_test=len(yte)))
        print(f"[{time.time()-t0:6.1f}s] SYNTH->REAL {held:10s} "
              f"per-scene rho={rho_ps:.3f}  synth2real rho={rho_sr:.3f}  "
              f"Δ={rho_sr-rho_ps:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "loso_table.csv", index=False)
    summary = {
        "ablation": args.ablation,
        "rf_kwargs": {k: v for k, v in RF_KW.items() if k != "n_jobs"},
        "loso": rows,
        "synth2real": s2r,
        "loso_mean": {
            "per_scene_rho": float(df["per_scene_rho"].mean()),
            "loso_rho": float(df["loso_rho"].mean()),
            "delta": float(df["delta"].mean()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"{'scene':10s} {'per-scene':>10s} {'LOSO':>8s} {'Δ':>8s}")
    for r in rows:
        print(f"{r['held_scene']:10s} {r['per_scene_rho']:10.3f} "
              f"{r['loso_rho']:8.3f} {r['delta']:+8.3f}")
    print(f"{'MEAN':10s} {df['per_scene_rho'].mean():10.3f} "
          f"{df['loso_rho'].mean():8.3f} {df['delta'].mean():+8.3f}")
    for r in s2r:
        print(f"\nsynth->real {r['held_scene']}: per-scene={r['per_scene_rho']:.3f} "
              f"synth2real={r['synth2real_rho']:.3f}  Δ={r['delta']:+.3f}")
    print(f"\n[{time.time()-t0:6.1f}s] saved → {out_dir}/summary.json + loso_table.csv")


if __name__ == "__main__":
    main()
