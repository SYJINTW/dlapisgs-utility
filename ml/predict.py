"""Inference helper: per-tile utility scores from a trained ML model.

Loads a model from model_dir/{model_type}.pkl and feature_names.json,
assembles features from precomputed static (C) + per-camera (A, includes
cos_cam_tile), returns (tile_idx, lod=0) pairs sorted descending by predicted
score.

The model output is used directly as tile priority — no invisible-tile floor.
v_k is in Group A so the model learns visibility importance end-to-end.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


def load_model(model_dir: str, model_type: str, expected_n_gs: Optional[int] = None):
    """Load a trained model once (startup, not the per-camera hot path) and force
    single-thread predict. Callers pass the returned object into predict_utility's
    model= kwarg so every camera call reuses it instead of reloading/repatching.

    n_jobs=1 is real for sklearn RandomForest (joblib n_jobs=-1 spins a thread pool per
    tiny-batch predict call, ~100ms dispatch overhead vs ~10-15ms single-thread traversal).
    No-op for XGBoost's sklearn wrapper -- its predict() takes the inplace_predict path,
    which never reads self.n_jobs (thread count is baked in at train time). Not currently
    a measurable stall for XGB (OpenMP pools persist across calls), so left in place but
    don't rely on this for XGB single-threading.

    expected_n_gs, if given, is cross-checked against metrics.json's source_ply_n_gs
    (written by ml/train.py at train time) -- catches pointing --ml-model-dir at the
    wrong scene's directory, which otherwise silently applies one scene's learned splits
    to another scene's correctly-computed features (feature names are scene-agnostic, so
    nothing else raises). Models saved before this check existed (no metrics.json, or no
    source_ply_n_gs field) skip the check with a warning, not a hard failure.
    """
    model_pkl = Path(model_dir) / f"{model_type}.pkl"
    if not model_pkl.exists():
        raise FileNotFoundError(
            f"Model not found: {model_pkl}. "
            f"Train first with: python ml/train.py --oracle-npz <path> --output-dir <dir>"
        )
    if expected_n_gs is not None:
        metrics_path = Path(model_dir) / "metrics.json"
        if metrics_path.exists():
            trained_n_gs = json.loads(metrics_path.read_text()).get("source_ply_n_gs")
            if trained_n_gs is not None and trained_n_gs != expected_n_gs:
                raise ValueError(
                    f"Model at {model_dir} was trained on a PLY with {trained_n_gs} "
                    f"Gaussians, but the current scene has {expected_n_gs}. "
                    "Wrong --ml-model-dir for this scene?"
                )
            elif trained_n_gs is None:
                print(f"[ml] {metrics_path} has no source_ply_n_gs -- skipping "
                      "scene-identity check (model trained before this check existed).",
                      flush=True)
        else:
            print(f"[ml] no metrics.json at {model_dir} -- skipping scene-identity "
                  "check (model trained before this check existed).", flush=True)
    model = joblib.load(model_pkl)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def predict_utility(
    model_dir: str,
    model_type: str,          # "lgbm" | "xgb" | "rf"
    *,
    static_features: np.ndarray,   # (N_tiles, 118) — Group C, precomputed at startup
    group_a: np.ndarray,            # (N_tiles, 15)  — Group A per camera (incl. cos_cam_tile)
    feature_names: list,            # list[str] from feature_names.json
    model=None,                     # pre-loaded model; if None, load from disk
) -> np.ndarray:
    """Predict tile utility and return sorted (tile_idx, lod=0) pairs.

    Args:
        model_dir:       Dir with {lgbm|xgb|rf}.pkl and feature_names.json.
        model_type:      Which model to use: "lgbm", "xgb", or "rf".
        static_features: Group C precomputed at startup. Shape (N_tiles, 118).
        group_a:         Group A per-camera features. Shape (N_tiles, 15).
        feature_names:   Column names matching training order.

    Returns:
        np.ndarray shape (N_tiles,), dtype float64: raw predicted utility scores.
    """
    if model is None:
        model_dir = Path(model_dir)
        model_pkl = model_dir / f"{model_type}.pkl"
        if not model_pkl.exists():
            raise FileNotFoundError(
                f"Model not found: {model_pkl}. "
                f"Train first with: python ml/train.py --oracle-npz <path> --output-dir <dir>"
            )
        model = joblib.load(model_pkl)

    # ── Assemble full feature matrix [A | C] ────────────────────────────────
    X_all = np.hstack([group_a, static_features])

    # ── Select columns matching training feature order ─────────────────────
    from ml.features import GROUP_A_NAMES, GROUP_C_NAMES
    all_col_names = GROUP_A_NAMES + GROUP_C_NAMES
    col_map = {name: i for i, name in enumerate(all_col_names)}
    try:
        feat_idx = [col_map[n] for n in feature_names]
    except KeyError as e:
        raise KeyError(
            f"Feature '{e.args[0]}' in feature_names.json not found in assembled matrix."
        ) from e

    X = X_all[:, feat_idx].astype(np.float32)

    # ── Predict ───────────────────────────────────────────────────────────
    return model.predict(X).astype(np.float64)   # (N_tiles,) raw scores


# Legitimate log(mse_loo) values live in ~[-70, 5] (features.py's MSE_EPS=1e-30 floor
# sets the low end; observed real ΔMSE values set the high end). exp_clipped bounds both
# ends independently of that upstream floor -- a single OOD feature row, or a future
# regression in features.py's floor, can't produce a linear score that overflows or
# dominates the marginal (per-byte) greedy sort. 2026-07-09: every exp(log_mse_loo) call
# site was unclamped before this.
LOG_MSE_CLIP = (-80.0, 20.0)


def exp_clipped(log_val: np.ndarray) -> np.ndarray:
    """exp() with a defensive clip on the log-space input -- see LOG_MSE_CLIP."""
    return np.exp(np.clip(log_val, *LOG_MSE_CLIP))


_BLEND_MODEL_TYPES = ("rf", "lgbm", "xgb")


def load_blend_models(model_dir: str, model_types=_BLEND_MODEL_TYPES,
                       expected_n_gs: Optional[int] = None) -> dict:
    """Load whichever of rf/lgbm/xgb.pkl exist in model_dir (startup, once).

    Mirrors load_model's n_jobs=1 + scene-identity guard for each present model type.
    Missing model files are skipped, not an error -- a dir with only 2 of 3 pkls still
    works for blending.
    """
    models = {}
    for model_type in model_types:
        if (Path(model_dir) / f"{model_type}.pkl").exists():
            models[model_type] = load_model(model_dir, model_type, expected_n_gs=expected_n_gs)
    if not models:
        raise FileNotFoundError(
            f"No {model_types} models found in {model_dir} for ml_blend scheme.")
    return models


def predict_utility_blend(
    model_dir: str,
    *,
    static_features: np.ndarray,
    group_a: np.ndarray,
    feature_names: list,
    models: "dict | None" = None,
) -> np.ndarray:
    """Average per-tile importance across the RF/LGBM/XGB models already trained for
    one scene/ablation (no retrain). Each model predicts log(mse_loo); exponentiate
    per-model first (recovers linear ΔMSE, the space the greedy per-byte key sorts in),
    then average -- averaging in linear space, not log space.

    models: dict model_type -> loaded model (from load_blend_models), reused across
    cameras. If None, loads every model in model_dir fresh (slow -- prefer preloading).

    Returns:
        np.ndarray shape (N_tiles,): blended positive-linear importance.
    """
    if models is None:
        models = load_blend_models(model_dir)

    linear_preds = [
        exp_clipped(predict_utility(
            model_dir, model_type,
            static_features=static_features, group_a=group_a,
            feature_names=feature_names, model=model,
        ))
        for model_type, model in models.items()
    ]
    return np.mean(linear_preds, axis=0)
