"""Inference helper: per-tile utility scores from a trained ML model.

Loads a model from model_dir/{model_type}.pkl and feature_names.json,
assembles features from precomputed static (C+D) + per-camera (A+B),
returns (tile_idx, lod=0) pairs sorted descending by predicted score.

The model output is used directly as tile priority — no invisible-tile floor.
v_k is in Group A so the model learns visibility importance end-to-end.
"""

import json
from pathlib import Path

import joblib
import numpy as np


def predict_utility(
    model_dir: str,
    model_type: str,          # "lgbm" | "xgb" | "rf"
    *,
    static_features: np.ndarray,   # (N_tiles, 134) — Groups C+D, precomputed at startup
    group_a: np.ndarray,            # (N_tiles, 14)  — Group A per camera
    group_b: "np.ndarray | None",   # (N_tiles, 12)  — Group B or None
    feature_names: list,            # list[str] from feature_names.json
    model=None,                     # pre-loaded model; if None, load from disk
) -> np.ndarray:
    """Predict tile utility and return sorted (tile_idx, lod=0) pairs.

    Args:
        model_dir:       Dir with {lgbm|xgb|rf}.pkl and feature_names.json.
        model_type:      Which model to use: "lgbm", "xgb", or "rf".
        static_features: Groups C+D precomputed at startup. Shape (N_tiles, 134).
        group_a:         Group A per-camera features. Shape (N_tiles, 14).
        group_b:         Group B per-camera features. Shape (N_tiles, 12), or None
                         if model was trained without Group B.
        feature_names:   Column names matching training order.

    Returns:
        np.ndarray shape (N_tiles, 2), dtype int64: (tile_idx, lod=0) sorted
        descending by predicted score. Matches calculate_utility_param output format.
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

    # ── Assemble full feature matrix [A | B | C | D] ──────────────────────
    if group_b is not None:
        X_all = np.hstack([group_a, group_b, static_features])  # (N_tiles, 160)
    else:
        X_all = np.hstack([group_a, static_features])            # (N_tiles, 148 or 132)

    # ── Select columns matching training feature order ─────────────────────
    # Build column name → index map from ALL_FEATURE_NAMES order used in X_all.
    # The order in X_all depends on whether group_b is present.
    from ml.features import (GROUP_A_NAMES, GROUP_B_NAMES,
                              GROUP_C_NAMES, GROUP_D_NAMES)
    if group_b is not None:
        all_col_names = GROUP_A_NAMES + GROUP_B_NAMES + GROUP_C_NAMES + GROUP_D_NAMES
    else:
        all_col_names = GROUP_A_NAMES + GROUP_C_NAMES + GROUP_D_NAMES

    col_map = {name: i for i, name in enumerate(all_col_names)}
    try:
        feat_idx = [col_map[n] for n in feature_names]
    except KeyError as e:
        raise KeyError(
            f"Feature '{e.args[0]}' in feature_names.json not found in assembled matrix. "
            "Ensure the model was trained with matching Group B setting."
        ) from e

    X = X_all[:, feat_idx].astype(np.float32)

    # ── Predict and sort ───────────────────────────────────────────────────
    scores = model.predict(X).astype(np.float64)

    N_tiles = len(scores)
    order = np.argsort(-scores, kind="stable")
    pairs = np.column_stack([order, np.zeros(N_tiles, dtype=np.int64)])
    return pairs.astype(np.int64)
