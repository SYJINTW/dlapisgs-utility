"""Inference helper: per-(tile, LOD=0) utility scores from a trained LightGBM model.

Returns sorted (tile_idx, lod) pairs in the same format as uc.calculate_utility_param,
so it drops into the existing _greedy_order_* functions unchanged.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
_WORKSPACE = _HERE.parent
for _p in (_WORKSPACE / "Frustum-for-3DGS", _WORKSPACE / "GGSP", _WORKSPACE / "GS-Interface"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import utility_calculation as uc  # noqa: E402

FEAT_COLS = ["lod", "v_k", "d_k", "C_k", "W_k"]


def predict_utility(
    model_dir: str,
    head: str,
    *,
    visibility_t,
    distances_t,
    tile_index_offsets_t,
    W_k_t,
    num_lod: int = 1,
) -> np.ndarray:
    """Predict tile utility scores and return sorted (tile_idx, lod) pairs.

    Args:
        model_dir: Directory with model_{head}.pkl and ols_coefs.json.
        head: "raw" or "resid".
        visibility_t: bool tensor [N_tiles].
        distances_t: float tensor [N_tiles].
        tile_index_offsets_t: long tensor [N_tiles+1] from tiling cache.
        W_k_t: float tensor [N_tiles], sum-normalized per-tile screen_area weight.
        num_lod: kept for API parity; v1 always emits lod=0.

    Returns:
        np.ndarray shape (N_tiles, 2), dtype int64 — (tile_idx, lod) pairs
        sorted descending by predicted utility. Matches calculate_utility_param output.
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / f"model_{head}.pkl")

    beta0, beta1 = 0.0, 0.0
    if head == "resid":
        coefs = json.loads((model_dir / "ols_coefs.json").read_text())
        beta0, beta1 = coefs["beta0"], coefs["beta1"]

    v_k = visibility_t.float().cpu().numpy()
    d_k = distances_t.cpu().numpy()
    offsets = tile_index_offsets_t.cpu().numpy()
    C_k = (offsets[1:] - offsets[:-1]).astype(np.float32)
    W_k = W_k_t.cpu().numpy()

    N_tiles = len(v_k)
    X = np.column_stack([
        np.zeros(N_tiles, dtype=np.float32),  # lod=0
        v_k,
        d_k,
        C_k,
        W_k,
    ])

    scores = model.predict(X).astype(np.float64)

    if head == "resid":
        scores += beta0 + beta1 * np.log(C_k.clip(1))

    scores[v_k == 0] = uc.INVISIBLE_PRIORITY_EPS

    order = np.argsort(-scores, kind="stable")
    pairs = np.column_stack([order, np.zeros(N_tiles, dtype=np.int64)])
    return pairs.astype(np.int64)
