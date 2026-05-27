"""PCA diagnostics on raw GS attributes per scene.

Plots:
  1. Cumulative explained variance vs number of PCs
  2. Attribute correlation heatmap (59×59)
  3. tSNE of GS (subsample, colored by tile index)

Usage:
    conda run -n gsquic python experiments/0527/plot_pca_gs.py \
        --scene ship \
        --oracle-npz output/0522/exp4_oracle_dq/ship/oracle_dq.npz \
        --out-dir output/quickplots/0527/pca_gs/ship
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from ml.features import ATTR_NAMES  # noqa: E402

# PLY attribute transforms (match ml/features.py build_static_features)
def _apply_transforms(raw: np.ndarray) -> np.ndarray:
    """raw: (N_gs, 59). Returns transformed copy."""
    out = raw.copy().astype(np.float32)
    op_idx = ATTR_NAMES.index("opacity")
    sc_idx = [ATTR_NAMES.index(a) for a in ("scale_0", "scale_1", "scale_2")]
    out[:, op_idx] = 1.0 / (1.0 + np.exp(-out[:, op_idx]))   # sigmoid
    for i in sc_idx:
        out[:, i] = np.exp(out[:, i])                          # exp
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--oracle-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tsne-n", type=int, default=5000,
                        help="GS subsample size for tSNE (default 5000)")
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip tSNE (slow for large scenes)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load oracle NPZ to get PLY path + tiling ──────────────────────────────
    npz = np.load(args.oracle_npz, allow_pickle=True)
    meta = json.loads(str(npz["gen_meta"].item()))
    ply_path = meta["scene_ply"]
    index_offsets = npz["index_offsets"].astype(np.int64)   # (N_tiles+1,)
    flat_indices  = npz["flat_indices"].astype(np.int64)    # (N_gs_tiled,)
    n_gs_per_tile = npz["n_gs_per_tile"].astype(np.int64)
    N_tiles = len(n_gs_per_tile)
    print(f"Scene: {args.scene}  tiles={N_tiles}  ply={ply_path}", flush=True)

    # ── Load PLY ──────────────────────────────────────────────────────────────
    WORKSPACE = HERE.parent
    for _p in (WORKSPACE/"GS-Interface", WORKSPACE/"Frustum-for-3DGS", WORKSPACE/"GGSP"):
        _s = str(_p)
        if _s not in sys.path:
            sys.path.insert(0, _s)
    import io_3dgs  # noqa: E402
    gs = io_3dgs.GaussianModelV2(ply_path)

    # Build (N_gs, 59) matrix in canonical order with transforms
    raw = np.stack([gs.data[a]["data"].astype(np.float32) for a in ATTR_NAMES], axis=1)
    X_gs = _apply_transforms(raw)  # (N_gs_full, 59)
    print(f"GS matrix: {X_gs.shape}", flush=True)

    # ── Global PCA on all scene GS ────────────────────────────────────────────
    print("Fitting PCA ...", flush=True)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_gs)
    pca = PCA(n_components=min(59, X_gs.shape[0]))
    pca.fit(X_sc)

    # 1. Cumulative explained variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum_var, 0.90)) + 1
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1
    n99 = int(np.searchsorted(cum_var, 0.99)) + 1
    print(f"PCs for 90%={n90}  95%={n95}  99%={n99}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cum_var)+1), cum_var * 100, marker="o", markersize=3)
    for thresh, n, color in [(90, n90, "orange"), (95, n95, "red"), (99, n99, "purple")]:
        ax.axhline(thresh, color=color, linestyle="--", linewidth=0.8,
                   label=f"{thresh}% → {n} PCs")
        ax.axvline(n, color=color, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title(f"PCA on GS attributes — {args.scene}\n"
                 f"({X_gs.shape[0]:,} Gaussians, 59 attrs after transforms)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_cumvar.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/pca_cumvar.png", flush=True)

    # 2. Attribute correlation heatmap
    print("Computing correlation heatmap ...", flush=True)
    corr = np.corrcoef(X_sc.T)  # (59, 59)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(np.abs(corr), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="|Pearson r|")
    # Label only non-SH attrs for readability
    tick_labels = []
    tick_pos = []
    for i, name in enumerate(ATTR_NAMES):
        if not name.startswith("f_rest"):
            tick_labels.append(name)
            tick_pos.append(i)
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title(f"GS attribute correlation |r| — {args.scene}")
    fig.tight_layout()
    fig.savefig(out_dir / "attr_corr.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/attr_corr.png", flush=True)

    # 3. tSNE (subsample, colored by tile)
    if not args.no_tsne:
        print(f"Running tSNE on {args.tsne_n} GS subsample ...", flush=True)
        try:
            from sklearn.manifold import TSNE
            rng = np.random.default_rng(0)
            # subsample uniformly from tiled GS
            n_sub = min(args.tsne_n, len(flat_indices))
            sub_idx = rng.choice(len(flat_indices), n_sub, replace=False)
            # map flat_indices to tile membership
            tile_of_gs = np.empty(len(flat_indices), dtype=np.int32)
            for t in range(N_tiles):
                s, e = index_offsets[t], index_offsets[t+1]
                tile_of_gs[s:e] = t
            gs_sub = flat_indices[sub_idx]
            tile_sub = tile_of_gs[sub_idx]
            X_sub = X_sc[gs_sub]  # (n_sub, 59)
            # project to top-20 PCs first (speed)
            X_pca20 = pca.transform(X_sub)[:, :20]
            tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=500)
            emb = tsne.fit_transform(X_pca20)
            fig, ax = plt.subplots(figsize=(8, 7))
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=tile_sub, cmap="tab20",
                            s=2, alpha=0.5, linewidths=0)
            plt.colorbar(sc, ax=ax, label="Tile index")
            ax.set_title(f"tSNE of GS attrs (PC1-20 input) — {args.scene}\n"
                         f"color=tile index, n={n_sub:,} subsample")
            ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_dir / "tsne_tiles.png", dpi=150)
            plt.close(fig)
            print(f"Saved {out_dir}/tsne_tiles.png", flush=True)
        except Exception as e:
            print(f"tSNE failed: {e}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
