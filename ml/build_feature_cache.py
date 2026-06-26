"""Generate per-scene static feature caches (Group C+D) shipped with the model.

Camera-invariant features are computed once from each scene's oracle_dq.npz and
saved as {model_dir}/feature_cache.npz. Selection loads this instead of rebuilding
from the PLY, so a deployed selector does zero per-scene GPU feature work.

Usage:
    conda run -n gsquic python ml/build_feature_cache.py \\
        --scenes bicycle garden stump chair drums ficus hotdog materials mic ship \\
        --oracle-root output/0527_clean/exp4_oracle_dq/eval \\
        --model-root ml/models_clean --ablation AC \\
        [--self-check]

Single scene:
    ... --oracle-npz <path>/oracle_dq.npz --out <model_dir>/feature_cache.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ml import features as ml_features  # noqa: E402


def _generate(oracle_npz: str, out_path: str, self_check: bool) -> None:
    static_feats, index_offsets = ml_features.static_cache_from_oracle(oracle_npz)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ml_features.save_feature_cache(out_path, static_feats, index_offsets)
    print(f"  saved {out_path}  shape={static_feats.shape}", flush=True)

    if self_check:
        # Reload through the validated path and confirm exact match with the build
        # we just persisted (guards tiling/ordering + round-trip integrity).
        loaded = ml_features.load_feature_cache(out_path, index_offsets)
        assert loaded.shape == static_feats.shape, (loaded.shape, static_feats.shape)
        assert np.array_equal(loaded, static_feats.astype(np.float32)), \
            "feature_cache round-trip mismatch"
        print("  self-check OK (exact round-trip match)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--oracle-root", default=None,
                    help="dir holding {scene}/oracle_dq.npz")
    ap.add_argument("--model-root", default=None,
                    help="dir holding {scene}/{ablation}/ — cache written there")
    ap.add_argument("--ablation", default="AC",
                    help="model subdir to drop feature_cache.npz into (default AC)")
    ap.add_argument("--oracle-npz", default=None, help="single-scene oracle npz")
    ap.add_argument("--out", default=None, help="single-scene output npz path")
    ap.add_argument("--self-check", action="store_true",
                    help="reload + assert exact match after writing")
    args = ap.parse_args()

    if args.oracle_npz:
        if not args.out:
            ap.error("--oracle-npz requires --out")
        print(f"[single] {args.oracle_npz}", flush=True)
        _generate(args.oracle_npz, args.out, args.self_check)
        return

    if not (args.scenes and args.oracle_root and args.model_root):
        ap.error("need --scenes + --oracle-root + --model-root (or --oracle-npz + --out)")

    for scene in args.scenes:
        oracle_npz = Path(args.oracle_root) / scene / "oracle_dq.npz"
        out_path = Path(args.model_root) / scene / args.ablation / "feature_cache.npz"
        if not oracle_npz.exists():
            print(f"[skip] {scene}: no {oracle_npz}", flush=True)
            continue
        print(f"[{scene}]", flush=True)
        _generate(str(oracle_npz), str(out_path), args.self_check)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
