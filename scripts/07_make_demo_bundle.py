"""
Step 7 - build the small bundle that ships in the repo and runs on Streamlit Cloud.

    python scripts/07_make_demo_bundle.py

The full dataset (~1.4 GB of .npy) and the 30 torch checkpoints stay local and gitignored.
This writes demo/ (~35 MB): pre-scaled test vectors plus the ONNX models the app scores with.
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scaler import Scaler

BUDGETS = [1, 5, 10, 25, 100]


def main(a):
    root = Path(__file__).resolve().parents[1]
    out = root / a.out
    (out / "models").mkdir(parents=True, exist_ok=True)

    # --- test vectors, already scaled + clipped so the app needs no scaler at runtime -------
    X = np.load(root / a.data_dir / "X_test.npy", mmap_mode="r")
    y = np.load(root / a.data_dir / "y_test.npy")
    te = np.load(root / a.data_dir / "splits.npz")["idx_test_keep"]
    sc = Scaler.load(root / a.model_dir / "scaler.npz")

    rng = np.random.default_rng(a.seed)
    per = a.n // 2
    pick = np.concatenate([rng.choice(te[y[te] == c], per, replace=False) for c in (0, 1)])
    pick = np.sort(pick)
    np.save(out / "X_demo.npy", sc.transform(np.asarray(X[pick])))
    np.save(out / "y_demo.npy", y[pick].astype(np.int8))
    print(f"demo rows: {len(pick)}  ({per} benign / {per} malicious)")

    # --- onnx models the app can actually run ---------------------------------------------
    kept = []
    for b in BUDGETS:
        src = root / a.model_dir / "quant" / f"{b:03d}_{a.method}_s{a.seed}"
        for name, dst in (("model_fp32.onnx", f"{b:03d}_fp32.onnx"),
                          ("model_int8_dynamic.onnx", f"{b:03d}_int8.onnx")):
            p = src / name
            if p.exists():
                shutil.copy(p, out / "models" / dst)
                kept.append(dst)
            else:
                print(f"  missing {p} - run scripts/06_quantize.py --all --keep-models")
    manifest = dict(n_rows=int(len(pick)), method=a.method, seed=a.seed,
                    budgets=[b for b in BUDGETS
                             if (out / "models" / f"{b:03d}_int8.onnx").exists()],
                    note="X_demo.npy is already scaled and clipped by src.scaler.Scaler")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {len(kept)} models -> {out}  ({total/1e6:.1f} MB total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--n", type=int, default=1000, help="test rows to ship (half of each class)")
    ap.add_argument("--method", default="supervised")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
