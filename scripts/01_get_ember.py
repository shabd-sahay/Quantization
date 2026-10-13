"""
Step 1 - get EMBER2018 and make a laptop-sized working copy.

    python scripts/01_get_ember.py download     # ~1.7 GB, resumable, sha256-checked
    python scripts/01_get_ember.py vectorize    # raw JSONL -> 2381-dim float32 .dat files (slow, one-time)
    python scripts/01_get_ember.py subsample    # stratified sample -> data/processed/*.npy

Layout produced:
    data/ember2018/            raw + vectorized files (gitignored)
    data/processed/            small .npy files the rest of the project uses
"""
import argparse, hashlib, tarfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
SHA256 = "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812"
RAW = Path("data")
ARCHIVE = RAW / "ember_dataset_2018_2.tar.bz2"
EMBER_DIR = RAW / "ember2018"
OUT = RAW / "processed"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    done = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(URL, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:            # already complete
            print("Archive already complete.")
        else:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) + done
            mode = "ab" if done and r.status_code == 206 else "wb"
            if mode == "wb":
                done = 0
            with open(ARCHIVE, mode) as f, tqdm(total=total, initial=done, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))

    print("Verifying sha256 ...")
    h = hashlib.sha256()
    with open(ARCHIVE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise SystemExit("Checksum mismatch - delete the archive and re-run download.")
    print("OK. Extracting ...")
    with tarfile.open(ARCHIVE, "r:bz2") as t:
        t.extractall(RAW, filter="data")
    print("Extracted to", EMBER_DIR)


def vectorize():
    import ember  # pip install git+https://github.com/elastic/ember.git  (needs lief)
    ember.create_vectorized_features(str(EMBER_DIR))
    ember.create_metadata(str(EMBER_DIR))
    print("Vectorized.")


def subsample(n_train=100_000, n_test=50_000, seed=42):
    import ember
    X_tr, y_tr, X_te, y_te = ember.read_vectorized_features(str(EMBER_DIR))
    meta = ember.read_metadata(str(EMBER_DIR))
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Train label counts (-1 = unlabeled):", dict(zip(*np.unique(y_tr, return_counts=True))))

    def stratified(y, n):
        idx = []
        classes = np.unique(y)
        for c in classes:
            pool = np.where(y == c)[0]
            idx.append(rng.choice(pool, size=min(n // len(classes), len(pool)), replace=False))
        return np.sort(np.concatenate(idx))

    tr = stratified(y_tr, n_train)
    te = stratified(y_te, n_test)
    np.save(OUT / "X_train.npy", np.asarray(X_tr[tr], dtype=np.float32))
    np.save(OUT / "y_train.npy", np.asarray(y_tr[tr], dtype=np.int8))   # keeps -1 = unlabeled
    np.save(OUT / "X_test.npy", np.asarray(X_te[te], dtype=np.float32))
    np.save(OUT / "y_test.npy", np.asarray(y_te[te], dtype=np.int8))
    # keep sha256 + coarse time so we can dedupe and do time-aware analysis later
    meta.iloc[tr].reset_index(drop=True)[["sha256", "appeared", "subset", "label"]].to_csv(OUT / "meta_train.csv", index=False)
    print("Saved", len(tr), "train /", len(te), "test rows to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "vectorize", "subsample"])
    a = ap.parse_args()
    {"download": download, "vectorize": vectorize, "subsample": subsample}[a.step]()
