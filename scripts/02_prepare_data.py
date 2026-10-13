"""
Step 2 - clean the data, track every row change, and build the label-budget masks.

    python scripts/02_prepare_data.py

Reads  data/processed/{X,y}_{train,test}.npy + meta_train.csv   (made by step 1)
Writes data/processed/splits.npz        index arrays (no data copied)
       data/processed/pipeline_log.json every step + row counts -> feeds app page 1
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BUDGETS = [0.01, 0.05, 0.10, 0.25, 1.0]   # fraction of the labeled training rows we "keep" labels for


def row_hashes(X):
    return np.array([hashlib.md5(np.ascontiguousarray(r).tobytes()).hexdigest() for r in X])


def main(data_dir, seed, val_frac):
    d = Path(data_dir)
    X_tr = np.load(d / "X_train.npy", mmap_mode="r")
    y_tr = np.load(d / "y_train.npy")
    X_te = np.load(d / "X_test.npy", mmap_mode="r")
    y_te = np.load(d / "y_test.npy")
    meta = pd.read_csv(d / "meta_train.csv")
    assert len(meta) == len(y_tr) == len(X_tr), "meta/X/y row counts differ - re-run step 1 subsample"

    log = []

    def add(stage, rows_in, rows_out, note):
        log.append(dict(stage=stage, rows_in=int(rows_in), rows_out=int(rows_out),
                        removed=int(rows_in - rows_out), note=note))

    # 1. duplicate file hashes -------------------------------------------------
    idx = np.where(~meta["sha256"].duplicated().values)[0]
    add("Drop duplicate file hashes", len(y_tr), len(idx), "same sha256 = same file")

    # 2. duplicate feature vectors (different files, identical features) --------
    h_tr = row_hashes(X_tr[idx])
    first = ~pd.Series(h_tr).duplicated().values
    add("Drop duplicate feature vectors", len(idx), first.sum(), "identical 2381-dim vectors add no information")
    idx = idx[first]
    h_tr = h_tr[first]

    # 3. test rows that also appear in train (leakage) --------------------------
    h_te = row_hashes(X_te)
    test_keep = np.where(~np.isin(h_te, h_tr))[0]
    add("Drop test rows that also appear in train", len(y_te), len(test_keep), "prevents train/test leakage")

    # 4. separate the truly unlabeled pool (-1) ---------------------------------
    y = y_tr
    unl_true = idx[y[idx] == -1]
    lab = idx[y[idx] != -1]
    add("Set aside truly unlabeled pool (-1)", len(idx), len(lab),
        f"{len(unl_true)} rows have no label at all; kept for semi-supervised use")

    # 5. validation split (labeled only, stratified) ----------------------------
    lab_tr, lab_val = train_test_split(lab, test_size=val_frac, stratify=y[lab], random_state=seed)
    add("Hold out validation set", len(lab), len(lab_tr), f"{len(lab_val)} labeled rows reserved for tuning/thresholds")

    # 6. label budgets: hide labels, nested so 1% is inside 5% is inside 10% ... -
    rng = np.random.default_rng(seed)
    perms = {c: rng.permutation(lab_tr[y[lab_tr] == c]) for c in (0, 1)}
    masks, budget_log = {}, []
    for p in BUDGETS:
        keep = np.concatenate([perms[c][: int(round(p * len(perms[c])))] for c in (0, 1)])
        keep = np.sort(keep)
        masks[f"budget_{int(round(p * 100)):03d}"] = keep
        hidden = len(lab_tr) - len(keep)
        budget_log.append(dict(budget=p, labeled=int(len(keep)), hidden_labels=int(hidden),
                               truly_unlabeled=int(len(unl_true)),
                               total_unlabeled=int(hidden + len(unl_true))))
    add("Mask labels per budget", len(lab_tr), len(lab_tr), "see 'budgets': labels hidden, true values kept only for scoring pseudo-labels")

    np.savez(d / "splits.npz", idx_lab_train=lab_tr, idx_val=lab_val, idx_unlabeled_true=unl_true,
             idx_test_keep=test_keep, **masks)
    with open(d / "pipeline_log.json", "w") as f:
        json.dump(dict(seed=seed, steps=log, budgets=budget_log,
                       final=dict(train_labeled_pool=int(len(lab_tr)), val=int(len(lab_val)),
                                  unlabeled_true=int(len(unl_true)), test=int(len(test_keep)))), f, indent=2)

    print(f"{'Step':45s}{'in':>9s}{'out':>9s}{'removed':>9s}")
    for s in log:
        print(f"{s['stage']:45s}{s['rows_in']:9d}{s['rows_out']:9d}{s['removed']:9d}")
    print("\nLabel budgets (labeled / hidden / truly unlabeled):")
    for b in budget_log:
        print(f"  {b['budget']:>5.0%}: {b['labeled']:6d} / {b['hidden_labels']:6d} / {b['truly_unlabeled']:6d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    main(a.data_dir, a.seed, a.val_frac)
