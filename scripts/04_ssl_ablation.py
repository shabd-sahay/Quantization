"""
Step 4 - why did self-training hurt, and can we fix it?  (multi-seed ablation)

    python scripts/04_ssl_ablation.py --seeds 42 43 44
    python scripts/04_ssl_ablation.py --seeds 42 --rounds 60 --st-rounds 2     # quick check

Compares, at label budgets 1/5/10/25%:
  supervised      - labeled rows only
  thr0.99/thr0.999- self-training with a much stricter confidence cut-off
  topk10          - curriculum: each round add only the top 10%/20%/30% most confident rows per class
  weighted0.2     - thr 0.95 as before, but pseudo-labels count only 20% as much as real labels
Each seed re-draws which labels are kept (seed 42 reproduces step 2) and reseeds LightGBM.
Writes results/ssl_ablation.json and prints mean +/- std over seeds.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=10, verbose=-1, num_threads=0)
BUDGETS = [0.01, 0.05, 0.10, 0.25]
VARIANTS = {
    "thr0.99":     dict(mode="thr", thr=0.99, w=1.0),
    "thr0.999":    dict(mode="thr", thr=0.999, w=1.0),
    "topk10":      dict(mode="topk", frac=0.10, w=1.0),
    "weighted0.2": dict(mode="thr", thr=0.95, w=0.2),
}


def fit(X, y, w, rounds, seed):
    return lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X, label=y, weight=w), num_boost_round=rounds)


def evaluate(m, Xv, yv, Xt, yt):
    sv, st = m.predict(Xv), m.predict(Xt)
    fpr, tpr, _ = roc_curve(yt, st)
    thr = float(np.quantile(sv[yv == 0], 0.99))
    return dict(auc=float(roc_auc_score(yt, st)), tpr_at_1pct_fpr=float(np.interp(0.01, fpr, tpr)),
                tpr_at_0p1pct_fpr=float(np.interp(0.001, fpr, tpr)),
                valthr_1pct_fpr=float((st[yt == 0] > thr).mean()))


def budget_idx(lab_tr, y, p, seed):
    rng = np.random.default_rng(seed)
    perms = {c: rng.permutation(lab_tr[y[lab_tr] == c]) for c in (0, 1)}
    return np.sort(np.concatenate([perms[c][: int(round(p * len(perms[c])))] for c in (0, 1)]))


def self_train(X, y, lab, pool, cfg, ev, a, seed):
    m = fit(X[lab], y[lab], None, a.rounds, seed)
    log = []
    for r in range(1, a.st_rounds + 1):
        p = m.predict(X[pool])
        pred, conf = (p >= 0.5).astype(np.int8), np.maximum(p, 1 - p)
        if cfg["mode"] == "thr":
            sel = np.where(conf >= cfg["thr"])[0]
        else:
            k = int(cfg["frac"] * r * len(pool) / 2)                  # per predicted class
            sel = np.concatenate([np.where(pred == c)[0][np.argsort(-conf[pred == c])][:k] for c in (0, 1)])
        if len(sel) == 0:
            break
        known = y[pool[sel]] >= 0
        acc = float((pred[sel][known] == y[pool[sel]][known]).mean()) if known.any() else None
        w = np.concatenate([np.ones(len(lab)), np.full(len(sel), cfg["w"])])
        m = fit(np.vstack([X[lab], X[pool[sel]]]), np.concatenate([y[lab], pred[sel]]), w, a.rounds, seed)
        log.append(dict(round=r, added=int(len(sel)), pseudo_label_acc=acc))
    return dict(**evaluate(m, *ev), rounds=log)


def main(a):
    d = Path(a.data_dir)
    X, y = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    Xte, yte = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
    s = np.load(d / "splits.npz")
    te = s["idx_test_keep"]
    ev = (X[s["idx_val"]], y[s["idx_val"]], Xte[te], yte[te])
    lab_tr, unl = s["idx_lab_train"], s["idx_unlabeled_true"]

    out = dict(seeds=a.seeds, boost_rounds=a.rounds, st_rounds=a.st_rounds, runs=[])
    for seed in a.seeds:
        for p in BUDGETS:
            lab = budget_idx(lab_tr, y, p, seed)
            pool = np.concatenate([np.setdiff1d(lab_tr, lab), unl])
            t0 = time.time()
            row = dict(seed=seed, budget=p, n_labeled=int(len(lab)),
                       supervised=evaluate(fit(X[lab], y[lab], None, a.rounds, seed), *ev))
            for name, cfg in VARIANTS.items():
                row[name] = self_train(X, y, lab, pool, cfg, ev, a, seed)
            out["runs"].append(row)
            print(f"seed {seed}  budget {p:.0%}  done in {time.time()-t0:.0f}s")
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(a.out, "w"), indent=2)

    methods = ["supervised"] + list(VARIANTS)
    print(f"\nTPR @ 1% FPR (test), mean +/- std over {len(a.seeds)} seed(s); delta = paired difference vs supervised")
    print(f"{'budget':>7s}" + "".join(f"{m:>27s}" for m in methods))
    for p in BUDGETS:
        rs = [r for r in out["runs"] if r["budget"] == p]
        base = np.array([r["supervised"]["tpr_at_1pct_fpr"] for r in rs])
        cells = []
        for m in methods:
            v = np.array([r[m]["tpr_at_1pct_fpr"] for r in rs])
            cells.append(f"{v.mean():.3f}+/-{v.std():.3f}" + ("" if m == "supervised" else f" ({(v-base).mean():+.3f})"))
        print(f"{p:>7.0%}" + "".join(f"{c:>27s}" for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="results/ssl_ablation.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--st-rounds", type=int, default=3)
    main(ap.parse_args())
