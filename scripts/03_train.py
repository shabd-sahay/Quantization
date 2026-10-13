"""
Step 3 - LightGBM under label scarcity: supervised vs self-training.

    python scripts/03_train.py                      # full run
    python scripts/03_train.py --rounds 60 --st-rounds 2   # quick pass to check everything works

For every label budget (1%, 5%, 10%, 25%, 100%):
  * supervised     - train only on the labels we kept
  * self-training  - predict the hidden/unlabeled rows, add the confident ones as pseudo-labels, retrain
Pseudo-label accuracy is measured against the true labels we hid in step 2.

Writes results/ssl_results.json (feeds app page 2) and models/lgbm_<budget>_<method>.txt
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=10, verbose=-1, num_threads=0)


def fit(X, y, Xv, yv, rounds, seed):
    ev = {}
    ds = lgb.Dataset(X, label=y)
    dv = lgb.Dataset(Xv, label=yv, reference=ds)
    m = lgb.train({**PARAMS, "seed": seed, "metric": ["binary_logloss", "auc"]}, ds,
                  num_boost_round=rounds, valid_sets=[dv], valid_names=["val"],
                  callbacks=[lgb.record_evaluation(ev)])
    return m, {k: [float(v) for v in vs] for k, vs in ev["val"].items()}


def tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target, fpr, tpr))


def evaluate(m, Xv, yv, Xt, yt):
    sv, st = m.predict(Xv), m.predict(Xt)
    out = dict(auc=float(roc_auc_score(yt, st)), pr_auc=float(average_precision_score(yt, st)),
               tpr_at_1pct_fpr=tpr_at_fpr(yt, st, 0.01), tpr_at_0p1pct_fpr=tpr_at_fpr(yt, st, 0.001))
    # deployment-style: pick the threshold on validation benign scores, then see what happens on test
    for name, f in (("1pct", 0.01), ("0p1pct", 0.001)):
        thr = float(np.quantile(sv[yv == 0], 1 - f))
        out[f"valthr_{name}_tpr"] = float((st[yt == 1] > thr).mean())
        out[f"valthr_{name}_fpr"] = float((st[yt == 0] > thr).mean())
    return out


def main(a):
    d, mdir = Path(a.data_dir), Path(a.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    X, y = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    Xte_all, yte_all = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
    s = np.load(d / "splits.npz")
    Xv, yv = X[s["idx_val"]], y[s["idx_val"]]
    te = s["idx_test_keep"]
    Xt, yt = Xte_all[te], yte_all[te]

    results = dict(seed=a.seed, boost_rounds=a.rounds, confidence=a.conf, budgets={})
    for key in sorted(k for k in s.files if k.startswith("budget_")):
        lab = s[key]
        hidden = np.setdiff1d(s["idx_lab_train"], lab)
        pool = np.concatenate([hidden, s["idx_unlabeled_true"]])
        t0 = time.time()
        m, curve = fit(X[lab], y[lab], Xv, yv, a.rounds, a.seed)
        m.save_model(str(mdir / f"lgbm_{key}_supervised.txt"))
        res = dict(n_labeled=int(len(lab)), n_pool=int(len(pool)),
                   supervised=dict(**evaluate(m, Xv, yv, Xt, yt), curve=curve, seconds=round(time.time() - t0, 1)),
                   self_training=[])
        print(f"[{key}] supervised: labeled={len(lab)}  TPR@1%FPR={res['supervised']['tpr_at_1pct_fpr']:.3f}")

        cur = m
        for r in range(1, a.st_rounds + 1):
            t0 = time.time()
            p = cur.predict(X[pool])
            conf = np.maximum(p, 1 - p) >= a.conf
            pseudo = (p >= 0.5).astype(np.int8)
            known = (y[pool] >= 0) & conf                     # confident rows whose true label we hid
            acc = float((pseudo[known] == y[pool][known]).mean()) if known.any() else None
            Xtr = np.vstack([X[lab], X[pool[conf]]])
            ytr = np.concatenate([y[lab], pseudo[conf]])
            cur, curve = fit(Xtr, ytr, Xv, yv, a.rounds, a.seed)
            res["self_training"].append(dict(round=r, added=int(conf.sum()), pool=int(len(pool)),
                                             added_malicious=int(pseudo[conf].sum()), pseudo_label_acc=acc,
                                             pseudo_acc_checked_on=int(known.sum()),
                                             **evaluate(cur, Xv, yv, Xt, yt), curve=curve,
                                             seconds=round(time.time() - t0, 1)))
            print(f"[{key}] self-train r{r}: added={int(conf.sum())}  pseudo-acc={acc}  "
                  f"TPR@1%FPR={res['self_training'][-1]['tpr_at_1pct_fpr']:.3f}")
        if a.st_rounds:
            cur.save_model(str(mdir / f"lgbm_{key}_selftrain.txt"))
        results["budgets"][key] = res
        with open(a.out, "w") as f:                            # save after each budget so nothing is lost
            json.dump(results, f, indent=2)
    print("Done ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="results/ssl_results.json")
    ap.add_argument("--rounds", type=int, default=200, help="LightGBM boosting rounds")
    ap.add_argument("--st-rounds", type=int, default=3, help="self-training rounds")
    ap.add_argument("--conf", type=float, default=0.95, help="min confidence to accept a pseudo-label")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
