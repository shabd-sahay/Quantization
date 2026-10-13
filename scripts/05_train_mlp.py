"""
Step 5 - MLP under label scarcity: supervised vs consistency regularization (FixMatch-style, no
confidence gate on the loss weight - the unlabeled term just asks for stable predictions under noise).

    python scripts/05_train_mlp.py --budgets 0.01 0.05 --epochs 15     # quick check
    python scripts/05_train_mlp.py --seeds 42 43 44                    # full run

Writes results/mlp_results.json, models/mlp_<budget>_<method>_s<seed>.pt, models/scaler.npz
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlp import MLP, Scaler, predict


def val_tpr(model, Xv, yv, dev):
    """TPR at 1% FPR measured on validation only - used for epoch selection."""
    sv = predict(model, Xv, dev)
    fpr, tpr, _ = roc_curve(yv, sv)
    return float(np.interp(0.01, fpr, tpr)), float(roc_auc_score(yv, sv))


def evaluate(model, Xv, yv, Xt, yt, dev):
    sv, st = predict(model, Xv, dev), predict(model, Xt, dev)
    fpr, tpr, _ = roc_curve(yt, st)
    thr = float(np.quantile(sv[yv == 0], 0.99))
    return dict(auc=float(roc_auc_score(yt, st)), tpr_at_1pct_fpr=float(np.interp(0.01, fpr, tpr)),
                tpr_at_0p1pct_fpr=float(np.interp(0.001, fpr, tpr)),
                valthr_1pct_fpr=float((st[yt == 0] > thr).mean()))


def budget_idx(lab_tr, y, p, seed):
    rng = np.random.default_rng(seed)
    perms = {c: rng.permutation(lab_tr[y[lab_tr] == c]) for c in (0, 1)}
    return np.sort(np.concatenate([perms[c][: int(round(p * len(perms[c])))] for c in (0, 1)]))


def train(Xl, yl, Xu, Xv, yv, Xt, yt, a, seed, dev, use_consistency):
    torch.manual_seed(seed)
    model = MLP(Xl.shape[1], p_drop=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    steps = max(1, len(Xl) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
    Xl_t = torch.as_tensor(Xl, device=dev)
    yl_t = torch.as_tensor(yl.astype(np.float32), device=dev)
    Xu_t = torch.as_tensor(Xu, device=dev) if use_consistency and len(Xu) else None
    g = torch.Generator(device="cpu").manual_seed(seed)
    curve, best = [], (-1.0, 0, None)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xl_t), generator=g).to(dev)
        tot_s = tot_u = 0.0
        ramp = min(1.0, (ep + 1) / max(1, a.rampup))
        for i in range(steps):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            if len(idx) < 2:
                continue
            loss_s = bce(model(Xl_t[idx]), yl_t[idx])
            loss = loss_s
            loss_u = torch.tensor(0.0)
            if Xu_t is not None:
                ui = torch.randint(0, len(Xu_t), (a.batch_u,), generator=g).to(dev)
                xu = Xu_t[ui]
                with torch.no_grad():                       # weak view -> target
                    model.eval()
                    tgt = torch.sigmoid(model(xu + a.noise_w * torch.randn_like(xu)))
                    model.train()
                strong = xu + a.noise_s * torch.randn_like(xu)
                strong = strong * (torch.rand_like(strong) > a.mask_p)      # feature dropout
                loss_u = ((torch.sigmoid(model(strong)) - tgt) ** 2).mean()
                loss = loss + a.lambda_u * ramp * loss_u
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot_s += float(loss_s.detach()); tot_u += float(loss_u.detach())
        vt, va = val_tpr(model, Xv, yv, dev)
        curve.append(dict(epoch=ep + 1, sup_loss=tot_s / steps, unsup_loss=tot_u / steps,
                          val_auc=va, val_tpr_at_1pct_fpr=vt))
        if vt > best[0]:                       # model selection on val TPR@1%FPR, not final epoch
            best = (vt, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"    ep{ep+1:02d} sup={curve[-1]['sup_loss']:.4f} unsup={curve[-1]['unsup_loss']:.4f} "
              f"val_auc={va:.4f} val_tpr={vt:.3f}", flush=True)
    final = dict(**evaluate(model, Xv, yv, Xt, yt, dev))
    model.load_state_dict(best[2])
    return model, dict(**evaluate(model, Xv, yv, Xt, yt, dev), curve=curve,
                       best_epoch=best[1], last_epoch_tpr_at_1pct_fpr=final["tpr_at_1pct_fpr"],
                       last_epoch_auc=final["auc"])


def main(a):
    d, mdir = Path(a.data_dir), Path(a.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    X, y = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    Xte, yte = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
    s = np.load(d / "splits.npz")
    lab_tr, unl, te = s["idx_lab_train"], s["idx_unlabeled_true"], s["idx_test_keep"]

    sc = Scaler().fit(X[lab_tr])            # fit on training rows only
    sc.save(mdir / "scaler.npz")
    Xs, Xv = sc.transform(X), sc.transform(X[s["idx_val"]])
    yv, Xt, yt = y[s["idx_val"]], sc.transform(Xte[te]), yte[te]

    out = dict(seeds=a.seeds, epochs=a.epochs, lambda_u=a.lambda_u, runs=[])
    for seed in a.seeds:
        for p in a.budgets:
            lab = budget_idx(lab_tr, y, p, seed)
            pool = np.concatenate([np.setdiff1d(lab_tr, lab), unl])
            row = dict(seed=seed, budget=p, n_labeled=int(len(lab)), n_pool=int(len(pool)))
            for name, cons in (("supervised", False), ("consistency", True)):
                print(f"  seed {seed} budget {p:.0%} {name} (labeled={len(lab)})", flush=True)
                t0 = time.time()
                m, r = train(Xs[lab], y[lab], Xs[pool], Xv, yv, Xt, yt, a, seed, dev, cons)
                r["seconds"] = round(time.time() - t0, 1)
                row[name] = r
                torch.save(m.state_dict(), mdir / f"mlp_{int(p*100):03d}_{name}_s{seed}.pt")
                print(f"    -> best ep{r['best_epoch']}: TPR@1%FPR={r['tpr_at_1pct_fpr']:.3f} auc={r['auc']:.4f}"
                      f"  | last ep: TPR={r['last_epoch_tpr_at_1pct_fpr']:.3f} ({r['seconds']}s)")
            out["runs"].append(row)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(a.out, "w"), indent=2)

    print(f"\nTPR @ 1% FPR (test) at the val-selected epoch, mean +/- std over {len(a.seeds)} seed(s)")
    print(f"{'budget':>7s}{'supervised':>24s}{'consistency':>26s}{'sup(last)':>14s}{'cons(last)':>14s}")
    for p in a.budgets:
        rs = [r for r in out["runs"] if r["budget"] == p]
        b = np.array([r["supervised"]["tpr_at_1pct_fpr"] for r in rs])
        c = np.array([r["consistency"]["tpr_at_1pct_fpr"] for r in rs])
        bl = np.array([r["supervised"]["last_epoch_tpr_at_1pct_fpr"] for r in rs])
        cl = np.array([r["consistency"]["last_epoch_tpr_at_1pct_fpr"] for r in rs])
        print(f"{p:>7.0%}{f'{b.mean():.3f}+/-{b.std():.3f}':>24s}"
              f"{f'{c.mean():.3f}+/-{c.std():.3f} ({(c-b).mean():+.3f})':>26s}"
              f"{f'{bl.mean():.3f}':>14s}{f'{cl.mean():.3f}':>14s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="results/mlp_results.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batch-u", type=int, default=512, help="unlabeled rows per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lambda-u", type=float, default=1.0, help="weight on the consistency loss")
    ap.add_argument("--rampup", type=int, default=5, help="epochs to ramp lambda_u from 0 to full")
    ap.add_argument("--noise-w", type=float, default=0.05, help="weak-view gaussian noise")
    ap.add_argument("--noise-s", type=float, default=0.25, help="strong-view gaussian noise")
    ap.add_argument("--mask-p", type=float, default=0.15, help="strong-view feature dropout prob")
    ap.add_argument("--device", default=None)
    main(ap.parse_args())
