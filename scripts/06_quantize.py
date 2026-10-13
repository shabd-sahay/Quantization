"""
Step 6 - quantize the MLP and measure what it costs on CPU.

    python scripts/06_quantize.py --budget 100 --method supervised --seed 42
    python scripts/06_quantize.py --all          # every budget x method, seed 42

Variants compared (all CPU, thread count pinned so the comparison is fair):
  fp32_torch     the trained model as-is
  int8_dynamic   torch dynamic quantization, Linear layers -> int8
  fp32_onnx      ONNX Runtime, float32
  int8_onnx_dyn  ONNX Runtime dynamic quantization
  int8_onnx_sta  ONNX Runtime static quantization, calibrated on validation rows

Reports for each: file size, p50/p95 single-sample latency, batch throughput,
AUC, TPR@1%FPR, TPR@0.1%FPR, and how far its scores drift from fp32.
Writes results/quant_results.json
"""
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlp import MLP, Scaler


def metrics(scores, yt):
    fpr, tpr, _ = roc_curve(yt, scores)
    return dict(auc=float(roc_auc_score(yt, scores)),
                tpr_at_1pct_fpr=float(np.interp(0.01, fpr, tpr)),
                tpr_at_0p1pct_fpr=float(np.interp(0.001, fpr, tpr)))


def bench(run_one, run_batch, X, n_single, batch):
    run_one(X[:1]); run_batch(X[:batch])                       # warm up
    lat = []
    for i in range(n_single):
        x = X[i:i + 1]
        t0 = time.perf_counter(); run_one(x); lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    t0 = time.perf_counter()
    nb = 0
    for i in range(0, min(len(X), batch * 20), batch):
        run_batch(X[i:i + batch]); nb += len(X[i:i + batch])
    thru = nb / (time.perf_counter() - t0)
    return dict(p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
                p99_ms=float(np.percentile(lat, 99)), batch_rows_per_sec=float(thru))


def torch_runner(model):
    def f(x):
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x)))).numpy()
    return f


def ort_runner(sess):
    name = sess.get_inputs()[0].name
    def f(x):
        return 1.0 / (1.0 + np.exp(-sess.run(None, {name: np.ascontiguousarray(x, dtype=np.float32)})[0].ravel()))
    return f


def predict_all(fn, X, bs=4096):
    return np.concatenate([fn(X[i:i + bs]) for i in range(0, len(X), bs)])


def run(a, budget, method, seed, Xv, yv, Xt, yt, out_dir):
    mdir = Path(a.model_dir)
    ckpt = mdir / f"mlp_{budget:03d}_{method}_s{seed}.pt"
    if not ckpt.exists():
        print(f"  skip (no checkpoint): {ckpt.name}"); return None
    torch.set_num_threads(a.threads)

    model = MLP(Xt.shape[1])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    res, tag = {}, f"{budget:03d}_{method}_s{seed}"
    work = out_dir / tag
    work.mkdir(parents=True, exist_ok=True)

    # --- fp32 torch -----------------------------------------------------------
    f32 = work / "fp32.pt"
    torch.save(model.state_dict(), f32)
    run32 = torch_runner(model)
    base = predict_all(run32, Xt)
    res["fp32_torch"] = dict(size_kb=f32.stat().st_size / 1024, **metrics(base, yt),
                             **bench(run32, run32, Xt, a.n_single, a.batch))

    # --- torch dynamic int8 ---------------------------------------------------
    try:
        from torch.ao.quantization import per_channel_dynamic_qconfig
        qmodel = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear: per_channel_dynamic_qconfig}, dtype=torch.qint8)
        qmodel.eval()
        q8 = work / "int8_dynamic.pt"
        torch.save(qmodel.state_dict(), q8)
        runq = torch_runner(qmodel)
        s = predict_all(runq, Xt)
        res["int8_dynamic"] = dict(size_kb=q8.stat().st_size / 1024, **metrics(s, yt),
                                   **bench(runq, runq, Xt, a.n_single, a.batch),
                                   **drift(base, s, Xv, yv, run32, runq))
    except Exception as e:
        res["int8_dynamic"] = dict(error=str(e))

    # --- ONNX -----------------------------------------------------------------
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader

        onnx_f32 = work / "model_fp32.onnx"
        torch.onnx.export(model, torch.zeros(1, Xt.shape[1]), str(onnx_f32),
                          input_names=["x"], output_names=["logit"], opset_version=18,
                          dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}}, dynamo=False)

        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1

        def add(name, path):
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            r = ort_runner(sess)
            s = predict_all(r, Xt)
            d = dict(size_kb=path.stat().st_size / 1024, **metrics(s, yt),
                     **bench(r, r, Xt, a.n_single, a.batch))
            if name != "fp32_onnx":
                d.update(drift(base, s, Xv, yv, run32, r))
            res[name] = d

        add("fp32_onnx", onnx_f32)

        onnx_dyn = work / "model_int8_dynamic.onnx"
        quantize_dynamic(str(onnx_f32), str(onnx_dyn), weight_type=QuantType.QInt8)
        add("int8_onnx_dyn", onnx_dyn)

        class Reader(CalibrationDataReader):
            def __init__(self, X, n):
                self.data = iter([{"x": X[i:i + 1]} for i in range(min(n, len(X)))])
            def get_next(self):
                return next(self.data, None)

        onnx_sta = work / "model_int8_static.onnx"
        quantize_static(str(onnx_f32), str(onnx_sta), Reader(Xv, a.calib),
                        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                        activation_type=QuantType.QUInt8)
        add("int8_onnx_sta", onnx_sta)
    except Exception as e:
        res["onnx_error"] = str(e)

    if not a.keep_models:
        shutil.rmtree(work, ignore_errors=True)
    return res


def drift(base, s, Xv, yv, run32, runq):
    """How far this variant's scores move from fp32, and what that does at the operating point."""
    thr = float(np.quantile(predict_all(run32, Xv)[yv == 0], 0.99))
    return dict(score_mae_vs_fp32=float(np.abs(s - base).mean()),
                score_max_abs_diff=float(np.abs(s - base).max()),
                decision_flip_rate_at_fp32_thr=float(((s > thr) != (base > thr)).mean()))


def main(a):
    d = Path(a.data_dir)
    sc = Scaler.load(Path(a.model_dir) / "scaler.npz")
    X, y = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    Xte, yte = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
    s = np.load(d / "splits.npz")
    Xv, yv = sc.transform(X[s["idx_val"]]), y[s["idx_val"]]
    te = s["idx_test_keep"]
    Xt, yt = sc.transform(Xte[te]), yte[te]

    jobs = ([(b, m) for b in (1, 5, 10, 25, 100) for m in ("supervised", "consistency")]
            if a.all else [(a.budget, a.method)])
    out = dict(threads=a.threads, n_single=a.n_single, batch=a.batch, runs={})
    out_dir = Path(a.work_dir)
    for b, m in jobs:
        print(f"[{b:3d}% {m}]", flush=True)
        r = run(a, b, m, a.seed, Xv, yv, Xt, yt, out_dir)
        if r is None:
            continue
        out["runs"][f"{b:03d}_{m}_s{a.seed}"] = r
        for k, v in r.items():
            if isinstance(v, dict) and "size_kb" in v:
                print(f"   {k:14s} {v['size_kb']:8.0f} KB  p50 {v['p50_ms']:6.3f} ms  p95 {v['p95_ms']:6.3f} ms"
                      f"  {v['batch_rows_per_sec']:9.0f} rows/s  TPR@1%FPR {v['tpr_at_1pct_fpr']:.3f}"
                      + (f"  flips {v['decision_flip_rate_at_fp32_thr']*100:.2f}%" if "decision_flip_rate_at_fp32_thr" in v else ""))
            elif isinstance(v, dict) and "error" in v:
                print(f"   {k:14s} FAILED: {v['error'][:300]}")
            elif k.endswith("_error"):
                print(f"   {k:14s} FAILED: {str(v)[:300]}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
    print("Done ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--work-dir", default="models/quant")
    ap.add_argument("--out", default="results/quant_results.json")
    ap.add_argument("--budget", type=int, default=100, help="label budget as a percent: 1, 5, 10, 25 or 100")
    ap.add_argument("--method", default="supervised", choices=["supervised", "consistency"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="every budget and method")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads; pinned so variants are comparable")
    ap.add_argument("--n-single", type=int, default=500, help="single-sample latency measurements")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--calib", type=int, default=512, help="validation rows used to calibrate static quantization")
    ap.add_argument("--keep-models", action="store_true", help="keep the exported .onnx/.pt files")
    main(ap.parse_args())
