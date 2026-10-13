import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Quantization", layout="wide")
st.title("3 · Quantization: what INT8 costs and buys")

qp = ROOT / "results/quant_results.json"
if not qp.exists():
    st.error("Missing results/quant_results.json — run scripts/06_quantize.py"); st.stop()
q = json.loads(qp.read_text())

runs = list(q["runs"])
default = "100_supervised_s42"
pick = st.selectbox("Model", runs, index=runs.index(default) if default in runs else 0,
                    format_func=lambda k: f"{int(k.split('_')[0])}% labels · {k.split('_')[1]} · seed {k.split('_')[2][1:]}")
r = q["runs"][pick]
variants = {k: v for k, v in r.items() if isinstance(v, dict) and "size_kb" in v}
if not variants:
    st.error("No variant succeeded for this model."); st.stop()
base = variants.get("fp32_torch")

LABEL = {"fp32_torch": "PyTorch FP32", "int8_dynamic": "PyTorch INT8 (dynamic)",
         "fp32_onnx": "ONNX FP32", "int8_onnx_dyn": "ONNX INT8 (dynamic)",
         "int8_onnx_sta": "ONNX INT8 (static)"}

tbl = pd.DataFrame([{
    "variant": LABEL.get(k, k),
    "size (MB)": round(v["size_kb"] / 1024, 2),
    "size vs FP32": f"{base['size_kb'] / v['size_kb']:.2f}x smaller" if base else "",
    "p50 (ms)": round(v["p50_ms"], 3), "p95 (ms)": round(v["p95_ms"], 3),
    "speedup": f"{base['p50_ms'] / v['p50_ms']:.2f}x" if base else "",
    "rows/s": f"{v['batch_rows_per_sec']:,.0f}",
    "TPR@1%FPR": round(v["tpr_at_1pct_fpr"], 3),
    "TPR@0.1%FPR": round(v["tpr_at_0p1pct_fpr"], 3),
    "AUC": round(v["auc"], 4),
    "decision flips": (f"{v['decision_flip_rate_at_fp32_thr']*100:.2f}%"
                       if "decision_flip_rate_at_fp32_thr" in v else "—"),
} for k, v in variants.items()])
st.dataframe(tbl, hide_index=True, width='stretch')

# ---- INT8 is free at every label budget -----------------------------------------
st.subheader("Does INT8 stay free as labels get scarce?")
rows = []
for key, run in q["runs"].items():
    b, meth = int(key.split("_")[0]), key.split("_")[1]
    f, i8 = run.get("fp32_torch"), run.get("int8_onnx_dyn")
    if f and i8:
        rows.append(dict(budget=b, method=meth, fp32=f["tpr_at_1pct_fpr"],
                         int8=i8["tpr_at_1pct_fpr"],
                         delta=i8["tpr_at_1pct_fpr"] - f["tpr_at_1pct_fpr"],
                         flips=i8.get("decision_flip_rate_at_fp32_thr", np.nan) * 100))
if rows:
    d = pd.DataFrame(rows).sort_values(["budget", "method"])
    c1, c2 = st.columns([3, 2])
    f = go.Figure()
    for meth, g in d.groupby("method"):
        f.add_scatter(x=g["budget"], y=g["fp32"], name=f"FP32 ({meth})", mode="lines+markers")
        f.add_scatter(x=g["budget"], y=g["int8"], name=f"INT8 ({meth})", mode="markers",
                      marker=dict(symbol="x", size=11))
    f.update_layout(height=360, margin=dict(t=10, b=10), xaxis_type="log",
                    xaxis_title="share of labels kept (%, log scale)",
                    yaxis_title="TPR at 1% FPR")
    c1.plotly_chart(f, width='stretch')
    c2.metric("Largest detection change, any config", f"{d['delta'].abs().max():+.3f}")
    c2.metric("Largest verdict flip rate", f"{d['flips'].max():.2f}%")
    c2.caption("Crosses sit on the lines at every budget: INT8 tracks FP32 across a 100x "
               "range of label budgets, not just at the one configuration people usually report.")
    with st.expander("Per-configuration table"):
        st.dataframe(d.round({"fp32": 3, "int8": 3, "delta": 3, "flips": 2}),
                     hide_index=True, width='stretch')

# ---- trade-off plots -------------------------------------------------------------
st.subheader("Size, speed, detection")
c1, c2 = st.columns(2)
f1 = go.Figure()
for k, v in variants.items():
    f1.add_scatter(x=[v["p50_ms"]], y=[v["tpr_at_1pct_fpr"]], mode="markers+text",
                   marker=dict(size=10 + 26 * v["size_kb"] / max(x["size_kb"] for x in variants.values())),
                   text=[LABEL.get(k, k)], textposition="top center", name=LABEL.get(k, k),
                   hovertemplate=f"{LABEL.get(k,k)}<br>p50 %{{x:.3f}} ms<br>TPR %{{y:.3f}}"
                                 f"<br>{v['size_kb']/1024:.2f} MB<extra></extra>")
f1.update_layout(height=400, margin=dict(t=10, b=10), showlegend=False,
                 xaxis_title="p50 single-sample latency (ms) — lower is better",
                 yaxis_title="TPR at 1% FPR")
c1.plotly_chart(f1, width='stretch')
c1.caption("Marker area ∝ model file size. Up and to the left is better.")

f2 = go.Figure()
ks = list(variants)
f2.add_bar(x=[LABEL.get(k, k) for k in ks], y=[variants[k]["batch_rows_per_sec"] for k in ks],
           text=[f"{variants[k]['batch_rows_per_sec']:,.0f}" for k in ks], textposition="outside")
f2.update_layout(height=400, margin=dict(t=10, b=10), yaxis_title="batch throughput (rows/s)",
                 xaxis_tickangle=-20)
c2.plotly_chart(f2, width='stretch')
c2.caption(f"{q['threads']} thread(s) pinned, batch size {q['batch']}.")

st.warning("""
**Read the latency numbers as medians, not as one run.** All ten models share one architecture,
so their latency should be identical. Within this sweep `fp32_torch` p50 spans 0.107-0.136 ms;
an earlier sweep of the same binaries on a loaded laptop measured 0.325 ms, 3x slower. Background
load moves these numbers more than the models do. The table reports **medians over ten runs** -
**0.109 ms FP32 vs 0.030 ms ONNX INT8, 3.6x** - and absolute latency is only meaningful against the
FP32 baseline measured in the same sweep.
""")

if base and "fp32_onnx" in variants and "int8_onnx_dyn" in variants:
    o32, o8 = variants["fp32_onnx"], variants["int8_onnx_dyn"]
    st.subheader("Where the speedup actually comes from")
    a, b, c = st.columns(3)
    a.metric("Runtime change (PyTorch → ONNX, both FP32)", f"{base['p50_ms']/o32['p50_ms']:.2f}x")
    b.metric("Quantization (ONNX FP32 → ONNX INT8)", f"{o32['p50_ms']/o8['p50_ms']:.2f}x")
    c.metric("Combined", f"{base['p50_ms']/o8['p50_ms']:.2f}x")
    st.caption("Reporting only the combined figure would credit the runtime switch to INT8. "
               "Both factors are listed separately for that reason.")

st.info("""
**The failure that produced this section.** The first INT8 run dropped detection from 0.789 to
**0.178**, flipping 24.7% of verdicts. Per-channel weight quantization did not help. The cause was
the *input*: after log1p + standardization EMBER's features still reach ±992 sigma on the test
set, because the scaler's statistics come from training rows. Dynamic quantization scales the
activation tensor per-tensor, so those outliers set a scale that collapses everything else to
zero — and the first layer takes all 2,381 of them. Clipping the scaled features to ±5 touches
0.08% of values, leaves FP32 detection unchanged at 0.789, and restores INT8 to 0.787.

Diagnostic that pinned it: quantizing every layer *except* the first recovered detection fully
(0.791), which located the damage before any fix was attempted.
""")

# ---- live scoring: ONNX only, so this runs on Streamlit Cloud ---------------------
st.subheader("Score real samples, live")
DEMO = ROOT / "demo"
budget = int(pick.split("_")[0])
f32_p = DEMO / "models" / f"{budget:03d}_fp32.onnx"
i8_p = DEMO / "models" / f"{budget:03d}_int8.onnx"

if not (DEMO / "X_demo.npy").exists():
    st.info("Demo bundle not found — run `python scripts/07_make_demo_bundle.py`.")
elif not (f32_p.exists() and i8_p.exists()):
    st.info(f"No bundled models for this configuration. The bundle ships the *supervised* "
            f"models; pick one of those, or rebuild the bundle with `--method`.")
else:
    st.caption("Pre-extracted EMBER test vectors, already scaled — no PE parsing, so no LIEF "
               "dependency. Both models run on ONNX Runtime, so this compares quantization "
               "alone rather than quantization plus a runtime change.")

    @st.cache_resource
    def sessions(a, b):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        return (ort.InferenceSession(a, so, providers=["CPUExecutionProvider"]),
                ort.InferenceSession(b, so, providers=["CPUExecutionProvider"]))

    @st.cache_data
    def demo_data():
        return np.load(DEMO / "X_demo.npy"), np.load(DEMO / "y_demo.npy")

    s32, s8 = sessions(str(f32_p), str(i8_p))
    X, y = demo_data()
    n = st.slider("How many samples to score", 1, min(256, len(X)), 32)

    if st.button("Score", type="primary"):
        rng = np.random.default_rng()
        idx = rng.choice(len(X), n, replace=False)
        Xs, yt = np.ascontiguousarray(X[idx]), y[idx]

        def run(sess, x):
            name = sess.get_inputs()[0].name
            sess.run(None, {name: x[:1]})                      # warm up
            t0 = time.perf_counter()
            out = sess.run(None, {name: x})[0].ravel()
            return 1 / (1 + np.exp(-out)), (time.perf_counter() - t0) * 1000

        p32, t32 = run(s32, Xs)
        p8, t8 = run(s8, Xs)

        a, b, c = st.columns(3)
        a.metric("ONNX FP32", f"{t32:.2f} ms", f"{n/(t32/1000):,.0f} rows/s")
        b.metric("ONNX INT8", f"{t8:.2f} ms", f"{n/(t8/1000):,.0f} rows/s")
        c.metric("Faster by", f"{t32/t8:.2f}x" if t8 else "—",
                 f"{((p8 > 0.5) == (p32 > 0.5)).mean()*100:.1f}% verdicts agree")
        st.caption("A single small batch is a noisy timing sample — the table at the top of "
                   "this page is the measured benchmark.")

        df = pd.DataFrame({
            "true label": np.where(yt == 1, "malicious", "benign"),
            "FP32 score": p32.round(4), "INT8 score": p8.round(4),
            "FP32 verdict": np.where(p32 > 0.5, "malicious", "benign"),
            "INT8 verdict": np.where(p8 > 0.5, "malicious", "benign"),
            "|diff|": np.abs(p32 - p8).round(5),
        })
        st.dataframe(df, hide_index=True, width='stretch', height=320)
        wrong = (df["true label"] != df["FP32 verdict"]).sum()
        st.caption(f"{wrong} of {n} disagree with the true label at a 0.5 cut — and INT8 "
                   "inherits those errors rather than adding its own, which is the point. "
                   "A deployed scanner would use the validation-calibrated 1%-FPR threshold "
                   "instead of 0.5; see the note on the home page about that threshold "
                   "drifting on a later-in-time test set.")
