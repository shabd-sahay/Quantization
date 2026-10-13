import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Training & detection", layout="wide")
st.title("2 · Training under label scarcity")

ssl_p, mlp_p, abl_p = (ROOT / "results/ssl_results.json", ROOT / "results/mlp_results.json",
                       ROOT / "results/ssl_ablation.json")
if not ssl_p.exists() or not mlp_p.exists():
    st.error("Missing results — run scripts/03_train.py and scripts/05_train_mlp.py"); st.stop()
ssl, mlp = json.loads(ssl_p.read_text()), json.loads(mlp_p.read_text())

# ---------- the headline curve -------------------------------------------------
st.subheader("Detection vs. label budget")
rows = []
for k, v in ssl["budgets"].items():
    rows.append(dict(budget=int(k.split("_")[1]) / 100, model="LightGBM (supervised)",
                     tpr=v["supervised"]["tpr_at_1pct_fpr"], n=v["n_labeled"], sd=np.nan))
for p in sorted({r["budget"] for r in mlp["runs"]}):
    for meth, label in (("supervised", "MLP (supervised)"), ("consistency", "MLP (+consistency)")):
        vals = [r[meth]["tpr_at_1pct_fpr"] for r in mlp["runs"] if r["budget"] == p and meth in r]
        if vals:
            rows.append(dict(budget=p, model=label, tpr=float(np.mean(vals)),
                             n=[r for r in mlp["runs"] if r["budget"] == p][0]["n_labeled"],
                             sd=float(np.std(vals))))
df = pd.DataFrame(rows)

fig = go.Figure()
for m, g in df.groupby("model"):
    g = g.sort_values("budget")
    fig.add_scatter(x=g["budget"] * 100, y=g["tpr"], name=m, mode="lines+markers",
                    error_y=dict(type="data", array=g["sd"].fillna(0), visible=True),
                    customdata=g["n"],
                    hovertemplate="%{x:.0f}%% of labels (%{customdata:,} rows)<br>TPR %{y:.3f}<extra>%{fullData.name}</extra>")
fig.update_layout(height=430, margin=dict(t=10, b=10), xaxis_type="log",
                  xaxis_title="share of labels kept (%, log scale)",
                  yaxis_title="TPR at 1% FPR (test)", legend=dict(y=0.02, x=0.98, xanchor="right"))
st.plotly_chart(fig, width='stretch')
st.caption("MLP points are the mean over 3 seeds, error bars ±1 sd; the epoch is chosen on "
           "validation TPR. LightGBM is a single seed.")

c1, c2 = st.columns(2)
c1.success("""
**Gradient boosting wins on tabular features.** LightGBM reaches 0.865 at full labels against
the MLP's 0.776, and holds a lead at every budget. The MLP earns its place by being the model
that can actually be quantized to INT8 — see page 3.
""")
c2.warning("""
**566 labels already get you AUC 0.936.** Detection at a strict false-positive budget is what
degrades: 0.865 → 0.500 as labels fall from 100% to 1%. AUC barely moves, which is exactly why
AUC is the wrong headline metric for a detector.
""")

# ---------- what the unlabeled pool bought: nothing -----------------------------
st.subheader("Can the unlabeled pool make up the gap?")
st.markdown("""
Two method families, eight configurations, three seeds. **None beat supervised training.**
""")
tab1, tab2 = st.tabs(["Self-training (LightGBM)", "Consistency (MLP)"])

with tab1:
    if abl_p.exists():
        abl = json.loads(abl_p.read_text())
        meths = ["supervised", "thr0.99", "thr0.999", "topk10", "weighted0.2"]
        out = []
        for p in sorted({r["budget"] for r in abl["runs"]}):
            rs = [r for r in abl["runs"] if r["budget"] == p]
            base = np.array([r["supervised"]["tpr_at_1pct_fpr"] for r in rs])
            row = {"budget": f"{p:.0%}"}
            for m in meths:
                v = np.array([r[m]["tpr_at_1pct_fpr"] for r in rs])
                row[m] = f"{v.mean():.3f}" + ("" if m == "supervised" else f" ({(v - base).mean():+.3f})")
            out.append(row)
        st.dataframe(pd.DataFrame(out), hide_index=True, width='stretch')
        st.markdown("""
**Why it fails.** The pseudo-labels are not the problem — at the top-10% confidence cut they are
**99.8–100% correct**, and detection still drops. The rows a model is most confident about are
the easy, typical ones; adding tens of thousands of them pulls training away from the hard cases
near the decision boundary, which is precisely the region a 1%-FPR threshold lives in. At the
0.95 cut the model also admits ~83% of the pool after one round, and its own errors then
reinforce across rounds: pseudo-label accuracy decays 92% → 88% at the 1% budget.
""")
    else:
        st.info("Run scripts/04_ssl_ablation.py for this table.")

with tab2:
    rows2 = []
    for p in sorted({r["budget"] for r in mlp["runs"]}):
        rs = [r for r in mlp["runs"] if r["budget"] == p]
        b = np.array([r["supervised"]["tpr_at_1pct_fpr"] for r in rs])
        c = np.array([r["consistency"]["tpr_at_1pct_fpr"] for r in rs])
        bl = np.array([r["supervised"].get("last_epoch_tpr_at_1pct_fpr", np.nan) for r in rs])
        cl = np.array([r["consistency"].get("last_epoch_tpr_at_1pct_fpr", np.nan) for r in rs])
        rows2.append({"budget": f"{p:.0%}", "supervised": f"{b.mean():.3f} ±{b.std():.3f}",
                      "consistency": f"{c.mean():.3f} ±{c.std():.3f} ({(c - b).mean():+.3f})",
                      "sup (last epoch)": f"{np.nanmean(bl):.3f}",
                      "cons (last epoch)": f"{np.nanmean(cl):.3f}"})
    st.dataframe(pd.DataFrame(rows2), hide_index=True, width='stretch')
    st.markdown("""
**A measurement trap worth showing.** Compare the first two columns with the last two. Trained
for a fixed 30 epochs and evaluated at the final epoch, consistency regularization looks like a
large win (0.257 vs 0.166 at the 5% budget). Select the epoch on validation instead and the win
disappears — consistency was regularizing away overfitting that proper model selection removes
anyway. The first version of this experiment reported the inflated number; the fix changed the
conclusion, not just the decimals.
""")

# ---------- training curves ----------------------------------------------------
st.subheader("Training behaviour")
budgets = sorted({r["budget"] for r in mlp["runs"]})
c1, c2, c3 = st.columns(3)
bud = c1.selectbox("Label budget", budgets, index=len(budgets) - 1, format_func=lambda p: f"{p:.0%}")
meth = c2.selectbox("Method", ["supervised", "consistency"])
seed = c3.selectbox("Seed", sorted({r["seed"] for r in mlp["runs"]}))
run = next((r for r in mlp["runs"] if r["budget"] == bud and r["seed"] == seed), None)
if run and meth in run:
    cur = pd.DataFrame(run[meth]["curve"])
    f = go.Figure()
    f.add_scatter(x=cur["epoch"], y=cur["val_auc"], name="val AUC", yaxis="y2")
    if "val_tpr_at_1pct_fpr" in cur:
        f.add_scatter(x=cur["epoch"], y=cur["val_tpr_at_1pct_fpr"], name="val TPR@1%FPR", yaxis="y2")
    f.add_scatter(x=cur["epoch"], y=cur["sup_loss"], name="supervised loss")
    if cur["unsup_loss"].abs().sum() > 0:
        f.add_scatter(x=cur["epoch"], y=cur["unsup_loss"], name="consistency loss")
    be = run[meth].get("best_epoch")
    if be:
        f.add_vline(x=be, line_dash="dot", annotation_text=f"selected epoch {be}")
    f.update_layout(height=400, margin=dict(t=10, b=10), xaxis_title="epoch",
                    yaxis_title="loss", yaxis2=dict(title="validation metric", overlaying="y", side="right"))
    st.plotly_chart(f, width='stretch')
    st.caption("Supervised loss keeps falling after validation TPR peaks — that divergence is the "
               "overfitting the epoch selection above is guarding against.")
