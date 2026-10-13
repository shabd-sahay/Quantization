import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Data handling", layout="wide")
st.title("1 · Handling an incomplete dataset")

log_p = ROOT / "data/processed/pipeline_log.json"
if not log_p.exists():
    st.error("Missing data/processed/pipeline_log.json — run scripts/02_prepare_data.py"); st.stop()
log = json.loads(log_p.read_text())

st.subheader("Every row that leaves the pipeline, and why")
steps = pd.DataFrame(log["steps"])
fig = go.Figure(go.Waterfall(
    orientation="v",
    measure=["absolute"] + ["relative"] * len(steps),
    x=["start"] + list(steps["stage"]),
    y=[int(steps.iloc[0]["rows_in"])] + [-int(r) for r in steps["removed"]],
    text=[f"{int(steps.iloc[0]['rows_in']):,}"] + [f"-{int(r):,}" if r else "0" for r in steps["removed"]],
    textposition="outside",
    connector={"line": {"dash": "dot"}},
))
fig.update_layout(height=430, margin=dict(t=10, b=10), yaxis_title="training rows",
                  xaxis_tickangle=-25, showlegend=False)
st.plotly_chart(fig, width='stretch')
st.dataframe(steps.rename(columns={"rows_in": "in", "rows_out": "out"}), hide_index=True,
             width='stretch')

st.info("""
**Dedup found almost nothing — and that is the point.** 4 duplicate feature vectors out of
99,999, and 1 leaked test row. EMBER is a properly constructed benchmark. The same two checks
run against the text-based CTI dataset this project replaced found that **86% of test rows had
an identical copy in training**, which inflated every number reported on it. The check is cheap;
skipping it is what costs you.
""")

st.subheader("What 'missing labels' actually means here")
b = pd.DataFrame(log["budgets"])
b["budget_pct"] = (b["budget"] * 100).round().astype(int).astype(str) + "%"
sel = st.select_slider("Label budget — how many labels we pretend to have",
                       options=list(b["budget_pct"]), value="10%")
row = b[b["budget_pct"] == sel].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Labeled rows we train on", f"{int(row['labeled']):,}")
c2.metric("Labels hidden by us", f"{int(row['hidden_labels']):,}",
          help="True label known, withheld — lets us score the model's own guesses")
c3.metric("Never labeled in EMBER", f"{int(row['truly_unlabeled']):,}",
          help="label = -1 in the dataset; no ground truth exists")

fig2 = go.Figure()
for name, col in (("labeled", "labeled"), ("hidden by us", "hidden_labels"),
                  ("never labeled (-1)", "truly_unlabeled")):
    fig2.add_bar(name=name, x=b["budget_pct"], y=b[col])
fig2.update_layout(barmode="stack", height=360, margin=dict(t=10, b=10),
                   yaxis_title="training rows", xaxis_title="label budget")
st.plotly_chart(fig2, width='stretch')

st.markdown(f"""
EMBER2018 ships **200,000 of 800,000** training rows with `label = -1` — the authors included
them deliberately, to make semi-supervised work possible on a real benchmark. The working
subsample draws evenly from benign / malicious / unlabeled, so unlabeled is **~1/3 here rather
than the dataset's 25%**.

Filling those labels in is the job: at the {sel} budget the model faces
**{int(row['total_unlabeled']):,} unlabeled rows** against {int(row['labeled']):,} labeled ones.
Because the features are dense numeric counts with no gaps, there is nothing to impute — what is
missing is supervision, not values. Page 2 shows what happens when the model tries to supply it.
""")
