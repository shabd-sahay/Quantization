"""
Streamlit demo. Run from the project root:   streamlit run app/Home.py
"""
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="EMBER Quant Detector", page_icon="shield", layout="wide")

ROOT = Path(__file__).resolve().parents[1]

st.title("Malware detection under label scarcity, then quantized for CPU")
st.caption("Static Windows PE features (EMBER2018) · 2,381 dims · CPU-only deployment")

st.markdown("""
Three questions this demo answers, one per page:

1. **Data** — what the cleaning pipeline removes, and what it means that a quarter of EMBER's
   training set has no label at all.
2. **Training & detection** — how detection degrades as labels disappear, and whether the
   unlabeled pool can be used to recover it.
3. **Quantization** — what INT8 costs in accuracy and buys in size and speed on a laptop CPU.
""")

st.subheader("Status")
need = {
    "data/processed/pipeline_log.json": "page 1 (data handling)",
    "results/ssl_results.json": "page 2 (LightGBM baseline)",
    "results/mlp_results.json": "page 2 (MLP label-budget curve)",
    "results/quant_results.json": "page 3 (quantization)",
}
cols = st.columns(len(need))
for c, (rel, what) in zip(cols, need.items()):
    ok = (ROOT / rel).exists()
    c.metric(what, "ready" if ok else "missing", help=rel)

missing = [r for r in need if not (ROOT / r).exists()]
if missing:
    st.warning("Run the scripts that produce: " + ", ".join(missing))

with st.expander("Hardware and measurement notes"):
    st.markdown("""
- Intel Core i7-1360P (Raptor Lake), AVX2 + AVX-VNNI, **no** AVX-512. CPU only, no GPU.
- All latency figures are single-sample, 1 thread pinned — the pessimistic endpoint-agent case.
- Detection is reported as TPR at a fixed 1% / 0.1% false-positive rate, never plain accuracy:
  on an 82/18-style split accuracy hides exactly the failure a security team cares about.
- The EMBER test set is **later in time** than the training set, so a threshold calibrated on
  validation does not transfer exactly. That gap is reported rather than tuned away.
""")
