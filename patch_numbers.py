#!/usr/bin/env python3
"""Recompute the quoted latency figures from results/quant_results.json and rewrite the
README table + the app's caveat box, so prose and data agree. Idempotent-ish: it replaces
whole blocks between markers it can find, and reports anything it could not match."""
import json, statistics as st
from pathlib import Path

root = Path(__file__).resolve().parent
q = json.loads((root / "results/quant_results.json").read_text())

KEYS = ["fp32_torch", "int8_dynamic", "fp32_onnx", "int8_onnx_dyn", "int8_onnx_sta"]
LABEL = {"fp32_torch": "PyTorch FP32", "int8_dynamic": "PyTorch INT8 (dynamic)",
         "fp32_onnx": "ONNX FP32", "int8_onnx_dyn": "**ONNX INT8 (dynamic)**",
         "int8_onnx_sta": "ONNX INT8 (static)"}

med, rng, size, thru = {}, {}, {}, {}
for k in KEYS:
    lat = [r[k]["p50_ms"] for r in q["runs"].values() if k in r]
    tp = [r[k]["batch_rows_per_sec"] for r in q["runs"].values() if k in r]
    sz = [r[k]["size_kb"] for r in q["runs"].values() if k in r]
    med[k], rng[k] = st.median(lat), (min(lat), max(lat))
    thru[k], size[k] = st.median(tp), st.median(sz)

full = q["runs"].get("100_supervised_s42", {})
tpr = {k: full.get(k, {}).get("tpr_at_1pct_fpr") for k in KEYS}
flip = {k: full.get(k, {}).get("decision_flip_rate_at_fp32_thr") for k in KEYS}

b, o32, o8, osta, t8 = med["fp32_torch"], med["fp32_onnx"], med["int8_onnx_dyn"], med["int8_onnx_sta"], med["int8_dynamic"]
speed, runtime_x, quant_x = b / o8, b / o32, o32 / o8
shrink = size["fp32_torch"] / size["int8_onnx_dyn"]

rows = "\n".join(
    f"| {LABEL[k]} | {size[k]/1024:.2f} MB | {med[k]:.3f} ms | {thru[k]:,.0f} rows/s | "
    f"{tpr[k]:.3f} | " + (f"{flip[k]*100:.2f}%" if flip[k] is not None else "—") + " |"
    for k in KEYS)
table = ("| Variant | Size | p50 | Throughput | TPR@1%FPR (100% labels) | Verdict flips |\n"
         "|---|---|---|---|---|---|\n" + rows)

caveat = (
    f"All ten models share one architecture, so their latency should be identical — and within a\n"
    f"single sweep it nearly is. Across sweeps it is not: an earlier run of the *same binaries* on a\n"
    f"loaded laptop measured `fp32_torch` at 0.325 ms where this one measures {b:.3f} ms, a 3x\n"
    f"difference from background load alone. Within this sweep `fp32_torch` p50 spans\n"
    f"{rng['fp32_torch'][0]:.3f}–{rng['fp32_torch'][1]:.3f} ms and `int8_onnx_dyn` spans\n"
    f"{rng['int8_onnx_dyn'][0]:.3f}–{rng['int8_onnx_dyn'][1]:.3f} ms. The table reports **medians over\n"
    f"the ten runs**; single-run figures are not quoted, and absolute latency is only meaningful\n"
    f"relative to the FP32 baseline measured in the same sweep.")

readme = (root / "README.md").read_text()
missed = []


def swap(text, start, end, new, name):
    i, j = text.find(start), text.find(end, text.find(start) + 1)
    if i < 0 or j < 0:
        missed.append(name); return text
    return text[:i] + new + "\n\n" + text[j:]


readme = swap(readme, "| Variant | Size | p50 |", "\n**Decomposed honestly:**", table, "quant table")
readme = swap(readme, "All ten models share one architecture", "\n---\n\n## Running it", caveat, "caveat")

for old, new in [
    ("**3.95× smaller, 4.1× faster** (0.318 → 0.077 ms p50)",
     f"**{shrink:.2f}x smaller, {speed:.1f}x faster** ({b:.3f} -> {o8:.3f} ms p50)"),
    ("of the 4.1× end-to-end speedup, **1.33× is the runtime change**",
     f"of the {speed:.1f}x end-to-end speedup, **{runtime_x:.2f}x is the runtime change**"),
    ("and **3.1× is quantization**", f"and **{quant_x:.1f}x is quantization**"),
]:
    if old in readme:
        readme = readme.replace(old, new)
    else:
        missed.append(old[:45])

(root / "README.md").write_text(readme)

# --- app caveat box ---------------------------------------------------------------
p = root / "app/pages/3_Quantization.py"
s = p.read_text()
i, j = s.find('st.warning("""'), s.find('""")', s.find('st.warning("""'))
if i >= 0 and j >= 0:
    s = s[:i] + ('st.warning("""\n**Read the latency numbers as medians, not as one run.** '
                 f'All ten models share one architecture,\nso their latency should be identical. Within this sweep `fp32_torch` p50 spans '
                 f'{rng["fp32_torch"][0]:.3f}-{rng["fp32_torch"][1]:.3f} ms;\nan earlier sweep of the same binaries on a loaded laptop measured 0.325 ms, 3x slower. '
                 f'Background\nload moves these numbers more than the models do. The table reports **medians over ten runs** '
                 f'-\n**{b:.3f} ms FP32 vs {o8:.3f} ms ONNX INT8, {speed:.1f}x** - and absolute latency is only meaningful against '
                 'the\nFP32 baseline measured in the same sweep.\n') + s[j:]
    p.write_text(s)
else:
    missed.append("app warning box")

print(f"medians  fp32_torch {b:.3f}  fp32_onnx {o32:.3f}  int8_torch {t8:.3f}  "
      f"int8_onnx_dyn {o8:.3f}  int8_onnx_sta {osta:.3f} ms")
print(f"speedup {speed:.2f}x  = runtime {runtime_x:.2f}x x quant {quant_x:.2f}x   size {shrink:.2f}x smaller")
print("UNMATCHED (edit by hand): " + ("; ".join(missed) if missed else "none"))
