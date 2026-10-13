# Malware detection under label scarcity, then quantized for CPU

Static Windows PE malware detection on **EMBER2018**, built around two questions:

1. **How far can detection get with almost no labels** — and can the dataset's 200,000 genuinely
   unlabeled files be put to use?
2. **What does INT8 quantization actually cost** once you measure it honestly on a laptop CPU?

Everything below is measured, including the parts that did not work.

**[Live demo](#)** · three pages: data handling, training under label scarcity, quantization.

---

## Headline results

| | |
|---|---|
| Detection at full labels (LightGBM) | **TPR 0.865 @ 1% FPR**, AUC 0.989 |
| Detection with 566 labels (1%) | TPR 0.500 @ 1% FPR, AUC 0.936 |
| INT8 vs FP32, worst case over 10 configs | **−0.008 TPR**, ≤0.49% of verdicts flipped |
| INT8 model size / latency | **3.96x smaller, 3.6x faster** (0.109 -> 0.030 ms p50) |
| Semi-supervised methods that beat supervised | **0 of 8**, across 3 seeds |

Hardware: Intel Core i7-1360P (Raptor Lake), AVX2 + AVX-VNNI, no AVX-512. CPU only, no GPU.

---

## Why TPR at a fixed FPR, and not accuracy

A malware scanner that cries wolf is unusable, so the operating point is pinned at a false-positive
budget and the question is how much malware is caught there. Accuracy hides exactly that: on this
data AUC moves from 0.936 to 0.989 as labels go from 1% to 100%, while **TPR at 1% FPR moves from
0.500 to 0.865**. AUC says the problem is nearly solved at 566 labels; the metric that matters says
it is not.

---

## 1. Data handling

EMBER2018 training set: 300k benign, 300k malicious, **200k with `label = -1`** — unlabeled on
purpose, so semi-supervised work is possible on a real benchmark. This project works on a
stratified 100k-row subsample (and 50k test rows), which draws evenly from the three groups, so
unlabeled is ~1/3 here rather than the dataset's 25%.

| Stage | In | Out | Removed |
|---|---|---|---|
| Drop duplicate file hashes | 99,999 | 99,999 | 0 |
| Drop duplicate feature vectors | 99,999 | 99,995 | 4 |
| Drop test rows also present in train | 50,000 | 49,999 | 1 |
| Set aside truly unlabeled pool (`-1`) | 99,995 | 66,664 | 33,331 |
| Hold out validation set | 66,664 | 56,664 | 10,000 |

**The dedup step found almost nothing, and that is worth reporting.** EMBER is properly constructed.
The same two checks run against the text-based CTI dataset this project replaced found that **86% of
its test rows had an identical copy in training** — every number published on it was inflated. The
check costs one function; skipping it costs the result.

Label budgets are nested (the 1% set is a subset of the 5% set, and so on) so the curves are
comparable. Hidden labels keep their true values out of band, which is what makes it possible to
score the model's own pseudo-labels later.

Because EMBER's features are dense numeric counts with no gaps, **nothing is imputed** — what is
missing is supervision, not values.

---

## 2. Training under label scarcity

### The curve

TPR @ 1% FPR on the held-out test set:

| Labels kept | Labeled rows | LightGBM | MLP (3 seeds) |
|---|---|---|---|
| 1% | 566 | 0.500 | 0.447 ± 0.021 |
| 5% | 2,834 | 0.673 | 0.516 ± 0.042 |
| 10% | 5,666 | 0.740 | 0.522 ± 0.036 |
| 25% | 14,166 | 0.774 | 0.634 ± 0.028 |
| 100% | 56,664 | 0.865 | 0.776 ± 0.010 |

Gradient boosting wins on tabular features, as expected. The MLP earns its place by being the model
that can be quantized to INT8 — that is its job here, not raw accuracy.

### The unlabeled pool did not help — eight configurations, three seeds

**Self-training on LightGBM** (TPR @ 1% FPR, mean of 3 seeds; parenthesis = paired difference vs supervised):

| Budget | supervised | conf ≥ 0.99 | conf ≥ 0.999 | top-10%/round | weighted 0.2 |
|---|---|---|---|---|---|
| 1% | 0.448 | 0.347 (−0.101) | 0.340 (−0.108) | 0.396 (−0.052) | 0.367 (−0.081) |
| 5% | 0.652 | 0.543 (−0.109) | 0.564 (−0.087) | 0.639 (−0.013) | 0.552 (−0.100) |
| 10% | 0.735 | 0.687 (−0.048) | 0.720 (−0.015) | 0.719 (−0.016) | 0.697 (−0.038) |
| 25% | 0.795 | 0.785 (−0.010) | 0.776 (−0.019) | 0.783 (−0.012) | 0.793 (−0.001) |

**Why it fails — and it is not label noise.** At the top-10% confidence cut the pseudo-labels are
**99.8–100% correct**, and detection still drops. The rows a model is most confident about are the
easy, typical ones; adding tens of thousands of them pulls training away from the hard cases near
the decision boundary, which is precisely where a 1%-FPR threshold lives. At the 0.95 cut the model
also admits ~83% of the pool in a single round, and its own errors then compound: pseudo-label
accuracy decays 92% → 88% over three rounds at the 1% budget.

**Consistency regularization on the MLP** — and a measurement trap worth showing:

| Budget | supervised | +consistency | sup (last epoch) | cons (last epoch) |
|---|---|---|---|---|
| 1% | 0.447 | 0.416 (−0.031) | 0.417 | 0.441 |
| 5% | 0.516 | 0.411 (−0.105) | 0.166 | 0.257 |
| 10% | 0.522 | 0.463 (−0.059) | 0.170 | 0.286 |
| 25% | 0.634 | 0.556 (−0.078) | 0.301 | 0.419 |
| 100% | 0.776 | 0.786 (+0.010) | 0.761 | 0.779 |

Compare the first two columns with the last two. Trained for a fixed 30 epochs and evaluated at the
final epoch, consistency looks like a large win (0.257 vs 0.166 at 5%). **Select the epoch on
validation instead and the win disappears** — consistency was only regularizing away overfitting
that proper model selection removes anyway. The first version of this experiment reported the
inflated number; fixing it reversed the conclusion.

> Scope: this is a negative result **for these two method families on this task**, not a claim about
> semi-supervised learning generally.

### A second finding: the threshold does not transfer

EMBER's test set is **later in time** than the training set. A threshold calibrated for 1% FPR on
validation lands at 1.3–2.4% FPR on test. Validation-selected checkpoints also transfer poorly
(seed 42, 10% consistency: validation TPR 0.643, test TPR 0.226). Reported rather than tuned away.

---

## 3. Quantization

Medians across all ten model configurations, 1 thread pinned, single-sample latency:

| Variant | Size | p50 | Throughput | TPR@1%FPR (100% labels) | Verdict flips |
|---|---|---|---|---|---|
| PyTorch FP32 | 5.17 MB | 0.109 ms | 51,619 rows/s | 0.789 | — |
| PyTorch INT8 (dynamic) | 1.32 MB | 0.099 ms | 86,268 rows/s | 0.787 | 0.39% |
| ONNX FP32 | 5.17 MB | 0.079 ms | 50,556 rows/s | 0.789 | — |
| **ONNX INT8 (dynamic)** | 1.31 MB | 0.030 ms | 135,283 rows/s | 0.788 | 0.14% |
| ONNX INT8 (static) | 1.31 MB | 0.033 ms | 138,500 rows/s | 0.787 | 0.16% |


**Decomposed honestly:** of the 3.6x end-to-end speedup, **1.38x is the runtime change**
(PyTorch → ONNX, both FP32) and **2.6x is quantization**. Reporting only the combined figure would
credit the runtime switch to INT8. Static calibration buys essentially nothing over dynamic here.

**And it holds everywhere.** Across all ten configurations (five label budgets × two training
methods), the largest detection change from INT8 is **0.008** and the largest verdict flip rate is
**0.49%**. INT8 being free at one well-trained configuration is the usual claim; it staying free
across a 100× range of label budgets is the stronger one.

### The failure that produced this section

The first INT8 run dropped detection from **0.789 to 0.178**, flipping 24.7% of verdicts.
Per-channel weight quantization did not help, which ruled out the weights.

The cause was the **input**. After log1p + standardization, EMBER's features still reach ±992 sigma
on the test set, because the scaler's statistics come from training rows. Dynamic quantization
scales the activation tensor per-tensor, so those few outliers set a scale that collapses every
ordinary value to zero — and the first layer takes all 2,381 of them.

The diagnostic that located it: quantizing every layer **except the first** recovered detection
fully (0.791). Only then was a fix attempted — clipping the scaled features to ±5, which touches
**0.08% of values**, leaves FP32 detection unchanged at 0.789, and restores INT8 to 0.787.

### Measurement caveat

All ten models share one architecture, so their latency should be identical — and within a
single sweep it nearly is. Across sweeps it is not: an earlier run of the *same binaries* on a
loaded laptop measured `fp32_torch` at 0.325 ms where this one measures 0.109 ms, a 3x
difference from background load alone. Within this sweep `fp32_torch` p50 spans
0.107–0.136 ms and `int8_onnx_dyn` spans
0.029–0.077 ms. The table reports **medians over
the ten runs**; single-run figures are not quoted, and absolute latency is only meaningful
relative to the FP32 baseline measured in the same sweep.


---

## Running it

### The app

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

The repo ships `results/*.json` and a small `demo/` bundle (1,000 pre-scaled test vectors plus ONNX
models), so all three pages work without downloading EMBER. The app is deliberately **torch-free** —
it scores with ONNX Runtime only, which keeps it inside Streamlit Cloud's resource limits and makes
the live comparison ONNX-FP32 vs ONNX-INT8, isolating quantization from the runtime change.

### Reproducing from scratch

```bash
pip install -r requirements-dev.txt
pip install lief
pip install git+https://github.com/elastic/ember.git
```

`ember` predates current scikit-learn and passes a bare string to `FeatureHasher`. Patch one line:

```bash
FILE=$(python -c "import ember.features as f; print(f.__file__)")
sed -i "s/transform(\[raw_obj\['entry'\]\])/transform([[raw_obj['entry']]])/" "$FILE"
```

Then, in order (step 1 downloads ~1.7 GB and expands to ~9 GB; budget ~25 GB free):

```bash
python scripts/01_get_ember.py download     # resumable, sha256-verified
python scripts/01_get_ember.py vectorize    # raw JSONL -> 2381-dim vectors
python scripts/01_get_ember.py subsample    # stratified working copy
python scripts/02_prepare_data.py           # dedup, leakage check, label budgets
python scripts/03_train.py                  # LightGBM baseline + self-training
python scripts/04_ssl_ablation.py           # 4 self-training variants x 3 seeds
python scripts/05_train_mlp.py --seeds 42 43 44
python scripts/06_quantize.py --all --keep-models
python scripts/07_make_demo_bundle.py       # small bundle for the app
```

---

## Layout

```
scripts/   01 data · 02 prepare · 03 train · 04 ablation · 05 mlp · 06 quantize · 07 demo bundle
src/       scaler.py (numpy-only, used by the app) · mlp.py (torch)
app/       Home.py + pages/ — one page per stage
results/   ssl_results.json · ssl_ablation.json · mlp_results.json · quant_results.json
demo/      1,000 scaled test vectors + ONNX models (committed; the full data is not)
```

Every figure in the app is generated from the JSON in `results/`; none are typed by hand.

---

## Known limitations

- **Validation set size.** The 10,000-row validation set is the same at every budget. In a real
  1%-label setting you would not have that many labels. It is used only for threshold calibration
  and epoch selection, never for tuning — but it is a convenience, and stated as one.
- **Scaler change after training.** The MLP checkpoints were trained with the unclipped scaler and
  are evaluated with the clipped one. FP32 detection is identical either way (0.789), so the results
  stand, but the two were not produced in one pass.
- **LightGBM is single-seed**; only the MLP and ablation results are averaged over three seeds.
- **Subsample, not the full dataset.** 100k of 800k training rows, so absolute numbers sit below
  what the full EMBER benchmark supports.
- **Pseudo-label accuracy is measurable only on the rows whose labels were hidden.** EMBER's truly
  unlabeled rows have no ground truth and may be harder.

## Citation

H. Anderson and P. Roth, *EMBER: An Open Dataset for Training Static PE Malware Machine Learning
Models*, arXiv:1804.04637, 2018. — <https://github.com/elastic/ember>
