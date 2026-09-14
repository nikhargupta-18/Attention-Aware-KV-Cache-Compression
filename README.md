# Attention-Aware KV-Cache Compression

## Overview

Large language models keep a **KV cache** during generation — a table of key-value
attention pairs for every previous token. This cache grows linearly with context length
and becomes the primary memory bottleneck for long-context inference.

This project asks: *can we evict most of the cache without hurting quality?*
The uncomfortable answer is: **it depends on how you measure quality.**

**The headline result:** StreamingLLM achieves perplexity within 4–5 units of the
full-cache baseline at any budget — yet at the same time, factual recall of mid-context
facts collapses to **0–11%**. Two metrics, same cache, opposite stories.

---

## Repository structure

```
attention-kv-cache/
├── src/
│   ├── kv_cache_policies.py      ← Core: all 4 policies + RoPECorrector
│   ├── attention_analysis.py     ← Phase 1: 5 attention visualisation plots
│   ├── run_policies.py           ← Phase 2: qualitative gen + NLL curves
│   ├── rope_correction_eval.py   ← Phase 3: RoPE correction comparison
│   ├── eval_perplexity.py        ← Phase 4: budget-swept perplexity
│   ├── eval_needle.py            ← Phase 4: needle-in-a-haystack heatmap
│   ├── eval_quality_vs_memory.py ← Phase 5: combined quality-vs-memory curve
│   ├── test_correctness.py       ← Correctness harness (14 unit tests)
│   ├── benchmark.py              ← Unified benchmark script
│   ├── test_model.py             ← Smoke test for model loading
│   ├── generate.py               ← Simple generation utility
│   ├── inspect_cache.py          ← Cache inspection utility
│   └── tokenize_text.py          ← Token inspection utility
├── results/
│   ├── plot1_*.png               ← Attention heatmaps
│   ├── plot2_sink_absorption.png
│   ├── plot3_sink_per_layer_head.png
│   ├── plot4_entropy_violin_*.png
│   ├── plot5_cumulative_mass_curve.png
│   ├── policy_nll_curves.png
│   ├── policy_cache_sizes.png
│   ├── rope_correction_nll.png
│   ├── rope_correction_delta.png
│   ├── phase5_perplexity.png
│   ├── phase5_needle_heatmap.png
│   ├── phase36_quality_vs_memory.png
│   └── *.json                    ← Raw data for all phases
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> All experiments run on **CPU**. No GPU required.
> The model (`Qwen/Qwen2.5-0.5B`, ~2 GB FP32) is downloaded automatically on first run.

### 2 — Verify correctness (no model download needed)

```bash
python src/test_correctness.py
```

Expected output:

```
============================================================
KV-Cache Policy Correctness Harness
============================================================
  ✅  NoEviction: identity (no K/V modification)
  ✅  SlidingWindow: stays within budget
  ✅  SlidingWindow: evicts sink (pos-0)
  ✅  StreamingLLM: stays within budget
  ✅  StreamingLLM: preserves sink tokens
  ✅  H2O: stays within budget
  ✅  H2O: raises ValueError without attentions
  ✅  H2O: reset() clears accumulated scores
  ✅  RoPECorrector: no-op on identity positions
  ✅  _evict_layers: correct slot mapping
  ✅  rotate_half: rh(rh(x)) == -x
  ✅  All policies: temporal order preserved
  ✅  StreamingLLM: budget maintained across 30 steps
  ✅  H2O: budget maintained across 20 steps
============================================================
ALL TESTS PASSED  (14/14)
============================================================
```

### 3 — Quick benchmark (3–5 min)

```bash
python src/benchmark.py --quick
```

### 4 — Full evaluation suite (~60 min total)

Run each phase individually (all outputs go to `results/`):

```bash
# Phase 1 — Attention analysis (5 plots, ~2 min)
python src/attention_analysis.py

# Phase 2 — Policy comparison: NLL curves + qualitative gen (~10 min)
python src/run_policies.py

# Phase 3 — RoPE correction comparison (~15 min)
python src/rope_correction_eval.py

# Phase 4a — Perplexity sweep (~8 min)
python src/eval_perplexity.py

# Phase 4b — Needle-in-a-haystack (~10 min)
python src/eval_needle.py

# Phase 5 — Quality-vs-memory curve (~15–20 min)
python src/eval_quality_vs_memory.py
```

---

## Eviction policies

| Policy | Description | Key property |
|---|---|---|
| **NoEviction** | Full cache — keep everything | Oracle upper bound |
| **SlidingWindow** | Keep the last `N` tokens; evict everything else | Intentionally **broken** baseline — evicts the attention sink |
| **StreamingLLM** | Keep first `sink_size` tokens + sliding window of recency | Preserves softmax normalisation; good PPL, poor mid-recall |
| **H2O** | Keep top-`budget` tokens by accumulated attention score | Content-adaptive; better recall at larger budgets |

### Architecture

Every policy implements the same two-method interface:

```python
cache = policy.step(past_key_values, attentions)  # apply eviction in-place
policy.reset()                                      # clear state between sequences
```

All policies operate directly on `DynamicCache` (HuggingFace's modern cache
format) without monkey-patching or model modifications.

---

## RoPE Correction (Phase 3)

When tokens are evicted, the surviving K vectors carry stale RoPE position
encodings. We implement the correction from StreamingLLM §3.3-3.4:

```
K_corrected = K * cos(Δ) + rotate_half(K) * sin(Δ)
```

where `Δ = new_slot - original_position` and the angle-difference identities give:

```
cos(Δ) = cos_new * cos_old + sin_new * sin_old
sin(Δ) = sin_new * cos_old - cos_new * sin_old
```

> **Important:** H2O does NOT use slot-based re-indexing. Its scattered,
> non-contiguous eviction pattern would create false adjacency if re-indexed,
> increasing NLL by ~1.2 nats. H2O keeps original RoPE positions instead.

---

## Key results

### Perplexity vs budget

| Policy | B=64 | B=128 | B=256 |
|---|---:|---:|---:|
| No eviction | 6.65 | 6.65 | 6.65 |
| StreamingLLM | 11.74 | 11.09 | 10.41 |
| Sliding window | 138.4 | 149.2 | 150.2 |
| H2O | 959.0 | 700.5 | 313.4 |

### Mid-context needle recall (25%–75% depth average)

| Policy | B=64 | B=128 | B=256 |
|---|---:|---:|---:|
| No eviction | 100% | 100% | 100% |
| StreamingLLM | 0% | 11% | 11% |
| Sliding window | 0% | 11% | 11% |
| H2O | 33% | 67% | 100% |

**The paradox:** StreamingLLM shows PPL ≈ 11 (looks acceptable) but mid-context
recall = 0–11% (catastrophic failure). Perplexity is the standard KV-compression
metric, and it is measuring the wrong thing.

### Plots

All plots are pre-generated in `results/`:

| File | What it shows |
|---|---|
| `plot1_attention_received_*.png` | Column-mean attention heatmaps per layer (log scale) |
| `plot2_sink_absorption.png` | First K tokens absorb 40–50% of total attention mass |
| `plot3_sink_per_layer_head.png` | Sink fraction is universal across all 24 layers × 14 heads |
| `plot4_entropy_violin_*.png` | Attention entropy distribution — low entropy = sparse = compressible |
| `plot5_cumulative_mass_curve.png` | 15% of keys capture 80% of mass — 6.7× compression potential |
| `policy_nll_curves.png` | Per-token NLL curves during teacher-forced evaluation |
| `policy_cache_sizes.png` | Cache occupancy over decode steps |
| `rope_correction_nll.png` | NLL with vs. without RoPE re-encoding |
| `rope_correction_delta.png` | Cumulative NLL gap (Phase 3.3 vs Phase 3.4) |
| `phase5_perplexity.png` | Budget-swept PPL — all curves cluster near the baseline |
| `phase5_needle_heatmap.png` | Recall heatmap (budget × depth) — green = hit, red = miss |
| `phase36_quality_vs_memory.png` | **The headline figure** — left: PPL looks fine; right: recall collapses |

---

## Model

**Qwen/Qwen2.5-0.5B** — chosen because:

- Fits in CPU RAM (~2 GB FP32); fully reproducible without GPU
- Uses modern GQA + RoPE architecture (same as frontier models)
- `attn_implementation="eager"` materialises the full attention matrix, required for H2O scoring and attention analysis

---

## Citation

Policies implemented from:

- Xiao et al., *"Efficient Streaming Language Models with Attention Sinks"*, ICLR 2024 (StreamingLLM)
- Zhang et al., *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs"*, NeurIPS 2023 (H2O)

---
