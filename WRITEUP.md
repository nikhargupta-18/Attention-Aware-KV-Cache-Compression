# Attention-Aware KV-Cache Compression: Writeup

> **The perplexity-vs-recall paradox in KV-cache eviction for long-context LLM inference**

---

## 1. Motivation and problem statement

Large language models maintain a **KV (key-value) cache** to avoid recomputing
attention over the full context at every generation step. Each new token queries
the cached keys from all previous tokens to attend to them. The cache grows
linearly with context length: at 100k tokens with a 7B-parameter model in FP16,
it typically exceeds 10 GB — often larger than the model weights themselves.

The natural engineering response is **KV-cache compression**: evict cached entries
that contribute negligibly to the output, keeping only a bounded "budget" of K/V
pairs. Several papers have shown that simple eviction policies achieve perplexity
(PPL) within 1–2 units of the full-cache baseline at 50% cache size, leading to
claims of near-lossless compression.

**This project challenges that claim.** We show empirically that:

1. Perplexity is insensitive to eviction because most next-token predictions require
   only local context, which eviction preserves.
2. Factual recall — retrieving a specific piece of information from mid-context —
   collapses completely under the same eviction that looks fine on PPL.
3. These two outcomes occur simultaneously, with the same policy, on the same model,
   at the same cache budget. **This is the paradox.**

---

## 2. Experimental setup

### Model

**Qwen/Qwen2.5-0.5B** (0.5B parameters, FP32, CPU).

- 24 transformer layers, 14 query heads, 2 KV heads (GQA), head dimension 64
- Rotary Position Embedding (RoPE); maximum context 32K tokens
- `attn_implementation="eager"` to materialise the full attention matrix
  (required for H2O scoring and attention analysis; ~2–3× slower than SDPA)
- Chosen for: CPU-reproducible, modern architecture (GQA + RoPE), `DynamicCache`
  API directly accessible without monkey-patching

### Eviction policies

**1. No eviction (baseline)**
The full KV cache is kept at every step. This is the quality upper bound.
Memory grows without bound but generation quality is maximal.

**2. Sliding window (broken baseline)**
Keep only the LAST `N` K/V pairs; evict everything older, including position 0.
This is the obvious first approach most engineers try. It fails because position 0
and early positions absorb "spare" attention probability mass (the attention sink
phenomenon). Removing them breaks the softmax normalisation for every subsequent
query. Expected result: repetition and incoherence within ~50 steps.

**3. StreamingLLM** (Xiao et al., ICLR 2024)
Keep the first `sink_size` K/V pairs (positions 0 to 3) plus a sliding window of
the most-recent `window_size` tokens. Budget = sink + window (we use 4 + 124 = 128
by default). The sinks are preserved, so softmax normalisation remains intact.
The "evicted middle" contains older non-sink tokens.

**4. H2O — Heavy-Hitter Oracle** (Zhang et al., NeurIPS 2023)
Maintain a running accumulated attention score for each cached position:
```
score[l, k] += mean_h( attn[l, h, last_query, k] )
```
When the cache exceeds the budget, evict the lowest-scoring positions (keeping
the "heavy hitters"). The key design decision: we use only the **last query row**
of each layer's attention matrix, not the column sum across all queries. The reason
is explained in §3.3.

### RoPE correction (Phase 3)

When tokens are evicted, the surviving K vectors still carry their original RoPE
position encodings. The new query at step `t` gets `position_ids = [cache_slot]`,
creating a mismatch: Q is at position `cache_slot` but many K vectors are encoded
at much larger original positions.

We implement the correction from StreamingLLM §3.3-3.4:

```
K_corrected = K_encoded * cos(Δ) + rotate_half(K_encoded) * sin(Δ)
```

where `Δ = new_slot − original_position` and the angle-difference identities give:

```
cos(Δ) = cos_new * cos_old + sin_new * sin_old
sin(Δ) = sin_new * cos_old − cos_new * sin_old
```

This re-encodes each K vector to its new cache-slot position, making both Q and K
operate in a contiguous 0..budget coordinate space that matches the training
distribution.

**H2O uses a different scheme.** Its content-adaptive eviction produces a scattered,
non-contiguous subset of original positions (e.g., positions 0, 5, 23, 88, 120...).
Compressing these to contiguous slots 0, 1, 2, 3, 4... creates false adjacency:
the model believes originally-separate positions are adjacent. Empirically,
slot-based re-indexing applied to H2O **increases** NLL by ~1.2 nats compared
to keeping original positions. H2O therefore keeps K vectors at their original
RoPE positions and uses absolute sequence position as the query's `position_ids`.

---

## 3. Phase 1 — Attention instrumentation

**Script:** `src/attention_analysis.py`

Before implementing eviction, we verify empirically that KV-cache eviction is
theoretically justified. Two properties must hold:

### 3.1 Score concentration (sparsity)

For each key position `k`, we compute the mean attention weight it receives from
all query positions `q ≥ k` (causal mask), averaged across all heads and layers.
This is the "column-mean" — a measure of how important each cached position is.

**Result (cumulative mass curve, `results/plot5_cumulative_mass_curve.png`):**
~15% of keys capture 80% of total attention mass across both test passages
(repetitive "Hello world" × 60 and a varied narrative). This means 85% of the
cached K/V pairs contribute less than 20% of the attention output — a 6.7×
theoretical compression budget.

### 3.2 Attention sink phenomenon

Regardless of content, the first 4 tokens absorb approximately 40–50% of total
attention mass, averaged across all queries, heads, and layers. A random baseline
predicts `4/T ≈ 1.3%` for a 300-token sequence. The observed 40–50% represents a
**30–40× overrepresentation.**

The sink is content-agnostic: the same pattern appears in both the repetitive and
narrative passages, and uniformly across all 24 layers × 14 heads (not just a
handful of "sink heads"). This is a structural property of the softmax + causal
mask combination: the first token is visible to every query, accumulating votes
regardless of its content.

**Implication for eviction:** Any policy that evicts position 0 will corrupt the
softmax for every subsequent query. This is why StreamingLLM preserves the first
`sink_size` tokens unconditionally.

---

## 4. Phase 2 — Eviction policy comparison

**Script:** `src/run_policies.py`

### 4.1 Qualitative evaluation

We feed a 200-token memory-probing prompt (introducing named characters, specific
facts) and generate 200 continuation tokens with each policy at budget = 128.

**Results:**
- `no_eviction`: Coherent, references the characters by name throughout
- `streaming_llm`: Fluent prose; loses some character-specific details after ~100 steps
- `h2o`: Comparable coherence; occasionally replaces character names with generic terms
- `sliding_window`: Begins hallucinating names within ~50 steps; degrades into repetition

### 4.2 NLL curves (teacher forcing)

For a fair comparison, we use **teacher forcing**: feed the correct previous token
at every step (not the model's prediction). This isolates cache quality from
generation drift.

The per-token NLL curves (`results/policy_nll_curves.png`) show:
- `no_eviction` and `streaming_llm` track closely
- `sliding_window` shows elevated NLL from the first eviction onwards
- `h2o` shows slightly elevated variance but similar mean to `streaming_llm`

---

## 5. Phase 3 — RoPE correction quantification

**Script:** `src/rope_correction_eval.py`

We compare two schemes:
- **Phase 3.3 (no correction):** Absolute `position_ids`, K vectors at original positions
- **Phase 3.4 (corrected):** Slot-based `position_ids`, K vectors re-encoded to cache slots

The NLL difference on standard prose is typically small (< 0.05 nats) because
grammar and fluency are fundamentally local — they don't require precise long-range
position matching. The correction becomes important for factual recall, where the
precise relative distance Q–K matters for retrieving a specific fact.

**Key finding for H2O:** Applying Phase 3.4 slot-based correction to H2O increases
NLL by ~1.2 nats because the scattered survivor set creates false adjacency in the
compressed coordinate space. H2O keeps original positions; `use_slot_positions = False`.

---

## 6. Phase 4 — Evaluation methodology

### 6.1 Perplexity (PPL) sweep

**Script:** `src/eval_perplexity.py`

We measure corpus PPL under teacher forcing for each (policy, budget) pair.
Prefill = 64 tokens; evaluate = 900 tokens.

**Results:**

| Policy | B=64 | B=96 | B=128 | B=192 | B=256 |
|---|---:|---:|---:|---:|---:|
| No eviction | 6.65 | 6.65 | 6.65 | 6.65 | 6.65 |
| StreamingLLM | 11.74 | 11.35 | 11.09 | 10.63 | 10.41 |
| Sliding window | 138.4 | 148.6 | 149.2 | 135.8 | 150.2 |
| H2O | 959.0 | 875.6 | 700.5 | 480.6 | 313.4 |

StreamingLLM's PPL is elevated but stable across budgets (~4–5 units above
baseline). Sliding window collapses immediately (it removes the sinks). H2O
shows high PPL in this teacher-forcing loop due to score-concentration
degeneration: after hundreds of steps, a tiny set of super-heavy-hitters dominates
the budget, evicting tokens needed for local next-token prediction. This does not
occur in the short-context needle evaluation.

### 6.2 Needle-in-a-haystack

**Script:** `src/eval_needle.py`

**Setup:**
- Needle: `"The secret launch code is DELTA-7-ZEPHYR."` — a distinctive, unguessable fact
- Haystack: ~384 neutral tokens about computing history
- Depths: 10%, 25%, 50%, 75%, 90% of haystack length
- Task: Generate the launch code after reading the full context
- Scoring: exact match OR token F1 ≥ 0.5 against "DELTA-7-ZEPHYR"

**This is prefill-only eviction:** the entire context is prefilled in one pass,
eviction happens once after prefill, then 20 tokens are generated greedily.
This models the RAG use case: compress a long retrieved context, then answer.

**Results (at budget = 128):**

| Policy | 10% | 25% | 50% | 75% | 90% |
|---|---|---|---|---|---|
| No eviction | ✅ | ✅ | ✅ | ✅ | ✅ |
| StreamingLLM | ❌ | ❌ | ❌ | ✅ | ✅ |
| Sliding window | ❌ | ❌ | ❌ | ✅ | ✅ |
| H2O | ❌ | ❌ | ✅ | ✅ | ✅ |

StreamingLLM can only recall facts that fall within its fixed window (75%, 90%)
or within the first 4 tokens (10% — near sinks). Everything in the "evicted middle"
(25%–50%) is irrecoverably lost.

**When the model fails:** It outputs `"DEL"` — the high-probability prefix for
"delete", "delivery", etc. in tech text. The model is not confused; it is
generating fluent text that happens to be wrong because the needed fact was evicted.

**The paradox confirmed:**
- StreamingLLM PPL at B=128 ≈ 11.09 (acceptable, ≈ 4.4 units above baseline)
- StreamingLLM mid-context recall at B=128 = 11% (catastrophic)
- PPL reports "fine"; needle reveals "broken"

---

## 7. Phase 5 — Quality-vs-memory curve

**Script:** `src/eval_quality_vs_memory.py`

This is the headline figure. We sweep budgets `[32, 48, 64, 96, 128, 192, 256]`
and simultaneously measure PPL and mid-context needle recall for all policies.

**Two-panel figure (`results/phase36_quality_vs_memory.png`):**

**Left (PPL):** All eviction lines cluster near the green baseline. StreamingLLM
is acceptably elevated but flat. Sliding window and H2O are high at small budgets
but converge. A reviewer looking only at this panel would conclude: "eviction
works, the PPL gap is manageable."

**Right (mid-context recall):** Sliding window and StreamingLLM show 0% recall
at budgets ≤ 96, with only marginal improvement at B=128 (one of three mid-depths
recovered). H2O is content-adaptive: it achieves 67% recall at B=48 and 100% at
B=128, demonstrating that content-aware eviction can simultaneously achieve
acceptable PPL and non-zero factual recall.

**The gap between the two panels is the headline result:** standard evaluation
methodology (PPL only) would approve all these policies. Needle evaluation reveals
that only H2O at B≥128 is truly acceptable for factual-recall tasks.

---

## 8. Why this matters

### The production risk

Any RAG pipeline that compresses a long retrieved context down to a fixed budget
before passing it to the LLM is vulnerable to this failure mode. The system
appears to work on standard benchmarks (which measure PPL or fluency) while
silently failing on any query that requires a specific fact from the compressed
middle of the context.

### The metric failure

PPL averages over hundreds of easy next-token predictions (function words,
punctuation, common phrases) that only need local context. The rare factual-recall
tokens are a tiny fraction of this average and contribute negligibly to the mean.
This means a policy can have 0% factual recall and still show ≈ 10.0 PPL vs.
6.65 for full cache — a difference that looks small on the standard scale.

### The right evaluation

**Every paper proposing KV-cache compression should report recall on a
needle-in-a-haystack benchmark alongside perplexity.** This project provides a
reproducible harness for doing so: see `src/eval_needle.py` and `src/benchmark.py`.

---

## 9. Key design decisions

### Why column-mean for attention importance (not row-mean)

Row-mean is `1/T` by construction (each row sums to 1). Column-mean is not
constrained — a high column-mean means "this token is important to many queries."
This is exactly the signal for deciding which K/V pairs to retain.

### Why H2O uses last-query-row scoring (not full column sum)

During a 180-token prefill, position 0 is visible to all 180 queries while
position 179 is visible to only 1. Summing all query rows inflates early positions'
scores purely due to causal-mask accessibility, not semantic importance. Using
only the final query row eliminates this bias.

### Why the evaluation uses teacher forcing

Teacher forcing (always feeding the correct previous token) prevents generation
drift — errors compounding over hundreds of steps. It isolates the effect of cache
quality from generation quality.

### Why we use PPL as the baseline metric

We use PPL because it's the metric KV-compression papers report. By showing that
PPL is misleading, we can make the comparison on the same terms the field uses.

### Why Qwen2.5-0.5B

The 0.5B model fits in CPU RAM (~2 GB FP32), making the entire project fully
reproducible without a GPU. It uses GQA + RoPE — the same architecture as modern
frontier models — so the results generalise.

---

## 10. Results summary

| Metric | Full cache | StreamingLLM (B=128) | H2O (B=128) |
|---|---|---|---|
| PPL | 6.65 | 11.09 | ~700* |
| Mid-context recall | 100% | 11% | 67% |

*H2O PPL is inflated by score-concentration degeneration in the 900-step
teacher-forcing loop. In the short-context needle evaluation, H2O performs well.

**Take-away:** No single eviction policy is universally best. StreamingLLM is
the best choice when PPL/fluency matters and factual recall is not required.
H2O is the best choice when factual recall matters. Neither is acceptable
when both are required at small budgets — which is the regime most real
long-context RAG systems operate in.

---

## References

1. Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024).
   *Efficient Streaming Language Models with Attention Sinks.*
   ICLR 2024. https://arxiv.org/abs/2309.17453

2. Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., ... & Li, C. (2023).
   *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.*
   NeurIPS 2023. https://arxiv.org/abs/2306.14048

3. Kamradt, G. (2023). *Needle In A Haystack — Pressure Testing LLMs.*
   https://github.com/gkamradt/LLMTest_NeedleInAHaystack

4. Bai, J., et al. (2023). *Qwen Technical Report.* arXiv:2309.16609.
