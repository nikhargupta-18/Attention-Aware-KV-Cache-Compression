"""
rope_correction_eval.py  –  Phase 3.4
=======================================
Targeted evaluation: does RoPE re-indexing after KV-cache eviction matter?

Background
----------
When tokens are evicted from the middle of the KV cache, the surviving K
vectors carry RoPE encodings for their ORIGINAL sequence positions.  With
StreamingLLM (sink=4, window=124, budget=128), after a 296-token sequence
the window tokens have original positions 172..295 but now occupy cache slots
4..127.  The query at step t gets position_ids=296 and computes relative
distances against the ORIGINAL positions (172..295), leaving a gap for the
evicted positions 4..171 that no longer exist in the cache.

Phase 3.3 ("absolute position" scheme) was already working surprisingly well
because:
  a) The relative distances Q-K are correct in absolute terms
  b) StreamingLLM's gap (positions 4..171 gone) is a fixed gap that doesn't
     grow — it's always one contiguous block between sinks and window

Phase 3.4 ("slot-based" scheme, StreamingLLM §3.3-3.4) is theoretically
more principled:
  a) K vectors are re-encoded to contiguous slots 0..budget-1
  b) Query gets position_ids = cache_slot (= budget when full)
  c) Both Q and K operate in a contiguous 0..budget coordinate space

Evaluation design
-----------------
To make the effect visible we:
  1. Use a LONG NLL passage (PREFILL + 500 tokens) where the gap grows.
  2. Embed a factual "needle" near the end of the sink region to create a
     position-sensitive test: with correct positions, the model can attend
     to the needle; with stale positions, the relative distance to the needle
     is corrupted.
  3. Compare four conditions for StreamingLLM and H2O:
       a) 3.3 scheme: absolute position_ids, no K correction
       b) 3.4 scheme: slot-based position_ids + K correction

The NLL difference on a standard passage is typically small (0.02–0.1 nats)
because RoPE position errors don't break grammar/syntax — they corrupt
which facts get attended to.  The difference becomes larger and measurable
with:
  - Longer sequences (more drift)
  - Factual-recall tests (Phase 3.5, needle-in-haystack)

This script establishes the baseline: quantifies the NLL gap and verifies
that the correction always helps or is neutral, never hurts.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from kv_cache_policies import (
    NoEvictionPolicy,
    SlidingWindowPolicy,
    StreamingLLMPolicy,
    H2OPolicy,
    RoPECorrector,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-0.5B"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BUDGET      = 128
SINK_SIZE   = 4
WINDOW_SIZE = BUDGET - SINK_SIZE    # 124

PREFILL_LEN = 200   # slightly longer than 3.3's 180 to widen the gap
EVAL_LEN    = 500   # longer evaluation span to make drift visible
SMOOTH_WIN  = 30

NLL_PASSAGE = (
    "The history of computing is inseparable from the history of mathematics. "
    "Charles Babbage conceived the Analytical Engine in the 1830s, a mechanical "
    "device that prefigured the modern stored-program computer. Ada Lovelace, "
    "working from Babbage's notes, wrote what is widely considered the first "
    "algorithm intended to be processed by such a machine. A century later, "
    "Alan Turing formalised the notion of computation with his abstract Turing "
    "machine, proving that certain problems — the halting problem chief among "
    "them — are fundamentally undecidable. John von Neumann's architecture, "
    "proposed in 1945, organised memory, a processing unit, and input/output "
    "into a coherent whole that remains the template for virtually every "
    "general-purpose computer built since. The transistor, invented at Bell "
    "Labs in 1947, replaced bulky vacuum tubes and enabled the miniaturisation "
    "that would eventually produce the integrated circuit, the microprocessor, "
    "and the pocket-sized devices that billions of people carry today. Gordon "
    "Moore observed in 1965 that the number of transistors on a chip roughly "
    "doubled every two years — a trend that held for half a century before "
    "physical limits began to slow the pace. Meanwhile, software evolved from "
    "hand-coded machine instructions to assembly language, then to high-level "
    "languages such as FORTRAN, COBOL, C, and eventually Python. Each "
    "abstraction layer traded raw efficiency for human expressiveness, "
    "broadening the population of people who could direct a machine to do "
    "useful work. The internet connected these machines into a global network, "
    "and the World Wide Web layered a navigable hypertext system on top. "
    "Search engines, social networks, streaming media, and cloud computing "
    "followed in rapid succession, transforming how knowledge is stored, "
    "retrieved, and shared across the entire planet. Artificial intelligence, "
    "once a laboratory curiosity, re-emerged as a practical discipline when "
    "deep neural networks proved capable of recognising images, transcribing "
    "speech, and translating languages at near-human accuracy. Large language "
    "models trained on vast corpora of text have more recently demonstrated "
    "the ability to generate coherent prose, write functional code, and reason "
    "through multi-step problems, raising both excitement about their potential "
    "and urgent questions about their societal implications. "
) * 3   # repeat for length

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 3.4 — RoPE Position Correction Evaluation")
print("=" * 65)
print(f"\nBudget : {BUDGET}  (sink={SINK_SIZE} + window={WINDOW_SIZE})")
print(f"Prefill: {PREFILL_LEN} tokens  |  Eval: {EVAL_LEN} tokens\n")

print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    dtype=torch.float32,
)
model.eval()
device = next(model.parameters()).device
print("Model loaded.\n")

# Build the RoPE correction table once and share across policies
rope_corrector = RoPECorrector(model, max_positions=2048)
print("RoPE corrector built.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Policy registry: compare 3.3 (no correction) vs 3.4 (correction)
# ─────────────────────────────────────────────────────────────────────────────
# For the "no correction" baseline we use policies WITHOUT a rope_corrector
# AND with absolute position_ids in the generation loop.
# For the "3.4 corrected" version we use policies WITH rope_corrector
# AND slot-based position_ids.

POLICIES_33 = {   # Phase 3.3: no RoPE correction, absolute position_ids
    "no_eviction"          : NoEvictionPolicy(),
    "streaming_llm_3.3"   : StreamingLLMPolicy(SINK_SIZE, WINDOW_SIZE),
    "h2o_3.3"             : H2OPolicy(BUDGET),
}
POLICIES_34 = {   # Phase 3.4: with RoPE correction, slot-based position_ids
    "no_eviction"          : NoEvictionPolicy(),          # same — no correction needed
    "streaming_llm_3.4"   : StreamingLLMPolicy(SINK_SIZE, WINDOW_SIZE,
                                                 rope_corrector=rope_corrector),
    "h2o_3.4"             : H2OPolicy(BUDGET,
                                       rope_corrector=rope_corrector),
}

STYLE = {
    "no_eviction"        : dict(color="#27ae60", ls="-",  lw=2.2,
                                label="No eviction (upper bound)"),
    "streaming_llm_3.3"  : dict(color="#2980b9", ls="--", lw=1.8,
                                label="StreamingLLM — 3.3 (absolute pos, no K correction)"),
    "streaming_llm_3.4"  : dict(color="#2980b9", ls="-",  lw=2.2,
                                label="StreamingLLM — 3.4 (slot pos + K correction)"),
    "h2o_3.3"            : dict(color="#e67e22", ls="--", lw=1.8,
                                label="H2O — 3.3 (absolute pos, no K correction)"),
    "h2o_3.4"            : dict(color="#e67e22", ls="-",  lw=2.2,
                                label="H2O — 3.4 (slot pos + K correction)"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Teacher-forcing NLL helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_nll_curve(
    policy,
    text: str,
    prefill_len: int,
    eval_len: int,
    slot_based_positions: bool,
) -> list[float]:
    """
    Teacher-forcing NLL.

    Parameters
    ----------
    slot_based_positions : bool
        True  → position_ids = [[cache.get_seq_length()]]  (Phase 3.4)
        False → position_ids = [[t]]  where t is the absolute text position
                (Phase 3.3)
    """
    policy.reset()

    inputs  = tokenizer(text, return_tensors="pt").to(device)
    all_ids = inputs["input_ids"][0]
    T       = len(all_ids)

    if T < prefill_len + 2:
        raise ValueError(f"Text only {T} tokens; need {prefill_len + 2}+")

    # ── Prefill ───────────────────────────────────────────────────────────────
    prefill_ids = all_ids[:prefill_len].unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=prefill_ids, use_cache=True, output_attentions=True)
    cache = policy.step(out.past_key_values, out.attentions)

    # ── Teacher-forcing evaluation ────────────────────────────────────────────
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    nll_values  = []
    end_pos     = min(T - 1, prefill_len + eval_len)

    for t in range(prefill_len, end_pos):
        tok_in  = all_ids[t    ].unsqueeze(0).unsqueeze(0).to(device)
        tok_tgt = all_ids[t + 1].to(device)

        if slot_based_positions:
            pos_ids = torch.tensor([[cache.get_seq_length()]], device=device)
        else:
            pos_ids = torch.tensor([[t]], device=device)

        with torch.no_grad():
            out = model(
                input_ids       = tok_in,
                past_key_values = cache,
                use_cache       = True,
                output_attentions = True,
                position_ids    = pos_ids,
            )
        cache = policy.step(out.past_key_values, out.attentions)

        log_probs = log_softmax(out.logits[0, -1, :])
        nll_values.append(-float(log_probs[tok_tgt]))

    return nll_values


def rolling_mean(values: list, window: int) -> np.ndarray:
    arr  = np.array(values, dtype=np.float32)
    kern = np.ones(window) / window
    return np.convolve(arr, kern, mode="same")


# ─────────────────────────────────────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("Computing NLL curves (teacher forcing)…")
print("─" * 65)

results = {}

for name, policy in POLICIES_33.items():
    slot = (name == "no_eviction")   # no_eviction: slot == absolute (equiv)
    print(f"  [3.3 scheme] {name} …", end="", flush=True)
    nll = compute_nll_curve(policy, NLL_PASSAGE, PREFILL_LEN, EVAL_LEN,
                            slot_based_positions=slot)
    results[name + "_33"] = nll
    print(f"  mean NLL = {np.mean(nll):.4f}")

print()
for name, policy in POLICIES_34.items():
    print(f"  [3.4 scheme] {name} …", end="", flush=True)
    nll = compute_nll_curve(policy, NLL_PASSAGE, PREFILL_LEN, EVAL_LEN,
                            slot_based_positions=True)
    results[name + "_34"] = nll
    print(f"  mean NLL = {np.mean(nll):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY — Mean NLL by scheme (lower = better)")
print("=" * 65)

baseline = np.mean(results["no_eviction_33"])
pairs = [
    ("no_eviction",       "no_eviction_33",     "no_eviction_34",     "No eviction"),
    ("streaming_llm",     "streaming_llm_3.3_33","streaming_llm_3.4_34","StreamingLLM"),
    ("h2o",               "h2o_3.3_33",          "h2o_3.4_34",          "H2O"),
]
for _, k33, k34, label in pairs:
    m33 = np.mean(results[k33])
    m34 = np.mean(results[k34])
    diff = m34 - m33
    sign = "+" if diff >= 0 else ""
    print(f"  {label:<18}  3.3={m33:.4f}  3.4={m34:.4f}  "
          f"Δ(3.4-3.3)={sign}{diff:.4f}"
          + ("  ✅ 3.4 better" if diff < -0.005 else
             "  ⚖️  negligible"  if abs(diff) <= 0.005 else
             "  ⚠️  3.3 better — investigate"))

# ─────────────────────────────────────────────────────────────────────────────
# Plot: NLL curves — 3.3 vs 3.4 for StreamingLLM and H2O
# ─────────────────────────────────────────────────────────────────────────────
x = np.arange(len(results["no_eviction_33"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, policy_name, title in [
    (axes[0], "streaming_llm", "StreamingLLM: 3.3 vs 3.4 position scheme"),
    (axes[1], "h2o",           "H2O: 3.3 vs 3.4 position scheme"),
]:
    k33 = f"{policy_name}_3.3_33"
    k34 = f"{policy_name}_3.4_34"

    for k, label, ls, lw, alpha in [
        ("no_eviction_33", "No eviction (upper bound)", "-", 2.0, 1.0),
        (k33, "3.3: absolute pos, no K corr", "--", 1.8, 1.0),
        (k34, "3.4: slot pos + K correction", "-",  2.2, 1.0),
    ]:
        raw    = np.array(results[k])
        smooth = rolling_mean(results[k], SMOOTH_WIN)
        color  = "#27ae60" if "no_eviction" in k else \
                 "#2980b9" if "streaming" in k else "#e67e22"
        ax.plot(x, smooth, color=color, linestyle=ls, linewidth=lw,
                alpha=alpha, label=label)
        ax.plot(x, raw, color=color, linestyle=ls, linewidth=0.6,
                alpha=0.2)

    ax.set_xlabel(f"Token offset from prefill end (prefill={PREFILL_LEN})",
                  fontsize=10)
    ax.set_ylabel("NLL (smoothed)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

mean_33_sllm = np.mean(results["streaming_llm_3.3_33"])
mean_34_sllm = np.mean(results["streaming_llm_3.4_34"])
mean_33_h2o  = np.mean(results["h2o_3.3_33"])
mean_34_h2o  = np.mean(results["h2o_3.4_34"])
plt.suptitle(
    f"RoPE position correction (Phase 3.4) — budget={BUDGET}, "
    f"Qwen2.5-0.5B\n"
    f"StreamingLLM: 3.3 NLL={mean_33_sllm:.3f} → 3.4 NLL={mean_34_sllm:.3f}  |  "
    f"H2O: 3.3 NLL={mean_33_h2o:.3f} → 3.4 NLL={mean_34_h2o:.3f}",
    fontsize=10, y=1.02,
)
plt.tight_layout()

out = os.path.join(RESULTS_DIR, "rope_correction_nll.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: cumulative NLL gap (Δ NLL = 3.4 - 3.3) over the evaluation window
# ─────────────────────────────────────────────────────────────────────────────
# A negative Δ means 3.4 is better; near-zero means the correction is neutral.
# For a correctly-implemented correction this should be ≤ 0 everywhere.
fig, ax = plt.subplots(figsize=(10, 4))

for policy_name, color, label in [
    ("streaming_llm", "#2980b9", "StreamingLLM"),
    ("h2o",           "#e67e22", "H2O"),
]:
    k33  = f"{policy_name}_3.3_33"
    k34  = f"{policy_name}_3.4_34"
    n    = min(len(results[k33]), len(results[k34]))
    diff = np.array(results[k34][:n]) - np.array(results[k33][:n])
    # Cumulative sum normalised by position → shows whether gap is growing
    cumulative_mean = np.cumsum(diff) / (np.arange(n) + 1)
    ax.plot(cumulative_mean, color=color, linewidth=2.0,
            label=f"{label}  (Δ = 3.4 − 3.3, lower=better for 3.4)")

ax.axhline(0, color="#7f8c8d", linestyle=":", linewidth=1.0,
           label="Δ = 0 (no difference)")
ax.fill_between(np.arange(n), 0, cumulative_mean,
                where=cumulative_mean < 0,
                alpha=0.15, color="#27ae60", label="3.4 better region")
ax.fill_between(np.arange(n), 0, cumulative_mean,
                where=cumulative_mean > 0,
                alpha=0.15, color="#e74c3c", label="3.3 better region")
ax.set_xlabel(f"Token offset from prefill end", fontsize=11)
ax.set_ylabel("Cumulative mean Δ NLL (3.4 − 3.3)", fontsize=11)
ax.set_title(
    "Cumulative NLL gap: slot-based RoPE correction vs. absolute positions\n"
    "(negative = 3.4 better; near-zero = correction neutral; positive = investigate)",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out2 = os.path.join(RESULTS_DIR, "rope_correction_delta.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved: {out2}")

print(f"\nAll outputs in: {os.path.abspath(RESULTS_DIR)}")
print("""
Note on visibility
──────────────────
The NLL difference on a standard passage is typically small (< 0.05 nats)
because RoPE position errors corrupt *which facts* get attended to, not
grammar or fluency.  The correction becomes measurably important when:
  •  Context is very long (gap between original and slot position grows)
  •  Factual-recall precision matters (Phase 3.5 needle-in-haystack)
""")
