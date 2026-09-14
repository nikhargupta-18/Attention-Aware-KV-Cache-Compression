"""
eval_perplexity.py  –  Phase 5.1
==================================
Corpus perplexity (PPL) swept over cache budgets for all three eviction
policies.  Demonstrates the first half of the perplexity-vs-recall paradox:

    Perplexity ≈ flat across policies and budgets.

This is the misleading metric — a bad eviction policy can look fine on PPL
while completely destroying factual retrieval (shown in eval_needle.py).

Methodology
-----------
Teacher-forcing on a long document (≥ 1 000 tokens).  For each
(policy, budget) pair:
  1. Prefill the first PREFILL_LEN tokens.
  2. Feed each subsequent token as the input (teacher forcing — always the
     correct token), record the negative log-likelihood of the *next* token.
  3. Aggregate: PPL = exp(mean(NLL)).

All eviction policies use Phase-3.4 RoPE correction.  Policies are
instantiated fresh for each (policy, budget) pair.

Output
------
  results/phase5_perplexity.png   — budget vs PPL line chart
  results/phase5_perplexity.json  — raw numbers
"""

import os
import sys
import json
import math
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

# Budget sweep: small enough to force meaningful evictions, large enough to
# show the curve flattening.
BUDGETS     = [64, 96, 128, 192, 256]

SINK_SIZE   = 4    # StreamingLLM: always keep first SINK_SIZE tokens
PREFILL_LEN = 64   # tokens used for prefill (policy starts evicting after this)
EVAL_LEN    = 900  # teacher-forcing tokens evaluated after prefill

# ─────────────────────────────────────────────────────────────────────────────
# Long document for evaluation
# One long passage that stays coherent so PPL is meaningful.
# ─────────────────────────────────────────────────────────────────────────────
LONG_DOC = (
    "The history of computing is inseparable from the history of mathematics. "
    "Charles Babbage conceived the Analytical Engine in the 1830s, a mechanical "
    "device that prefigured the modern stored-program computer. Ada Lovelace, "
    "working from Babbage's notes, wrote what is widely considered the first "
    "algorithm intended to be processed by such a machine. A century later, "
    "Alan Turing formalised the notion of computation with his abstract Turing "
    "machine, proving that certain problems — the halting problem chief among them — "
    "are fundamentally undecidable. John von Neumann's architecture, proposed in 1945, "
    "organised memory, a processing unit, and input/output into a coherent whole "
    "that remains the template for virtually every general-purpose computer built "
    "since. The transistor, invented at Bell Labs in 1947, replaced bulky vacuum "
    "tubes and enabled the miniaturisation that would eventually produce the "
    "integrated circuit, the microprocessor, and the pocket-sized devices that "
    "billions of people carry today. Gordon Moore observed in 1965 that the number "
    "of transistors on a chip roughly doubled every two years — a trend that held "
    "for half a century before physical limits began to slow the pace. Meanwhile, "
    "software evolved from hand-coded machine instructions to assembly language, "
    "then to high-level languages such as FORTRAN, COBOL, C, and eventually Python. "
    "Each abstraction layer traded raw efficiency for human expressiveness, broadening "
    "the population of people who could direct a machine to do useful work. The "
    "internet connected these machines into a global network, and the World Wide Web "
    "layered a navigable hypertext system on top. Search engines, social networks, "
    "streaming media, and cloud computing followed in rapid succession, transforming "
    "how knowledge is stored, retrieved, and shared across the entire planet. "
    "Artificial intelligence, once a laboratory curiosity, re-emerged as a practical "
    "discipline when deep neural networks proved capable of recognising images, "
    "transcribing speech, and translating languages at near-human accuracy. Large "
    "language models trained on vast corpora of text have more recently demonstrated "
    "the ability to generate coherent prose, write functional code, and reason through "
    "multi-step problems, raising both excitement about their potential and urgent "
    "questions about their societal implications. The semiconductor industry responded "
    "to the slowdown in transistor scaling with heterogeneous architectures: graphics "
    "processing units repurposed for deep learning, tensor processing units custom-"
    "built for matrix multiplication, and neuromorphic chips that mimic the sparse "
    "firing patterns of biological neurons. Programming languages evolved in tandem: "
    "dynamic typing lowered the barrier to entry, while static type systems and formal "
    "verification tools emerged to tame the complexity of large codebases. Version "
    "control systems, continuous integration pipelines, and containerisation platforms "
    "transformed software engineering from a solitary craft into a highly collaborative "
    "discipline practised by distributed teams spanning multiple time zones. Open-source "
    "communities demonstrated that complex, high-quality software could be built and "
    "maintained without centralised commercial control, challenging traditional notions "
    "of intellectual property and creating a shared infrastructure upon which much of "
    "the modern digital economy depends. Security researchers uncovered layer after layer "
    "of vulnerabilities in systems once thought robust, prompting an arms race between "
    "attackers exploiting newly discovered flaws and defenders patching them. "
    "Cryptography matured from a military specialty into a public discipline, with "
    "public-key encryption, digital signatures, and zero-knowledge proofs forming the "
    "backbone of electronic commerce and private communication. Quantum computing "
    "emerged from theoretical physics laboratories as a potential disruptor: quantum "
    "algorithms promise exponential speedups for certain problems, most notably "
    "factoring large integers, which underlies the security of widely deployed "
    "cryptographic schemes. Whether practical, fault-tolerant quantum computers will "
    "arrive in five years or fifty remains an open question, but the possibility has "
    "already prompted standardisation bodies to develop post-quantum cryptographic "
    "algorithms intended to remain secure even against quantum adversaries. "
) * 2   # ~1 100 tokens when repeated twice


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 5.1 — Corpus Perplexity vs Cache Budget")
print("=" * 65)
print(f"\nModel   : {MODEL_NAME}")
print(f"Budgets : {BUDGETS}")
print(f"Prefill : {PREFILL_LEN} tokens")
print(f"Eval    : {EVAL_LEN} tokens after prefill\n")

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

# Shared RoPE corrector for all corrected policies
rope_corrector = RoPECorrector(model, max_positions=2048)
print("RoPE corrector built.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Verify document length
# ─────────────────────────────────────────────────────────────────────────────
_all_ids = tokenizer(LONG_DOC, return_tensors="pt")["input_ids"][0]
doc_len  = len(_all_ids)
needed   = PREFILL_LEN + EVAL_LEN + 1
print(f"Document: {doc_len} tokens  (need ≥ {needed})")
if doc_len < needed:
    raise RuntimeError(
        f"LONG_DOC is too short ({doc_len} tokens). "
        f"Need at least {needed}. Increase LONG_DOC or reduce EVAL_LEN."
    )
print()


# ─────────────────────────────────────────────────────────────────────────────
# Policy factory — fresh instances per (name, budget)
# ─────────────────────────────────────────────────────────────────────────────

def make_policy(name: str, budget: int):
    """Return a freshly initialised policy for the given name and budget."""
    window = budget - SINK_SIZE
    if name == "no_eviction":
        return NoEvictionPolicy()
    elif name == "sliding_window":
        return SlidingWindowPolicy(window_size=budget,
                                   rope_corrector=rope_corrector)
    elif name == "streaming_llm":
        return StreamingLLMPolicy(sink_size=SINK_SIZE,
                                  window_size=window,
                                  rope_corrector=rope_corrector)
    elif name == "h2o":
        return H2OPolicy(budget=budget,
                         rope_corrector=rope_corrector)
    else:
        raise ValueError(f"Unknown policy: {name}")


POLICY_NAMES = ["no_eviction", "sliding_window", "streaming_llm", "h2o"]


# ─────────────────────────────────────────────────────────────────────────────
# Core: teacher-forcing NLL → PPL
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppl(policy, text: str, prefill_len: int, eval_len: int) -> float:
    """
    Teacher-forcing perplexity on ``text``.

    Prefill on the first ``prefill_len`` tokens; then feed each successive
    token as the input (teacher forcing) and record the NLL of the correct
    next token.  Returns PPL = exp(mean(NLL)).

    Supports both slot-based (Phase 3.4) and absolute position_ids.
    """
    policy.reset()

    inputs  = tokenizer(text, return_tensors="pt").to(device)
    all_ids = inputs["input_ids"][0]
    T       = len(all_ids)

    end_pos = min(T - 1, prefill_len + eval_len)
    if end_pos <= prefill_len:
        raise ValueError("Text too short for prefill + eval.")

    # ── Prefill ───────────────────────────────────────────────────────────────
    prefill_ids = all_ids[:prefill_len].unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids       = prefill_ids,
            use_cache       = True,
            output_attentions = True,
        )
    cache = policy.step(out.past_key_values, out.attentions)

    # ── Teacher-forcing ───────────────────────────────────────────────────────
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    total_nll   = 0.0
    n_steps     = 0

    use_slot = getattr(policy, "use_slot_positions", True)

    for t in range(prefill_len, end_pos):
        tok_in  = all_ids[t    ].unsqueeze(0).unsqueeze(0).to(device)
        tok_tgt = all_ids[t + 1].to(device)

        if use_slot:
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

        log_probs  = log_softmax(out.logits[0, -1, :])
        total_nll += -float(log_probs[tok_tgt])
        n_steps   += 1

    mean_nll = total_nll / n_steps
    return math.exp(mean_nll)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep: (policy, budget) → PPL
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("Sweeping (policy × budget) → PPL")
print("─" * 65)

results: dict[str, dict[int, float]] = {name: {} for name in POLICY_NAMES}

for name in POLICY_NAMES:
    for budget in BUDGETS:
        # no_eviction has no budget parameter — run once, same result for all
        if name == "no_eviction" and budget != BUDGETS[0]:
            results[name][budget] = results[name][BUDGETS[0]]
            continue

        print(f"  {name:<18} budget={budget:>3} … ", end="", flush=True)
        policy = make_policy(name, budget)
        ppl    = compute_ppl(policy, LONG_DOC, PREFILL_LEN, EVAL_LEN)
        results[name][budget] = ppl
        print(f"PPL = {ppl:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Console summary table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY — Perplexity (lower = better)")
print("=" * 65)
header = f"{'Policy':<20}" + "".join(f"  B={b:<4}" for b in BUDGETS)
print(header)
print("─" * len(header))
for name in POLICY_NAMES:
    row = f"{name:<20}"
    baseline = results["no_eviction"][BUDGETS[0]]
    for b in BUDGETS:
        ppl  = results[name][b]
        delta = ppl - baseline
        row += f"  {ppl:>5.2f}"
    print(row)
print()
print("  Δ vs no_eviction baseline:")
baseline_ppl = results["no_eviction"][BUDGETS[0]]
for name in POLICY_NAMES:
    if name == "no_eviction":
        continue
    row = f"  {name:<18}"
    for b in BUDGETS:
        delta = results[name][b] - baseline_ppl
        sign  = "+" if delta >= 0 else ""
        row  += f"  {sign}{delta:>+.2f}"
    print(row)

# ─────────────────────────────────────────────────────────────────────────────
# Save JSON
# ─────────────────────────────────────────────────────────────────────────────
json_out = os.path.join(RESULTS_DIR, "phase5_perplexity.json")
with open(json_out, "w") as f:
    json.dump({"budgets": BUDGETS, "ppl": results}, f, indent=2)
print(f"\nSaved raw data: {json_out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: budget vs PPL
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "no_eviction"   : dict(color="#27ae60", ls="-",  lw=2.4,
                           marker="o", ms=7,
                           label="No eviction (full cache — upper bound)"),
    "sliding_window": dict(color="#e74c3c", ls="--", lw=2.0,
                           marker="s", ms=6,
                           label="Sliding window  ← intentionally broken"),
    "streaming_llm" : dict(color="#2980b9", ls="-",  lw=2.0,
                           marker="^", ms=6,
                           label="StreamingLLM (sink + window)"),
    "h2o"           : dict(color="#e67e22", ls="-",  lw=2.0,
                           marker="D", ms=6,
                           label="H2O (heavy-hitter oracle)"),
}

fig, ax = plt.subplots(figsize=(10, 5))

for name in POLICY_NAMES:
    y_vals = [results[name][b] for b in BUDGETS]
    ax.plot(BUDGETS, y_vals, **STYLE[name])

ax.set_xlabel("Cache budget (K/V token slots)", fontsize=12)
ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
ax.set_title(
    "Corpus perplexity vs cache budget\n"
    "All policies track the full-cache baseline → PPL is a misleading metric",
    fontsize=12,
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xticks(BUDGETS)

# Annotate the paradox
ax.annotate(
    "PPL ≈ flat across all\npolicies — even the\nbroken sliding window",
    xy=(BUDGETS[len(BUDGETS) // 2],
        results["sliding_window"][BUDGETS[len(BUDGETS) // 2]]),
    xytext=(BUDGETS[1], max(results["sliding_window"].values()) + 0.3),
    fontsize=9,
    color="#e74c3c",
    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
)

plt.tight_layout()
png_out = os.path.join(RESULTS_DIR, "phase5_perplexity.png")
plt.savefig(png_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plot   : {png_out}")

print(f"\nAll Phase 5.1 outputs in: {os.path.abspath(RESULTS_DIR)}")
print("""
Interpretation
──────────────
If PPL gaps between policies are < 1.0 PPL units at budget=128, this
confirms the first half of the paradox: perplexity is insensitive to
which tokens are evicted, even when factual recall has collapsed.
Run eval_needle.py to see the other half.
""")
