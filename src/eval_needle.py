"""
eval_needle.py  –  Phase 5.2
==============================
Needle-in-a-haystack evaluation: confirms the second half of the paradox.

    Factual recall collapses under bad eviction — invisible in perplexity.

Design (Greg Kamradt-style, scaled for a 0.5B model)
------------------------------------------------------
A short, unambiguous "needle" fact is inserted at varying *depths* into a
neutral "haystack" passage.  The model is then prompted to retrieve the fact.
We score whether it succeeds.

Key variables
─────────────
• Depth  : position of the needle as a fraction of haystack length.
           10% → near the beginning (likely inside the attention-sink region).
           50% → the evicted middle — this is where retrieval collapses first.
           90% → near the end (in the recency window — all policies should pass).
• Budget : how many K/V pairs the eviction policy retains.
• Policy : sliding_window (broken), streaming_llm, h2o, no_eviction (oracle).

Expected pattern
────────────────
  No eviction : recall ≈ 100% at every depth and budget.
  StreamingLLM: recall ≈ 100% at depths ≤ SINK_SIZE/budget fraction (sinks
                retained), ≈ 100% at depth → 100% (recency window).
                Collapses at mid depths — the "evicted middle".
  H2O         : better than StreamingLLM at mid depths (content-aware keeps
                frequently-attended tokens), but degrades at small budgets.
  Sliding win : collapses at all but the most recent depths — the worst case.

Scoring
───────
• Exact match : generated text contains the answer string verbatim.
• Fuzzy match : token-level F1 ≥ FUZZY_THRESHOLD against answer tokens
                (tolerates minor tokenisation splits).
• Combined score : max(exact, fuzzy).  Both are reported in the JSON.

Output
──────
  results/phase5_needle_heatmap.png   — 4-panel grid of heatmaps
  results/phase5_needle_results.json  — raw scores
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
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

BUDGETS        = [64, 96, 128]           # cache token budgets to sweep
DEPTHS         = [0.10, 0.25, 0.50, 0.75, 0.90]  # needle position fractions
SINK_SIZE      = 4                       # StreamingLLM fixed sink count
MAX_NEW_TOKENS = 20                      # tokens to generate for the answer
FUZZY_THRESHOLD = 0.5                    # token F1 threshold for fuzzy match

# The needle: a short, unmistakable fact the model cannot hallucinate correctly
NEEDLE        = "The secret launch code is DELTA-7-ZEPHYR."
NEEDLE_ANSWER = "DELTA-7-ZEPHYR"        # the token(s) we look for in output

# Query appended after the full context
QUERY = (
    "\n\nBased on the text above, what is the secret launch code? "
    "The launch code is:"
)

# Neutral filler haystack — semantically unrelated to the needle so the model
# cannot guess the answer; long enough to force real eviction at small budgets.
HAYSTACK_BASE = (
    "The history of computing is inseparable from the history of mathematics. "
    "Charles Babbage conceived the Analytical Engine in the 1830s, a mechanical "
    "device that prefigured the modern stored-program computer. Ada Lovelace, "
    "working from Babbage's notes, wrote what is widely considered the first "
    "algorithm intended to be processed by such a machine. A century later, "
    "Alan Turing formalised the notion of computation with his abstract Turing "
    "machine, proving that certain problems — the halting problem chief among them — "
    "are fundamentally undecidable. John von Neumann's architecture, proposed in "
    "1945, organised memory, a processing unit, and input/output into a coherent "
    "whole that remains the template for virtually every general-purpose computer "
    "built since. The transistor, invented at Bell Labs in 1947, replaced bulky "
    "vacuum tubes and enabled the miniaturisation that would eventually produce "
    "the integrated circuit, the microprocessor, and the pocket-sized devices that "
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
    "multi-step problems. "
)


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 5.2 — Needle-in-a-Haystack Evaluation")
print("=" * 65)
print(f"\nModel       : {MODEL_NAME}")
print(f"Needle      : '{NEEDLE}'")
print(f"Answer key  : '{NEEDLE_ANSWER}'")
print(f"Depths      : {DEPTHS}")
print(f"Budgets     : {BUDGETS}\n")

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

rope_corrector = RoPECorrector(model, max_positions=2048)
print("RoPE corrector built.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Policy factory — fresh per (name, budget)
# ─────────────────────────────────────────────────────────────────────────────

def make_policy(name: str, budget: int):
    window = max(1, budget - SINK_SIZE)
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
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context_tokens(depth: float, target_haystack_tokens: int) -> torch.Tensor:
    """
    Build a token sequence:
      [haystack_prefix] [NEEDLE_SENTENCE] [haystack_suffix] [QUERY]

    The needle is inserted at ``depth`` fraction of ``target_haystack_tokens``.

    Returns
    -------
    torch.Tensor  shape (1, seq_len)  on CPU
    """
    # Tokenise the building blocks
    needle_tok  = tokenizer(NEEDLE  + " ", add_special_tokens=False)["input_ids"]
    query_tok   = tokenizer(QUERY,         add_special_tokens=False)["input_ids"]

    # Build haystack token sequence by repeating HAYSTACK_BASE until we have
    # enough tokens, then slice to exactly target_haystack_tokens.
    hs_repeats = 1
    while True:
        hs_tok = tokenizer(
            HAYSTACK_BASE * hs_repeats,
            add_special_tokens=False,
        )["input_ids"]
        if len(hs_tok) >= target_haystack_tokens:
            break
        hs_repeats += 1

    hs_tok = hs_tok[:target_haystack_tokens]
    split  = int(len(hs_tok) * depth)

    context_ids = hs_tok[:split] + needle_tok + hs_tok[split:] + query_tok
    return torch.tensor([context_ids])


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def token_f1(pred_ids: list[int], gold_ids: list[int]) -> float:
    """Token-level F1 between two token-ID sequences."""
    if not pred_ids or not gold_ids:
        return 0.0
    pred_c = Counter(pred_ids)
    gold_c = Counter(gold_ids)
    common = sum((pred_c & gold_c).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_ids)
    recall    = common / len(gold_ids)
    return 2 * precision * recall / (precision + recall)


def score_output(generated_text: str) -> dict:
    """
    Score a generated answer string against NEEDLE_ANSWER.

    Returns dict with keys:
      exact  : 1 if NEEDLE_ANSWER appears verbatim in generated_text, else 0
      fuzzy  : token-level F1 against NEEDLE_ANSWER tokens
      combined : max(exact, 1 if fuzzy >= FUZZY_THRESHOLD else 0)
    """
    exact = 1 if NEEDLE_ANSWER.lower() in generated_text.lower() else 0

    pred_ids = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
    gold_ids = tokenizer(NEEDLE_ANSWER,  add_special_tokens=False)["input_ids"]
    f1       = token_f1(pred_ids, gold_ids)
    fuzzy    = 1 if f1 >= FUZZY_THRESHOLD else 0

    return {
        "exact"   : exact,
        "fuzzy"   : fuzzy,
        "f1"      : round(f1, 4),
        "combined": max(exact, fuzzy),
        "text"    : generated_text.strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_needle_trial(
    policy,
    context_ids: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """
    Prefill ``context_ids`` under ``policy``, then generate ``max_new_tokens``
    greedily.  Returns the generated string (continuation only).
    """
    policy.reset()
    ctx = context_ids.to(device)
    T   = ctx.shape[1]

    # ── Prefill ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(
            input_ids       = ctx,
            use_cache       = True,
            output_attentions = True,
        )
    cache = policy.step(out.past_key_values, out.attentions)

    # ── Greedy decode ─────────────────────────────────────────────────────────
    next_tok = int(out.logits[0, -1, :].argmax())
    generated = [next_tok]
    actual_pos = T   # absolute position of the last prefill token

    use_slot = getattr(policy, "use_slot_positions", True)

    for _ in range(max_new_tokens - 1):
        tok_tensor = torch.tensor([[next_tok]], device=device)

        if use_slot:
            pos_tensor = torch.tensor([[cache.get_seq_length()]], device=device)
        else:
            pos_tensor = torch.tensor([[actual_pos]], device=device)

        with torch.no_grad():
            out = model(
                input_ids       = tok_tensor,
                past_key_values = cache,
                use_cache       = True,
                output_attentions = True,
                position_ids    = pos_tensor,
            )
        cache = policy.step(out.past_key_values, out.attentions)

        next_tok = int(out.logits[0, -1, :].argmax())
        generated.append(next_tok)
        actual_pos += 1

        if next_tok == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Determine haystack size: aim for context ~ max(BUDGETS) * 3 tokens so that
# eviction is forced at all budget levels.
# ─────────────────────────────────────────────────────────────────────────────
TARGET_HAYSTACK = max(BUDGETS) * 3   # ~384 tokens at budget=128
print(f"Target haystack size: {TARGET_HAYSTACK} tokens")
# Verify with a quick tokenisation
_test_ctx = build_context_tokens(0.5, TARGET_HAYSTACK)
print(f"Full context at depth=50%: {_test_ctx.shape[1]} tokens\n")

# ─────────────────────────────────────────────────────────────────────────────
# Run sweep
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("Running needle-in-a-haystack sweep …")
print("─" * 65)

# results[policy_name][budget][depth_str] = score_dict
all_results: dict = {name: {b: {} for b in BUDGETS} for name in POLICY_NAMES}

for policy_name in POLICY_NAMES:
    print(f"\n  Policy: {policy_name}")
    for budget in BUDGETS:
        for depth in DEPTHS:
            ctx_ids = build_context_tokens(depth, TARGET_HAYSTACK)
            policy  = make_policy(policy_name, budget)

            gen_text = run_needle_trial(policy, ctx_ids, MAX_NEW_TOKENS)
            scores   = score_output(gen_text)

            depth_key = f"{int(depth * 100)}%"
            all_results[policy_name][budget][depth_key] = scores

            symbol = "✅" if scores["combined"] else "❌"
            print(
                f"    budget={budget:>3}  depth={depth_key:>4}  "
                f"{symbol}  F1={scores['f1']:.2f}  → \"{gen_text.strip()[:50]}\""
            )


# ─────────────────────────────────────────────────────────────────────────────
# Console summary table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY — Combined recall (exact OR fuzzy F1 ≥ 0.5)")
print("=" * 65)

depth_keys = [f"{int(d*100)}%" for d in DEPTHS]

for budget in BUDGETS:
    print(f"\n  Budget = {budget}")
    header = f"  {'Policy':<20}" + "".join(f"  {dk:>5}" for dk in depth_keys)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name in POLICY_NAMES:
        row = f"  {name:<20}"
        for dk in depth_keys:
            score = all_results[name][budget][dk]["combined"]
            row  += f"  {'✅' if score else '❌'}   "
        print(row)

# ─────────────────────────────────────────────────────────────────────────────
# Save JSON
# ─────────────────────────────────────────────────────────────────────────────
json_out = os.path.join(RESULTS_DIR, "phase5_needle_results.json")
with open(json_out, "w") as f:
    json.dump({"budgets": BUDGETS, "depths": DEPTHS, "results": all_results},
              f, indent=2)
print(f"\nSaved raw data: {json_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: 4-panel heatmap (one per policy)
# ─────────────────────────────────────────────────────────────────────────────
POLICY_LABELS = {
    "no_eviction"   : "No eviction\n(full cache — oracle)",
    "sliding_window": "Sliding window\n(broken baseline)",
    "streaming_llm" : "StreamingLLM\n(sink + recency window)",
    "h2o"           : "H2O\n(heavy-hitter oracle)",
}

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    "Needle-in-a-Haystack Recall\n"
    f"Needle: '{NEEDLE_ANSWER}'  |  Qwen2.5-0.5B\n"
    f"Green = recalled · Red = lost   (combined: exact OR fuzzy F1 ≥ {FUZZY_THRESHOLD})",
    fontsize=11, y=1.04,
)

depth_labels  = [f"{int(d*100)}%" for d in DEPTHS]
budget_labels = [str(b) for b in BUDGETS]

for ax, name in zip(axes, POLICY_NAMES):
    # Build recall matrix: rows = budgets (y), cols = depths (x)
    matrix = np.zeros((len(BUDGETS), len(DEPTHS)), dtype=float)
    for bi, budget in enumerate(BUDGETS):
        for di, dk in enumerate(depth_labels):
            matrix[bi, di] = all_results[name][budget][dk]["combined"]

    im = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto",
                   cmap="RdYlGn", interpolation="nearest")

    # Annotate cells with HIT / MISS (ASCII — emoji glyphs missing in DejaVu Sans)
    for bi in range(len(BUDGETS)):
        for di in range(len(DEPTHS)):
            val   = matrix[bi, di]
            text  = "HIT"  if val >= 0.5 else "MISS"
            color = "white" if val < 0.5 else "black"
            ax.text(di, bi, text, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels(depth_labels, fontsize=9)
    ax.set_yticks(range(len(BUDGETS)))
    ax.set_yticklabels(budget_labels, fontsize=9)
    ax.set_xlabel("Needle depth", fontsize=10)
    ax.set_ylabel("Cache budget (tokens)", fontsize=10)
    ax.set_title(POLICY_LABELS[name], fontsize=10, fontweight="bold")

# Shared colour bar
cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="Recall score")
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["0  (missed)", "1  (correct)"])

plt.tight_layout()
png_out = os.path.join(RESULTS_DIR, "phase5_needle_heatmap.png")
plt.savefig(png_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved heatmap : {png_out}")

# ─────────────────────────────────────────────────────────────────────────────
# Paradox confirmation printout
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("PARADOX CONFIRMATION")
print("=" * 65)
print()

# Compute mean recall at "evicted middle" depths (25%–75%)
middle_depths = ["25%", "50%", "75%"]
print("  Mean recall at mid-context depths (25%–75%):")
for name in POLICY_NAMES:
    recalls = []
    for b in BUDGETS:
        for dk in middle_depths:
            recalls.append(all_results[name][b][dk]["combined"])
    mean_recall = np.mean(recalls)
    print(f"    {name:<20}  mid-recall = {mean_recall:.2f}")

print()
print("  Check eval_perplexity.py results — if PPL gaps are < 1.0 nats")
print("  but mid-recall gaps are > 0.3, the paradox is confirmed:")
print("  perplexity ≈ flat  ·  needle retrieval collapses  ← headline result")
print()
print(f"All Phase 5.2 outputs in: {os.path.abspath(RESULTS_DIR)}")
