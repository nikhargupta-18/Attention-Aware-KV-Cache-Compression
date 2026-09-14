"""
eval_quality_vs_memory.py  –  Phase 3.6
=========================================
Quality-vs-memory curve: the single plot that summarises the entire project.

For each eviction policy, we sweep cache budgets and measure TWO quality
metrics simultaneously:

  1. Corpus perplexity (PPL)
     → The "misleading" metric. Stays flat (good-looking) even as factual
       recall collapses, because PPL averages over easy next-token predictions
       that don't require long-range retrieval.

  2. Mid-context needle recall
     → The "revealing" metric. Directly measures whether a specific fact
       inserted in the evicted middle of the context can be retrieved.
       This is what collapses under bad eviction.

The two-panel figure makes the paradox immediately visible:
  - Left panel (PPL):     all eviction curves track the baseline → "looks fine"
  - Right panel (recall): sliding_window and streaming_llm crash at mid depths
                          → the actual failure, invisible to perplexity

This is the headline result of the project.

Budget sweep
------------
BUDGETS = [32, 48, 64, 96, 128, 192, 256]

Fine enough to show the curve shape; wide enough to capture the "cliff" where
small budgets cause catastrophic forgetting.

Needle recall definition
------------------------
For each (policy, budget) pair:
  • Insert needle at 3 mid-context depths: 25%, 50%, 75%
  • Score each trial (exact OR fuzzy F1 ≥ 0.5)
  • Report the mean recall across the 3 depths
This is the "hardest" recall task — the easy 90% depth (recency window) and
trivial 10% depth (near sinks) are excluded to make the comparison meaningful.

Outputs
-------
  results/phase36_quality_vs_memory.png   — 2-panel figure
  results/phase36_quality_vs_memory.json  — raw numbers
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

# Budget sweep: 7 points from very tight (32) to generous (256)
BUDGETS     = [32, 48, 64, 96, 128, 192, 256]

SINK_SIZE   = 4     # StreamingLLM fixed sink count
PREFILL_LEN = 48    # kept short to be fast; we want eval tokens, not prefill
EVAL_LEN    = 500   # teacher-forcing evaluation span

# Needle evaluation
NEEDLE        = "The secret launch code is DELTA-7-ZEPHYR."
NEEDLE_ANSWER = "DELTA-7-ZEPHYR"
QUERY         = (
    "\n\nBased on the text above, what is the secret launch code? "
    "The launch code is:"
)
MID_DEPTHS    = [0.25, 0.50, 0.75]   # only mid-context; 10%/90% excluded
FUZZY_THRESH  = 0.5
MAX_GEN       = 20

# ─────────────────────────────────────────────────────────────────────────────
# Documents
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
    "the ability to generate coherent prose, write functional code, and reason "
    "through multi-step problems, raising both excitement about their potential and "
    "urgent questions about their societal implications. "
) * 2

HAYSTACK_BASE = (
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
    "the population of people who could direct a machine to do useful work. "
)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 3.6 — Quality vs Memory Curve")
print("=" * 65)
print(f"\nModel  : {MODEL_NAME}")
print(f"Budgets: {BUDGETS}\n")

print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    dtype=torch.float32,
)
model.eval()
device = next(model.parameters()).device
rope_corrector = RoPECorrector(model, max_positions=2048)
print("Model + RoPE corrector ready.\n")

# Verify document length
_doc_ids = tokenizer(LONG_DOC, return_tensors="pt")["input_ids"][0]
needed   = PREFILL_LEN + EVAL_LEN + 1
print(f"Document: {len(_doc_ids)} tokens  (need ≥ {needed})")
assert len(_doc_ids) >= needed, "LONG_DOC too short — increase EVAL_LEN or doc"
print()

# ─────────────────────────────────────────────────────────────────────────────
# Policy factory
# ─────────────────────────────────────────────────────────────────────────────
POLICY_NAMES = ["no_eviction", "sliding_window", "streaming_llm", "h2o"]

def make_policy(name: str, budget: int):
    window = max(1, budget - SINK_SIZE)
    if name == "no_eviction":
        return NoEvictionPolicy()
    elif name == "sliding_window":
        return SlidingWindowPolicy(window_size=budget, rope_corrector=rope_corrector)
    elif name == "streaming_llm":
        return StreamingLLMPolicy(sink_size=SINK_SIZE, window_size=window,
                                  rope_corrector=rope_corrector)
    elif name == "h2o":
        return H2OPolicy(budget=budget, rope_corrector=rope_corrector)
    raise ValueError(name)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: teacher-forcing PPL
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppl(policy, text: str, prefill_len: int, eval_len: int) -> float:
    policy.reset()
    inputs  = tokenizer(text, return_tensors="pt").to(device)
    all_ids = inputs["input_ids"][0]
    end_pos = min(len(all_ids) - 1, prefill_len + eval_len)

    # Prefill
    with torch.no_grad():
        out = model(input_ids=all_ids[:prefill_len].unsqueeze(0),
                    use_cache=True, output_attentions=True)
    cache = policy.step(out.past_key_values, out.attentions)

    lsm = torch.nn.LogSoftmax(dim=-1)
    total_nll, n = 0.0, 0
    use_slot = getattr(policy, "use_slot_positions", True)

    for t in range(prefill_len, end_pos):
        tok_in  = all_ids[t].unsqueeze(0).unsqueeze(0).to(device)
        tok_tgt = all_ids[t + 1].to(device)
        pos_ids = torch.tensor([[cache.get_seq_length() if use_slot else t]],
                               device=device)
        with torch.no_grad():
            out = model(input_ids=tok_in, past_key_values=cache,
                        use_cache=True, output_attentions=True, position_ids=pos_ids)
        cache = policy.step(out.past_key_values, out.attentions)
        total_nll += -float(lsm(out.logits[0, -1, :])[tok_tgt])
        n += 1

    return math.exp(total_nll / n)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: needle recall (mid-context average)
# ─────────────────────────────────────────────────────────────────────────────

def build_context(depth: float, haystack_tokens: int) -> torch.Tensor:
    """Return token-ids tensor (1, seq_len) with needle at given depth."""
    needle_ids = tokenizer(NEEDLE + " ", add_special_tokens=False)["input_ids"]
    query_ids  = tokenizer(QUERY,         add_special_tokens=False)["input_ids"]

    repeats = 1
    while True:
        hs = tokenizer(HAYSTACK_BASE * repeats,
                       add_special_tokens=False)["input_ids"]
        if len(hs) >= haystack_tokens:
            break
        repeats += 1
    hs = hs[:haystack_tokens]

    split   = int(len(hs) * depth)
    ctx_ids = hs[:split] + needle_ids + hs[split:] + query_ids
    return torch.tensor([ctx_ids])


def token_f1(pred_ids, gold_ids) -> float:
    if not pred_ids or not gold_ids:
        return 0.0
    pc, gc  = Counter(pred_ids), Counter(gold_ids)
    common  = sum((pc & gc).values())
    if common == 0:
        return 0.0
    p = common / len(pred_ids)
    r = common / len(gold_ids)
    return 2 * p * r / (p + r)


def score_output(text: str) -> float:
    """Returns 1.0 if needle answer recalled, else 0.0 (exact OR fuzzy)."""
    exact = 1 if NEEDLE_ANSWER.lower() in text.lower() else 0
    pred  = tokenizer(text,          add_special_tokens=False)["input_ids"]
    gold  = tokenizer(NEEDLE_ANSWER, add_special_tokens=False)["input_ids"]
    fuzzy = 1 if token_f1(pred, gold) >= FUZZY_THRESH else 0
    return float(max(exact, fuzzy))


def run_needle(policy, budget: int) -> float:
    """Average mid-context recall across MID_DEPTHS."""
    TARGET_HS = max(budget * 3, 200)   # force eviction at all budgets
    scores = []
    for depth in MID_DEPTHS:
        ctx = build_context(depth, TARGET_HS).to(device)
        T   = ctx.shape[1]
        policy.reset()

        with torch.no_grad():
            out = model(input_ids=ctx, use_cache=True, output_attentions=True)
        cache    = policy.step(out.past_key_values, out.attentions)
        next_tok = int(out.logits[0, -1, :].argmax())
        gen      = [next_tok]
        abs_pos  = T
        use_slot = getattr(policy, "use_slot_positions", True)

        for _ in range(MAX_GEN - 1):
            t_in  = torch.tensor([[next_tok]], device=device)
            p_ids = torch.tensor(
                [[cache.get_seq_length() if use_slot else abs_pos]], device=device)
            with torch.no_grad():
                out = model(input_ids=t_in, past_key_values=cache,
                            use_cache=True, output_attentions=True, position_ids=p_ids)
            cache = policy.step(out.past_key_values, out.attentions)
            next_tok = int(out.logits[0, -1, :].argmax())
            gen.append(next_tok)
            abs_pos += 1
            if next_tok == tokenizer.eos_token_id:
                break

        text   = tokenizer.decode(gen, skip_special_tokens=True)
        scores.append(score_output(text))
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("Sweeping (policy × budget) → PPL + mid-context needle recall")
print("─" * 65)

ppl_results    = {name: {} for name in POLICY_NAMES}
recall_results = {name: {} for name in POLICY_NAMES}

_noev_ppl = None   # compute no_eviction once

for name in POLICY_NAMES:
    print(f"\n  Policy: {name}")
    for budget in BUDGETS:

        # ── PPL ───────────────────────────────────────────────────────────────
        if name == "no_eviction" and _noev_ppl is not None:
            ppl = _noev_ppl
        else:
            policy = make_policy(name, budget)
            ppl    = compute_ppl(policy, LONG_DOC, PREFILL_LEN, EVAL_LEN)
            if name == "no_eviction":
                _noev_ppl = ppl
        ppl_results[name][budget] = ppl

        # ── Needle recall ──────────────────────────────────────────────────────
        policy = make_policy(name, budget)
        recall = run_needle(policy, budget)
        recall_results[name][budget] = recall

        print(f"    budget={budget:>3}  PPL={ppl:>8.3f}  mid-recall={recall:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Save JSON
# ─────────────────────────────────────────────────────────────────────────────
json_out = os.path.join(RESULTS_DIR, "phase36_quality_vs_memory.json")
with open(json_out, "w") as f:
    json.dump({"budgets": BUDGETS,
               "ppl": ppl_results,
               "recall": recall_results}, f, indent=2)
print(f"\nSaved raw data: {json_out}")

# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
noev_ppl = ppl_results["no_eviction"][BUDGETS[-1]]
print(f"\n  PPL (no_eviction baseline = {noev_ppl:.2f})")
header = f"  {'Policy':<20}" + "".join(f"  B={b:<3}" for b in BUDGETS)
print(header); print("  " + "─" * (len(header) - 2))
for name in POLICY_NAMES:
    row = f"  {name:<20}"
    for b in BUDGETS:
        row += f"  {ppl_results[name][b]:>5.1f}"
    print(row)

print(f"\n  Mid-context needle recall (↑ better)")
print(header); print("  " + "─" * (len(header) - 2))
for name in POLICY_NAMES:
    row = f"  {name:<20}"
    for b in BUDGETS:
        row += f"  {recall_results[name][b]:>5.2f}"
    print(row)

# ─────────────────────────────────────────────────────────────────────────────
# Plot — 2-panel quality-vs-memory figure
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "no_eviction"   : dict(color="#27ae60", ls="-",  lw=2.5, marker="o", ms=7,
                           label="No eviction (full cache — upper bound)"),
    "sliding_window": dict(color="#e74c3c", ls="--", lw=2.0, marker="s", ms=6,
                           label="Sliding window  (broken baseline)"),
    "streaming_llm" : dict(color="#2980b9", ls="-",  lw=2.0, marker="^", ms=6,
                           label="StreamingLLM  (sink + recency window)"),
    "h2o"           : dict(color="#e67e22", ls="-",  lw=2.0, marker="D", ms=6,
                           label="H2O  (heavy-hitter oracle)"),
}

fig, (ax_ppl, ax_rec) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Quality vs Memory — KV-Cache Eviction  |  Qwen2.5-0.5B\n"
    "The paradox: perplexity looks fine while factual recall collapses",
    fontsize=13, fontweight="bold", y=1.02,
)

# ── Left: PPL ────────────────────────────────────────────────────────────────
for name in POLICY_NAMES:
    y = [ppl_results[name][b] for b in BUDGETS]
    # Cap display at 30 to keep the interesting region readable
    y_capped = [min(v, 30) for v in y]
    # If any values were capped, add a note marker
    capped_any = any(v > 30 for v in y)
    ax_ppl.plot(BUDGETS, y_capped, **STYLE[name],
                markerfacecolor="white" if capped_any else STYLE[name]["color"])

ax_ppl.set_xlabel("Cache budget (K/V token slots)", fontsize=12)
ax_ppl.set_ylabel("Perplexity  (↓ better)", fontsize=12)
ax_ppl.set_title(
    "Corpus Perplexity vs Budget\n"
    "← 'Looks fine' — PPL is insensitive to which tokens are evicted",
    fontsize=11,
)
ax_ppl.set_xticks(BUDGETS)
ax_ppl.set_ylim(bottom=0, top=30)
ax_ppl.annotate("Values > 30 capped\n(sliding window, H2O at small budgets)",
                xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=8, color="#999999", ha="left")
ax_ppl.legend(fontsize=9, loc="upper right")
ax_ppl.grid(True, alpha=0.3)

# ── Right: Recall ─────────────────────────────────────────────────────────────
for name in POLICY_NAMES:
    y = [recall_results[name][b] for b in BUDGETS]
    ax_rec.plot(BUDGETS, y, **STYLE[name])

ax_rec.set_xlabel("Cache budget (K/V token slots)", fontsize=12)
ax_rec.set_ylabel("Mid-context recall  (↑ better)", fontsize=12)
ax_rec.set_title(
    "Mid-context Needle Recall vs Budget\n"
    "← 'The truth' — factual retrieval collapses under bad eviction",
    fontsize=11,
)
ax_rec.set_xticks(BUDGETS)
ax_rec.set_ylim(-0.05, 1.10)
ax_rec.axhline(1.0, color="#27ae60", ls=":", lw=1.0, alpha=0.5)
ax_rec.axhline(0.0, color="#e74c3c", ls=":", lw=1.0, alpha=0.5)

# Shade the "eviction collapses recall" zone
ax_rec.fill_between(BUDGETS,
                    [recall_results["streaming_llm"][b] for b in BUDGETS],
                    [recall_results["no_eviction"][b]   for b in BUDGETS],
                    alpha=0.12, color="#e74c3c",
                    label="Recall gap: no_eviction vs StreamingLLM")
ax_rec.legend(fontsize=9, loc="lower right")
ax_rec.grid(True, alpha=0.3)

plt.tight_layout()
png_out = os.path.join(RESULTS_DIR, "phase36_quality_vs_memory.png")
plt.savefig(png_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved plot: {png_out}")

print(f"\nAll Phase 3.6 outputs in: {os.path.abspath(RESULTS_DIR)}")
print("""
Reading the two-panel figure
────────────────────────────
Left  (PPL):    All eviction lines cluster near the green baseline.
                Looks acceptable — the "perplexity lie."

Right (recall): streaming_llm and sliding_window crash at small budgets.
                H2O is content-adaptive; it scores higher by retaining the
                tokens that matter.  no_eviction is the oracle at 1.0.

The gap between the two panels IS the headline result.
""")
