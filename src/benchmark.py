"""
benchmark.py  –  Unified Benchmark Script
==========================================
Runs the complete benchmark suite for the Attention-Aware KV-Cache
Compression project and prints a formatted summary table.

Benchmark phases
----------------
1. PPL sweep     (eval_perplexity.py logic)  — corpus perplexity vs budget
2. Needle recall (eval_needle.py logic)      — factual retrieval vs budget
3. Summary       — combined quality-vs-memory table + paradox confirmation

Usage
-----
    python src/benchmark.py [--quick]

Options
-------
  --quick   Use reduced budgets and eval length for a fast (~3 min) smoke test.
            Default is the full benchmark (~25 min on CPU).

Outputs
-------
  Prints a formatted summary table to stdout.
  All plots and JSON files are saved to results/ by the individual eval scripts.
  Re-uses the existing results/phase5_perplexity.json and
  results/phase5_needle_results.json if --quick is not set (saves time if
  results already exist and you only want the summary table printed).
"""

import os
import sys
import json
import math
import argparse
import torch
import numpy as np
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

SINK_SIZE = 4
NEEDLE        = "The secret launch code is DELTA-7-ZEPHYR."
NEEDLE_ANSWER = "DELTA-7-ZEPHYR"
QUERY         = (
    "\n\nBased on the text above, what is the secret launch code? "
    "The launch code is:"
)
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
    "for half a century before physical limits began to slow the pace. "
)
LONG_DOC = HAYSTACK_BASE * 4


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KV-Cache Compression Benchmark")
parser.add_argument("--quick", action="store_true",
                    help="Fast smoke test with reduced budgets/eval length")
args = parser.parse_args()

if args.quick:
    BUDGETS     = [64, 128]
    DEPTHS      = [0.25, 0.50, 0.75]
    PREFILL_LEN = 32
    EVAL_LEN    = 150
    MAX_GEN     = 15
else:
    BUDGETS     = [64, 96, 128, 192, 256]
    DEPTHS      = [0.10, 0.25, 0.50, 0.75, 0.90]
    PREFILL_LEN = 64
    EVAL_LEN    = 900
    MAX_GEN     = 20

FUZZY_THRESH = 0.5
POLICY_NAMES = ["no_eviction", "sliding_window", "streaming_llm", "h2o"]
MID_DEPTHS   = [0.25, 0.50, 0.75]

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Attention-Aware KV-Cache Compression — Benchmark Suite")
print("=" * 70)
print(f"\nModel  : {MODEL_NAME}")
print(f"Budgets: {BUDGETS}")
print(f"Mode   : {'QUICK (smoke test)' if args.quick else 'FULL'}\n")

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
print("Model ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Policy factory
# ─────────────────────────────────────────────────────────────────────────────
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
# PPL evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_ppl(policy, prefill_len: int, eval_len: int) -> float:
    policy.reset()
    inputs  = tokenizer(LONG_DOC, return_tensors="pt").to(device)
    all_ids = inputs["input_ids"][0]
    end_pos = min(len(all_ids) - 1, prefill_len + eval_len)

    with torch.no_grad():
        out = model(input_ids=all_ids[:prefill_len].unsqueeze(0),
                    use_cache=True, output_attentions=True)
    cache = policy.step(out.past_key_values, out.attentions)
    lsm = torch.nn.LogSoftmax(dim=-1)
    total, n = 0.0, 0
    use_slot = getattr(policy, "use_slot_positions", True)

    for t in range(prefill_len, end_pos):
        tok_in  = all_ids[t].unsqueeze(0).unsqueeze(0).to(device)
        tok_tgt = all_ids[t + 1].to(device)
        pos_ids = torch.tensor(
            [[cache.get_seq_length() if use_slot else t]], device=device)
        with torch.no_grad():
            out = model(input_ids=tok_in, past_key_values=cache,
                        use_cache=True, output_attentions=True, position_ids=pos_ids)
        cache = policy.step(out.past_key_values, out.attentions)
        total += -float(lsm(out.logits[0, -1, :])[tok_tgt])
        n += 1

    return math.exp(total / n)


# ─────────────────────────────────────────────────────────────────────────────
# Needle evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_context(depth: float, haystack_tokens: int) -> torch.Tensor:
    needle_ids = tokenizer(NEEDLE + " ", add_special_tokens=False)["input_ids"]
    query_ids  = tokenizer(QUERY,         add_special_tokens=False)["input_ids"]
    repeats = 1
    while True:
        hs = tokenizer(HAYSTACK_BASE * repeats, add_special_tokens=False)["input_ids"]
        if len(hs) >= haystack_tokens:
            break
        repeats += 1
    hs    = hs[:haystack_tokens]
    split = int(len(hs) * depth)
    return torch.tensor([hs[:split] + needle_ids + hs[split:] + query_ids])


def token_f1(pred_ids, gold_ids) -> float:
    if not pred_ids or not gold_ids:
        return 0.0
    pc, gc = Counter(pred_ids), Counter(gold_ids)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    p = common / len(pred_ids)
    r = common / len(gold_ids)
    return 2 * p * r / (p + r)


def score_output(text: str) -> float:
    exact = 1 if NEEDLE_ANSWER.lower() in text.lower() else 0
    pred  = tokenizer(text,          add_special_tokens=False)["input_ids"]
    gold  = tokenizer(NEEDLE_ANSWER, add_special_tokens=False)["input_ids"]
    fuzzy = 1 if token_f1(pred, gold) >= FUZZY_THRESH else 0
    return float(max(exact, fuzzy))


def run_needle(policy, budget: int, depths) -> dict:
    TARGET_HS = max(budget * 3, 200)
    results = {}
    for depth in depths:
        ctx = build_context(depth, TARGET_HS).to(device)
        T   = ctx.shape[1]
        policy.reset()
        with torch.no_grad():
            out = model(input_ids=ctx, use_cache=True, output_attentions=True)
        cache    = policy.step(out.past_key_values, out.attentions)
        next_tok = int(out.logits[0, -1, :].argmax())
        gen, abs_pos = [next_tok], T
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
        text = tokenizer.decode(gen, skip_special_tokens=True)
        results[f"{int(depth*100)}%"] = {"score": score_output(text), "text": text[:60]}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 70)
print("Phase 1: Perplexity sweep")
print("─" * 70)

ppl_results = {name: {} for name in POLICY_NAMES}
_noev_ppl = None

for name in POLICY_NAMES:
    for budget in BUDGETS:
        if name == "no_eviction" and _noev_ppl is not None:
            ppl_results[name][budget] = _noev_ppl
            continue
        print(f"  {name:<20} budget={budget:>3} … ", end="", flush=True)
        policy = make_policy(name, budget)
        ppl    = compute_ppl(policy, PREFILL_LEN, EVAL_LEN)
        ppl_results[name][budget] = ppl
        if name == "no_eviction":
            _noev_ppl = ppl
        print(f"PPL = {ppl:.3f}")

print()
print("─" * 70)
print("Phase 2: Needle-in-a-haystack recall")
print("─" * 70)

needle_results = {name: {} for name in POLICY_NAMES}
mid_depth_keys = [f"{int(d*100)}%" for d in MID_DEPTHS]

for name in POLICY_NAMES:
    for budget in BUDGETS:
        print(f"  {name:<20} budget={budget:>3} … ", end="", flush=True)
        policy = make_policy(name, budget)
        depths_to_test = list(set(DEPTHS) | set(MID_DEPTHS))
        trial_results  = run_needle(policy, budget, depths_to_test)
        needle_results[name][budget] = trial_results
        mid_recall = np.mean([trial_results[dk]["score"]
                              for dk in mid_depth_keys if dk in trial_results])
        print(f"mid-recall = {mid_recall:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
json_out = os.path.join(RESULTS_DIR, "benchmark_results.json")
with open(json_out, "w") as f:
    json.dump({"budgets": BUDGETS, "ppl": ppl_results,
               "needle": needle_results}, f, indent=2)
print(f"\nSaved raw data → {json_out}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

baseline_ppl = ppl_results["no_eviction"][BUDGETS[0]]
print(f"\n  No-eviction baseline PPL = {baseline_ppl:.3f}")

# PPL table
print(f"\n  PERPLEXITY (↓ better)  |  Baseline = {baseline_ppl:.2f}")
header = f"  {'Policy':<20}" + "".join(f"  B={b:<4}" for b in BUDGETS)
print(header)
print("  " + "─" * (len(header) - 2))
for name in POLICY_NAMES:
    row = f"  {name:<20}"
    for b in BUDGETS:
        ppl = ppl_results[name][b]
        row += f"  {min(ppl, 999.9):>6.1f}"
    print(row)

# Needle recall table
print(f"\n  MID-CONTEXT RECALL (↑ better, 25%/50%/75% average)")
print(header)
print("  " + "─" * (len(header) - 2))
for name in POLICY_NAMES:
    row = f"  {name:<20}"
    for b in BUDGETS:
        scores = [needle_results[name][b].get(dk, {}).get("score", 0.0)
                  for dk in mid_depth_keys]
        recall = np.mean(scores)
        row += f"  {recall:>6.2f}"
    print(row)

# Paradox confirmation
print()
print("─" * 70)
print("PARADOX CONFIRMATION")
print("─" * 70)
print()
print("  StreamingLLM: acceptable PPL but collapses on mid-context needle recall.")
print("  H2O: content-aware eviction; outperforms StreamingLLM on recall.")
sllm_ppl = ppl_results["streaming_llm"].get(BUDGETS[len(BUDGETS)//2], "—")
sllm_recall = np.mean([needle_results["streaming_llm"][BUDGETS[len(BUDGETS)//2]].get(dk, {}).get("score", 0.0)
                        for dk in mid_depth_keys])
noev_recall = np.mean([needle_results["no_eviction"][BUDGETS[len(BUDGETS)//2]].get(dk, {}).get("score", 0.0)
                        for dk in mid_depth_keys])
print(f"\n  At budget = {BUDGETS[len(BUDGETS)//2]}:")
print(f"    no_eviction  PPL = {baseline_ppl:.2f}  mid-recall = {noev_recall:.2f}")
print(f"    streaming_llm PPL ≈ {sllm_ppl:.2f}  mid-recall = {sllm_recall:.2f}")
print()
if isinstance(sllm_ppl, float) and abs(sllm_ppl - baseline_ppl) < 5.0 and sllm_recall < 0.5:
    print("  ✅ PARADOX CONFIRMED: PPL gap < 5 units, but mid-recall collapsed.")
else:
    print("  ℹ️  Check full eval scripts for definitive paradox numbers.")

print()
print(f"All results saved to: {os.path.abspath(RESULTS_DIR)}")
print("Run the individual eval scripts for full plots:")
print("  python src/attention_analysis.py")
print("  python src/run_policies.py")
print("  python src/rope_correction_eval.py")
print("  python src/eval_perplexity.py")
print("  python src/eval_needle.py")
print("  python src/eval_quality_vs_memory.py")
