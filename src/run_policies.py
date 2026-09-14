"""
run_policies.py  –  Phase 3.3 / 3.4 evaluation harness
=======================================================
Demonstrates and quantitatively compares the three KV-cache eviction
policies implemented in ``kv_cache_policies.py``, with Phase 3.4
RoPE-position correction applied by default.

Two evaluations
---------------
1.  **Qualitative — greedy generation from a memory-probing prompt**

    The prompt introduces several named characters and specific facts in the
    first 200+ tokens.  A model with intact long-range context will reproduce
    those names and facts in the continuation.  A model with a broken cache
    will produce generic, incoherent, or hallucinated text.

    Expected ordering of coherence:
        no_eviction  >  h2o  ≈  streaming_llm  >>  sliding_window

2.  **Quantitative — per-token NLL (teacher-forcing)**

    A held-out passage is tokenised.  The first ``PREFILL_LEN`` tokens are fed
    as a prefill; then each subsequent token is fed one at a time (teacher
    forcing — always the correct token, not the model's prediction).  After
    each step the eviction policy is applied and the NLL of the *correct*
    next token is recorded.  Low NLL means the model is predicting well.

    Teacher forcing isolates the effect of cache quality from generation drift:
    we ask "given the evicted cache AND the correct previous token, how well
    can the model predict the next token?"

    A smoothed NLL curve is saved to results/policy_nll_curves.png.

Position-ID correctness
-----------------------
After eviction the cache is shorter than the true sequence.  Without
correction, the model would assign the wrong RoPE position to the new query
token (it infers position from cache length).  This script passes an explicit
``position_ids`` tensor in every decode-phase call so that the query token
always lands at its correct position in the full sequence.  This makes the
comparison fair: only the *content* of the cache differs between policies,
not the query's positional encoding.

Budget
------
All three eviction policies share the same token budget (``BUDGET``).
StreamingLLM splits it as sink_size + window_size.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src/ to the path so we can import from the same directory
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

BUDGET      = 128          # total K/V budget for all eviction policies
SINK_SIZE   = 4            # StreamingLLM: sink tokens to always keep
WINDOW_SIZE = BUDGET - SINK_SIZE   # = 124; shared by SlidingWindow too

PREFILL_LEN    = 180       # tokens used as prefill in NLL evaluation
EVAL_LEN       = 250       # tokens evaluated (teacher-forcing) after prefill
MAX_NEW_TOKENS = 200       # tokens generated in qualitative evaluation
SMOOTH_WINDOW  = 25        # rolling-average window for NLL plot

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

# Memory-probing prompt: introduces characters and facts in the first ~200 t.
# A good cache policy will preserve enough context to continue the story
# coherently; sliding window will lose those details almost immediately.
MEMORY_PROBE_PROMPT = (
    "Dr. Maya Chen, a marine biologist specialising in deep-sea bioluminescence, "
    "and her colleague Professor Alistair Webb, a retired oceanographer who had "
    "spent four decades studying abyssal currents, set out on an expedition aboard "
    "the research vessel Nereid in early March. Their mission: deploy a new array "
    "of pressure sensors along the western rim of the Mariana Trench, where a "
    "series of puzzling temperature anomalies had been recorded the previous autumn. "
    "Maya had designed the sensor array herself, spending eighteen months calibrating "
    "each node in her laboratory at the Scripps Institution of Oceanography. "
    "Alistair, sceptical by nature but endlessly curious, insisted on including a "
    "secondary acoustic transponder network so that data could be cross-validated in "
    "real time. The crew of twelve worked in two shifts; the night shift was led by "
    "Chief Engineer Rosa Delgado, a taciturn woman who communicated almost entirely "
    "through precise instrument readings and brief handwritten notes left in the "
    "navigation log. By the fourth day the weather had deteriorated: a low-pressure "
    "system pushed whitecapped swells across the bow, and the Nereid rolled heavily "
    "despite her stabilisers. Maya worked through the nausea, cross-referencing "
    "incoming bathymetric data with the charts she had memorised from the preliminary "
    "survey. Late that afternoon, Chief Engineer Rosa appeared at the door of the "
    "science cabin and placed a single sheet on the desk. It read: 'Anomaly confirmed. "
    "Depth 8,400 m. Temperature delta +1.7 °C. Source unknown.' "
    "Maya looked at Alistair. Alistair looked at Maya. Neither spoke for a long "
    "moment. Then Maya reached for the intercom and said:"
)

# Passage used for the NLL evaluation (independent of the memory probe)
NLL_PASSAGE = (
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
    "Artificial intelligence, once a laboratory curiosity, re-emerged as a "
    "practical discipline when deep neural networks proved capable of recognising "
    "images, transcribing speech, and translating languages at near-human accuracy. "
    "Large language models trained on vast corpora of text have more recently "
    "demonstrated the ability to generate coherent prose, write functional code, "
    "and reason through multi-step problems, raising both excitement about their "
    "potential and urgent questions about their societal implications. "
) * 2   # repeat so we have well over PREFILL_LEN + EVAL_LEN tokens


# ─────────────────────────────────────────────────────────────────────────────
# Load model (eager attention required for output_attentions=True)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 3.3 — KV-Cache Eviction Policy Evaluation")
print("=" * 65)
print(f"\nBudget : {BUDGET} tokens  "
      f"(sink={SINK_SIZE}, window={WINDOW_SIZE})")
print(f"Model  : {MODEL_NAME}\n")

print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",  # SDPA does not materialise the attn matrix
    dtype=torch.float32,
)
model.eval()
device = next(model.parameters()).device
print("Model loaded.\n")

# Phase 3.4: build the shared RoPE correction table
rope_corrector = RoPECorrector(model, max_positions=2048)
print("RoPE corrector built.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Policy registry
# ─────────────────────────────────────────────────────────────────────────────
# All eviction policies now receive the RoPE corrector (Phase 3.4).
# NoEvictionPolicy needs no corrector: cache slot == true sequence position.
POLICIES = {
    "no_eviction"   : NoEvictionPolicy(),
    "sliding_window": SlidingWindowPolicy(window_size=WINDOW_SIZE,
                                           rope_corrector=rope_corrector),
    "streaming_llm" : StreamingLLMPolicy(sink_size=SINK_SIZE,
                                          window_size=WINDOW_SIZE,
                                          rope_corrector=rope_corrector),
    "h2o"           : H2OPolicy(budget=BUDGET,
                                 rope_corrector=rope_corrector),
}

# Visual style for plots
STYLE = {
    "no_eviction"   : dict(color="#27ae60", linestyle="-",  linewidth=2.2,
                           label="No eviction (full cache, upper bound)"),
    "sliding_window": dict(color="#e74c3c", linestyle="--", linewidth=2.2,
                           label=f"Sliding window (window={WINDOW_SIZE})"),
    "streaming_llm" : dict(color="#2980b9", linestyle="-",  linewidth=2.0,
                           label=f"StreamingLLM (sink={SINK_SIZE}+window={WINDOW_SIZE})"),
    "h2o"           : dict(color="#e67e22", linestyle="-",  linewidth=2.0,
                           label=f"H2O (budget={BUDGET})"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Core: custom generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_policy(
    policy,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    """
    Greedy autoregressive generation with a custom decode loop so we can
    intercept and evict the KV cache after every single forward pass.

    Returns
    -------
    generated_text : str
        The model's continuation of ``prompt``.
    cache_sizes : list[int]
        Number of K/V pairs in the cache at each decode step (after eviction).
    """
    policy.reset()

    # ── Prefill: process the full prompt in one forward pass ─────────────────
    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    prefill_T = inputs["input_ids"].shape[1]   # true sequence length so far

    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,            # needed by H2O; cheap for others
        )

    cache = policy.step(outputs.past_key_values, outputs.attentions)
    cache_sizes = [cache.get_seq_length()]

    # Greedy first token (predicted from the last prefill logit)
    next_tok_id = int(outputs.logits[0, -1, :].argmax())
    generated   = [next_tok_id]

    # ── Decode loop ───────────────────────────────────────────────────────────
    # Phase 3.4: if policy.use_slot_positions is True, we use SLOT-BASED 
    # position_ids so that both Q and K operate in the same contiguous 
    # 0..budget coordinate space after eviction. Otherwise (e.g. H2O), 
    # we use the absolute sequence position.
    
    actual_pos = prefill_T

    for _ in range(max_new_tokens - 1):
        tok_tensor = torch.tensor([[next_tok_id]], device=device)
        
        if getattr(policy, "use_slot_positions", True):
            pos_tensor = torch.tensor([[cache.get_seq_length()]], device=device)
        else:
            pos_tensor = torch.tensor([[actual_pos]], device=device)

        with torch.no_grad():
            outputs = model(
                input_ids       = tok_tensor,
                past_key_values = cache,
                use_cache       = True,
                output_attentions = True,
                position_ids    = pos_tensor,
            )

        cache = policy.step(outputs.past_key_values, outputs.attentions)
        cache_sizes.append(cache.get_seq_length())

        next_tok_id = int(outputs.logits[0, -1, :].argmax())
        generated.append(next_tok_id)
        actual_pos += 1

        if next_tok_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, cache_sizes


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative: teacher-forcing NLL
# ─────────────────────────────────────────────────────────────────────────────

def compute_nll_curve(
    policy,
    text: str,
    prefill_len: int,
    eval_len: int,
) -> list[float]:
    """
    Teacher-forcing NLL evaluation.

    Prefill on ``text[:prefill_len]`` tokens, then for each subsequent
    position up to ``prefill_len + eval_len`` feed the *correct* token (not
    the model's prediction) and record the NLL of the correct *next* token.
    The eviction policy is applied after every step.

    The ``position_ids`` argument ensures the query is always placed at its
    true sequence position, making the NLL comparable across policies.

    Returns
    -------
    list[float] : NLL value at each position (length ≤ eval_len).
    """
    policy.reset()

    inputs  = tokenizer(text, return_tensors="pt").to(device)
    all_ids = inputs["input_ids"][0]   # (T,)
    T       = len(all_ids)

    if T < prefill_len + 2:
        raise ValueError(
            f"Text has only {T} tokens; need at least {prefill_len + 2}."
        )

    # ── Prefill ───────────────────────────────────────────────────────────────
    prefill_ids = all_ids[:prefill_len].unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids       = prefill_ids,
            use_cache       = True,
            output_attentions = True,
        )
    cache = policy.step(outputs.past_key_values, outputs.attentions)

    # ── Teacher-forcing decode (Phase 3.4: slot-based position_ids) ───────────
    # position_ids = cache.get_seq_length() gives the next available cache slot.
    # For NoEvictionPolicy this equals the true sequence position (no gap).
    # For eviction policies with RoPE correction, it equals the budget (= the
    # slot immediately after the re-indexed cache), which is the correct
    # position in the 0..budget coordinate space.
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    nll_values  = []
    end_pos     = min(T - 1, prefill_len + eval_len)

    for t in range(prefill_len, end_pos):
        tok_in  = all_ids[t    ].unsqueeze(0).unsqueeze(0).to(device)   # (1,1)
        tok_tgt = all_ids[t + 1].to(device)                             # scalar
        
        if getattr(policy, "use_slot_positions", True):
            pos_ids = torch.tensor([[cache.get_seq_length()]], device=device)
        else:
            pos_ids = torch.tensor([[t]], device=device)

        with torch.no_grad():
            outputs = model(
                input_ids       = tok_in,
                past_key_values = cache,
                use_cache       = True,
                output_attentions = True,
                position_ids    = pos_ids,
            )
        cache = policy.step(outputs.past_key_values, outputs.attentions)

        log_probs = log_softmax(outputs.logits[0, -1, :])
        nll_values.append(-float(log_probs[tok_tgt]))

    return nll_values



# ─────────────────────────────────────────────────────────────────────────────
# Smoothing helper
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Simple centred rolling average (same length as input via 'same' padding)."""
    arr  = np.array(values, dtype=np.float32)
    kern = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kern, mode="same")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Qualitative evaluation — greedy generation
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("EVALUATION 1 — Qualitative: greedy generation")
print("─" * 65)
print(f"Prompt length : ~{len(tokenizer.encode(MEMORY_PROBE_PROMPT))} tokens")
print(f"Max new tokens: {MAX_NEW_TOKENS}\n")

all_cache_sizes = {}

for name, policy in POLICIES.items():
    print(f"  Generating with policy: {name} …", end="", flush=True)
    gen_text, sizes = generate_with_policy(policy, MEMORY_PROBE_PROMPT,
                                           MAX_NEW_TOKENS)
    all_cache_sizes[name] = sizes
    print(f" done  (cache peaked at {max(sizes)} tokens)")

    sep = "─" * 60
    print(f"\n{'─'*20}  {name}  {'─'*20}")
    print(gen_text[:800])   # first 800 chars is enough to see quality
    print()

# ── Plot: cache occupancy over generation steps ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for name, sizes in all_cache_sizes.items():
    ax.plot(sizes, **STYLE[name])
ax.axhline(BUDGET, color="#7f8c8d", linestyle=":", linewidth=1.0,
           label=f"Budget = {BUDGET}")
ax.set_xlabel("Decode step", fontsize=12)
ax.set_ylabel("K/V pairs in cache", fontsize=12)
ax.set_title("Cache occupancy during generation\n"
             "(eviction policies stay bounded; no_eviction grows without limit)",
             fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_cache = os.path.join(RESULTS_DIR, "policy_cache_sizes.png")
plt.savefig(out_cache, dpi=150)
plt.close()
print(f"\nSaved: {out_cache}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Quantitative evaluation — NLL curves (teacher forcing)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("EVALUATION 2 — Quantitative: per-token NLL (teacher forcing)")
print("─" * 65)
print(f"Prefill length : {PREFILL_LEN} tokens")
print(f"Evaluation span: {EVAL_LEN} tokens after prefill")
print(f"Smoothing window: {SMOOTH_WINDOW} tokens\n")

nll_results = {}
for name, policy in POLICIES.items():
    print(f"  Computing NLL for policy: {name} …", end="", flush=True)
    nll = compute_nll_curve(policy, NLL_PASSAGE, PREFILL_LEN, EVAL_LEN)
    nll_results[name] = nll
    mean_nll = np.mean(nll)
    print(f" done  (mean NLL = {mean_nll:.3f})")

# ── Plot: NLL curves ──────────────────────────────────────────────────────────
fig, (ax_raw, ax_smooth) = plt.subplots(1, 2, figsize=(15, 5))

x = np.arange(len(next(iter(nll_results.values()))))

for name, nll in nll_results.items():
    raw    = np.array(nll)
    smooth = rolling_mean(nll, SMOOTH_WINDOW)
    # Raw (faint)
    ax_raw.plot(x, raw, alpha=0.35, **{**STYLE[name], "linewidth": 1.0,
                                        "label": None})
    ax_raw.plot(x, smooth, **STYLE[name])   # smoothed (bold)
    ax_smooth.plot(x, smooth, **STYLE[name])

for ax, title in [
    (ax_raw,    "Per-token NLL — raw + smoothed"),
    (ax_smooth, "Per-token NLL — smoothed only"),
]:
    ax.set_xlabel(f"Token position (offset from prefill_len={PREFILL_LEN})",
                  fontsize=11)
    ax.set_ylabel("Negative log-likelihood", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"KV-cache eviction quality comparison  |  budget={BUDGET} tokens  |  "
    f"Qwen2.5-0.5B",
    fontsize=12, y=1.02,
)
plt.tight_layout()
out_nll = os.path.join(RESULTS_DIR, "policy_nll_curves.png")
plt.savefig(out_nll, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_nll}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY — Mean NLL (lower = better prediction quality)")
print("=" * 65)
baseline_nll = np.mean(nll_results["no_eviction"])
for name, nll in nll_results.items():
    mean  = np.mean(nll)
    delta = mean - baseline_nll
    sign  = "+" if delta >= 0 else ""
    print(f"  {name:<20} mean NLL = {mean:.4f}  "
          f"(Δ vs baseline: {sign}{delta:.4f})")

print(f"\nAll output files in: {os.path.abspath(RESULTS_DIR)}")
