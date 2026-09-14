"""
Phase 2 – Attention Instrumentation
=====================================

Goal: Empirically show two well-known-but-rarely-visualised properties of
      transformer attention in a real open-weight model:

  1. **Score concentration** – most of the probability mass in every attention
     row is carried by just a handful of key positions.  The rest of the row
     is near zero.  This is the theoretical motivation for KV-cache eviction:
     if a key barely receives any attention weight, its cached K/V pair is
     "wasted" memory.

  2. **Initial-token sink** – tokens at the very beginning of the sequence
     (often BOS or the first few content tokens) systematically accumulate a
     disproportionate share of the attention mass regardless of what those
     tokens actually say.  This was named the "attention sink" phenomenon in
     the StreamingLLM paper (Xiao et al., 2023) and is now a foundational
     observation that justifies *always* keeping the first K tokens in a
     compressed KV cache.

We visualise these properties through five complementary plots that together
build an airtight empirical case.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless – write PNGs, do not open a GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-0.5B"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# We deliberately mix two semantically distinct passages so we can check that
# the sink phenomenon is *content-agnostic*.  If sinks only appeared in one
# passage we might suspect it is an artefact of that content.
PASSAGES = {
    "repetitive": "Hello world. " * 60,          # ~120 tokens – repetitive, low entropy
    "narrative" : (
        "The quick brown fox jumps over the lazy dog. "
        "In a distant galaxy far beyond the Milky Way, explorers discovered "
        "ruins of an ancient civilisation. Scientists debated the origin of "
        "life for centuries. The mountain stood silent as the storm passed. "
        "She read the letter twice before folding it carefully and placing it "
        "back inside the worn envelope. The algorithm ran in O(n log n) time. "
        "Water molecules form hydrogen bonds that give it unusual properties. "
        "The orchestra tuned their instruments before the conductor arrived. "
        "Markets opened lower on fears of rising interest rates. "
    ) * 3,                                        # ~300 tokens – varied content
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load model once (reuse for all passages)
# ──────────────────────────────────────────────────────────────────────────────
# attn_implementation="eager" is REQUIRED.  The default "sdpa" (scaled-dot-
# product-attention via torch.nn.functional.scaled_dot_product_attention) is
# a fused kernel that never materialises the full [B, H, T, T] attention matrix
# in Python-visible memory – outputs.attentions would be None.  Eager mode
# computes the matrix explicitly and returns it.
print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    torch_dtype=torch.float32,   # fp32 to avoid any numerical quirks in scores
)
model.eval()
print("Model loaded.\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper: run one forward pass and extract attention tensors
# ──────────────────────────────────────────────────────────────────────────────
def get_attention_scores(text: str):
    """
    Returns
    -------
    attn : np.ndarray  shape (num_layers, num_heads, seq_len, seq_len)
        Softmax attention weights (each row sums to 1 in the causal mask
        region, 0 elsewhere).
    tokens : list[str]
        Decoded token strings (for axis labels).
    """
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len} tokens")

    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=False,          # we don't need the KV cache here; saves
                                      # memory so we can handle longer contexts
            output_attentions=True,   # tells the model to return attn weights
        )

    # outputs.attentions is a tuple of length num_layers; each element is a
    # tensor of shape (batch=1, heads, seq_len, seq_len).  We squeeze the
    # batch dimension and stack into a single numpy array for easy slicing.
    attn = np.stack(
        [layer_attn[0].cpu().numpy() for layer_attn in outputs.attentions]
    )  # (L, H, T, T)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    return attn, tokens


# ──────────────────────────────────────────────────────────────────────────────
# 4. Analysis helpers
# ──────────────────────────────────────────────────────────────────────────────

def mean_attention_received(attn: np.ndarray) -> np.ndarray:
    """
    For each key position k, compute the average attention weight it receives
    across *all* query positions that can attend to it (causal: q >= k),
    across all heads and all layers.

    Shape: (num_layers, num_heads, seq_len)

    Why column-mean rather than row-mean?
    --------------------------------------
    Row-mean (outgoing attention from a query) is uniform by construction –
    every row in the causal mask sums to 1, so the row mean is just 1/context.
    Column-mean (incoming attention to a key) is *not* constrained to be
    uniform.  A high column-mean means "many queries think this token is
    important."  This is exactly the quantity we want to maximise when we
    pick which K/V pairs to *keep* in a compressed cache.
    """
    L, H, T, _ = attn.shape
    col_mean = np.zeros((L, H, T), dtype=np.float32)
    for k in range(T):
        # Queries that can attend to key k are rows k..T-1 (causal masking).
        # Taking the mean over those rows gives the "average importance" of
        # key k as seen by all queries that could use it.
        col_mean[:, :, k] = attn[:, :, k:, k].mean(axis=-1)
    return col_mean   # (L, H, T)


def entropy_per_row(attn: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of each attention distribution (one per query token).

    Low entropy  → peaked/concentrated distribution → few keys dominate.
    High entropy → flat distribution → attention is spread evenly.

    We clip the log to avoid log(0) = -inf for the masked-out positions.
    Shape returned: (num_layers, num_heads, seq_len)
    """
    eps = 1e-9
    return -(attn * np.log(attn + eps)).sum(axis=-1)   # (L, H, T)


def effective_key_count(attn: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    For each query row, find the *minimum* number of top-attended keys needed
    to capture `threshold` fraction of the total probability mass.

    This is sometimes called the "effective branching factor" or "attention
    sparsity" measure.  A value of 3 means: on average, 3 tokens explain 80 %
    of what this query is looking at.  Anything else is noise from a KV-cache
    perspective.

    Shape returned: (num_layers, num_heads, seq_len)
    """
    L, H, T, _ = attn.shape
    counts = np.zeros((L, H, T), dtype=np.float32)
    for l in range(L):
        for h in range(H):
            for q in range(T):
                row = np.sort(attn[l, h, q])[::-1]   # descending
                cumulative = np.cumsum(row)
                # np.searchsorted finds the first index where cumsum >= threshold
                n = int(np.searchsorted(cumulative, threshold)) + 1
                counts[l, h, q] = n
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# 5. Run analysis for both passages
# ──────────────────────────────────────────────────────────────────────────────
results = {}
for name, text in PASSAGES.items():
    print(f"Processing passage: '{name}'")
    attn, tokens = get_attention_scores(text)
    col_mean = mean_attention_received(attn)          # (L, H, T)
    H_entropy = entropy_per_row(attn)                 # (L, H, T)
    results[name] = dict(attn=attn, tokens=tokens,
                         col_mean=col_mean, H_entropy=H_entropy)
    print(f"  Attention tensor shape: {attn.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Plot 1 – Heatmap of mean attention received per token position
#             (averaged over heads, shown per layer)
# ──────────────────────────────────────────────────────────────────────────────
# WHY: A heatmap with layers on the Y axis and token position on the X axis
# lets us see at a glance whether *all* layers or only some layers exhibit the
# sink behaviour.  If the first few columns are bright across *all* rows, the
# sink is universal and not a layer-specific artefact.
for name, res in results.items():
    attn     = res["attn"]      # (L, H, T, T)
    col_mean = res["col_mean"]  # (L, H, T)
    T        = attn.shape[-1]

    # Average over heads → shape (L, T)
    mean_over_heads = col_mean.mean(axis=1)

    fig, ax = plt.subplots(figsize=(min(T / 3, 20), 6))
    im = ax.imshow(
        mean_over_heads,
        aspect="auto",
        cmap="plasma",
        # LogNorm makes small differences at low values visible while still
        # showing the dominant sink positions.
        norm=LogNorm(vmin=max(mean_over_heads.min(), 1e-6),
                     vmax=mean_over_heads.max()),
    )
    ax.set_xlabel("Key token position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        f"[{name}] Mean attention received per key position\n"
        f"(averaged over {attn.shape[1]} heads, log scale)",
        fontsize=13,
    )
    plt.colorbar(im, ax=ax, label="Mean attention weight (log scale)")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"plot1_attention_received_{name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Plot 2 – Fraction of attention mass absorbed by first K tokens
#             across all layers and heads
# ──────────────────────────────────────────────────────────────────────────────
# WHY: Rather than eyeballing the heatmap, we want a precise number: "what
# fraction of total attention mass goes to token positions 0..K-1?"  If this
# fraction is large (say, >50 % with K=4 on a 300-token sequence), that is an
# extraordinary concentration – those 4 tokens represent 1.3 % of the
# positions but absorb >50 % of the mass.  This single number is the key
# empirical result motivating the StreamingLLM approach.
SINK_Ks = [1, 2, 4, 8, 16]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (name, res) in zip(axes, results.items()):
    attn = res["attn"]   # (L, H, T, T)
    L, H, T, _ = attn.shape

    # For each query q, the attention it distributes is attn[l,h,q,:].
    # The mass that lands on the first K key positions is attn[l,h,q,:K].sum().
    # We average this over all queries, heads, and layers.
    sink_fractions = []
    for K in SINK_Ks:
        # Sum over first K key positions for every query, then mean over q,h,l
        # Shape computation: attn[:,:,:,:K].sum(-1) → (L,H,T) → .mean() → scalar
        frac = attn[:, :, :, :K].sum(axis=-1).mean()
        sink_fractions.append(float(frac))

    bars = ax.bar(
        [str(k) for k in SINK_Ks],
        sink_fractions,
        color=plt.cm.plasma(np.linspace(0.2, 0.8, len(SINK_Ks))),
        edgecolor="white",
        linewidth=0.8,
    )
    # Annotate bars with the exact percentages
    for bar, frac in zip(bars, sink_fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{frac*100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.axhline(
        T / max(SINK_Ks) / T, color="grey", linestyle="--", linewidth=0.8,
        label="Uniform baseline",
    )
    ax.set_xlabel("Number of initial tokens K", fontsize=12)
    ax.set_ylabel("Fraction of total attention mass", fontsize=12)
    ax.set_title(f"[{name}]\nSink absorption (seq len={T})", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)

plt.suptitle(
    "Attention mass absorbed by the first K token positions\n"
    "(averaged over all layers, heads, and query positions)",
    fontsize=13, y=1.02,
)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "plot2_sink_absorption.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Plot 3 – Per-layer, per-head view of sink fraction (first 4 tokens)
#             shown as a grid heatmap
# ──────────────────────────────────────────────────────────────────────────────
# WHY: Averaging over layers and heads can hide heterogeneity.  Some heads
# might be "uniform" while others are extreme sinks.  A L×H grid tells us
# whether the sink is truly universal or driven by a minority of heads.
# If the grid is uniformly warm (high fraction), the phenomenon is structural,
# not the artefact of a single "sink head."
SINK_K = 4

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (name, res) in zip(axes, results.items()):
    attn = res["attn"]   # (L, H, T, T)
    L, H, T, _ = attn.shape

    # For each (layer, head), compute average fraction of mass on first SINK_K
    # tokens.  attn[l,h,:,:SINK_K].sum(-1) gives per-query sink mass → (T,).
    # Mean over queries gives one number per (l,h).
    sink_grid = attn[:, :, :, :SINK_K].sum(axis=-1).mean(axis=-1)  # (L, H)

    im = ax.imshow(sink_grid, aspect="auto", cmap="plasma",
                   vmin=0, vmax=sink_grid.max())
    plt.colorbar(im, ax=ax, label=f"Fraction absorbed by first {SINK_K} tokens")
    ax.set_xlabel("Head index", fontsize=11)
    ax.set_ylabel("Layer index", fontsize=11)
    ax.set_title(
        f"[{name}]\nSink fraction per layer × head\n(first {SINK_K} tokens, seq len={T})",
        fontsize=12,
    )
    # Mark max cell
    l_max, h_max = np.unravel_index(np.argmax(sink_grid), sink_grid.shape)
    ax.plot(h_max, l_max, "w*", markersize=12, label=f"Max={sink_grid[l_max,h_max]:.2f}")
    ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "plot3_sink_per_layer_head.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Plot 4 – Entropy distribution across layers
#             (violin plot: one violin per layer, pooled over heads & queries)
# ──────────────────────────────────────────────────────────────────────────────
# WHY: Entropy is the information-theoretic measure of how spread-out a
# probability distribution is.  If every attention row had entropy equal to
# log(T) (i.e. completely uniform), there would be no sparsity to exploit.
# If entropy is low, most rows are peaky – a small number of tokens dominate.
# A violin plot shows the *distribution* of entropy values (not just the mean)
# so we can see whether the peakedness is a consistent property of all rows
# or just a few extreme ones.
for name, res in results.items():
    H_ent = res["H_entropy"]    # (L, H, T)
    L = H_ent.shape[0]

    # Flatten over heads and queries → one list of entropy values per layer
    data_by_layer = [H_ent[l].reshape(-1).tolist() for l in range(L)]

    fig, ax = plt.subplots(figsize=(max(L * 0.7, 10), 5))
    parts = ax.violinplot(data_by_layer, positions=range(L),
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#9b59b6")
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    uniform_entropy = np.log(np.arange(1, L + 2))   # upper bound: log(q+1)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Shannon entropy (nats)", fontsize=12)
    ax.set_title(
        f"[{name}] Per-row attention entropy distribution across layers\n"
        "(low entropy = concentrated attention = KV compression opportunity)",
        fontsize=12,
    )
    ax.set_xticks(range(L))
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"plot4_entropy_violin_{name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Plot 5 – Cumulative attention mass curve (sorted, descending)
#              "How many keys does it take to cover 80 % of attention mass?"
# ──────────────────────────────────────────────────────────────────────────────
# WHY: This is the most direct visualisation of sparsity.  We sort the
# attention weights in each row (descending) and plot the cumulative sum.  If
# the curve rises steeply and then plateaus, attention is sparse.  If it rises
# linearly (like a uniform distribution), attention is dense.  We compare the
# empirical curves against the theoretical uniform baseline.  The gap between
# them is the "compressibility budget."
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, res) in zip(axes, results.items()):
    attn = res["attn"]   # (L, H, T, T)
    L, H, T, _ = attn.shape

    # Sample 1000 rows at random to keep computation fast; the result is the
    # same as using all rows because distribution is homogeneous.
    rng = np.random.default_rng(42)
    all_rows = attn.reshape(-1, T)              # (L*H*T, T) – every query row
    idx = rng.choice(len(all_rows), size=min(1000, len(all_rows)), replace=False)
    sample_rows = all_rows[idx]                 # (1000, T)

    # For each sampled row, sort descending and compute cumulative sum
    sorted_rows = np.sort(sample_rows, axis=-1)[:, ::-1]   # (1000, T)
    cumsum_rows = sorted_rows.cumsum(axis=-1)               # (1000, T)

    # x-axis: fraction of keys used (0 to 1)
    x = np.arange(1, T + 1) / T

    # Plot mean ± std band
    mean_curve = cumsum_rows.mean(axis=0)
    std_curve  = cumsum_rows.std(axis=0)

    ax.fill_between(x, mean_curve - std_curve, np.minimum(mean_curve + std_curve, 1.0),
                    alpha=0.3, color="#e74c3c", label="±1 std")
    ax.plot(x, mean_curve, color="#e74c3c", linewidth=2, label="Empirical (mean)")
    ax.plot(x, x, color="grey", linestyle="--", linewidth=1.2, label="Uniform baseline")

    # Mark the 80 % threshold
    thresh = 0.80
    idx_80 = np.searchsorted(mean_curve, thresh)
    if idx_80 < T:
        ax.axvline(x[idx_80], color="#2ecc71", linestyle=":", linewidth=1.5,
                   label=f"80% mass @ {x[idx_80]*100:.1f}% of keys")
        ax.axhline(thresh, color="#2ecc71", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Fraction of keys used (sorted by weight, desc)", fontsize=11)
    ax.set_ylabel("Cumulative attention mass", fontsize=11)
    ax.set_title(f"[{name}]\nCumulative attention mass curve (seq len={T})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.suptitle(
    "Cumulative mass curve: how many keys needed to cover 80% of attention?\n"
    "(each curve = average over 1 000 sampled query rows, all layers & heads)",
    fontsize=12, y=1.02,
)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "plot5_cumulative_mass_curve.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 11. Print summary statistics
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
for name, res in results.items():
    attn  = res["attn"]
    L, H, T, _ = attn.shape
    col_mean = res["col_mean"]   # (L, H, T)

    # --- Sink absorption ---
    sink4 = float(attn[:, :, :, :4].sum(axis=-1).mean())
    random_baseline = 4 / T

    # --- Entropy ---
    H_ent = res["H_entropy"]   # (L, H, T)
    mean_ent   = H_ent.mean()
    max_uniform = np.log(T)    # entropy of a flat distribution over T tokens

    print(f"\nPassage: '{name}'  (T={T})")
    print(f"  Sink (first 4 tokens):")
    print(f"    Observed fraction : {sink4*100:.2f}%")
    print(f"    Random baseline   : {random_baseline*100:.2f}%  (= 4/{T})")
    print(f"    Overrepresentation: {sink4/random_baseline:.1f}×")
    print(f"  Entropy (mean across all rows/heads/layers):")
    print(f"    Observed          : {mean_ent:.3f} nats")
    print(f"    Uniform upper bound: {max_uniform:.3f} nats  (log {T})")
    print(f"    Concentration ratio: {1 - mean_ent/max_uniform:.1%} below uniform")

print("\nAll plots saved to:", os.path.abspath(RESULTS_DIR))
