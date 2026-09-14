"""
kv_cache_policies.py  –  Phase 3.3
=====================================
Three pluggable KV-cache eviction policies that operate directly on the
``past_key_values`` object returned by a HuggingFace CausalLM.

Public interface
----------------
Every policy is a *stateful* class with two methods::

    evicted_cache = policy.step(past_key_values, attentions=None)
    policy.reset()          # call between independent sequences

``step`` is called once per forward pass (prefill AND each decode step).
It modifies the cache **in-place** (reassigning the per-layer key/value
tensors) and returns the same cache object so it can be chained naturally.

Cache format (Qwen2.5 / modern transformers)
--------------------------------------------
``past_key_values`` is a ``DynamicCache`` whose ``layers`` attribute is a
list of ``DynamicLayer`` objects.  Each layer exposes:

    layer.keys   : Tensor  (batch, kv_heads, seq_len, head_dim)
    layer.values : Tensor  (batch, kv_heads, seq_len, head_dim)

Reassigning these tensors (``layer.keys = new_k``) is sufficient; the cache
object tracks sequence length through ``get_seq_length()``, which reads the
first layer's tensor shape automatically.

Attention tensor format
-----------------------
``attentions`` is a tuple of length ``num_layers``.  Each element has shape
``(batch=1, num_heads, q_len, kv_len)``.

    •  During prefill:    q_len == kv_len == T  (full attention matrix)
    •  During generation: q_len == 1            (one new query per step)

Only ``H2OPolicy`` requires attention weights; the other two policies accept
the argument and silently ignore it.

Position-ID note
----------------
When ``step`` evicts tokens, the cache becomes shorter than the actual
sequence length.  The *caller* (run_policies.py) is responsible for passing
explicit ``position_ids`` to every subsequent model forward call so that the
new query token is placed at its *true* sequence position rather than at the
(shorter) cache length.  This is the only way to get correct RoPE distances
after eviction.
"""

import torch
from transformers.cache_utils import DynamicCache


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seq_len(cache: DynamicCache) -> int:
    """Current number of K/V pairs stored in the cache."""
    return cache.get_seq_length()


def _evict_layers(cache: DynamicCache, keep_idx: torch.Tensor) -> DynamicCache:
    """
    Slice every layer's key/value tensors to the positions in ``keep_idx``
    (a 1-D LongTensor of sorted position indices on CPU).

    Modifies the cache in-place and returns it.
    """
    device = cache.layers[0].keys.device
    idx = keep_idx.to(device)
    for layer in cache.layers:
        layer.keys   = layer.keys[:, :, idx, :]
        layer.values = layer.values[:, :, idx, :]
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseKVCachePolicy:
    """Abstract base — subclasses implement ``step``."""
    name: str = "base"

    def step(self, cache: DynamicCache, attentions=None) -> DynamicCache:
        """
        Apply the eviction policy to ``cache`` (in-place).

        Parameters
        ----------
        cache : DynamicCache
            The cache returned by the latest model forward pass (already
            contains the new token's K/V pair, appended by the model itself).
        attentions : tuple[Tensor] | None
            Per-layer attention weights (batch, heads, q_len, kv_len).
            Required by H2OPolicy; others ignore this argument.

        Returns
        -------
        DynamicCache  (same object, possibly modified in-place)
        """
        raise NotImplementedError

    def reset(self):
        """Reset any internal state.  Call between independent sequences."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3a.  Sliding Window  (naive baseline — intentionally broken)
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowPolicy(BaseKVCachePolicy):
    """
    Naive baseline: keep only the LAST ``window_size`` K/V pairs; drop
    everything older — **including position 0 and the attention sinks**.

    Why this is the control group
    ─────────────────────────────
    Phase 3.2 proved empirically that the very first tokens absorb a
    disproportionate share of attention mass regardless of content (the
    "attention sink" phenomenon).  Evicting them does not merely lose old
    information — it *breaks the softmax normalisation* for every subsequent
    query:

    •  Before eviction: ``softmax([sink_score, …, recent_scores])``
    •  After  eviction: ``softmax([…,           recent_scores])``

    The sink positions were absorbing spare probability mass.  Removing them
    forces that mass onto the remaining tokens, distorting the attention
    distribution away from what the model was trained on.  The residual
    stream, which expects well-formed distributions, receives corrupted
    signals → visible quality degradation: repetition, hallucination, loss
    of long-range coreference, and early incoherence.

    This is your control group.  Bad results here are the point.
    """
    name = "sliding_window"

    def __init__(self, window_size: int):
        """
        Parameters
        ----------
        window_size : int
            Number of most-recent K/V pairs to keep.  Everything older,
            including position 0, is discarded at each step.
        """
        self.window_size = window_size

    def step(self, cache: DynamicCache, attentions=None) -> DynamicCache:
        seq_len = _seq_len(cache)
        if seq_len > self.window_size:
            # Keep the LAST window_size positions (drop the front)
            keep = torch.arange(seq_len - self.window_size, seq_len)
            _evict_layers(cache, keep)
        return cache


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  Attention-Sink-Aware  (StreamingLLM-style)
# ─────────────────────────────────────────────────────────────────────────────

class StreamingLLMPolicy(BaseKVCachePolicy):
    """
    StreamingLLM-style rolling cache: always keep the first ``sink_size``
    tokens (the attention sinks) **plus** a sliding window of the most-recent
    ``window_size`` tokens; evict everything in between.

    Reference
    ─────────
    Xiao et al., "Efficient Streaming Language Models with Attention Sinks",
    ICLR 2024, Section 3.  The core insight: the sinks must be preserved
    because they absorb "spare" attention mass; losing them breaks the softmax
    even for queries that have no semantic dependence on the sink *content*.

    Cache layout after eviction (budget = sink_size + window_size)
    ──────────────────────────────────────────────────────────────

        [sink_0 … sink_{K-1} | win_{t-N} … win_{t-1} | win_t]
          ←── sink region ──→ ←──────── window region ────────→

    Properties
    ──────────
    •  Fixed, content-independent — no per-token scoring or bookkeeping.
    •  O(1) cost per decode step (two tensor slices + cat).
    •  Predictable, bounded memory: exactly ``sink_size + window_size`` pairs.
    •  Trades long-range non-sink memory for guaranteed sink stability.
    """
    name = "streaming_llm"

    def __init__(self, sink_size: int, window_size: int):
        """
        Parameters
        ----------
        sink_size : int
            Number of initial tokens always retained.  StreamingLLM uses 4
            for most models; 1 also works for Qwen.
        window_size : int
            Number of most-recent tokens retained.
        """
        self.sink_size   = sink_size
        self.window_size = window_size

    def step(self, cache: DynamicCache, attentions=None) -> DynamicCache:
        seq_len = _seq_len(cache)
        budget  = self.sink_size + self.window_size
        if seq_len > budget:
            # Sink indices: 0 … sink_size-1
            sink_idx = torch.arange(self.sink_size)
            # Window indices: last window_size positions
            win_idx  = torch.arange(seq_len - self.window_size, seq_len)
            keep     = torch.cat([sink_idx, win_idx])
            _evict_layers(cache, keep)
        return cache


# ─────────────────────────────────────────────────────────────────────────────
# 3c.  Accumulated-Score / Heavy-Hitter Oracle  (H2O-style)
# ─────────────────────────────────────────────────────────────────────────────

class H2OPolicy(BaseKVCachePolicy):
    """
    H2O (Heavy-Hitter Oracle) content-aware cache: maintain a running
    *accumulated attention score* for each cached position based on the
    attention weight it receives from the **latest query position** (averaged
    across heads and layers) at each step.  When the cache exceeds ``budget``,
    evict the lowest-scoring tokens.

    Reference
    ─────────
    Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
    Inference of LLMs", NeurIPS 2023, Sections 3–4.

    Score update (each forward pass)
    ─────────────────────────────────
    For layer l, head h, key position k::

        score[l, k] += mean_h( attn[l, h, -1, k] )

    Why last-query-only instead of summing over all queries:
    Summing over all query positions during prefill creates a severe causal-mask
    bias: position 0 is visible to (and can be attended to by) all T queries,
    whereas position T-1 is only visible to 1 query.  Naively summing all rows
    inflates early tokens' scores purely due to their position, causing H2O
    to degenerate into a keep-first policy.  Using only the final query row
    (q = -1) evaluates relevance without causal-mask position bias.

    Eviction rule
    ─────────────
    When |cache| > budget::

        agg_score[k] = mean_l( score[l, k] )         # aggregate over layers
        keep = argsort(agg_score, descending)[:budget]
        keep = sort(keep)                             # restore temporal order
        evict the rest; mirror eviction in score tensor

    Properties
    ──────────
    •  Content-adaptive: positions that consistently receive high attention
       (heavy hitters, including sinks) are retained automatically.
    •  Requires ``output_attentions=True`` at every step (needed for scoring).
    •  Stateful: call ``reset()`` between independent sequences.
    •  Most bookkeeping-intensive policy; budget extra implementation time.
    """
    name = "h2o"

    def __init__(self, budget: int):
        """
        Parameters
        ----------
        budget : int
            Maximum K/V pairs to keep (same limit applied across all layers).
        """
        self.budget = budget
        # Shape: (num_layers, seq_len) — grows each step, shrunk after eviction.
        # CPU tensor; avoids accumulating on GPU across many steps.
        self.scores: torch.Tensor | None = None

    def reset(self):
        """Clear accumulated scores.  Must be called between sequences."""
        self.scores = None

    def step(self, cache: DynamicCache, attentions) -> DynamicCache:
        """
        Parameters
        ----------
        cache : DynamicCache
            Cache *after* the latest forward pass (new token already appended).
        attentions : tuple[Tensor]
            Shape per layer: (1, heads, q_len, kv_len).
            q_len == seq_len during prefill; q_len == 1 during generation.
        """
        if attentions is None:
            raise ValueError(
                "H2OPolicy requires output_attentions=True so that attention "
                "weights are available for heavy-hitter score bookkeeping."
            )

        num_layers = len(cache.layers)
        seq_len    = _seq_len(cache)   # cache length *including* new token

        # ── 1.  Initialise or extend the score tensor ─────────────────────────
        if self.scores is None:
            # First call (prefill).  Initialise to zeros; we fill below.
            self.scores = torch.zeros(num_layers, seq_len, dtype=torch.float32)
        else:
            # Subsequent calls (generation).  The model appended one new K/V;
            # append a matching zero column so dimensions stay aligned.
            new_col = torch.zeros(num_layers, 1, dtype=torch.float32)
            self.scores = torch.cat([self.scores, new_col], dim=1)

        # ── 2.  Accumulate attention scores ───────────────────────────────────
        # attn_l : (1, heads, q_len, kv_len)
        #
        # Why we use ONLY THE LAST QUERY ROW (not the column sum over all rows):
        # ────────────────────────────────────────────────────────────────────────
        # During prefill, causal masking means position k can be attended to by
        # (T-k) queries while position 0 is attended to by ALL T queries.
        # Summing ALL query rows during prefill therefore inflates scores of
        # early positions purely because more queries can see them — not because
        # they are semantically more important.  With a 180-token prefill and a
        # 128-token budget, the naive column-sum approach would keep roughly
        # positions 0..127 and evict the most recent 52 tokens of the prompt,
        # causing H2O to perform no better than a "keep-first-N" policy.
        #
        # Using only the LAST QUERY's attention row eliminates this bias:
        #   • During prefill  : scores reflect which positions the final prefill
        #                       token finds most relevant (unbiased by causal mask).
        #   • During generation: scores reflect which cached positions the most-
        #                       recently-generated token attends to.
        #
        # This is also the natural streaming interpretation of the H2O paper:
        # each new query casts one "vote" for every cached position it attends
        # to, and the cumulative votes define the heavy hitters.
        for l, attn_l in enumerate(attentions):
            if attn_l is None:
                continue
            # Last query row: (heads, kv_len) — unbiased by causal mask
            last_q = attn_l[0, :, -1, :]
            # Average over heads → (kv_len,)
            score_update = last_q.mean(dim=0)
            kv_len = score_update.shape[0]
            update_len = min(kv_len, seq_len)
            self.scores[l, :update_len] += score_update[:update_len].cpu()


        # ── 3.  Evict if over budget ──────────────────────────────────────────
        if seq_len <= self.budget:
            return cache   # nothing to evict yet

        # Aggregate scores across layers → one scalar per position
        agg = self.scores.mean(dim=0)                # (seq_len,)

        # Top-budget positions by accumulated score
        _, keep_idx = torch.topk(agg, self.budget, largest=True)
        # Restore temporal order so the model sees a causally ordered cache
        keep_idx, _ = torch.sort(keep_idx)           # ascending (CPU)

        # Apply eviction to the cache tensors and to the score tensor
        _evict_layers(cache, keep_idx)
        self.scores = self.scores[:, keep_idx]       # mirror eviction

        return cache


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: no-eviction baseline
# ─────────────────────────────────────────────────────────────────────────────

class NoEvictionPolicy(BaseKVCachePolicy):
    """
    Full-cache baseline: keep every K/V pair, no eviction.

    This is the performance upper bound.  Memory grows without bound during
    generation — not practical for long contexts, but sets the ceiling for
    quality comparisons.
    """
    name = "no_eviction"

    def step(self, cache: DynamicCache, attentions=None) -> DynamicCache:
        return cache   # pass-through; cache is already fully populated
