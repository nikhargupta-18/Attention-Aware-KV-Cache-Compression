"""
kv_cache_policies.py  –  Phase 3.3 / 3.4
==========================================
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

Position-ID note (Phase 3.4)
-----------------------------
When ``step`` evicts tokens, the remaining K vectors' RoPE encodings no
longer match their new (slot-based) relative positions.  Phase 3.4 fixes
this via ``RoPECorrector``, which re-encodes each K vector to reflect its
new cache-slot position.

After re-encoding, **the caller must use slot-based position_ids** for
every subsequent model forward call:

    pos_ids = torch.tensor([[cache.get_seq_length()]])   # ← slot, not abs

This makes both Q and K operate in the same contiguous 0..budget coordinate
space, matching the model's training distribution exactly.

Without this fix, position_ids = [[actual_sequence_pos]] keeps Q at the
correct absolute position but leaves gaps in the K position sequence (the
evicted middle tokens are gone but their position slots are not).  The model
was trained on contiguous contexts; non-contiguous relative distances subtly
corrupt attention patterns.  The degradation stays syntactically fluent but
causes factual-recall failures — invisible without targeted needle-in-a-
haystack evaluation (Phase 3.5).
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
    Note: after this call, K vectors at new slot i still carry the RoPE
    encoding for their OLD slot (``keep_idx[i]``).  Call
    ``RoPECorrector.correct`` afterwards to fix them.
    """
    device = cache.layers[0].keys.device
    idx = keep_idx.to(device)
    for layer in cache.layers:
        layer.keys   = layer.keys[:, :, idx, :]
        layer.values = layer.values[:, :, idx, :]
    return cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    The 'rotate-half' operation used by Qwen2.5's RoPE.

    Splits the last dimension in half and returns [-x2 | x1], which
    implements a 90° rotation within each (x1, x2) pair:

        rotate_half([a, b]) = [-b, a]

    This is used in: k_encoded = k * cos + rotate_half(k) * sin
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE correction (Phase 3.4)
# ─────────────────────────────────────────────────────────────────────────────

class RoPECorrector:
    """
    Corrects stale RoPE encodings in cached K tensors after token eviction.

    The core problem
    ─────────────────
    When a token is written into the KV cache, its K vector is RoPE-encoded
    at the token's *original* sequence position.  After eviction, the
    remaining K vectors keep these original encodings — but their *effective*
    positions in the evicted cache are now different (some earlier slots are
    gone, so every surviving token "moves left").

    If we then generate with Q at position ``cache_slot`` and K at
    ``original_pos``, the relative distance ``Q_pos - K_pos`` is computed
    correctly in absolute terms BUT creates a position-space gap where the
    evicted tokens used to live.  The model was trained on contiguous
    sequences and has never seen such gaps; the mismatch subtly corrupts
    attention patterns.

    The fix (StreamingLLM §3.3–3.4)
    ─────────────────────────────────
    After each eviction, re-encode every surviving K from its OLD slot
    position to its NEW slot position (0, 1, 2, …, budget-1), so that the
    cache always looks like a contiguous context to the model.  The query
    then gets ``position_ids = [[cache_slot]]`` (= budget when full), and
    both Q and K operate in the same 0..budget coordinate space.

    Mathematical derivation
    ────────────────────────
    RoPE applies a rotation R(p) to each K at position p:

        K_encoded = R(p_old) @ K_raw

    To move K from p_old to p_new without access to K_raw:

        K_corrected = R(p_new) @ K_raw
                    = R(p_new) @ R(p_old)^{-1} @ K_encoded
                    = R(p_new - p_old) @ K_encoded

    In the half-split RoPE formulation used by Qwen2.5 this becomes:

        K_corrected = K_encoded * cos(Δ) + rotate_half(K_encoded) * sin(Δ)

    where Δ = p_new - p_old and the angle-difference identities give:

        cos(Δ) = cos_new * cos_old + sin_new * sin_old
        sin(Δ) = sin_new * cos_old - cos_new * sin_old

    Incremental corrections
    ────────────────────────
    At steady state (cache always full), eviction removes exactly ONE token
    per decode step.  The window tokens each shift left by one slot, so
    Δ = -1 per step.  The numerical error from applying many small rotations
    is negligible (each correction is a unitary transform).

    Usage
    ─────
    ::
        rope_corrector = RoPECorrector(model)
        # ... after _evict_layers(cache, keep_idx) ...
        rope_corrector.correct(cache, old_positions=keep_idx,
                               new_positions=torch.arange(budget))
    """

    def __init__(self, model, max_positions: int = 2048):
        """
        Parameters
        ----------
        model : AutoModelForCausalLM
            The loaded model — used to extract the shared RoPE module.
        max_positions : int
            Pre-compute the cos/sin table up to this many positions.
            Must be >= the largest cache slot ever used.
        """
        device = next(model.parameters()).device

        # Qwen2.5 stores the shared rotary embedding on model.model
        rotary_emb = model.model.rotary_emb

        # Pre-compute cos/sin for positions 0..max_positions-1
        # pos_ids: (1, max_positions)
        pos_ids = torch.arange(max_positions, device=device).unsqueeze(0)
        # dummy is used only for its dtype; any shape works
        dummy = torch.zeros(1, device=device, dtype=torch.float32)

        with torch.no_grad():
            cos, sin = rotary_emb(dummy, pos_ids)
        # cos, sin: (1, max_positions, head_dim) with first half == second half
        # Store on CPU; moved to device in correct() as needed
        self.cos = cos.squeeze(0).float().cpu()  # (max_positions, head_dim)
        self.sin = sin.squeeze(0).float().cpu()

    def correct(
        self,
        cache: DynamicCache,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
    ) -> None:
        """
        In-place: recode each K vector from ``old_positions[i]`` to
        ``new_positions[i]``.

        Parameters
        ----------
        cache : DynamicCache
            The cache *after* ``_evict_layers`` has already run (so the
            K tensor at slot i came from slot ``old_positions[i]``).
        old_positions : 1-D LongTensor, CPU
            The slot indices that each surviving token occupied BEFORE
            eviction.  Typically the ``keep_idx`` passed to
            ``_evict_layers``.
        new_positions : 1-D LongTensor, CPU
            The contiguous slot indices to assign: ``torch.arange(budget)``.
        """
        if torch.equal(old_positions, new_positions):
            return   # nothing to do — positions haven't changed

        cos_old = self.cos[old_positions]   # (num_kept, head_dim)
        sin_old = self.sin[old_positions]
        cos_new = self.cos[new_positions]
        sin_new = self.sin[new_positions]

        # Angle-difference identity: cos(new-old) and sin(new-old)
        cos_diff = cos_new * cos_old + sin_new * sin_old   # (num_kept, head_dim)
        sin_diff = sin_new * cos_old - cos_new * sin_old

        for layer in cache.layers:
            k  = layer.keys.float()   # (batch, kv_heads, num_kept, head_dim)
            dev = k.device

            # Broadcast (num_kept, head_dim) → (1, 1, num_kept, head_dim)
            cd = cos_diff.to(dev).unsqueeze(0).unsqueeze(0)
            sd = sin_diff.to(dev).unsqueeze(0).unsqueeze(0)

            # Apply: K_corrected = K * cos(Δ) + rotate_half(K) * sin(Δ)
            layer.keys = (k * cd + rotate_half(k) * sd).to(layer.keys.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseKVCachePolicy:
    """Abstract base — subclasses implement ``step``."""
    name: str = "base"

    # True  → caller should use slot-based position_ids (cache.get_seq_length())
    #          and K vectors are re-encoded to contiguous slots after eviction.
    # False → caller should use absolute sequence position_ids; K vectors
    #          keep their original RoPE encodings (no correction applied).
    use_slot_positions: bool = True

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

    RoPE correction (Phase 3.4)
    ────────────────────────────
    Even though this policy is broken by design, RoPE correction is applied
    when a ``rope_corrector`` is provided, for consistency.  The window
    tokens are simply re-indexed to slots 0..window_size-1 after eviction.
    """
    name = "sliding_window"

    def __init__(
        self,
        window_size: int,
        rope_corrector: "RoPECorrector | None" = None,
    ):
        """
        Parameters
        ----------
        window_size : int
            Number of most-recent K/V pairs to keep.  Everything older,
            including position 0, is discarded at each step.
        rope_corrector : RoPECorrector | None
            If provided, K vectors are re-encoded to slot positions 0..W-1
            after each eviction.  The caller must then use slot-based
            ``position_ids``.
        """
        self.window_size    = window_size
        self.rope_corrector = rope_corrector

    def step(self, cache: DynamicCache, attentions=None) -> DynamicCache:
        seq_len = _seq_len(cache)
        if seq_len > self.window_size:
            # Keep the LAST window_size positions (drop the front)
            keep = torch.arange(seq_len - self.window_size, seq_len)
            _evict_layers(cache, keep)
            if self.rope_corrector is not None:
                new_pos = torch.arange(self.window_size)
                self.rope_corrector.correct(cache, keep.cpu(), new_pos)
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

    RoPE correction (Phase 3.4)
    ────────────────────────────
    Without correction, the window tokens retain their original large
    position values (e.g. positions 172..295 in a 296-token sequence) while
    occupying slots 4..127 in the cache.  The resulting position gaps — where
    the evicted tokens used to be — are out-of-distribution for the model.

    With correction (StreamingLLM §3.3-3.4): after each eviction the
    window tokens are re-encoded to cache slots 4..budget-1, and the query
    receives ``position_ids = [[budget]]``.  Both Q and K now operate in
    a contiguous 0..budget coordinate space, matching the training
    distribution exactly.

    Sink tokens (positions 0..sink_size-1) are unchanged — their slot
    positions equal their original positions, so Δ = 0 and no rotation
    is applied.

    Properties
    ──────────
    •  Fixed, content-independent — no per-token scoring or bookkeeping.
    •  O(1) cost per decode step (two tensor slices + cat + RoPE correct).
    •  Predictable, bounded memory: exactly ``sink_size + window_size`` pairs.
    •  Trades long-range non-sink memory for guaranteed sink stability.
    """
    name = "streaming_llm"

    def __init__(
        self,
        sink_size: int,
        window_size: int,
        rope_corrector: "RoPECorrector | None" = None,
    ):
        """
        Parameters
        ----------
        sink_size : int
            Number of initial tokens always retained.  StreamingLLM uses 4
            for most models; 1 also works for Qwen.
        window_size : int
            Number of most-recent tokens retained.
        rope_corrector : RoPECorrector | None
            If provided, K vectors are re-encoded to contiguous slot positions
            after each eviction (Phase 3.4 fix).
        """
        self.sink_size      = sink_size
        self.window_size    = window_size
        self.rope_corrector = rope_corrector

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
            if self.rope_corrector is not None:
                # Sinks map 0→0, 1→1, … (Δ=0, no rotation)
                # Window tokens map old_slot → 4,5,…,127 (Δ = new - old, often negative)
                new_pos = torch.arange(budget)
                self.rope_corrector.correct(cache, keep.cpu(), new_pos)
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

    RoPE correction (Phase 3.4)
    ────────────────────────────
    H2O's content-adaptive eviction can produce any subset of original slot
    indices, so the position gaps after eviction are irregular.  With
    ``rope_corrector`` provided, the kept tokens are re-encoded to contiguous
    slots 0..budget-1, making H2O's evicted cache as well-positioned as
    StreamingLLM's.

    Properties
    ──────────
    •  Content-adaptive: positions that consistently receive high attention
       (heavy hitters, including sinks) are retained automatically.
    •  Requires ``output_attentions=True`` at every step (needed for scoring).
    •  Stateful: call ``reset()`` between independent sequences.
    •  Most bookkeeping-intensive policy; budget extra implementation time.
    """
    name = "h2o"

    # H2O must NOT use slot-based positions.
    #
    # StreamingLLM re-indexes to slot positions because its eviction pattern
    # (sink block + recency window) produces two CONTIGUOUS blocks.  Re-mapping
    # them to 0..budget-1 creates a coherent "virtual short context" and the
    # model attends correctly to nearby and distant tokens alike.
    #
    # H2O's content-adaptive eviction produces a SCATTERED, non-contiguous
    # subset of the original sequence.  Re-indexing these scattered positions
    # to contiguous slots 0..budget-1 would tell the model that e.g. originally-
    # adjacent positions 45 and 67 are now only 1 slot apart (a false adjacency),
    # corrupting relative-distance attention for all retained heavy hitters.
    # Empirically: applying slot-based re-indexing to H2O increases NLL by
    # ~1.2 nats (see rope_correction_eval.py results).
    #
    # The correct scheme for H2O: keep K vectors at their original RoPE
    # positions and pass the true absolute sequence position as position_ids
    # for the query.  The non-contiguous position gaps are "honest" — the model
    # correctly computes distances to whichever tokens are actually present.
    use_slot_positions: bool = False

    def __init__(
        self,
        budget: int,
        rope_corrector: "RoPECorrector | None" = None,
    ):
        """
        Parameters
        ----------
        budget : int
            Maximum K/V pairs to keep (same limit applied across all layers).
        rope_corrector : RoPECorrector | None
            If provided, K vectors are re-encoded to contiguous slot positions
            after each eviction (Phase 3.4 fix).
        """
        self.budget         = budget
        self.rope_corrector = rope_corrector
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

        # Apply eviction to the cache tensors
        _evict_layers(cache, keep_idx)

        # Phase 3.4: correct K encodings to new contiguous slot positions
        if self.rope_corrector is not None:
            new_pos = torch.arange(self.budget)
            self.rope_corrector.correct(cache, keep_idx.cpu(), new_pos)

        # Mirror eviction in the score tensor
        self.scores = self.scores[:, keep_idx]

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

    No RoPE correction is needed: K vectors were written at positions 0, 1,
    2, … which equal their cache-slot indices.  The query uses
    ``position_ids = [[cache.get_seq_length()]]`` which equals the true
    sequence length (identical to the absolute position).
    """
    name = "no_eviction"

    def step(self, cache: DynamicCache, attentions=None) -> DynamicCache:
        return cache   # pass-through; cache is already fully populated
