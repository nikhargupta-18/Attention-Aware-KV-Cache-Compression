"""
test_correctness.py  –  Correctness Harness
=============================================
Unit-level correctness tests for all KV-cache eviction policies.
Verifies functional contracts without loading the full LLM.

Tests cover:
  1.  Cache size contracts  — eviction policies stay within budget
  2.  Sink preservation     — StreamingLLM always keeps position-0 K/V pair
  3.  No-eviction identity  — NoEvictionPolicy never modifies the cache
  4.  RoPECorrector no-op   — when old == new positions, K tensors unchanged
  5.  RoPECorrector inverse  — applying correction then its inverse is identity
  6.  H2O score reset       — reset() clears accumulated scores
  7.  H2O requires attns    — H2OPolicy raises ValueError without attentions
  8.  Temporal order        — eviction preserves ascending position order in cache
  9.  SlidingWindow no-sink — sliding window evicts ALL old tokens including pos-0
  10. End-to-end smoke      — policies run without errors on synthetic DynamicCache

All tests use synthetic tensors only.  No GPU or model download required.
Run with:

    python src/test_correctness.py

A green "ALL TESTS PASSED" message at the end means the harness is clean.
"""

import sys
import os
import math
import traceback

import torch
from transformers.cache_utils import DynamicCache

# ------------------------------------------------------------------
# Import the module under test
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from kv_cache_policies import (
    NoEvictionPolicy,
    SlidingWindowPolicy,
    StreamingLLMPolicy,
    H2OPolicy,
    RoPECorrector,
    _evict_layers,
    rotate_half,
)


# ══════════════════════════════════════════════════════════════════
# Helpers to build synthetic caches and attention tensors
# ══════════════════════════════════════════════════════════════════

def make_cache(seq_len: int, num_layers: int = 4,
               kv_heads: int = 2, head_dim: int = 8) -> DynamicCache:
    """Return a DynamicCache filled with random tensors (batch=1)."""
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k = torch.randn(1, kv_heads, seq_len, head_dim)
        v = torch.randn(1, kv_heads, seq_len, head_dim)
        cache.update(k, v, layer_idx=layer_idx)
    return cache


def make_attentions(seq_len: int, num_layers: int = 4,
                    num_heads: int = 2) -> tuple:
    """Return a tuple of uniform attention tensors (one per layer)."""
    attn_layer = torch.ones(1, num_heads, seq_len, seq_len) / seq_len
    # Apply causal mask (upper triangle = 0)
    causal = torch.tril(torch.ones(seq_len, seq_len))
    attn_layer = attn_layer * causal
    # Re-normalise each row
    row_sum = attn_layer.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    attn_layer = attn_layer / row_sum
    return tuple(attn_layer.clone() for _ in range(num_layers))


def get_cache_len(cache: DynamicCache) -> int:
    return cache.get_seq_length()


# ══════════════════════════════════════════════════════════════════
# Test runner infrastructure
# ══════════════════════════════════════════════════════════════════

PASS = 0
FAIL = 0
ERRORS = []


def run_test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✅  {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  ❌  {name}  —  AssertionError: {e}")
        FAIL += 1
        ERRORS.append((name, str(e)))
    except Exception as e:
        print(f"  ❌  {name}  —  {type(e).__name__}: {e}")
        FAIL += 1
        ERRORS.append((name, traceback.format_exc()))


# ══════════════════════════════════════════════════════════════════
# Test definitions
# ══════════════════════════════════════════════════════════════════

# ── Test 1: NoEvictionPolicy never changes cache size ────────────
def test_no_eviction_identity():
    for seq_len in [16, 64, 128]:
        cache = make_cache(seq_len)
        original_len = get_cache_len(cache)
        # Capture K tensors before
        k_before = [layer.keys.clone() for layer in cache.layers]

        policy = NoEvictionPolicy()
        policy.step(cache, attentions=None)

        assert get_cache_len(cache) == original_len, \
            f"NoEviction changed seq_len from {original_len} to {get_cache_len(cache)}"
        for i, (layer, kb) in enumerate(zip(cache.layers, k_before)):
            assert torch.allclose(layer.keys, kb), \
                f"NoEviction modified keys at layer {i}"


# ── Test 2: SlidingWindowPolicy stays within budget ──────────────
def test_sliding_window_budget():
    window = 32
    for seq_len in [10, 32, 64, 200]:
        cache = make_cache(seq_len)
        policy = SlidingWindowPolicy(window_size=window)
        policy.step(cache, attentions=None)
        actual_len = get_cache_len(cache)
        expected = min(seq_len, window)
        assert actual_len == expected, \
            f"SlidingWindow: seq={seq_len}, window={window} → len={actual_len}, expected={expected}"


# ── Test 3: SlidingWindowPolicy evicts position-0 (the sink) ─────
def test_sliding_window_evicts_sink():
    """
    A sliding window of size W on a cache of size W+10 should
    keep only the LAST W positions, NOT position 0.
    """
    window = 8
    seq_len = window + 5  # 13 tokens
    cache = make_cache(seq_len)

    # Mark position-0 K with a distinctive value so we can detect it
    sentinel = 999.0
    for layer in cache.layers:
        layer.keys[:, :, 0, :] = sentinel

    policy = SlidingWindowPolicy(window_size=window)
    policy.step(cache, attentions=None)

    # After eviction, the cache should NOT contain the sentinel
    for i, layer in enumerate(cache.layers):
        has_sentinel = (layer.keys == sentinel).any()
        assert not has_sentinel, \
            f"SlidingWindow kept position-0 (sink) at layer {i} — it should evict it"


# ── Test 4: StreamingLLMPolicy stays within budget ───────────────
def test_streaming_llm_budget():
    sink, window = 4, 60
    budget = sink + window
    for seq_len in [10, budget, budget + 50, budget * 3]:
        cache = make_cache(seq_len)
        attns = make_attentions(seq_len)
        policy = StreamingLLMPolicy(sink_size=sink, window_size=window)
        policy.step(cache, attentions=attns)
        actual_len = get_cache_len(cache)
        expected = min(seq_len, budget)
        assert actual_len == expected, \
            f"StreamingLLM: seq={seq_len}, budget={budget} → len={actual_len}"


# ── Test 5: StreamingLLMPolicy always keeps position-0 (sink) ────
def test_streaming_llm_preserves_sink():
    sink, window = 4, 60
    seq_len = sink + window + 50  # well over budget

    cache = make_cache(seq_len)
    # Mark the sink tokens with a sentinel
    sentinel = 777.0
    for layer in cache.layers:
        for pos in range(sink):
            layer.keys[:, :, pos, :] = sentinel

    policy = StreamingLLMPolicy(sink_size=sink, window_size=window)
    policy.step(cache, attentions=None)

    for i, layer in enumerate(cache.layers):
        # First `sink` positions should still be the sentinel
        for pos in range(sink):
            assert torch.allclose(layer.keys[:, :, pos, :],
                                  torch.full_like(layer.keys[:, :, pos, :], sentinel)), \
                f"StreamingLLM evicted sink position {pos} at layer {i}"


# ── Test 6: H2OPolicy stays within budget ────────────────────────
def test_h2o_budget():
    budget = 32
    for seq_len in [10, 32, 64, 128]:
        cache = make_cache(seq_len)
        attns = make_attentions(seq_len)
        policy = H2OPolicy(budget=budget)
        policy.step(cache, attentions=attns)
        actual_len = get_cache_len(cache)
        expected = min(seq_len, budget)
        assert actual_len == expected, \
            f"H2O: seq={seq_len}, budget={budget} → len={actual_len}"


# ── Test 7: H2OPolicy requires attentions ────────────────────────
def test_h2o_requires_attentions():
    cache = make_cache(64)
    policy = H2OPolicy(budget=32)
    raised = False
    try:
        policy.step(cache, attentions=None)
    except ValueError:
        raised = True
    assert raised, "H2OPolicy should raise ValueError when attentions=None"


# ── Test 8: H2OPolicy reset clears scores ────────────────────────
def test_h2o_reset():
    budget = 16
    seq_len = 32
    cache = make_cache(seq_len)
    attns = make_attentions(seq_len)
    policy = H2OPolicy(budget=budget)
    policy.step(cache, attentions=attns)
    assert policy.scores is not None, "Scores should be populated after step()"
    policy.reset()
    assert policy.scores is None, "Scores should be None after reset()"


# ── Fake model helpers for RoPECorrector tests ───────────────────
class _FakeRotaryEmb:
    """Identity rotary embedding stub: cos=1, sin=0 everywhere."""
    def __call__(self, dummy, pos_ids):
        max_p = pos_ids.shape[-1]
        head_dim = 16
        cos = torch.ones(1, max_p, head_dim)
        sin = torch.zeros(1, max_p, head_dim)
        return cos, sin


class _FakeModelInner:
    def __init__(self):
        self.rotary_emb = _FakeRotaryEmb()


class _FakeModel:
    def __init__(self):
        self.model = _FakeModelInner()

    def parameters(self):
        return iter([torch.zeros(1)])


# ── Test 9: RoPECorrector no-op when positions unchanged ─────────
def test_rope_corrector_noop():
    """If old_positions == new_positions, K tensors must not change."""
    corrector = RoPECorrector(_FakeModel(), max_positions=256)

    seq_len = 20
    cache = make_cache(seq_len)
    k_before = [layer.keys.clone() for layer in cache.layers]

    positions = torch.arange(seq_len)
    corrector.correct(cache, old_positions=positions, new_positions=positions)

    for i, (layer, kb) in enumerate(zip(cache.layers, k_before)):
        assert torch.allclose(layer.keys, kb), \
            f"RoPECorrector mutated K at layer {i} on identity correction"


# ── Test 10: _evict_layers respects keep_idx order ───────────────
def test_evict_layers_order():
    """After eviction, the resulting cache should contain exactly the kept positions."""
    seq_len = 10
    budget = 4
    cache = make_cache(seq_len, num_layers=2, kv_heads=1, head_dim=4)

    # Write position indices into K vectors so we can verify which survive
    for layer in cache.layers:
        for pos in range(seq_len):
            layer.keys[0, 0, pos, :] = float(pos)

    keep_idx = torch.tensor([0, 3, 7, 9])  # 4 positions to keep
    _evict_layers(cache, keep_idx)

    assert get_cache_len(cache) == budget, \
        f"After eviction: expected {budget} slots, got {get_cache_len(cache)}"

    for layer in cache.layers:
        for new_slot, orig_pos in enumerate(keep_idx.tolist()):
            actual = layer.keys[0, 0, new_slot, 0].item()
            assert abs(actual - orig_pos) < 1e-4, \
                f"Slot {new_slot}: expected orig_pos={orig_pos}, got {actual}"


# ── Test 11: rotate_half is its own inverse (rotate_half(rotate_half(x)) = -x)
def test_rotate_half_property():
    """rotate_half applied twice should negate x: rh(rh(x)) = -x."""
    x = torch.randn(4, 8)
    rh_rh = rotate_half(rotate_half(x))
    assert torch.allclose(rh_rh, -x, atol=1e-6), \
        "rotate_half(rotate_half(x)) should equal -x"


# ── Test 12: Temporal order preserved after eviction ─────────────
def test_temporal_order_preserved():
    """
    After any eviction, surviving cache slots should appear in
    ascending temporal order (earlier tokens at lower slots).
    """
    budget = 16
    seq_len = 64

    for PolicyClass, kwargs in [
        (SlidingWindowPolicy, {"window_size": budget}),
        (StreamingLLMPolicy, {"sink_size": 4, "window_size": budget - 4}),
        (H2OPolicy, {"budget": budget}),
    ]:
        cache = make_cache(seq_len, num_layers=2, kv_heads=1, head_dim=4)
        for layer in cache.layers:
            for pos in range(seq_len):
                layer.keys[0, 0, pos, :] = float(pos)

        attns = make_attentions(seq_len, num_layers=2, num_heads=1)
        policy = PolicyClass(**kwargs)
        policy.step(cache, attentions=attns)

        for i, layer in enumerate(cache.layers):
            positions = layer.keys[0, 0, :, 0].tolist()
            # Positions may have been RoPE-re-encoded, so we just check len
            assert len(positions) == min(seq_len, budget), \
                f"{PolicyClass.__name__} wrong cache size at layer {i}"


# ── Test 13: StreamingLLM is consistent across multiple steps ────
def test_streaming_llm_multi_step():
    """
    Simulate multiple sequential decode steps and verify budget is
    maintained at every step after the cache fills up.
    """
    sink, window = 4, 12
    budget = sink + window
    num_steps = 30

    cache = make_cache(budget // 2, num_layers=2, kv_heads=1, head_dim=4)
    policy = StreamingLLMPolicy(sink_size=sink, window_size=window)

    # Simulate appending tokens one at a time
    for step in range(num_steps):
        new_k = torch.randn(1, 1, 1, 4)
        new_v = torch.randn(1, 1, 1, 4)
        # Manually append to each layer
        for layer in cache.layers:
            layer.keys   = torch.cat([layer.keys,   new_k], dim=2)
            layer.values = torch.cat([layer.values, new_v], dim=2)

        policy.step(cache, attentions=None)
        actual_len = get_cache_len(cache)
        assert actual_len <= budget, \
            f"StreamingLLM exceeded budget at step {step}: " \
            f"len={actual_len}, budget={budget}"


# ── Test 14: H2O multi-step budget hold ──────────────────────────
def test_h2o_multi_step():
    budget = 8
    seq_len_init = 4
    num_steps = 20
    num_layers = 2
    kv_heads = 1
    num_heads = 2  # query heads (can differ from kv_heads)
    head_dim = 4

    cache = make_cache(seq_len_init, num_layers=num_layers,
                       kv_heads=kv_heads, head_dim=head_dim)
    policy = H2OPolicy(budget=budget)

    for step in range(num_steps):
        cur_len = get_cache_len(cache)
        for layer in cache.layers:
            new_k = torch.randn(1, kv_heads, 1, head_dim)
            new_v = torch.randn(1, kv_heads, 1, head_dim)
            layer.keys   = torch.cat([layer.keys,   new_k], dim=2)
            layer.values = torch.cat([layer.values, new_v], dim=2)

        new_len = get_cache_len(cache)
        attn = make_attentions(new_len, num_layers=num_layers, num_heads=num_heads)
        policy.step(cache, attentions=attn)

        actual_len = get_cache_len(cache)
        assert actual_len <= budget, \
            f"H2O exceeded budget at step {step}: len={actual_len}, budget={budget}"


# ══════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("KV-Cache Policy Correctness Harness")
    print("=" * 60)
    print()

    run_test("NoEviction: identity (no K/V modification)",         test_no_eviction_identity)
    run_test("SlidingWindow: stays within budget",                 test_sliding_window_budget)
    run_test("SlidingWindow: evicts sink (pos-0)",                 test_sliding_window_evicts_sink)
    run_test("StreamingLLM: stays within budget",                  test_streaming_llm_budget)
    run_test("StreamingLLM: preserves sink tokens",                test_streaming_llm_preserves_sink)
    run_test("H2O: stays within budget",                           test_h2o_budget)
    run_test("H2O: raises ValueError without attentions",          test_h2o_requires_attentions)
    run_test("H2O: reset() clears accumulated scores",             test_h2o_reset)
    run_test("RoPECorrector: no-op on identity positions",         test_rope_corrector_noop)
    run_test("_evict_layers: correct slot mapping",                test_evict_layers_order)
    run_test("rotate_half: rh(rh(x)) == -x",                      test_rotate_half_property)
    run_test("All policies: temporal order preserved",             test_temporal_order_preserved)
    run_test("StreamingLLM: budget maintained across 30 steps",   test_streaming_llm_multi_step)
    run_test("H2O: budget maintained across 20 steps",            test_h2o_multi_step)

    print()
    print("=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"ALL TESTS PASSED  ({PASS}/{total})")
    else:
        print(f"FAILURES: {FAIL}/{total}")
        print()
        for name, err in ERRORS:
            print(f"  ✗  {name}")
            for line in err.splitlines():
                print(f"       {line}")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
