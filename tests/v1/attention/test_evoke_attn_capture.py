# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.attention.evoke_attn_capture import (
    advance_decode_step,
    aggregate_per_block,
    all_captures,
    clear,
    compute_attention_weights,
    get_capture,
    is_enabled,
    maybe_capture,
    register_layer,
    unregister_layer,
)


@pytest.fixture(autouse=True)
def _reset_state():
    clear()
    yield
    clear()


def _q(num_tokens: int = 1, num_heads: int = 4, head_size: int = 64) -> torch.Tensor:
    return torch.zeros(num_tokens, num_heads, head_size)


def _k(num_tokens: int = 1, num_kv_heads: int = 4, head_size: int = 64) -> torch.Tensor:
    return torch.zeros(num_tokens, num_kv_heads, head_size)


def test_register_and_is_enabled():
    assert not is_enabled("layer_20")
    register_layer("layer_20")
    assert is_enabled("layer_20")
    unregister_layer("layer_20")
    assert not is_enabled("layer_20")


def test_maybe_capture_noop_for_unregistered_layer():
    maybe_capture("not_registered", _q(), _k(), _k())
    assert get_capture("not_registered") is None


def test_maybe_capture_records_for_registered_layer():
    register_layer("layer_20")
    maybe_capture("layer_20", _q(num_tokens=1), _k(num_tokens=32), _k(num_tokens=32))

    record = get_capture("layer_20")
    assert record is not None
    assert record.query_shape == (1, 4, 64)
    assert record.key_shape == (32, 4, 64)
    assert record.value_shape == (32, 4, 64)


def test_maybe_capture_handles_none_kv():
    # Cross-attention paths may not pass key/value at the layer boundary;
    # the capture must record an empty shape rather than crash.
    register_layer("layer_20")
    maybe_capture("layer_20", _q(), None, None)
    record = get_capture("layer_20")
    assert record is not None
    assert record.key_shape == ()
    assert record.value_shape == ()


def test_clear_drops_state():
    register_layer("layer_20")
    maybe_capture("layer_20", _q(), _k(), _k())
    clear()
    assert get_capture("layer_20") is None
    assert not is_enabled("layer_20")


def test_decode_step_advances():
    s0 = advance_decode_step()
    s1 = advance_decode_step()
    s2 = advance_decode_step()
    assert s1 == s0 + 1
    assert s2 == s1 + 1


def test_capture_records_current_decode_step():
    register_layer("layer_20")
    advance_decode_step()
    advance_decode_step()
    advance_decode_step()
    maybe_capture("layer_20", _q(), _k(), _k())
    record = get_capture("layer_20")
    assert record is not None
    assert record.decode_step == 3


def test_multi_layer_capture():
    register_layer("layer_5")
    register_layer("layer_20")
    maybe_capture("layer_5", _q(num_heads=8), _k(num_kv_heads=8), _k(num_kv_heads=8))
    maybe_capture(
        "layer_20", _q(num_heads=16), _k(num_kv_heads=16), _k(num_kv_heads=16)
    )

    captures = all_captures()
    assert set(captures.keys()) == {"layer_5", "layer_20"}
    assert captures["layer_5"].query_shape == (1, 8, 64)
    assert captures["layer_20"].query_shape == (1, 16, 64)


def test_unregister_stops_capture():
    register_layer("layer_20")
    maybe_capture("layer_20", _q(), _k(), _k())
    assert get_capture("layer_20") is not None

    unregister_layer("layer_20")
    clear()
    maybe_capture("layer_20", _q(num_tokens=99), _k(num_tokens=99), _k(num_tokens=99))
    assert get_capture("layer_20") is None


def test_compute_attention_weights_softmax_sums_to_one():
    # Per-(q, head) row of weights must sum to 1.0 within numerical tolerance.
    query = torch.randn(2, 4, 64)
    key = torch.randn(8, 4, 64)
    weights = compute_attention_weights(query, key, causal=False)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_compute_attention_weights_shape():
    query = torch.randn(3, 8, 64)
    key = torch.randn(16, 8, 64)
    weights = compute_attention_weights(query, key, causal=False)
    assert tuple(weights.shape) == (3, 8, 16)


def test_compute_attention_weights_causal_mask():
    """In causal mode the last K positions correspond to the Q tokens; query
    at offset i can only attend to keys [0..num_kv - num_q + i]."""
    num_q, num_heads, head_size = 2, 4, 16
    num_kv = 6
    query = torch.randn(num_q, num_heads, head_size)
    key = torch.randn(num_kv, num_heads, head_size)
    weights = compute_attention_weights(query, key, causal=True)
    # Query 0 corresponds to KV position num_kv - num_q + 0 = 4. Position 5 forbidden.
    # Query 1 corresponds to KV position 5. No positions forbidden.
    assert torch.allclose(weights[0, :, 5], torch.zeros(num_heads), atol=1e-6)
    # Query 0 should still attend to positions 0..4 (some non-zero mass each).
    assert weights[0, :, :5].sum().item() > 0.99
    # Causal row sums are still 1.
    assert torch.allclose(weights.sum(dim=-1), torch.ones(num_q, num_heads), atol=1e-5)


def test_compute_attention_weights_gqa():
    """GQA: num_heads > num_kv_heads. Each KV head is shared across multiple
    Q heads. compute_attention_weights replicates K to span Q's heads."""
    query = torch.randn(1, 8, 64)
    # 2 KV heads serving 8 Q heads (groups of 4)
    key = torch.randn(4, 2, 64)
    weights = compute_attention_weights(query, key, causal=False)
    assert tuple(weights.shape) == (1, 8, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 8), atol=1e-5)


def test_aggregate_per_block_sums_correctly():
    # Two queries, 2 heads, 8 KV tokens. 4 blocks of 2 tokens each.
    weights = torch.zeros(2, 2, 8)
    weights[:, :, :2] = 0.25  # block 0 carries all mass
    weights[:, :, 2:4] = 0.25  # block 1
    weights[:, :, 4:6] = 0.25  # block 2
    weights[:, :, 6:8] = 0.25  # block 3
    block_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    per_block = aggregate_per_block(weights, block_ids, num_blocks=4)
    assert tuple(per_block.shape) == (2, 4)
    # Each block sums to 0.5 (two tokens at 0.25, averaged across 2 heads)
    assert torch.allclose(per_block, torch.full((2, 4), 0.5), atol=1e-6)


def test_maybe_capture_stores_weights_when_key_full_provided():
    register_layer("layer_20")
    query = torch.randn(1, 4, 64)
    key_full = torch.randn(8, 4, 64)
    maybe_capture(
        "layer_20",
        query,
        key=key_full[-1:],
        value=key_full[-1:],
        key_full=key_full,
        causal=False,
    )
    record = get_capture("layer_20")
    assert record is not None
    assert record.weights is not None
    assert tuple(record.weights.shape) == (1, 4, 8)
    row_sums = record.weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_maybe_capture_weights_none_without_key_full():
    register_layer("layer_20")
    maybe_capture("layer_20", _q(), _k(), _k())
    record = get_capture("layer_20")
    assert record is not None
    assert record.weights is None
