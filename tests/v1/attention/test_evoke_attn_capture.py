# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.attention.evoke_attn_capture import (
    advance_decode_step,
    all_captures,
    clear,
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
