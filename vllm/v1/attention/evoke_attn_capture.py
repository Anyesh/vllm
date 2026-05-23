# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVOKE attention-weight capture for the multi-signal eviction scorer.

The EVOKE policy in `vllm/v1/kv_offload/cpu/policies/evoke.py` wants the
model's own per-block attention pattern as one of the signals that drives
eviction. FlashAttention fuses softmax inside its CUDA kernel and never
materializes the QK^T weights as a tensor, so we run a side-compute path
for one (or a small set of) chosen layers and write the result to a
host-resident buffer the policy can read.

The capture point is `Attention.forward` (vllm/model_executor/layers/
attention/attention.py): a single hook call fires before the FA dispatch,
runs only when the current layer's name is registered for capture, and is
a strict no-op otherwise. The main attention path is unchanged: FA still
runs, the model output is byte-identical with capture off versus capture
on.

This module exposes the registration/read API. The numeric softmax(QK^T)
computation against the paged KV cache lands in a follow-on commit; the
current capture writes shape metadata so the wiring can be tested
end-to-end first.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch


@dataclass
class CaptureRecord:
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    decode_step: int


@dataclass
class _CaptureState:
    enabled_layers: set[str] = field(default_factory=set)
    last_capture: dict[str, CaptureRecord] = field(default_factory=dict)
    decode_step: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATE = _CaptureState()


def register_layer(layer_name: str) -> None:
    with _STATE.lock:
        _STATE.enabled_layers.add(layer_name)


def unregister_layer(layer_name: str) -> None:
    with _STATE.lock:
        _STATE.enabled_layers.discard(layer_name)


def is_enabled(layer_name: str) -> bool:
    with _STATE.lock:
        return layer_name in _STATE.enabled_layers


def clear() -> None:
    with _STATE.lock:
        _STATE.enabled_layers.clear()
        _STATE.last_capture.clear()
        _STATE.decode_step = 0


def get_capture(layer_name: str) -> CaptureRecord | None:
    with _STATE.lock:
        return _STATE.last_capture.get(layer_name)


def all_captures() -> dict[str, CaptureRecord]:
    with _STATE.lock:
        return dict(_STATE.last_capture)


def advance_decode_step() -> int:
    with _STATE.lock:
        _STATE.decode_step += 1
        return _STATE.decode_step


def maybe_capture(
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
) -> None:
    # Strict no-op for layers not registered for capture; the cost on the
    # hot attention path is a single dict lookup under a lock.
    if layer_name not in _STATE.enabled_layers:
        return
    with _STATE.lock:
        if layer_name not in _STATE.enabled_layers:
            return
        _STATE.last_capture[layer_name] = CaptureRecord(
            query_shape=tuple(query.shape),
            key_shape=tuple(key.shape) if key is not None else (),
            value_shape=tuple(value.shape) if value is not None else (),
            decode_step=_STATE.decode_step,
        )
