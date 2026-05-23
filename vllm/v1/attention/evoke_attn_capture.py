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
    # Computed attention weights softmax(Q @ K^T / sqrt(d_h)) when the caller
    # supplied a K_full reconstruction. None if only shape was recorded.
    weights: torch.Tensor | None = None


def reconstruct_key_full_paged(
    key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Reconstruct K_full for one sequence from a paged KV cache.

    vLLM stores K in pages of `block_size` tokens, indexed by per-sequence
    block_table rows. For position i in [0, seq_len) the physical block id
    is block_table_row[i // block_size] and the within-block offset is
    i % block_size. This function gathers those positions into a contiguous
    [seq_len, num_kv_heads, head_size] tensor suitable for the attention
    side-compute.

    Args:
        key_cache: shape [num_blocks, block_size, num_kv_heads, head_size]
            (the K half of the per-layer paged KV cache).
        block_table_row: shape [num_pages_for_this_seq], physical block ids
            for the sequence's logical pages.
        seq_len: number of valid tokens in this sequence.
        block_size: page size in tokens.

    Returns:
        shape [seq_len, num_kv_heads, head_size].
    """
    device = key_cache.device
    positions = torch.arange(seq_len, device=device)
    page_idx = positions // block_size
    intra_offset = positions % block_size
    # block_table_row may be longer than the number of pages we need.
    physical_blocks = block_table_row[page_idx]
    return key_cache[physical_blocks, intra_offset]


def compute_attention_weights(
    query: torch.Tensor,
    key_full: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Compute softmax(Q @ K_full^T / sqrt(d_h)) for the capture layer.

    Mirrors what FlashAttention's fused kernel computes internally, run as a
    side-compute on the same Q and K tensors. The numeric result is suitable
    for the EVOKE multi-signal scorer; it is not used to feed the model.

    Args:
        query: shape [num_q, num_heads, head_size].
        key_full: shape [num_kv, num_kv_heads, head_size]. Caller must
            reconstruct this from the paged KV cache for the relevant
            sequence; passing only the per-step `key` from the layer's
            forward gives current-step-only attention, not the historical
            distribution.
        causal: when True, mask out positions i,j where j > num_kv -
            num_q + i so the query can only attend to past keys. Set False
            for bidirectional attention.

    Returns:
        shape [num_q, num_heads, num_kv] in float32, summing to 1.0 along
        the last axis at every (q, head) position.
    """
    num_q, num_heads, head_size = query.shape
    num_kv, num_kv_heads, _ = key_full.shape
    if num_heads != num_kv_heads:
        # GQA: replicate each KV head to span the heads that share it.
        repeats = num_heads // num_kv_heads
        key_full = key_full.repeat_interleave(repeats, dim=1)
    # einsum: [num_q, num_heads, head_size] @ [num_kv, num_heads, head_size]
    # -> [num_q, num_heads, num_kv]
    scores = torch.einsum("qhd,khd->qhk", query.float(), key_full.float())
    scores /= head_size**0.5
    if causal:
        # The last num_q positions of K correspond to the Q tokens. A query
        # at position i attends to keys [0 .. (num_kv - num_q + i)].
        kv_positions = torch.arange(num_kv, device=query.device)
        q_positions = torch.arange(num_q, device=query.device) + (num_kv - num_q)
        # mask[i, j] = True where j > q_pos_i (future position)
        mask = kv_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        scores = scores.masked_fill(
            mask.unsqueeze(1),  # broadcast over heads
            float("-inf"),
        )
    return torch.softmax(scores, dim=-1)


def aggregate_per_block(
    weights: torch.Tensor,
    block_ids: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    """Aggregate per-token attention weights into per-block sums.

    Args:
        weights: shape [num_q, num_heads, num_kv] from
            compute_attention_weights.
        block_ids: shape [num_kv], integer block id for each KV position.
        num_blocks: total number of blocks the aggregation spans.

    Returns:
        shape [num_q, num_blocks] in float32: total attention from each query
        to each block, averaged across heads. This is the per-block signal
        the EVOKE scorer consumes.
    """
    num_q, num_heads, num_kv = weights.shape
    head_avg = weights.mean(dim=1)  # [num_q, num_kv]
    out = torch.zeros(num_q, num_blocks, dtype=head_avg.dtype, device=weights.device)
    out.scatter_add_(1, block_ids.unsqueeze(0).expand(num_q, -1), head_avg)
    return out


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
    key_full: torch.Tensor | None = None,
    causal: bool = True,
) -> None:
    # Strict no-op for layers not registered for capture; the cost on the
    # hot attention path is a single dict lookup under a lock. When the
    # caller supplies key_full (the reconstruction of K from the paged KV
    # cache for the active sequences), the softmax(Q @ K^T) weights are
    # computed and stored on the record. Without key_full, only the shape
    # metadata is recorded.
    if layer_name not in _STATE.enabled_layers:
        return
    weights = (
        compute_attention_weights(query, key_full, causal=causal)
        if key_full is not None
        else None
    )
    with _STATE.lock:
        if layer_name not in _STATE.enabled_layers:
            return
        _STATE.last_capture[layer_name] = CaptureRecord(
            query_shape=tuple(query.shape),
            key_shape=tuple(key.shape) if key is not None else (),
            value_shape=tuple(value.shape) if value is not None else (),
            decode_step=_STATE.decode_step,
            weights=weights,
        )
