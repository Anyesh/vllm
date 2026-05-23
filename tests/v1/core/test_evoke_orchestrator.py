# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.attention.evoke_attn_capture import (
    CaptureRecord,
    compute_attention_weights,
)
from vllm.v1.core.eviction_policy import EvokeBlockEvictionPolicy
from vllm.v1.core.evoke_orchestrator import EvokeCaptureOrchestrator


@dataclass
class _FakeBlock:
    block_id: int


def _seed_policy(policy: EvokeBlockEvictionPolicy, block_ids: list[int]) -> None:
    for bid in block_ids:
        policy.on_block_freed(_FakeBlock(block_id=bid))


def _make_capture(num_q: int, num_heads: int, num_kv: int) -> CaptureRecord:
    query = torch.randn(num_q, num_heads, 8)
    key = torch.randn(num_kv, num_heads, 8)
    weights = compute_attention_weights(query, key, causal=False)
    return CaptureRecord(
        query_shape=tuple(query.shape),
        key_shape=tuple(key.shape),
        value_shape=tuple(key.shape),
        decode_step=0,
        weights=weights,
    )


def test_on_capture_step_pushes_scores_to_policy():
    policy = EvokeBlockEvictionPolicy()
    _seed_policy(policy, [7, 11, 13, 19])
    orch = EvokeCaptureOrchestrator(policy)

    capture = _make_capture(num_q=1, num_heads=2, num_kv=8)
    # Block table: tokens 0..3 in block 7, 4..7 in block 11
    block_table = torch.tensor([7, 11])
    block_size = 4

    scores = orch.on_capture_step(capture, block_table, block_size)
    assert scores is not None
    assert set(scores.keys()) <= {7, 11}
    assert policy.meta[7].attention_score > 0.0
    assert policy.meta[11].attention_score > 0.0


def test_on_capture_step_noop_when_weights_none():
    policy = EvokeBlockEvictionPolicy()
    _seed_policy(policy, [0])
    orch = EvokeCaptureOrchestrator(policy)

    capture = CaptureRecord(
        query_shape=(1, 2, 8),
        key_shape=(4, 2, 8),
        value_shape=(4, 2, 8),
        decode_step=0,
        weights=None,
    )
    assert orch.on_capture_step(capture, torch.tensor([0]), block_size=4) is None
    assert policy.meta[0].attention_score == 0.0


def test_on_capture_step_only_emits_nonzero_scores():
    """Block ids that span the range but receive zero mass should not flood
    the policy's update path with no-op writes."""
    policy = EvokeBlockEvictionPolicy()
    # Mass distribution: all weight on positions in block 7, none on block 11
    weights = torch.zeros(1, 1, 8)
    weights[:, :, :4] = 0.25
    capture = CaptureRecord(
        query_shape=(1, 1, 8),
        key_shape=(8, 1, 8),
        value_shape=(8, 1, 8),
        decode_step=0,
        weights=weights,
    )
    _seed_policy(policy, [7])
    orch = EvokeCaptureOrchestrator(policy)
    block_table = torch.tensor([7, 11])
    scores = orch.on_capture_step(capture, block_table, block_size=4)
    assert scores is not None
    assert 7 in scores
    assert 11 not in scores


def test_on_turn_boundary_decays_recovery_strength():
    policy = EvokeBlockEvictionPolicy()
    _seed_policy(policy, [0, 1])
    policy.set_recovery_strength(0, 1.0)
    policy.set_recovery_strength(1, 0.5)
    orch = EvokeCaptureOrchestrator(policy, recovery_decay=0.5)

    orch.on_turn_boundary()
    assert abs(policy.meta[0].recovery_strength - 0.5) < 1e-6
    assert abs(policy.meta[1].recovery_strength - 0.25) < 1e-6


def test_on_turn_boundary_accepts_per_call_decay_override():
    policy = EvokeBlockEvictionPolicy()
    _seed_policy(policy, [0])
    policy.set_recovery_strength(0, 1.0)
    orch = EvokeCaptureOrchestrator(policy, recovery_decay=0.7)

    orch.on_turn_boundary(recovery_decay=0.1)
    assert abs(policy.meta[0].recovery_strength - 0.1) < 1e-6


def test_mark_recovered_tags_block():
    policy = EvokeBlockEvictionPolicy()
    _seed_policy(policy, [0])
    orch = EvokeCaptureOrchestrator(policy)

    orch.mark_recovered(0)
    assert policy.meta[0].recovery_strength == 1.0
    orch.mark_recovered(0, recovery_strength=0.4)
    assert policy.meta[0].recovery_strength == 0.4


def test_update_block_embedding_pushes_to_policy():
    policy = EvokeBlockEvictionPolicy()
    _seed_policy(policy, [0])
    orch = EvokeCaptureOrchestrator(policy)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    orch.update_block_embedding(0, emb)
    got = policy.get_embedding(0)
    assert got is not None
    assert np.array_equal(got, emb)


def test_high_attention_block_outscores_low_after_capture():
    """End-to-end: after a capture step pushes asymmetric attention into
    the policy, the policy's eviction selection should reflect the
    attention signal (low-attention block evicts first)."""
    policy = EvokeBlockEvictionPolicy()
    policy.w_attention = 1.0
    policy.w_recency = 0.0
    _seed_policy(policy, [7, 11])

    # Construct weights that concentrate on block 7's positions
    weights = torch.zeros(1, 1, 8)
    weights[:, :, :4] = 0.25  # block 7
    weights[:, :, 4:] = 0.0  # block 11 (would be zero scores)
    capture = CaptureRecord(
        query_shape=(1, 1, 8),
        key_shape=(8, 1, 8),
        value_shape=(8, 1, 8),
        decode_step=0,
        weights=weights,
    )
    orch = EvokeCaptureOrchestrator(policy)
    orch.on_capture_step(capture, torch.tensor([7, 11]), block_size=4)

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 11
