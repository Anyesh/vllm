# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np

from vllm.v1.core.eviction_policy import (
    SOURCE_ASSISTANT,
    SOURCE_DOCUMENT,
    SOURCE_USER,
    EvokeBlockEvictionPolicy,
    EvokeRequestMeta,
    LRUBlockEvictionPolicy,
)


@dataclass
class _FakeBlock:
    # Minimal duck-typed stand-in for vllm.v1.core.kv_cache_utils.KVCacheBlock.
    # The eviction-policy surface only reads block.block_id, so the policy
    # tests avoid pulling the full vllm config/kv_cache stack into scope.
    block_id: int


def _block(block_id: int) -> _FakeBlock:
    return _FakeBlock(block_id=block_id)


def test_lru_evicts_oldest_first():
    policy = LRUBlockEvictionPolicy()
    for i in range(5):
        policy.on_block_freed(_block(i))

    chosen = policy.select_eviction_candidates(n=2, protected=set())
    assert chosen is not None
    assert [b.block_id for b in chosen] == [0, 1]


def test_lru_skips_protected():
    policy = LRUBlockEvictionPolicy()
    for i in range(5):
        policy.on_block_freed(_block(i))

    chosen = policy.select_eviction_candidates(n=2, protected={0, 1})
    assert chosen is not None
    assert [b.block_id for b in chosen] == [2, 3]


def test_lru_returns_none_when_insufficient():
    policy = LRUBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))

    assert policy.select_eviction_candidates(n=3, protected=set()) is None


def test_lru_on_block_allocated_removes_from_pool():
    policy = LRUBlockEvictionPolicy()
    b0 = _block(0)
    b1 = _block(1)
    policy.on_block_freed(b0)
    policy.on_block_freed(b1)
    policy.on_block_allocated(b0)

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_lru_zero_n_returns_empty():
    policy = LRUBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    assert policy.select_eviction_candidates(n=0, protected=set()) == []


def test_lru_clear():
    policy = LRUBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.clear()
    assert policy.num_evictable() == 0
    assert policy.select_eviction_candidates(n=1, protected=set()) is None


def test_evoke_evicts_oldest_when_only_recency_is_active():
    policy = EvokeBlockEvictionPolicy()
    for i in range(3):
        policy.on_block_freed(_block(i))

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 0


def test_evoke_skips_pinned():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.set_pinned(0, True)

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_evoke_skips_protected_set():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))

    chosen = policy.select_eviction_candidates(n=1, protected={0})
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_evoke_priority_protects_high_priority_block():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.set_priority(0, 10.0)

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    # block 0 is older (freed first) but the priority multiplier makes its
    # score 10x. So block 1 is the eviction candidate.
    assert chosen[0].block_id == 1


def test_evoke_attention_score_protects_high_attention_block():
    policy = EvokeBlockEvictionPolicy()
    policy.w_attention = 1.0
    policy.w_recency = 0.0
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.set_attention_score(0, 0.9)
    policy.set_attention_score(1, 0.1)

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_evoke_source_type_floor_protects_user_above_document():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.set_source_type(0, SOURCE_USER)
    for _ in range(200):
        policy._tick += 1
    policy.on_block_freed(_block(1))

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    # The user-floor (0.6) for block 0 keeps it above block 1's raw recency
    assert chosen[0].block_id == 1


def test_evoke_source_type_floor_user_above_assistant():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.set_source_type(0, SOURCE_USER)
    policy.on_block_freed(_block(1))
    policy.set_source_type(1, SOURCE_ASSISTANT)
    for _ in range(500):
        policy._tick += 1

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_evoke_returns_none_when_all_protected():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.set_pinned(0, True)
    policy.set_pinned(1, True)

    assert policy.select_eviction_candidates(n=1, protected=set()) is None


def test_evoke_update_attention_scores_bulk():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))

    policy.update_attention_scores({0: 0.8, 1: 0.4, 99: 0.9})

    assert policy.meta[0].attention_score == 0.8
    assert policy.meta[1].attention_score == 0.4
    assert 99 not in policy.meta


def test_evoke_decay_recovery_strength():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.set_recovery_strength(0, 1.0)
    policy.set_recovery_strength(1, 0.5)

    policy.decay_recovery_strength(0.7)
    assert abs(policy.meta[0].recovery_strength - 0.7) < 1e-6
    assert abs(policy.meta[1].recovery_strength - 0.35) < 1e-6


def test_evoke_embedding_storage():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    policy.set_embedding(0, emb)
    got = policy.get_embedding(0)
    assert got is not None
    assert np.array_equal(got, emb)


def test_evoke_on_block_allocated_takes_out_of_free_pool():
    policy = EvokeBlockEvictionPolicy()
    b0 = _block(0)
    b1 = _block(1)
    policy.on_block_freed(b0)
    policy.on_block_freed(b1)
    policy.on_block_allocated(b0)
    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_evoke_drop_meta_removes_state():
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.set_priority(0, 5.0)
    policy.drop_meta([0])
    assert 0 not in policy.meta


def test_evoke_multi_signal_recommended_recipe():
    """The recommended config (w_attention=0.5, w_recency=0.2,
    w_coherence=0.3) should pick the block with the lowest combined
    score across all three signals."""
    policy = EvokeBlockEvictionPolicy()
    policy.w_attention = 0.5
    policy.w_recency = 0.2
    policy.w_coherence = 0.3
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.set_attention_score(0, 0.9)
    policy.set_coherence_score(0, 0.9)
    policy.set_attention_score(1, 0.1)
    policy.set_coherence_score(1, 0.0)

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_request_meta_round_trip():
    policy = EvokeBlockEvictionPolicy()
    meta = EvokeRequestMeta(source_type=SOURCE_USER, priority=2.0, pinned=False)
    policy.set_request_meta("req-1", meta)
    assert policy.request_meta["req-1"] is meta
    policy.drop_request_meta("req-1")
    assert "req-1" not in policy.request_meta


def test_assign_block_to_request_applies_meta():
    policy = EvokeBlockEvictionPolicy()
    policy.set_request_meta(
        "req-1",
        EvokeRequestMeta(source_type=SOURCE_USER, priority=3.0, pinned=False),
    )
    policy.on_block_freed(_block(0))
    policy.assign_block_to_request(0, "req-1")

    assert policy.meta[0].source_type == SOURCE_USER
    assert policy.meta[0].priority == 3.0
    assert policy.meta[0].request_id == "req-1"


def test_assign_block_to_request_without_meta_leaves_defaults():
    """Assigning a block to a request whose meta is not yet registered
    sets the block's request_id but leaves source_type/priority/pinned
    at defaults. The harness may register the request meta after blocks
    are allocated for early prefill, or never register meta at all."""
    policy = EvokeBlockEvictionPolicy()
    policy.on_block_freed(_block(0))
    policy.assign_block_to_request(0, "req-2")
    assert policy.meta[0].request_id == "req-2"
    assert policy.meta[0].source_type is None
    assert policy.meta[0].priority == 1.0
    assert not policy.meta[0].pinned


def test_request_meta_drives_eviction_via_floor():
    """End-to-end: a user-source request's blocks should outlive a
    document-source request's blocks even when both are similarly aged,
    because the user floor (0.6) is higher than what document blocks
    score from raw recency."""
    policy = EvokeBlockEvictionPolicy()
    policy.set_request_meta("user-req", EvokeRequestMeta(source_type=SOURCE_USER))
    policy.set_request_meta("doc-req", EvokeRequestMeta(source_type=SOURCE_DOCUMENT))
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.assign_block_to_request(0, "user-req")
    policy.assign_block_to_request(1, "doc-req")
    for _ in range(200):
        policy._tick += 1

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1


def test_request_meta_pinned_prevents_eviction():
    policy = EvokeBlockEvictionPolicy()
    policy.set_request_meta(
        "pinned-req",
        EvokeRequestMeta(source_type=SOURCE_USER, priority=1.0, pinned=True),
    )
    policy.on_block_freed(_block(0))
    policy.on_block_freed(_block(1))
    policy.assign_block_to_request(0, "pinned-req")

    chosen = policy.select_eviction_candidates(n=1, protected=set())
    assert chosen is not None
    assert chosen[0].block_id == 1
