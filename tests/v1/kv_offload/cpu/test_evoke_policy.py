# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.evoke import (
    SOURCE_ASSISTANT,
    SOURCE_DOCUMENT,
    SOURCE_USER,
    EvokeCachePolicy,
    cosine_similarity,
)


def _make_block(block_id: int, ref_cnt: int = 0) -> BlockStatus:
    block = BlockStatus(block_id)
    block.ref_cnt = ref_cnt
    return block


def test_insert_get_remove_roundtrip():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"a", _make_block(0))
    policy.insert(b"b", _make_block(1))

    assert policy.get(b"a") is not None
    assert policy.get(b"a").block_id == 0
    assert policy.get(b"missing") is None

    policy.remove(b"a")
    assert policy.get(b"a") is None


def test_evict_picks_oldest_when_signals_default():
    """With w_coherence/w_recovery effectively zero (coherence stub returns 0,
    recovery_strength defaults to 0), the score reduces to recency: oldest
    block should evict first."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"oldest", _make_block(0))
    policy.insert(b"middle", _make_block(1))
    policy.insert(b"newest", _make_block(2))

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    assert len(evicted) == 1
    assert evicted[0][0] == b"oldest"


def test_evict_skips_protected():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"oldest", _make_block(0))
    policy.insert(b"newer", _make_block(1))

    evicted = policy.evict(n=1, protected={b"oldest"})
    assert evicted is not None
    assert evicted[0][0] == b"newer"


def test_evict_skips_active_refcnt():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"active", _make_block(0, ref_cnt=1))
    policy.insert(b"idle", _make_block(1))

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == b"idle"


def test_evict_returns_none_when_insufficient():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"only", _make_block(0, ref_cnt=1))

    # Need 1 eviction; only block is ref_cnt > 0 so not evictable
    assert policy.evict(n=1, protected=set()) is None


def test_touch_resets_recency():
    """touch() bumps a block's last_touch_tick so it survives an eviction
    that would otherwise pick it as the oldest."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"target", _make_block(0))
    policy.insert(b"newer", _make_block(1))
    policy.touch([b"target"])

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    # target was touched after newer was inserted, so newer is now oldest
    assert evicted[0][0] == b"newer"


def test_clear_removes_everything():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"a", _make_block(0))
    policy.insert(b"b", _make_block(1))
    policy.clear()
    assert policy.get(b"a") is None
    assert policy.get(b"b") is None


def test_priority_multiplier_protects_high_priority_block():
    """A block with high priority should outscore an older block with default
    priority, even though the older block has been touched more recently."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"important", _make_block(0))
    policy.insert(b"recent", _make_block(1))
    # Boost "important" to be load-bearing
    policy.meta[b"important"].priority = 10.0

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == b"recent"


def test_recovery_strength_protects_recovered_block():
    """A freshly-recovered block (recovery_strength=1.0, w_recovery enabled)
    should outscore a default block with recency 1.0 but no recovery signal,
    once w_recovery is turned on."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.w_recovery = 1.0
    policy.insert(b"recovered", _make_block(0))
    policy.insert(b"plain", _make_block(1))
    policy.meta[b"recovered"].recovery_strength = 1.0

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    # recovered has recency lower (older) but recovery_strength=1.0 boosts it
    assert evicted[0][0] == b"plain"


def test_source_type_floor_lifts_user_above_document():
    """A user-source block should outlive a document-source block even when
    both are equally old. The default user floor (0.6) is much higher than
    the default-recency-only score a document block can achieve from age
    alone."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"user_turn", _make_block(0))
    policy.insert(b"doc_chunk", _make_block(1))
    policy.set_source_type(b"user_turn", SOURCE_USER)
    policy.set_source_type(b"doc_chunk", SOURCE_DOCUMENT)
    # Age the user_turn so its raw recency drops below the floor
    for _ in range(200):
        policy.touch([b"doc_chunk"])

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == b"doc_chunk"


def test_source_type_floor_assistant_below_user():
    """Default floors order user > assistant. When both are aged out to
    near-zero recency, the assistant block evicts before the user block."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"user", _make_block(0))
    policy.insert(b"assistant", _make_block(1))
    policy.set_source_type(b"user", SOURCE_USER)
    policy.set_source_type(b"assistant", SOURCE_ASSISTANT)
    # Age both
    for _ in range(500):
        policy._tick += 1

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == b"assistant"


def test_pinned_block_never_evicts():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"pinned", _make_block(0))
    policy.insert(b"evictable", _make_block(1))
    policy.set_pinned(b"pinned", True)

    evicted = policy.evict(n=1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == b"evictable"


def test_pinned_block_blocks_n_eviction():
    """If the only-evictable set is too small after pinning, evict returns
    None (caller cannot make space)."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"a", _make_block(0))
    policy.insert(b"b", _make_block(1))
    policy.set_pinned(b"a", True)
    policy.set_pinned(b"b", True)

    assert policy.evict(n=1, protected=set()) is None


def test_set_priority_round_trip():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"a", _make_block(0))
    policy.set_priority(b"a", 5.0)
    assert policy.meta[b"a"].priority == 5.0


def test_embedding_storage_round_trip():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"a", _make_block(0))
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    policy.set_embedding(b"a", emb)
    got = policy.get_embedding(b"a")
    assert got is not None
    assert np.array_equal(got, emb)


def test_cosine_similarity_known_values():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0], dtype=np.float32)
    d = np.array([-1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    assert abs(cosine_similarity(a, c) - 0.0) < 1e-6
    assert abs(cosine_similarity(a, d) - (-1.0)) < 1e-6


def test_cosine_similarity_zero_vector():
    # Zero vector returns 0.0 so callers never see NaN from a numerical edge
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0


def test_select_for_recovery_ranks_by_similarity():
    """Smart-recovery: among evicted candidates, return the top-k by cosine
    similarity to the query, beating both the absolute floor and the
    strongest-resident threshold."""
    policy = EvokeCachePolicy(cache_capacity=4)
    for key in (b"hit", b"weak", b"strong", b"off_topic"):
        policy.insert(key, _make_block(0))
    policy.set_embedding(b"hit", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    policy.set_embedding(b"weak", np.array([0.7, 0.7, 0.0], dtype=np.float32))
    policy.set_embedding(b"strong", np.array([0.9, 0.1, 0.0], dtype=np.float32))
    policy.set_embedding(b"off_topic", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    selected = policy.select_for_recovery(
        query_embedding=query,
        candidate_keys=[b"hit", b"weak", b"strong", b"off_topic"],
        resident_max_similarity=0.6,
        top_k=2,
        min_similarity=0.0,
    )
    keys = [k for k, _ in selected]
    # hit (1.0) and strong (0.9) beat resident 0.6; weak (~0.7) only marginally
    assert keys[0] == b"hit"
    assert keys[1] == b"strong"
    assert b"off_topic" not in keys


def test_select_for_recovery_respects_resident_gate():
    """A candidate whose similarity does not beat the strongest resident
    match must not be returned."""
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"weak", _make_block(0))
    policy.set_embedding(b"weak", np.array([0.5, 0.5, 0.0], dtype=np.float32))

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    selected = policy.select_for_recovery(
        query_embedding=query,
        candidate_keys=[b"weak"],
        resident_max_similarity=0.9,
        top_k=4,
    )
    assert selected == []


def test_select_for_recovery_skips_candidates_without_embedding():
    policy = EvokeCachePolicy(cache_capacity=4)
    policy.insert(b"no_emb", _make_block(0))
    policy.insert(b"with_emb", _make_block(1))
    policy.set_embedding(b"with_emb", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    selected = policy.select_for_recovery(
        query_embedding=query,
        candidate_keys=[b"no_emb", b"with_emb"],
        resident_max_similarity=0.0,
        top_k=4,
    )
    keys = [k for k, _ in selected]
    assert keys == [b"with_emb"]
