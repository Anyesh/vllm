# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.evoke import EvokeCachePolicy


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
