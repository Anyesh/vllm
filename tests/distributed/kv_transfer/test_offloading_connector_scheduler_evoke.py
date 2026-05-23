"""Unit tests for EVOKE smart-recovery integration in
OffloadingConnectorScheduler.

The full scheduler flow has heavy dependencies (Request, KVCacheBlocks,
VllmConfig). These tests cover the new methods at a lower level by
constructing a minimal stand-in for the scheduler that exposes just the
attributes the smart-recovery helper needs (self.manager). End-to-end
chihiro testing of the connector flow lands in a follow-on bench harness.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


_EMPTY_REQ_CTX = ReqContext(req_id="")


def _key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def _stage_offloaded_block(
    manager: CPUOffloadingManager,
    key_int: int,
    embedding: np.ndarray,
    original_position: int = 0,
) -> OffloadKey:
    key = _key(key_int)
    manager.prepare_store([key], _EMPTY_REQ_CTX, original_positions=[original_position])
    manager.complete_store([key], _EMPTY_REQ_CTX, success=True)
    manager._policy.set_embedding(key, embedding)
    return key


def _stub_scheduler(manager: CPUOffloadingManager) -> SimpleNamespace:
    """Just enough surface for _compute_smart_recovery_keys: it needs
    self.manager. The method also accesses logger via the module, not self."""
    stub = SimpleNamespace(manager=manager)
    stub._compute_smart_recovery_keys = (
        OffloadingConnectorScheduler._compute_smart_recovery_keys.__get__(stub)
    )
    return stub


def test_compute_smart_recovery_no_meta_returns_empty():
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="evoke")
    _stage_offloaded_block(manager, 1, np.array([1.0, 0.0], dtype=np.float32))

    request = SimpleNamespace(request_id="r1", evoke_request_meta=None)
    stub = _stub_scheduler(manager)
    assert stub._compute_smart_recovery_keys(request, prefix_keys_to_load=set()) == (
        [],
        [],
    )


def test_compute_smart_recovery_zero_top_k_returns_empty():
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="evoke")
    _stage_offloaded_block(manager, 1, np.array([1.0, 0.0], dtype=np.float32))
    request = SimpleNamespace(
        request_id="r1",
        evoke_request_meta={
            "query_embedding": [1.0, 0.0],
            "recover_top_k": 0,
        },
    )
    stub = _stub_scheduler(manager)
    assert stub._compute_smart_recovery_keys(request, prefix_keys_to_load=set()) == (
        [],
        [],
    )


def test_compute_smart_recovery_missing_embedding_returns_empty():
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="evoke")
    _stage_offloaded_block(manager, 1, np.array([1.0, 0.0], dtype=np.float32))
    request = SimpleNamespace(
        request_id="r1",
        evoke_request_meta={"recover_top_k": 4},
    )
    stub = _stub_scheduler(manager)
    assert stub._compute_smart_recovery_keys(request, prefix_keys_to_load=set()) == (
        [],
        [],
    )


def test_compute_smart_recovery_returns_ranked_keys():
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="evoke")
    k_hit = _stage_offloaded_block(manager, 1, np.array([1.0, 0.0], dtype=np.float32))
    k_strong = _stage_offloaded_block(
        manager, 2, np.array([0.9, 0.1], dtype=np.float32)
    )
    _stage_offloaded_block(manager, 3, np.array([0.0, 1.0], dtype=np.float32))

    request = SimpleNamespace(
        request_id="r1",
        evoke_request_meta={
            "query_embedding": [1.0, 0.0],
            "recover_top_k": 2,
            "min_similarity": 0.0,
        },
    )
    stub = _stub_scheduler(manager)
    keys, positions = stub._compute_smart_recovery_keys(
        request, prefix_keys_to_load=set()
    )
    assert k_hit in keys
    assert k_strong in keys
    assert len(keys) == len(positions)


def test_compute_smart_recovery_dedupes_against_prefix_keys():
    """When a recovery candidate is already scheduled by the prefix-extension
    path, it must be excluded from the recovery list to avoid double-loading."""
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="evoke")
    k_hit = _stage_offloaded_block(manager, 1, np.array([1.0, 0.0], dtype=np.float32))
    k_strong = _stage_offloaded_block(
        manager, 2, np.array([0.9, 0.1], dtype=np.float32)
    )

    request = SimpleNamespace(
        request_id="r1",
        evoke_request_meta={
            "query_embedding": [1.0, 0.0],
            "recover_top_k": 4,
        },
    )
    stub = _stub_scheduler(manager)
    keys, _ = stub._compute_smart_recovery_keys(request, prefix_keys_to_load={k_hit})
    assert k_hit not in keys
    assert k_strong in keys


def test_compute_smart_recovery_accepts_dataclass_meta():
    """Meta may arrive as an EvokeRequestMeta dataclass (set by harness
    code) or as a dict (deserialized from sampling_params.extra_args).
    Both must work."""
    from vllm.v1.core.eviction_policy import EvokeRequestMeta

    manager = CPUOffloadingManager(num_blocks=4, cache_policy="evoke")
    _stage_offloaded_block(manager, 1, np.array([1.0, 0.0], dtype=np.float32))

    meta_obj = EvokeRequestMeta(
        query_embedding=[1.0, 0.0],
        recover_top_k=4,
        min_similarity=0.0,
    )
    request = SimpleNamespace(request_id="r1", evoke_request_meta=meta_obj)
    stub = _stub_scheduler(manager)
    keys, _ = stub._compute_smart_recovery_keys(request, prefix_keys_to_load=set())
    assert len(keys) >= 1


def test_compute_smart_recovery_non_evoke_policy_no_op():
    """The OffloadingManager default returns an empty recommend_recovery for
    non-EVOKE policies (LRU, ARC). The connector scheduler then naturally
    returns no recovery keys without errors."""
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="lru")
    request = SimpleNamespace(
        request_id="r1",
        evoke_request_meta={
            "query_embedding": [1.0, 0.0],
            "recover_top_k": 4,
        },
    )
    stub = _stub_scheduler(manager)
    assert stub._compute_smart_recovery_keys(request, prefix_keys_to_load=set()) == (
        [],
        [],
    )
