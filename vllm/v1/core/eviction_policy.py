# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable block-eviction policy for vLLM's GPU active KV cache.

The default GPU eviction in `block_pool.py` is FIFO over a doubly-linked
free list (LRU when blocks are appended in access order). This module
defines an ABC analogous to `vllm.v1.kv_offload.cpu.policies.base.
CachePolicy` so research extensions (EVOKE multi-signal scoring,
attention-weighted eviction, source-type priority) can replace the
selection without rewriting block_pool's hot path. Integration with
block_pool is opt-in and lives in a separate commit; this file defines
the surface and ships a default LRU adapter plus the EVOKE plugin.

The interface is symmetric to the CPU-tier `CachePolicy`:

- `on_block_freed(block)` / `on_block_allocated(block)` — observability
  hooks the block_pool calls so policies can maintain their own
  scoring state.
- `select_eviction_candidates(n, protected)` — returns up to n blocks
  to evict, ordered by eviction preference (first = first to evict).
  Mirrors `CachePolicy.evict(n, protected)`.
- Setters (`set_attention_score`, `set_priority`, etc.) allow harness
  and orchestrator code to push signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


class _BlockLike(Protocol):
    """Minimal surface this module needs from a KV-cache block.

    Using a Protocol instead of the concrete `KVCacheBlock` keeps this
    file importable in lightweight test contexts that do not load all
    of vllm.config / kv_cache_interface (which pull cbor2, pydantic,
    and the full attention-backend registry transitively).
    """

    block_id: int


# Source-type constants shared with the CPU-tier EvokeCachePolicy. Re-exported
# here so callers do not have to reach across packages.
from vllm.v1.kv_offload.cpu.policies.evoke import (
    SOURCE_ASSISTANT,
    SOURCE_DOCUMENT,
    SOURCE_SYSTEM,
    SOURCE_USER,
)

__all__ = [
    "BlockEvictionPolicy",
    "EvokeBlockEvictionPolicy",
    "EvokeGpuBlockMeta",
    "EvokeRequestMeta",
    "LRUBlockEvictionPolicy",
    "SOURCE_ASSISTANT",
    "SOURCE_DOCUMENT",
    "SOURCE_SYSTEM",
    "SOURCE_USER",
]


class BlockEvictionPolicy(ABC):
    @abstractmethod
    def on_block_freed(self, block: KVCacheBlock) -> None: ...

    @abstractmethod
    def on_block_allocated(self, block: KVCacheBlock) -> None: ...

    @abstractmethod
    def select_eviction_candidates(
        self, n: int, protected: set[int]
    ) -> list[KVCacheBlock] | None:
        """Return n blocks to evict, ordered first-to-evict.

        Args:
            n: number of blocks needed.
            protected: block_ids that must NOT be selected (e.g. blocks
                currently referenced by an in-flight request).

        Returns:
            A list of n KVCacheBlock instances in eviction order. Returns
            None if fewer than n evictable candidates exist (caller cannot
            satisfy the allocation).
        """

    @abstractmethod
    def num_evictable(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...


class LRUBlockEvictionPolicy(BlockEvictionPolicy):
    """Default LRU policy. Matches the existing FreeKVCacheBlockQueue
    behavior: blocks freed earliest are evicted first. Provided so
    block_pool can take a uniform `BlockEvictionPolicy` argument without
    changing the default semantics for any caller."""

    def __init__(self) -> None:
        self._free: OrderedDict[int, KVCacheBlock] = OrderedDict()

    def on_block_freed(self, block: KVCacheBlock) -> None:
        self._free[block.block_id] = block

    def on_block_allocated(self, block: KVCacheBlock) -> None:
        self._free.pop(block.block_id, None)

    def select_eviction_candidates(
        self, n: int, protected: set[int]
    ) -> list[KVCacheBlock] | None:
        if n == 0:
            return []
        chosen: list[KVCacheBlock] = []
        for block_id, block in self._free.items():
            if block_id in protected:
                continue
            chosen.append(block)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            return None
        return chosen

    def num_evictable(self) -> int:
        return len(self._free)

    def clear(self) -> None:
        self._free.clear()


@dataclass
class EvokeGpuBlockMeta:
    last_touch_tick: int
    priority: float = 1.0
    recovery_strength: float = 0.0
    attention_score: float = 0.0
    coherence_score: float = 0.0
    source_type: str | None = None
    pinned: bool = False
    embedding: np.ndarray | None = field(default=None, repr=False)
    request_id: str | None = None


@dataclass
class EvokeRequestMeta:
    """Per-request EVOKE metadata extracted from the OpenAI-compatible
    request body's `vllm_xargs.evoke_request_meta`. Applied to every
    block allocated for the request at on_block_freed time so source-type
    floors, harness priority, and pinning take effect without further
    plumbing through the scheduler hot path.

    Smart-recovery fields (`query_embedding`, `recover_top_k`) are consulted
    at request admission: the connector scheduler calls
    `EvokeCaptureOrchestrator.recommend_recovery` against the offload
    manager's policy, augments the request's `keys_to_load` with the
    returned offloaded blocks, and the worker re-anchors them via the
    RoPE-delta rotator on load. `query_embedding` is stored as a
    `list[float]` (not numpy) so it serializes cleanly across the engine's
    ZMQ boundary; it is converted to `np.ndarray` at the consumption site
    in the orchestrator.
    """

    source_type: str | None = None
    priority: float = 1.0
    pinned: bool = False
    query_embedding: list[float] | None = None
    recover_top_k: int = 0
    min_similarity: float = 0.0


class EvokeBlockEvictionPolicy(BlockEvictionPolicy):
    """Multi-signal scoring policy for the GPU active KV cache.

    Symmetric in design to `EvokeCachePolicy` (CPU offload tier): the
    same recipe of attention + recency + coherence + recovery_strength,
    lifted by source-type floors and multiplied by harness-supplied
    priority. Lowest-scoring blocks evict first.

    Configured with the same defaults as the CPU policy. Deployments with
    the attention-capture path wired in should bump `w_attention` to 0.5
    and set `w_recency=0.2, w_coherence=0.3` for the recommended recipe.
    """

    def __init__(self) -> None:
        self._free: dict[int, KVCacheBlock] = {}
        self.meta: dict[int, EvokeGpuBlockMeta] = {}
        self._tick: int = 0
        self.w_attention: float = 0.0
        self.w_recency: float = 0.4
        self.w_coherence: float = 0.6
        self.w_recovery: float = 0.0
        self.recency_half_life: int = 64
        self.source_floors: dict[str, float] = {
            SOURCE_USER: 0.6,
            SOURCE_ASSISTANT: 0.5,
            SOURCE_SYSTEM: 0.6,
        }
        # Per-request metadata pushed in by the caller (e.g. via the
        # ChatCompletionRequest extension path). Looked up when a block is
        # assigned to a request so source_type / priority / pinned can be
        # applied without further hot-path plumbing.
        self.request_meta: dict[str, EvokeRequestMeta] = {}

    def _ensure_meta(self, block_id: int) -> EvokeGpuBlockMeta:
        meta = self.meta.get(block_id)
        if meta is None:
            self._tick += 1
            meta = EvokeGpuBlockMeta(last_touch_tick=self._tick)
            self.meta[block_id] = meta
        return meta

    def _recency(self, block_id: int) -> float:
        age = self._tick - self.meta[block_id].last_touch_tick
        return 0.5 ** (age / max(1, self.recency_half_life))

    def _score(self, block_id: int) -> float:
        meta = self.meta[block_id]
        raw = (
            self.w_attention * meta.attention_score
            + self.w_recency * self._recency(block_id)
            + self.w_coherence * meta.coherence_score
            + self.w_recovery * meta.recovery_strength
        )
        if meta.source_type is not None:
            floor = self.source_floors.get(meta.source_type, 0.0)
            raw = max(raw, floor)
        return raw * meta.priority

    def on_block_freed(self, block: KVCacheBlock) -> None:
        self._tick += 1
        self._free[block.block_id] = block
        meta = self._ensure_meta(block.block_id)
        meta.last_touch_tick = self._tick

    def on_block_allocated(self, block: KVCacheBlock) -> None:
        self._tick += 1
        self._free.pop(block.block_id, None)
        meta = self._ensure_meta(block.block_id)
        meta.last_touch_tick = self._tick

    def select_eviction_candidates(
        self, n: int, protected: set[int]
    ) -> list[KVCacheBlock] | None:
        if n == 0:
            return []
        evictable: list[tuple[KVCacheBlock, float]] = []
        for block_id, block in self._free.items():
            meta = self.meta.get(block_id)
            if meta is None or meta.pinned or block_id in protected:
                continue
            evictable.append((block, self._score(block_id)))
        if len(evictable) < n:
            return None
        evictable.sort(key=lambda kv: kv[1])
        return [block for block, _ in evictable[:n]]

    def num_evictable(self) -> int:
        return sum(
            1
            for bid, meta in self.meta.items()
            if bid in self._free and not meta.pinned
        )

    def clear(self) -> None:
        self._free.clear()
        self.meta.clear()
        self._tick = 0

    def set_source_type(self, block_id: int, source_type: str | None) -> None:
        self._ensure_meta(block_id).source_type = source_type

    def set_priority(self, block_id: int, priority: float) -> None:
        self._ensure_meta(block_id).priority = priority

    def set_pinned(self, block_id: int, pinned: bool) -> None:
        self._ensure_meta(block_id).pinned = pinned

    def set_attention_score(self, block_id: int, score: float) -> None:
        self._ensure_meta(block_id).attention_score = score

    def set_coherence_score(self, block_id: int, score: float) -> None:
        self._ensure_meta(block_id).coherence_score = score

    def set_recovery_strength(self, block_id: int, strength: float) -> None:
        self._ensure_meta(block_id).recovery_strength = strength

    def set_embedding(self, block_id: int, embedding: np.ndarray) -> None:
        self._ensure_meta(block_id).embedding = embedding

    def get_embedding(self, block_id: int) -> np.ndarray | None:
        meta = self.meta.get(block_id)
        return meta.embedding if meta is not None else None

    def set_request_meta(self, request_id: str, meta: EvokeRequestMeta) -> None:
        """Register per-request EVOKE metadata. The scheduler / kv_cache
        manager calls `assign_block_to_request` for each block allocated
        on behalf of the request; from then on the block inherits this
        request's source_type / priority / pinned at scoring time."""
        self.request_meta[request_id] = meta

    def drop_request_meta(self, request_id: str) -> None:
        self.request_meta.pop(request_id, None)

    def assign_block_to_request(self, block_id: int, request_id: str) -> None:
        """Tag a freshly-allocated block as belonging to a request. Applies
        the request's EvokeRequestMeta (if registered) to the block's
        per-block meta so eviction scoring picks up source_type/priority/
        pinned. Called by the kv_cache_manager at block-allocation time."""
        block_meta = self._ensure_meta(block_id)
        block_meta.request_id = request_id
        req_meta = self.request_meta.get(request_id)
        if req_meta is not None:
            block_meta.source_type = req_meta.source_type
            block_meta.priority = req_meta.priority
            block_meta.pinned = req_meta.pinned

    def decay_recovery_strength(self, decay: float) -> None:
        for meta in self.meta.values():
            meta.recovery_strength *= decay

    def update_attention_scores(self, scores: dict[int, float]) -> None:
        # Silently skip block_ids that no longer have meta (block was
        # allocated out and its meta cleared by the orchestrator).
        for block_id, score in scores.items():
            meta = self.meta.get(block_id)
            if meta is not None:
                meta.attention_score = score

    def drop_meta(self, keys: Iterable[int]) -> None:
        for block_id in keys:
            self.meta.pop(block_id, None)
