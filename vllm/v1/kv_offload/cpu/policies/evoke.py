# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


@dataclass
class EvokeBlockMeta:
    last_touch_tick: int
    priority: float = 1.0
    recovery_strength: float = 0.0


class EvokeCachePolicy(CachePolicy):
    """Multi-signal scoring policy for the EVOKE eviction layer.

    Scores each block as a weighted sum of recency, coherence (similarity to
    the current task focus), recovery_strength (the model's own signal that a
    block was load-bearing on a recent turn), and a harness-supplied priority
    multiplier. The lowest-scoring blocks are evicted first.

    This is the phase-1 stub: recency is wired and the priority multiplier is
    in place; coherence and recovery_strength are reserved fields with zero
    weight by default. Smart-recovery via retrieval encoder and the
    attention-capture signal land in subsequent phases. The plugin surface
    here is what the wider policy will hook into.
    """

    def __init__(self, cache_capacity: int) -> None:
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.meta: dict[OffloadKey, EvokeBlockMeta] = {}
        self._tick: int = 0
        self.w_recency: float = 0.4
        self.w_coherence: float = 0.6
        self.w_recovery: float = 0.0
        self.recency_half_life: int = 64

    def _recency(self, key: OffloadKey) -> float:
        age = self._tick - self.meta[key].last_touch_tick
        # half-life decay: at age = recency_half_life, recency = 0.5
        return 0.5 ** (age / max(1, self.recency_half_life))

    def _score(self, key: OffloadKey) -> float:
        meta = self.meta[key]
        recency = self._recency(key)
        coherence = 0.0
        recovery = meta.recovery_strength
        raw = (
            self.w_recency * recency
            + self.w_coherence * coherence
            + self.w_recovery * recovery
        )
        return raw * meta.priority

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self._tick += 1
        self.blocks[key] = block
        self.meta[key] = EvokeBlockMeta(last_touch_tick=self._tick)

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        del self.meta[key]

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if key in self.meta:
                self._tick += 1
                self.meta[key].last_touch_tick = self._tick

    def clear(self) -> None:
        self.blocks.clear()
        self.meta.clear()

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        evictable: list[tuple[OffloadKey, BlockStatus, float]] = []
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                evictable.append((key, block, self._score(key)))
        if len(evictable) < n:
            return None
        evictable.sort(key=lambda triple: triple[2])
        chosen = evictable[:n]
        result: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block, _ in chosen:
            del self.blocks[key]
            del self.meta[key]
            result.append((key, block))
        return result
