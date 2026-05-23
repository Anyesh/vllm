# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

# Source-type strings used by the floor map. Harnesses are free to use any
# string; the canonical EVOKE set is below. Unknown source types fall through
# to no floor.
SOURCE_USER = "user"
SOURCE_ASSISTANT = "assistant"
SOURCE_DOCUMENT = "document"
SOURCE_SYSTEM = "system"


@dataclass
class EvokeBlockMeta:
    last_touch_tick: int
    priority: float = 1.0
    recovery_strength: float = 0.0
    source_type: str | None = None
    pinned: bool = False
    embedding: np.ndarray | None = field(default=None, repr=False)
    # Attention-mass signal: aggregated per-block attention from the
    # capture layer's softmax(QK^T). Stays at 0.0 until the
    # capture-orchestrator pushes values in.
    attention_score: float = 0.0
    # Coherence: cosine similarity of this block's embedding to the
    # current task-focus embedding. Stays at 0.0 until the orchestrator
    # pushes values in at turn boundary.
    coherence_score: float = 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EvokeCachePolicy(CachePolicy):
    """Multi-signal scoring policy for the EVOKE eviction layer.

    Scores each block as a weighted sum of recency, coherence (similarity to
    the current task focus), recovery_strength (the model's own signal that a
    block was load-bearing on a recent turn), and a harness-supplied priority
    multiplier. The score is lifted to a source-type floor before being
    multiplied by priority, so conversation-backbone content (user and
    assistant turns) outlives document content under budget pressure unless
    the harness overrides the priority. The lowest-scoring blocks evict
    first.

    Smart-recovery selection (top-K bring-back at turn boundary, gated by
    retrieval-encoder similarity) is supported via the embedding-storage
    hooks (set_embedding / get_embedding); the orchestrator that performs
    selection lives one layer above the policy and consumes the embeddings
    stored here.
    """

    def __init__(self, cache_capacity: int) -> None:
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.meta: dict[OffloadKey, EvokeBlockMeta] = {}
        self._tick: int = 0
        # Default recipe is the fork-independent fallback: recency + coherence
        # carry the score, attention and recovery are reserved. Deployments
        # with the attention-capture path wired in should set
        # w_attention=0.5, w_recency=0.2, w_coherence=0.3 as the recommended
        # multi-signal config.
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

    def _recency(self, key: OffloadKey) -> float:
        age = self._tick - self.meta[key].last_touch_tick
        return 0.5 ** (age / max(1, self.recency_half_life))

    def _score(self, key: OffloadKey) -> float:
        meta = self.meta[key]
        recency = self._recency(key)
        attention = meta.attention_score
        coherence = meta.coherence_score
        recovery = meta.recovery_strength
        raw = (
            self.w_attention * attention
            + self.w_recency * recency
            + self.w_coherence * coherence
            + self.w_recovery * recovery
        )
        if meta.source_type is not None:
            floor = self.source_floors.get(meta.source_type, 0.0)
            raw = max(raw, floor)
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
            meta = self.meta[key]
            if block.ref_cnt == 0 and key not in protected and not meta.pinned:
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

    def set_source_type(self, key: OffloadKey, source_type: str | None) -> None:
        if key in self.meta:
            self.meta[key].source_type = source_type

    def set_priority(self, key: OffloadKey, priority: float) -> None:
        if key in self.meta:
            self.meta[key].priority = priority

    def set_pinned(self, key: OffloadKey, pinned: bool) -> None:
        if key in self.meta:
            self.meta[key].pinned = pinned

    def set_embedding(self, key: OffloadKey, embedding: np.ndarray) -> None:
        if key in self.meta:
            self.meta[key].embedding = embedding

    def get_embedding(self, key: OffloadKey) -> np.ndarray | None:
        meta = self.meta.get(key)
        return meta.embedding if meta is not None else None

    def set_attention_score(self, key: OffloadKey, score: float) -> None:
        if key in self.meta:
            self.meta[key].attention_score = score

    def set_coherence_score(self, key: OffloadKey, score: float) -> None:
        if key in self.meta:
            self.meta[key].coherence_score = score

    def set_recovery_strength(self, key: OffloadKey, strength: float) -> None:
        if key in self.meta:
            self.meta[key].recovery_strength = strength

    def decay_recovery_strength(self, decay: float) -> None:
        """Apply per-turn decay to every block's recovery_strength.

        Called by the orchestrator at the start of each user turn so a
        freshly-recovered block survives one or two more eviction passes
        before fading back to ordinary candidate status.
        """
        for meta in self.meta.values():
            meta.recovery_strength *= decay

    def update_attention_scores(self, scores: dict[OffloadKey, float]) -> None:
        """Bulk update attention scores from the capture orchestrator.

        Keys missing from the policy are silently ignored (the block may
        have been evicted between capture and update).
        """
        for key, score in scores.items():
            if key in self.meta:
                self.meta[key].attention_score = score

    def select_for_recovery(
        self,
        query_embedding: np.ndarray,
        candidate_keys: Iterable[OffloadKey],
        resident_max_similarity: float,
        top_k: int,
        min_similarity: float = 0.0,
    ) -> list[tuple[OffloadKey, float]]:
        """Smart-recovery selection: rank candidate (evicted) blocks by
        cosine similarity to the query, gated against the strongest
        already-resident match so that weak recoveries cannot pollute the
        cache with off-topic content.

        Returns the top-k (key, similarity) pairs ordered by similarity
        desc. A candidate must beat both `min_similarity` and
        `resident_max_similarity` to be returned.
        """
        ranked: list[tuple[OffloadKey, float]] = []
        for key in candidate_keys:
            meta = self.meta.get(key)
            if meta is None or meta.embedding is None:
                continue
            sim = cosine_similarity(query_embedding, meta.embedding)
            if sim >= min_similarity and sim > resident_max_similarity:
                ranked.append((key, sim))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
