# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVOKE capture orchestrator: glue between attention capture and policy.

Phase 3 emits softmax(QK^T) per-decode-step into a `CaptureRecord`. Phase 4
and 5a expose `update_attention_scores` / `decay_recovery_strength` on the
EVOKE policies. The orchestrator is the small piece between them: after each
decode step where capture fired, aggregate per-block attention weights,
push them into the eviction policy. At end of each user turn, decay the
recovery_strength so freshly recovered blocks fade back to ordinary
candidate status.

The orchestrator is intentionally separate from the policy and the
capture module so that:
- Capture stays a pure data-collection layer (no policy dependency).
- Policy stays an in-place scorer (no torch dependency leaks beyond what
  is already imported for embeddings).
- The orchestrator can be replaced (e.g. with one that averages across
  multiple captured layers, or one that runs an exponential moving
  average) without touching either module.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.attention.evoke_attn_capture import (
    CaptureRecord,
    aggregate_per_block,
)
from vllm.v1.kv_offload.cpu.policies.evoke import cosine_similarity

if TYPE_CHECKING:
    from vllm.v1.core.eviction_policy import EvokeBlockEvictionPolicy
    from vllm.v1.kv_offload.base import OffloadKey
    from vllm.v1.kv_offload.cpu.policies.evoke import EvokeCachePolicy


class EvokeCaptureOrchestrator:
    def __init__(
        self,
        policy: "EvokeBlockEvictionPolicy",
        recovery_decay: float = 0.7,
    ) -> None:
        self.policy = policy
        self.recovery_decay = recovery_decay

    def on_capture_step(
        self,
        capture: CaptureRecord,
        block_table_row: torch.Tensor,
        block_size: int,
    ) -> dict[int, float] | None:
        """Consume one capture step and push per-block attention scores to
        the policy.

        Args:
            capture: latest CaptureRecord; must have non-None weights
                (otherwise this is a no-op).
            block_table_row: shape [num_pages_for_this_seq]. The physical
                block ids backing the sequence whose attention was captured.
            block_size: page size in tokens for this layer's cache.

        Returns:
            The block_id -> attention_score dict that was pushed to the
            policy, or None if the capture had no weights.
        """
        if capture.weights is None:
            return None
        weights = capture.weights
        num_q, _, seq_len = weights.shape
        block_ids_per_token = torch.tensor(
            [int(block_table_row[i // block_size].item()) for i in range(seq_len)]
        )
        max_block_id = int(block_ids_per_token.max().item())
        per_block = aggregate_per_block(
            weights, block_ids_per_token, num_blocks=max_block_id + 1
        )
        avg_per_block = per_block.mean(dim=0)
        scores: dict[int, float] = {}
        for block_id in range(max_block_id + 1):
            mass = float(avg_per_block[block_id].item())
            if mass > 0.0:
                scores[block_id] = mass
        self.policy.update_attention_scores(scores)
        return scores

    def on_turn_boundary(self, recovery_decay: float | None = None) -> None:
        """Apply recovery_strength decay across every block in the policy.

        Called at the start of each new user turn so a recently recovered
        block survives the next eviction pass when the watermark trips,
        then fades back to ordinary candidate status over a handful of
        turns.
        """
        decay = self.recovery_decay if recovery_decay is None else recovery_decay
        self.policy.decay_recovery_strength(decay)

    def mark_recovered(self, block_id: int, recovery_strength: float = 1.0) -> None:
        """Tag a block as freshly recovered. Called from the recovery path
        immediately after splicing the K/V tensors back into the cache so
        the next eviction pass respects the model's prior relevance signal
        rather than evicting on recency/coherence alone."""
        self.policy.set_recovery_strength(block_id, recovery_strength)

    def update_block_embedding(self, block_id: int, embedding: np.ndarray) -> None:
        """Push a block's representative embedding into the policy so
        smart-recovery selection can score this block against future
        user-message queries."""
        self.policy.set_embedding(block_id, embedding)

    def resident_max_similarity(self, query_embedding: np.ndarray) -> float:
        """Return the strongest cosine similarity between the query and any
        currently-resident block embedding on the GPU policy. Used as the
        gate that an evicted candidate must beat to be worth recovering: a
        block already on GPU that matches the query as well as the candidate
        would makes the recovery wasted bandwidth.
        """
        best = 0.0
        for meta in self.policy.meta.values():
            if meta.embedding is None:
                continue
            sim = cosine_similarity(query_embedding, meta.embedding)
            if sim > best:
                best = sim
        return best

    def recommend_recovery(
        self,
        query_embedding: np.ndarray,
        offload_policy: "EvokeCachePolicy",
        top_k: int,
        candidate_keys: Iterable["OffloadKey"] | None = None,
        min_similarity: float = 0.0,
    ) -> list[tuple["OffloadKey", float]]:
        """Select evicted (offloaded) blocks worth bringing back for the
        next user turn.

        Combines GPU-side and CPU-side EVOKE state:
        - resident max similarity gate is computed against the GPU policy's
          embeddings, so a block that the model already has on-cache cannot
          be pre-empted by a weaker offloaded candidate.
        - actual ranking + threshold logic lives in
          `EvokeCachePolicy.select_for_recovery`.

        Args:
            query_embedding: encoder embedding of the incoming user message.
            offload_policy: the CPU-tier EVOKE policy whose blocks we are
                ranking for recovery.
            top_k: maximum number of keys to return.
            candidate_keys: restrict to this subset of keys (defaults to all
                blocks currently in the offload policy).
            min_similarity: absolute floor a candidate must beat in addition
                to the resident max similarity.

        Returns:
            List of (OffloadKey, similarity) ordered by similarity desc, at
            most top_k entries. Empty when no candidate beats the gate.
        """
        if top_k <= 0:
            return []
        resident_max = self.resident_max_similarity(query_embedding)
        keys = (
            list(candidate_keys)
            if candidate_keys is not None
            else list(offload_policy.blocks.keys())
        )
        return offload_policy.select_for_recovery(
            query_embedding=query_embedding,
            candidate_keys=keys,
            resident_max_similarity=resident_max,
            top_k=top_k,
            min_similarity=min_similarity,
        )
