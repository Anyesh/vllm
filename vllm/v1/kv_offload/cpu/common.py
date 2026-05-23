# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import numpy as np

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.

    `original_positions` and `new_positions` are parallel arrays to `block_ids`
    used by EVOKE smart-recovery: when a block is loaded back at a different
    absolute token position than where it was originally written, the worker
    applies a RoPE delta (un-rotate at original, re-rotate at new) so the
    cached K bytes align with the new position. Both arrays default to all-
    zeros (or all -1) which means "no rotation needed" -- the existing prefix-
    extension load path leaves them at zero and the worker skips the rotation
    step.

    Rotation is skipped per-block when `original_positions[i] == new_positions[i]`
    or when either is -1 (position unknown). This keeps the legacy load path
    unchanged for non-EVOKE policies and for prefix-extension loads where the
    block lands at the same position it was offloaded from.
    """

    def __init__(
        self,
        block_ids: list[int],
        original_positions: Sequence[int] | None = None,
        new_positions: Sequence[int] | None = None,
    ):
        super().__init__(block_ids)
        if original_positions is None:
            self.original_positions = np.full_like(self.block_ids, -1)
        else:
            self.original_positions = np.array(original_positions, dtype=np.int64)
            assert len(self.original_positions) == len(self.block_ids), (
                "original_positions must align with block_ids"
            )
        if new_positions is None:
            self.new_positions = np.full_like(self.block_ids, -1)
        else:
            self.new_positions = np.array(new_positions, dtype=np.int64)
            assert len(self.new_positions) == len(self.block_ids), (
                "new_positions must align with block_ids"
            )

    @staticmethod
    def medium() -> str:
        return "CPU"
