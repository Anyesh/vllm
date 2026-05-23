"""RoPE-delta rotation for EVOKE smart-recovery on vLLM.

When a KV block is loaded back from the CPU offload tier into a different
absolute token position than where it was originally written, the K bytes in
the block are RoPE-encoded for the wrong positions. This module applies a
rotation delta on the loaded K tensor so it aligns with the new position.

The math: RoPE is a rotation, so rotate-by-a then rotate-by-b equals
rotate-by-(a+b). Two calls to vllm._custom_ops.rotary_embedding -- one
inverse at the original position, one forward at the new position -- produce
output bit-equivalent to rotating from scratch at the new position. Validated
numerically end-to-end in
tests/v1/kv_offload/test_rope_delta_recovery.py (both pure-PyTorch math
and the actual CUDA C++ op on chihiro).

The rotator is enqueued on the same CUDA stream as the swap_blocks_batch
transfer, between the swap and the completion event, so the event gates on
rotation finishing. The next forward pass that reads from these blocks sees
correctly-positioned K.

V is untouched: RoPE only rotates K (and Q, which is recomputed every step
from hidden states so it never needs recovery).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops


class EvokeRopeDeltaRotator:
    """Applies RoPE-delta rotation per recovered block, on a caller-provided
    CUDA stream. Construction takes a list of per-layer K tensor views and
    the model's RoPE configuration; each call enqueues the two-step rotation
    (inverse then forward) for the blocks that need it.

    Pure NO-OP for blocks where original_position == new_position or either
    position is -1 (unknown). This keeps the prefix-extension load path
    (where blocks land at the same token positions they were offloaded from)
    free of any rotation cost.
    """

    def __init__(
        self,
        k_views_per_layer: list[torch.Tensor],
        cos_sin_cache: torch.Tensor,
        head_size: int,
        is_neox: bool = True,
    ):
        """
        Args:
            k_views_per_layer: list[Tensor], one per attention layer, each
                with shape (num_gpu_blocks, block_size, num_kv_heads,
                head_size) and the model's K dtype (typically bfloat16).
                The rotator mutates these views in place.
            cos_sin_cache: torch.Tensor of shape (max_position, head_size),
                the standard vLLM RoPE cache (first half cos, second half sin).
                Must reside on the same CUDA device as k_views.
            head_size: per-head dimension (e.g. 128 for Qwen2.5 7B).
            is_neox: True for neox-style two-halves layout (Qwen2.5 default).
        """
        assert len(k_views_per_layer) > 0
        for v in k_views_per_layer:
            assert v.is_cuda
            assert v.dim() == 4, (
                f"expected K view of shape (num_blocks, block_size, "
                f"num_kv_heads, head_size), got {tuple(v.shape)}"
            )
            assert v.shape[-1] == head_size, (
                f"last dim of K view {v.shape[-1]} != head_size {head_size}"
            )
        assert cos_sin_cache.is_cuda
        assert cos_sin_cache.shape[-1] == head_size
        self.k_views_per_layer = k_views_per_layer
        self.cos_sin_cache = cos_sin_cache
        self.head_size = head_size
        self.is_neox = is_neox

        num_blocks, block_size, num_kv_heads, _ = k_views_per_layer[0].shape
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.num_layers = len(k_views_per_layer)
        # Reusable dummy Q tensor that the C++ op requires but we don't use.
        # num_heads must satisfy num_heads % num_kv_heads == 0 (GQA invariant),
        # so we size it to num_kv_heads which is the smallest valid count.
        self._q_dummy = torch.zeros(
            (block_size, num_kv_heads, head_size),
            dtype=k_views_per_layer[0].dtype,
            device=k_views_per_layer[0].device,
        )

    def maybe_rotate_blocks(
        self,
        stream: torch.cuda.Stream,
        block_ids: Sequence[int],
        original_positions: Sequence[int],
        new_positions: Sequence[int],
    ) -> int:
        """Enqueue rotation kernels on `stream` for the blocks whose
        original_position differs from new_position (and neither is -1).

        Args:
            stream: CUDA stream to enqueue rotation ops on. The caller is
                responsible for ordering this with the upstream
                swap_blocks_batch transfer and the downstream completion
                event so this stream observes the transferred bytes before
                rotation runs.
            block_ids: GPU block ids of the blocks just loaded.
            original_positions: per-block absolute token position the block
                held at offload time. -1 means unknown.
            new_positions: per-block absolute token position the block will
                hold in the destination sequence. -1 means unknown.

        Returns:
            The number of blocks rotated (counts the blocks, not the layer
            calls). 0 when every block is either no-op or skip.
        """
        assert len(block_ids) == len(original_positions) == len(new_positions)
        rotated = 0
        device = self.k_views_per_layer[0].device
        with torch.cuda.stream(stream):
            for block_id, orig_pos, new_pos in zip(
                block_ids, original_positions, new_positions
            ):
                orig_pos = int(orig_pos)
                new_pos = int(new_pos)
                if orig_pos < 0 or new_pos < 0 or orig_pos == new_pos:
                    continue
                pos_orig = torch.arange(
                    orig_pos, orig_pos + self.block_size, device=device
                )
                pos_new = torch.arange(
                    new_pos, new_pos + self.block_size, device=device
                )
                for k_view in self.k_views_per_layer:
                    k_block = k_view[block_id]
                    # `rotary_embedding` requires a contiguous K tensor.
                    # When k_view is a non-contiguous slice (e.g. the HND
                    # cross-layer layout where per-layer K is strided),
                    # clone into a temp, rotate, then copy back so the
                    # rotated K reaches the underlying GPU cache.
                    if k_block.is_contiguous():
                        k_target = k_block
                        needs_writeback = False
                    else:
                        k_target = k_block.contiguous()
                        needs_writeback = True
                    q_dummy = self._q_dummy.zero_()
                    ops.rotary_embedding(
                        pos_orig,
                        q_dummy,
                        k_target,
                        self.head_size,
                        self.cos_sin_cache,
                        self.is_neox,
                        rope_dim_offset=0,
                        inverse=True,
                    )
                    q_dummy = self._q_dummy.zero_()
                    ops.rotary_embedding(
                        pos_new,
                        q_dummy,
                        k_target,
                        self.head_size,
                        self.cos_sin_cache,
                        self.is_neox,
                    )
                    if needs_writeback:
                        k_block.copy_(k_target)
                rotated += 1
        return rotated
