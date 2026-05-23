"""End-to-end test for EvokeRopeDeltaRotator on CUDA.

Stages a paged K cache, RoPE-encodes one block at position p_orig (simulating
"the block was written and offloaded at p_orig"), then calls the rotator to
re-anchor it at p_new (simulating "the block is being loaded back at p_new").
Verifies the resulting K matches a freshly-rotated K_raw at p_new within
bfloat16 numerical tolerance.

CUDA-only. Skip on machines without GPU.
"""

from __future__ import annotations

import pytest
import torch


def _build_cos_sin_cache(
    head_dim: int, max_pos: int, base: float = 1000000.0
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).cuda()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RoPE rotator runs only on CUDA",
)
def test_rotator_recovers_correct_k_at_new_position():
    from vllm import _custom_ops as ops
    from vllm.v1.kv_offload.cpu.evoke_rope_delta import EvokeRopeDeltaRotator

    torch.manual_seed(7)
    num_blocks = 4
    block_size = 16
    num_kv_heads = 4
    head_dim = 128
    num_layers = 3
    max_pos = 4096
    base = 1000000.0
    p_orig = 100
    p_new = 1000
    block_id_to_recover = 2

    cos_sin_cache = _build_cos_sin_cache(head_dim, max_pos, base)

    k_views_per_layer = [
        torch.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(num_layers)
    ]

    k_raw_per_layer = [
        torch.randn(
            (block_size, num_kv_heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(num_layers)
    ]
    # Populate the cache as if the block was written at p_orig: copy raw K
    # in, then RoPE-rotate at p_orig (matching what reshape_and_cache_flash
    # writes at write time).
    pos_orig_t = torch.arange(p_orig, p_orig + block_size, device="cuda")
    for layer_idx in range(num_layers):
        k_views_per_layer[layer_idx][block_id_to_recover] = k_raw_per_layer[
            layer_idx
        ].clone()
        q_dummy = torch.zeros(
            block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        ops.rotary_embedding(
            pos_orig_t,
            q_dummy,
            k_views_per_layer[layer_idx][block_id_to_recover],
            head_dim,
            cos_sin_cache,
            True,
        )

    # Ground truth: what the model would have seen if the block had been
    # written at p_new instead.
    pos_new_t = torch.arange(p_new, p_new + block_size, device="cuda")
    k_direct_per_layer = []
    for layer_idx in range(num_layers):
        k_direct = k_raw_per_layer[layer_idx].clone()
        q_dummy = torch.zeros(
            block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        ops.rotary_embedding(
            pos_new_t, q_dummy, k_direct, head_dim, cos_sin_cache, True
        )
        k_direct_per_layer.append(k_direct)

    rotator = EvokeRopeDeltaRotator(
        k_views_per_layer=k_views_per_layer,
        cos_sin_cache=cos_sin_cache,
        head_size=head_dim,
        is_neox=True,
    )
    stream = torch.cuda.current_stream()
    rotated = rotator.maybe_rotate_blocks(
        stream,
        block_ids=[block_id_to_recover],
        original_positions=[p_orig],
        new_positions=[p_new],
    )
    torch.cuda.synchronize()

    assert rotated == 1
    # bf16 has ~8 mantissa bits; the rotator runs two rotations (inverse +
    # forward) vs the ground truth's one direct rotation, so expected delta
    # is ~2 ULPs of bf16. A larger blow-up would indicate a real math error.
    bf16_two_rotation_tol = 3e-2
    for layer_idx in range(num_layers):
        recovered = k_views_per_layer[layer_idx][block_id_to_recover]
        diff = (
            (recovered.float() - k_direct_per_layer[layer_idx].float())
            .abs()
            .max()
            .item()
        )
        assert diff < bf16_two_rotation_tol, (
            f"layer {layer_idx}: rotator output differs from direct rotation "
            f"by {diff:.3e} (tol {bf16_two_rotation_tol:.0e})"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RoPE rotator runs only on CUDA",
)
def test_rotator_skips_when_positions_match_or_unknown():
    from vllm.v1.kv_offload.cpu.evoke_rope_delta import EvokeRopeDeltaRotator

    cos_sin_cache = _build_cos_sin_cache(128, 4096)
    k_views = [
        torch.randn(4, 16, 4, 128, dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    ]
    saved_k = [v.clone() for v in k_views]
    rotator = EvokeRopeDeltaRotator(
        k_views_per_layer=k_views,
        cos_sin_cache=cos_sin_cache,
        head_size=128,
        is_neox=True,
    )

    stream = torch.cuda.current_stream()
    rotated = rotator.maybe_rotate_blocks(
        stream,
        block_ids=[0, 1, 2],
        original_positions=[100, -1, 200],
        new_positions=[100, 500, -1],
    )
    torch.cuda.synchronize()

    assert rotated == 0
    for v, s in zip(k_views, saved_k):
        assert torch.equal(v, s), "no-op call mutated K views"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RoPE rotator runs only on CUDA",
)
def test_rotator_only_rotates_targeted_blocks():
    from vllm.v1.kv_offload.cpu.evoke_rope_delta import EvokeRopeDeltaRotator

    cos_sin_cache = _build_cos_sin_cache(128, 4096)
    torch.manual_seed(11)
    k_views = [
        torch.randn(4, 16, 4, 128, dtype=torch.bfloat16, device="cuda")
        for _ in range(1)
    ]
    untouched = k_views[0][[0, 2, 3]].clone()

    rotator = EvokeRopeDeltaRotator(
        k_views_per_layer=k_views,
        cos_sin_cache=cos_sin_cache,
        head_size=128,
        is_neox=True,
    )
    stream = torch.cuda.current_stream()
    rotator.maybe_rotate_blocks(
        stream,
        block_ids=[1],
        original_positions=[64],
        new_positions=[320],
    )
    torch.cuda.synchronize()

    assert torch.equal(k_views[0][0], untouched[0])
    assert torch.equal(k_views[0][2], untouched[1])
    assert torch.equal(k_views[0][3], untouched[2])
