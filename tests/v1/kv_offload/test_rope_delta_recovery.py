"""Numerical validation of RoPE-delta-on-load for EVOKE smart-recovery.

EVOKE recovery loads a previously-evicted KV block at a different absolute
position than where it was originally written. Because vLLM applies RoPE
eagerly at write time (and the K bytes stored in the paged cache are already
rotated), naively loading the block into a different slot yields K vectors
encoded for the WRONG positions.

The candidate fix: after the swap_blocks transfer brings the K bytes to GPU,
apply a rotation delta -- un-rotate at the original position, re-rotate at
the new position. This test validates the algebra in pure PyTorch (no C++
ops, no CUDA) so the design is provably correct before runtime integration
lands.

The math: RoPE at position p rotates each 2D pair (x1, x2) by angle p*omega_i
where omega_i = base^(-2i/d). Composing rotations: rotate-by-a then
rotate-by-b equals rotate-by-(a+b). So un-rotate-at-p then rotate-at-q
equals rotate-at-(q-p) on the original un-rotated vectors, which equals what
we would have gotten by rotating at q directly.
"""

from __future__ import annotations

import torch


def _rope_freqs(head_dim: int, base: float = 1000000.0) -> torch.Tensor:
    half = head_dim // 2
    return 1.0 / (base ** (torch.arange(0, half, dtype=torch.float64) * 2 / head_dim))


def _at_positions(
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    head_dim = x.shape[-1]
    half = head_dim // 2
    angles = positions.to(torch.float64).unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    if inverse:
        sin = -sin
    x1 = x[..., :half].to(torch.float64)
    x2 = x[..., half:].to(torch.float64)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


def test_rope_delta_recovers_direct_rotation_pure_pytorch():
    """Central invariant: un-rotate at p_orig + re-rotate at p_new produces
    the same K tensor as rotating from scratch at p_new. If this holds, the
    recovery design is mathematically sound."""
    torch.manual_seed(0)
    block_size = 16
    num_kv_heads = 4
    head_dim = 128
    base = 1000000.0
    p_orig = 100
    p_new = 1000

    freqs = _rope_freqs(head_dim, base)
    k_raw = torch.randn(block_size, num_kv_heads, head_dim, dtype=torch.float32)

    pos_orig = torch.arange(p_orig, p_orig + block_size)
    pos_new = torch.arange(p_new, p_new + block_size)

    k_direct_new = _at_positions(k_raw, pos_new, freqs, inverse=False)

    k_cached = _at_positions(k_raw, pos_orig, freqs, inverse=False)
    k_unrotated = _at_positions(k_cached, pos_orig, freqs, inverse=True)
    k_delta_new = _at_positions(k_unrotated, pos_new, freqs, inverse=False)

    max_abs_diff = (k_delta_new - k_direct_new).abs().max().item()
    rel_tol = 1e-4
    assert max_abs_diff < rel_tol, (
        f"RoPE-delta math broken: max_abs_diff={max_abs_diff:.3e} "
        f"(should be < {rel_tol:.0e}). Round-trip through inverse + forward "
        f"did not reproduce direct rotation at new position."
    )

    sanity_diff = (k_unrotated - k_raw).abs().max().item()
    assert sanity_diff < rel_tol, (
        f"inverse rotation did not recover the original tensor: "
        f"max_abs_diff={sanity_diff:.3e}"
    )


def test_rope_delta_handles_zero_position_shift():
    """p_orig == p_new: delta is zero, the recovered K should equal the
    cached K exactly under the inverse+forward round trip."""
    torch.manual_seed(1)
    block_size = 16
    head_dim = 128
    freqs = _rope_freqs(head_dim)
    k_raw = torch.randn(block_size, 4, head_dim, dtype=torch.float32)
    positions = torch.arange(500, 500 + block_size)

    k_cached = _at_positions(k_raw, positions, freqs, inverse=False)
    k_after_round_trip = _at_positions(
        _at_positions(k_cached, positions, freqs, inverse=True),
        positions,
        freqs,
        inverse=False,
    )
    diff = (k_after_round_trip - k_cached).abs().max().item()
    assert diff < 1e-4


def test_rope_delta_position_shift_equivalence():
    """un-rotate-at-p then rotate-at-q equals rotate-at-(q-p) on the raw
    vectors. Sanity check on the rotation group property."""
    torch.manual_seed(2)
    block_size = 8
    head_dim = 64
    freqs = _rope_freqs(head_dim, base=1000.0)
    k_raw = torch.randn(block_size, 2, head_dim, dtype=torch.float32)

    pos_orig = torch.arange(10, 10 + block_size)
    pos_new = torch.arange(70, 70 + block_size)

    k_cached = _at_positions(k_raw, pos_orig, freqs, inverse=False)
    k_via_delta = _at_positions(
        _at_positions(k_cached, pos_orig, freqs, inverse=True),
        pos_new,
        freqs,
        inverse=False,
    )
    k_via_direct = _at_positions(k_raw, pos_new, freqs, inverse=False)
    diff = (k_via_delta - k_via_direct).abs().max().item()
    assert diff < 1e-4


def test_rope_delta_with_vllm_cuda_op():
    """Same invariant, but using vLLM's actual C++ rotary_embedding op
    instead of the pure-PyTorch reference. Requires CUDA. Validates that the
    op's inverse=True path produces the expected un-rotation, and that two
    op calls (inverse + forward) reproduce a direct forward call at the new
    position. This is the runtime path that EVOKE recovery will actually use.
    """
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required for vLLM C++ rotary_embedding op")

    from vllm import _custom_ops as ops

    torch.manual_seed(3)
    block_size = 16
    num_q_heads = 28
    num_kv_heads = 4
    head_dim = 128
    base = 1000000.0
    max_pos = 4096
    p_orig = 100
    p_new = 1000

    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(
        device="cuda", dtype=torch.float32
    )

    pos_orig = torch.arange(p_orig, p_orig + block_size, device="cuda")
    pos_new = torch.arange(p_new, p_new + block_size, device="cuda")

    k_raw = torch.randn(
        block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.float32
    )
    q_dummy_template = torch.zeros(
        block_size, num_q_heads, head_dim, device="cuda", dtype=torch.float32
    )

    k_direct = k_raw.clone()
    q_dummy = q_dummy_template.clone()
    ops.rotary_embedding(pos_new, q_dummy, k_direct, head_dim, cos_sin_cache, True)

    k_cached = k_raw.clone()
    q_dummy = q_dummy_template.clone()
    ops.rotary_embedding(pos_orig, q_dummy, k_cached, head_dim, cos_sin_cache, True)

    k_delta = k_cached.clone()
    q_dummy = q_dummy_template.clone()
    ops.rotary_embedding(
        pos_orig,
        q_dummy,
        k_delta,
        head_dim,
        cos_sin_cache,
        True,
        rope_dim_offset=0,
        inverse=True,
    )
    q_dummy = q_dummy_template.clone()
    ops.rotary_embedding(pos_new, q_dummy, k_delta, head_dim, cos_sin_cache, True)

    max_abs_diff = (k_delta - k_direct).abs().max().item()
    rel_tol = 1e-3
    assert max_abs_diff < rel_tol, (
        f"vLLM C++ rotary_embedding inverse+forward did not reproduce direct "
        f"rotation: max_abs_diff={max_abs_diff:.3e} (tol={rel_tol})"
    )


if __name__ == "__main__":
    test_rope_delta_recovers_direct_rotation_pure_pytorch()
    print("test_rope_delta_recovers_direct_rotation_pure_pytorch PASSED")
    test_rope_delta_handles_zero_position_shift()
    print("test_rope_delta_handles_zero_position_shift PASSED")
    test_rope_delta_position_shift_equivalence()
    print("test_rope_delta_position_shift_equivalence PASSED")
    if torch.cuda.is_available():
        test_rope_delta_with_vllm_cuda_op()
        print("test_rope_delta_with_vllm_cuda_op PASSED")
    else:
        print("test_rope_delta_with_vllm_cuda_op SKIPPED (no CUDA)")
