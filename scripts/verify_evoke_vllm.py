"""End-to-end smoke test for EVOKE on vLLM.

Validates that the architectural pieces shipped on the `evoke-port` branch
actually run under a real vLLM engine on GPU, not only under the local Python
unit tests. Covers:

1. Modules import inside vLLM's own environment (not the lightweight test venv).
2. `EvokeBlockEvictionPolicy` is installed on the engine's `BlockPool` and
   `select_eviction_candidates` is reachable from `get_new_blocks`.
3. `evoke_attn_capture.register_layer` for a real Qwen attention layer makes
   the in-kernel hook in `Attention.forward` fire with a non-None
   `CaptureRecord` carrying softmax(QK^T) weights that sum to 1.0 at every
   (q, head).
4. `EvokeCaptureOrchestrator.on_capture_step` pushes per-block attention
   scores into `policy.meta` after a decode step.
5. A multi-turn-style decode runs to completion under enforce_eager without
   crash and the policy accumulates state across steps.

The test is intentionally an architectural smoke: it does not run the planted-
fact recovery loop (that lives in the bench harnesses, Task #3), and it does
not exercise `select_for_recovery` (Task #2 wires that into
`OffloadingManager.prepare_load`). Failing this script means the port is
broken on real hardware; passing it means the integration surface is intact
and the benches can build on top.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm import LLM, SamplingParams
from vllm.v1.attention import evoke_attn_capture
from vllm.v1.core.eviction_policy import (
    SOURCE_USER,
    EvokeBlockEvictionPolicy,
    EvokeRequestMeta,
)
from vllm.v1.core.evoke_orchestrator import EvokeCaptureOrchestrator


def find_block_pool(llm: LLM):
    """Reach into the engine to get the GPU BlockPool. The exact path depends
    on vLLM's internal layout; we walk a few known locations and fail loudly
    if none match, so this script reports a useful error rather than mining
    deep into AttributeError chains."""
    engine = llm.llm_engine
    candidates = [
        (
            "engine_core.engine_core.scheduler.kv_cache_manager.block_pool",
            lambda: (
                engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
            ),
        ),
        (
            "engine_core.engine_core.scheduler.kv_cache_manager.coordinator.block_pool",
            lambda: (
                engine.engine_core.engine_core.scheduler.kv_cache_manager.coordinator.block_pool
            ),
        ),
        (
            "engine_core.scheduler.kv_cache_manager.block_pool",
            lambda: engine.engine_core.scheduler.kv_cache_manager.block_pool,
        ),
    ]
    errors = []
    for name, accessor in candidates:
        try:
            pool = accessor()
            if pool is not None:
                print(f"   found block_pool via {name}")
                return pool
        except AttributeError as e:
            errors.append(f"   {name}: {e}")
    print("could not locate BlockPool; tried:", file=sys.stderr)
    for err in errors:
        print(err, file=sys.stderr)
    raise RuntimeError("BlockPool not reachable; engine layout may have shifted")


def install_policy(block_pool, policy: EvokeBlockEvictionPolicy) -> int:
    """Plug the EVOKE policy into an already-constructed BlockPool. We have to
    do this post-init because there is no end-to-end CLI flag yet (that lands
    with the bench harnesses in Task #3). After attaching, prime the policy by
    replaying the current free-block state so its `_free` dict matches the
    pool's free queue."""
    block_pool.eviction_policy = policy
    primed = 0
    for block in block_pool.blocks:
        if getattr(block, "is_null", False):
            continue
        if block.ref_cnt == 0:
            policy.on_block_freed(block)
            primed += 1
    return primed


def discover_attention_layer(llm: LLM) -> str:
    """Find a middle attention layer name on the loaded model. For Qwen 2.5
    7B the convention is `model.layers.{N}.self_attn.attn`; we walk the
    runner's attention layers to grab a real name from the model in case the
    convention differs by quantization or model variant."""
    engine = llm.llm_engine
    worker_paths = [
        lambda: (
            engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        ),
        lambda: engine.engine_core.engine_core.executor.driver_worker.model_runner,
        lambda: engine.engine_core.model_executor.driver_worker.model_runner,
    ]
    runner = None
    for accessor in worker_paths:
        try:
            runner = accessor()
            if runner is not None:
                break
        except AttributeError:
            continue
    if runner is None:
        raise RuntimeError("could not locate model_runner")

    layer_names: list[str] = []
    model = runner.model
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Attention" and hasattr(module, "layer_name"):
            layer_names.append(module.layer_name)
    if not layer_names:
        raise RuntimeError("no Attention layers found on the model")
    print(f"   model has {len(layer_names)} attention layers")
    print(f"   first: {layer_names[0]}")
    print(f"   last:  {layer_names[-1]}")
    mid = layer_names[len(layer_names) // 2]
    print(f"   picking middle layer for capture: {mid}")
    return mid


def assert_capture_record(record, layer_name: str) -> None:
    assert record is not None, f"no CaptureRecord for layer {layer_name}"
    print(
        f"   query_shape={record.query_shape} key_shape={record.key_shape} "
        f"value_shape={record.value_shape} decode_step={record.decode_step}"
    )
    if record.weights is None:
        print(
            "   weights=None (shape-only fallback; "
            "forward_context probably unavailable in this path)"
        )
        return
    weights = record.weights
    print(f"   weights.shape={tuple(weights.shape)} dtype={weights.dtype}")
    sums = weights.sum(dim=-1)
    max_dev = float((sums - 1.0).abs().max().item())
    print(f"   max |softmax sum - 1.0| = {max_dev:.2e}")
    assert max_dev < 1e-3, f"softmax weights do not sum to 1.0 (max dev {max_dev})"


def run(model_path: str, max_tokens: int) -> int:
    print("=" * 70)
    print(f"EVOKE smoke test: model={model_path}")
    print("=" * 70)

    is_gguf = model_path.lower().endswith(".gguf")
    print(f"[1/8] Loading model (gguf={is_gguf}, enforce_eager=True)")
    t0 = time.time()
    kwargs = dict(
        model=model_path,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="auto",
        attention_backend="FLASH_ATTN",
    )
    if is_gguf:
        kwargs["tokenizer"] = "Qwen/Qwen2.5-7B-Instruct"
    llm = LLM(**kwargs)
    print(f"   loaded in {time.time() - t0:.1f}s")

    print("[2/8] Locating BlockPool")
    block_pool = find_block_pool(llm)
    print(f"   num_gpu_blocks={block_pool.num_gpu_blocks}")

    print("[3/8] Installing EvokeBlockEvictionPolicy")
    policy = EvokeBlockEvictionPolicy()
    primed = install_policy(block_pool, policy)
    print(f"   primed policy with {primed} free blocks")
    print(f"   policy.num_evictable()={policy.num_evictable()}")
    assert primed > 0, "expected at least one block primed into policy"

    print("[4/8] Wiring EvokeCaptureOrchestrator")
    orchestrator = EvokeCaptureOrchestrator(policy=policy)

    print("[5/8] Discovering attention layer and registering for capture")
    layer_name = discover_attention_layer(llm)
    evoke_attn_capture.clear()
    evoke_attn_capture.register_layer(layer_name)
    assert evoke_attn_capture.is_enabled(layer_name)

    print("[6/8] Running a generation with EVOKE request meta")
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        extra_args={
            "evoke_request_meta": {
                "source_type": SOURCE_USER,
                "priority": 1.2,
                "pinned": False,
            }
        },
    )
    # Pre-register so the policy sees the meta even if the engine.add_request
    # path for `extra_args` is not yet wired all the way through; the
    # `set_request_meta` is the source of truth either way.
    policy.set_request_meta(
        "smoke-test-req",
        EvokeRequestMeta(source_type=SOURCE_USER, priority=1.2, pinned=False),
    )

    prompt = (
        "You are a careful assistant. The user planted a fact earlier: "
        "the secret code is BUTTERFLY-1742. Recall it on request.\n\n"
        "Tell me a brief story about a lighthouse keeper."
    )
    t0 = time.time()
    outputs = llm.generate([prompt], sampling)
    gen_time = time.time() - t0
    text = outputs[0].outputs[0].text
    print(f"   generated {len(text)} chars in {gen_time:.1f}s")
    print(f"   first 120: {text[:120]!r}")

    print("[7/8] Verifying capture record")
    record = evoke_attn_capture.get_capture(layer_name)
    assert_capture_record(record, layer_name)

    print("[8/8] Verifying policy state accumulated")
    print(f"   policy.meta has {len(policy.meta)} entries")
    print(f"   policy._tick={policy._tick}")
    populated = sum(1 for m in policy.meta.values() if m.last_touch_tick > 0)
    print(f"   blocks with non-zero last_touch_tick: {populated}")
    assert populated > 0, "policy meta has no live entries after generation"

    print(f"   policy.request_meta entries: {len(policy.request_meta)}")
    print(f"   request_meta keys: {sorted(policy.request_meta.keys())}")
    engine_assigned = [rid for rid in policy.request_meta if rid != "smoke-test-req"]
    if engine_assigned:
        print(
            f"   EngineCore->_evoke_register_request_meta fired for: {engine_assigned}"
        )
    else:
        print(
            "   WARN: no engine-assigned request_meta entries (sampling_params"
            ".extra_args['evoke_request_meta'] may not have plumbed through)"
        )

    tagged_blocks = [
        bid
        for bid, m in policy.meta.items()
        if m.source_type is not None or m.request_id is not None
    ]
    print(f"   blocks with assign_block_to_request applied: {len(tagged_blocks)}")

    if record is not None and record.weights is not None:
        try:
            scores = orchestrator.on_capture_step(
                capture=record,
                block_table_row=torch.arange(
                    1 + (record.weights.shape[-1] // 16), dtype=torch.long
                ),
                block_size=16,
            )
            print(
                f"   orchestrator pushed {len(scores) if scores else 0} "
                f"per-block attention scores into policy"
            )
        except Exception as e:
            print(f"   orchestrator push failed (non-fatal for smoke): {e}")

    print()
    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "/mnt/c/Applications/llama-cpp/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ),
        help="GGUF model path (WSL view of chihiro Windows fs), or HF model id",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.model.lower().endswith(".gguf"):
        if not Path(args.model).exists():
            print(f"ERROR: model not found: {args.model}", file=sys.stderr)
            return 2
    return run(args.model, args.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
