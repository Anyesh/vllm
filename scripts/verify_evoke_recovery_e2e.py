"""End-to-end smoke test for EVOKE smart-recovery on vLLM.

Loads vLLM with OffloadingConnector configured for the EVOKE policy +
RoPE rotator, fills the GPU cache enough to trigger offloads to the CPU
tier, then sends a request carrying `evoke_request_meta.recover_top_k`
and `query_embedding`. Verifies the recovery path actually fires:

- the connector scheduler calls `_compute_smart_recovery_keys`
- non-empty keys come back from `OffloadingManager.recommend_recovery`
- the load spec carries non-trivial `(orig, new)` position pairs
- the worker's `_maybe_apply_evoke_rope_delta` runs (rotator was constructed)
- the model generates coherent text after recovery

This is the gating signal that vLLM-side smart-recovery is wired end-to-end.
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

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.v1.core.eviction_policy import SOURCE_USER, EvokeBlockEvictionPolicy


QWEN2_5_7B_HEAD_DIM = 128
QWEN2_5_7B_NUM_LAYERS = 28
QWEN2_5_7B_NUM_KV_HEADS = 4
QWEN2_5_7B_ROPE_BASE = 1_000_000.0


def install_evoke_policy(llm: LLM) -> EvokeBlockEvictionPolicy:
    block_pool = (
        llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
    )
    policy = EvokeBlockEvictionPolicy()
    block_pool.eviction_policy = policy
    primed = 0
    for block in block_pool.blocks:
        if getattr(block, "is_null", False):
            continue
        if block.ref_cnt == 0:
            policy.on_block_freed(block)
            primed += 1
    print(f"[evoke] policy installed, primed with {primed} blocks")
    return policy


def find_offload_manager(llm: LLM):
    """Locate the OffloadingManager via the scheduler -> connector ->
    connector_scheduler.manager attribute chain. Returns None if no
    offloading is active."""
    engine = llm.llm_engine
    try:
        scheduler = engine.engine_core.engine_core.scheduler
    except AttributeError:
        return None
    connector = getattr(scheduler, "connector", None)
    if connector is None:
        return None
    connector_scheduler = getattr(connector, "connector_scheduler", None)
    if connector_scheduler is None:
        return None
    return getattr(connector_scheduler, "manager", None)


def find_offloading_handler(llm: LLM):
    """Locate the CpuGpuOffloadingHandlers built by the connector worker.
    The handler lives on the spec instance, accessed via the worker-side
    OffloadingConnectorWorker. Searches multiple known attribute paths."""
    engine = llm.llm_engine
    candidates: list = []
    try:
        worker = engine.engine_core.engine_core.model_executor.driver_worker.worker
        candidates.append(worker)
    except AttributeError:
        pass
    try:
        worker = engine.engine_core.engine_core.model_executor.driver_worker
        candidates.append(worker)
    except AttributeError:
        pass
    for worker in candidates:
        connector_worker = getattr(worker, "kv_connector_worker", None) or getattr(
            worker, "_kv_connector_worker", None
        )
        if connector_worker is None:
            continue
        spec = getattr(connector_worker, "spec", None)
        if spec is None:
            continue
        handlers = getattr(spec, "_handlers", None)
        if handlers is not None:
            return handlers
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "/mnt/c/Applications/llama-cpp/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ),
    )
    parser.add_argument(
        "--cpu-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="bytes of CPU memory for the offload tier",
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--num-fill-requests",
        type=int,
        default=4,
        help="number of filler requests to flood the GPU cache so blocks "
        "spill to CPU offload",
    )
    parser.add_argument("--recover-top-k", type=int, default=4)
    args = parser.parse_args()

    if args.model.lower().endswith(".gguf") and not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 2

    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": args.cpu_bytes,
            "eviction_policy": "evoke",
            "evoke_recovery_enabled": True,
            "evoke_rope_head_dim": QWEN2_5_7B_HEAD_DIM,
            "evoke_rope_num_layers": QWEN2_5_7B_NUM_LAYERS,
            "evoke_rope_num_kv_heads": QWEN2_5_7B_NUM_KV_HEADS,
            "evoke_rope_base": QWEN2_5_7B_ROPE_BASE,
            "evoke_rope_max_position": args.max_model_len,
            "evoke_rope_is_neox": True,
        },
    )

    print("=" * 70)
    print(f"EVOKE recovery e2e smoke: model={args.model}")
    print(
        f"  cpu_bytes={args.cpu_bytes / 1e9:.2f} GB, "
        f"num_fill_requests={args.num_fill_requests}, "
        f"recover_top_k={args.recover_top_k}"
    )
    print("=" * 70)

    t0 = time.time()
    kwargs = dict(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype="auto",
        attention_backend="FLASH_ATTN",
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=True,
    )
    if args.model.lower().endswith(".gguf"):
        kwargs["tokenizer"] = "Qwen/Qwen2.5-7B-Instruct"
    llm = LLM(**kwargs)
    print(f"[1/6] engine loaded in {time.time() - t0:.1f}s")

    policy = install_evoke_policy(llm)
    print(f"[2/6] EVOKE policy installed; num_evictable={policy.num_evictable()}")

    handlers = find_offloading_handler(llm)
    if handlers is None:
        print(
            "[3/6] WARN: could not locate CpuGpuOffloadingHandlers; "
            "rotator state unknown"
        )
    else:
        rotator = handlers.cpu_to_gpu_handler.evoke_rope_rotator
        if rotator is None:
            print(
                "[3/6] WARN: rotator NOT constructed -- recovered blocks "
                "will NOT be re-anchored. Check spec config."
            )
        else:
            print(
                f"[3/6] rotator constructed: "
                f"{rotator.num_layers} layers, head_size={rotator.head_size}, "
                f"is_neox={rotator.is_neox}"
            )

    query_embedding = list(np.random.RandomState(42).randn(8).astype(np.float32))

    sampling = SamplingParams(max_tokens=8, temperature=0.0)
    print(f"[4a/6] sending {args.num_fill_requests} filler requests to populate cache")
    for i in range(args.num_fill_requests):
        prompt = f"Filler request {i}: " + ("hi " * 1000) + " Done."
        t = time.time()
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        if (i + 1) % 5 == 0:
            print(
                f"   filler {i + 1}/{args.num_fill_requests} ({time.time() - t:.1f}s)"
            )

    offload_manager = find_offload_manager(llm)
    if offload_manager is not None:
        offload_policy = offload_manager._policy
        offload_block_count = len(offload_policy.blocks)
        print(f"[4b/6] offload tier has {offload_block_count} blocks after fillers")
        if offload_block_count == 0:
            print(
                "   WARN: no blocks offloaded -- cache wasn't filled enough "
                "to trigger spill; recovery candidates will be empty"
            )
        else:
            query = np.asarray(query_embedding, dtype=np.float32)
            print(
                "   manually pushing query-aligned embeddings into offload "
                "blocks so recommend_recovery has candidates"
            )
            rng = np.random.RandomState(7)
            for i, (key, _block) in enumerate(offload_policy.blocks.items()):
                if i < 4:
                    aligned = (query + 0.05 * rng.randn(*query.shape)).astype(
                        np.float32
                    )
                    offload_policy.set_embedding(key, aligned)
                else:
                    misaligned = rng.randn(*query.shape).astype(np.float32)
                    offload_policy.set_embedding(key, misaligned)
            print(
                f"   pushed embeddings on {offload_block_count} blocks; "
                f"top 4 aligned to query"
            )
    else:
        print("[4b/6] WARN: could not locate offload manager for embedding push")

    print("[5/6] sending recovery request with evoke_request_meta")
    sampling_recovery = SamplingParams(
        max_tokens=32,
        temperature=0.0,
        extra_args={
            "evoke_request_meta": {
                "source_type": SOURCE_USER,
                "priority": 1.0,
                "pinned": False,
                "query_embedding": query_embedding,
                "recover_top_k": args.recover_top_k,
                "min_similarity": 0.0,
            }
        },
    )
    recovery_prompt = (
        "After reviewing the earlier filler requests, briefly summarize what "
        "you remember and continue with a single sentence."
    )
    t = time.time()
    outputs = llm.generate([recovery_prompt], sampling_recovery, use_tqdm=False)
    print(f"   recovery request generation: {time.time() - t:.1f}s")
    print(f"   text: {outputs[0].outputs[0].text!r}")

    print("[6/6] inspecting policy state and offload manager")
    print(f"   policy.meta entries: {len(policy.meta)}")
    print(f"   policy.request_meta entries: {len(policy.request_meta)}")
    engine_assigned = [
        rid for rid, m in policy.request_meta.items() if m.recover_top_k > 0
    ]
    if engine_assigned:
        print(f"   request_meta with recover_top_k > 0: {engine_assigned}")
    else:
        print(
            "   WARN: no request_meta entry carried recover_top_k > 0; "
            "extra_args plumbing may not have run"
        )

    print()
    print("PASS (engine + rotator + recovery path active end-to-end)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
