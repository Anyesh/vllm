"""Agentic multi-turn eval for EVOKE on vLLM.

Mirrors `unlearn/scripts/agent_bench.py` workload: simulates an agent
accumulating context across turns (system prompt, an early planted fact, then
many unrelated file reads), then probes a fact from the early planted file
that must survive eviction.

This is the multi-turn analog NIAH cannot exercise. EVOKE policy difference
shows here: vLLM's prefix cache reuses blocks across sequential requests, and
when the cache fills, the eviction policy decides which old blocks survive.
With EVOKE's source-type floors, USER blocks (carrying the planted fact)
should outlive filler under pressure; with default LRU, FIFO wins.

Caveat: without smart-recovery scheduler integration (the deferred half of
Task #2), an evicted block is gone for good in vLLM. Pass rates reflect
eviction quality but not recovery-after-eviction. That gap is the next-paper
work.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm import LLM, SamplingParams
from vllm.v1.core.eviction_policy import (
    SOURCE_DOCUMENT,
    SOURCE_USER,
    EvokeBlockEvictionPolicy,
)


SYSTEM = (
    "You are an autonomous coding assistant working inside a software "
    "repository. Inspect files before editing them, make minimal correct "
    "changes, run the test suite after every change, and explain your "
    "reasoning concisely. Never invent file contents you have not read."
)

PLANTED_FACT = (
    "config.py: central application configuration. "
    "The maximum retry limit is set to 17 attempts. "
    "Connection timeouts and pool sizes are also defined in this file."
)
FACT_KEY = "config.py"
EXPECTED = "17"
PROBE = "What is the maximum retry limit set in config.py?"

FILLER_TEMPLATE = (
    "{name}: module wires together service components. It defines helpers "
    "for request parsing, response shaping, and error propagation, and "
    "favours small composable functions over deep inheritance hierarchies. "
    "Internal naming follows the project's conventions for snake_case "
    "module-level constants and PascalCase class names. Unit tests live "
    "alongside the implementation, organised by behaviour rather than by "
    "interface boundary."
)

FILE_NAMES = [
    "database",
    "auth",
    "handlers",
    "models",
    "cache",
    "router",
    "metrics",
    "serializer",
    "validators",
    "tasks",
]


def build_session() -> list[tuple[str, str]]:
    """List of (file_key, file_text) tuples in the order the agent reads them.
    Planted fact is item 1 (right after system), so a FIFO eviction sweeps it
    out first; an EVOKE-protected source_type=user assignment should keep it
    alive longer."""
    items = [(FACT_KEY, PLANTED_FACT)]
    for name in FILE_NAMES:
        items.append((f"{name}.py", FILLER_TEMPLATE.format(name=name)))
    return items


def build_turn_prompt(history: list[tuple[str, str]], new_file: tuple[str, str]) -> str:
    """Construct the prompt for a single agent turn. `history` is files
    already-read in earlier turns; `new_file` is the one being read now."""
    parts = [f"System: {SYSTEM}", ""]
    for key, text in history:
        parts.append(f"=== Earlier file read: {key} ===\n{text}")
    key, text = new_file
    parts.append(f"=== Current file read: {key} ===\n{text}")
    parts.append("")
    parts.append(
        "Acknowledge that you have read the current file in one short sentence."
    )
    return "\n".join(parts)


def build_probe_prompt(history: list[tuple[str, str]]) -> str:
    parts = [f"System: {SYSTEM}", ""]
    for key, text in history:
        parts.append(f"=== Earlier file read: {key} ===\n{text}")
    parts.append("")
    parts.append(f"User question: {PROBE}")
    parts.append("Answer:")
    return "\n".join(parts)


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
    print(f"  [evoke] policy installed, primed with {primed} blocks")
    return policy


def install_eviction_counter(policy):
    original = policy.select_eviction_candidates
    counter = {"calls": 0, "blocks_evicted": 0}

    def wrapped(n, protected):
        counter["calls"] += 1
        result = original(n, protected)
        if result is not None:
            counter["blocks_evicted"] += len(result)
        return result

    policy.select_eviction_candidates = wrapped
    return counter


@dataclass
class RunResult:
    strategy: str
    budget_blocks: int
    n_turns: int
    probe_ok: bool
    probe_answer: str
    probe_elapsed_s: float
    eviction_calls: int
    blocks_evicted: int
    final_meta_size: int


def load_engine(model: str, num_gpu_blocks_override: int, max_model_len: int) -> LLM:
    kwargs = dict(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        max_model_len=max_model_len,
        dtype="auto",
        attention_backend="FLASH_ATTN",
        num_gpu_blocks_override=num_gpu_blocks_override,
        enable_prefix_caching=True,
    )
    if model.lower().endswith(".gguf"):
        kwargs["tokenizer"] = "Qwen/Qwen2.5-7B-Instruct"
    return LLM(**kwargs)


def run_one_session(
    strategy: str,
    model: str,
    budget_blocks: int,
    max_model_len: int,
    ack_tokens: int,
    probe_tokens: int,
) -> RunResult:
    print(f"\n=== strategy={strategy} budget_blocks={budget_blocks} ===")
    t0 = time.time()
    llm = load_engine(
        model=model,
        num_gpu_blocks_override=budget_blocks,
        max_model_len=max_model_len,
    )
    print(f"  engine loaded in {time.time() - t0:.1f}s")

    policy = install_evoke_policy(llm) if strategy == "evoke" else None
    counter = install_eviction_counter(policy) if policy is not None else None

    session = build_session()
    history: list[tuple[str, str]] = []
    # Per-turn: USER source-type so EVOKE's source_floors[USER]=0.6 protects
    # these blocks (assuming the policy difference matters under pressure).
    ack_sampling = SamplingParams(
        max_tokens=ack_tokens,
        temperature=0.0,
        extra_args={
            "evoke_request_meta": {
                "source_type": SOURCE_USER,
                "priority": 1.0,
                "pinned": False,
            }
        }
        if strategy == "evoke"
        else None,
    )
    for i, file in enumerate(session):
        prompt = build_turn_prompt(history, file)
        try:
            t1 = time.time()
            outputs = llm.generate([prompt], ack_sampling, use_tqdm=False)
            ack = outputs[0].outputs[0].text.strip().replace("\n", " ")[:80]
            print(
                f"  turn {i + 1}/{len(session)} ({file[0]}) {time.time() - t1:.2f}s ack: {ack!r}"
            )
        except Exception as e:
            print(f"  turn {i + 1} ERROR: {e}")
            ack = f"ERROR: {e}"
        history.append(file)

    probe_sampling = SamplingParams(max_tokens=probe_tokens, temperature=0.0)
    probe_prompt = build_probe_prompt(history)
    t2 = time.time()
    try:
        outputs = llm.generate([probe_prompt], probe_sampling, use_tqdm=False)
        probe_answer = outputs[0].outputs[0].text.strip()
    except Exception as e:
        probe_answer = f"ERROR: {e}"
    probe_elapsed = time.time() - t2
    probe_ok = EXPECTED.lower() in probe_answer.lower()
    print(
        f"  PROBE [{'OK' if probe_ok else 'MISS'}] {probe_elapsed:.2f}s : {probe_answer[:120]!r}"
    )

    result = RunResult(
        strategy=strategy,
        budget_blocks=budget_blocks,
        n_turns=len(session),
        probe_ok=probe_ok,
        probe_answer=probe_answer,
        probe_elapsed_s=probe_elapsed,
        eviction_calls=counter["calls"] if counter else 0,
        blocks_evicted=counter["blocks_evicted"] if counter else 0,
        final_meta_size=len(policy.meta) if policy else 0,
    )

    try:
        llm.llm_engine.engine_core.engine_core.shutdown()
    except Exception as e:
        print(f"  shutdown warn: {e}")
    del llm
    if policy is not None:
        del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "/mnt/c/Applications/llama-cpp/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ),
    )
    parser.add_argument("--strategies", default="evoke,lru")
    parser.add_argument(
        "--budgets",
        default="256,512,1024",
        help="num_gpu_blocks_override values; small = forces eviction",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--ack-tokens", type=int, default=16)
    parser.add_argument("--probe-tokens", type=int, default=48)
    parser.add_argument("--out", default="agent_results.json")
    args = parser.parse_args()

    if args.model.lower().endswith(".gguf"):
        if not Path(args.model).exists():
            print(f"ERROR: model not found: {args.model}", file=sys.stderr)
            return 2

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    results: list[RunResult] = []
    for strategy in strategies:
        for budget in budgets:
            try:
                r = run_one_session(
                    strategy=strategy,
                    model=args.model,
                    budget_blocks=budget,
                    max_model_len=args.max_model_len,
                    ack_tokens=args.ack_tokens,
                    probe_tokens=args.probe_tokens,
                )
                results.append(r)
            except Exception as e:
                print(f"  session FAILED ({strategy}@{budget}): {e}")
                results.append(
                    RunResult(
                        strategy=strategy,
                        budget_blocks=budget,
                        n_turns=0,
                        probe_ok=False,
                        probe_answer=f"SESSION_FAIL: {e}",
                        probe_elapsed_s=0.0,
                        eviction_calls=0,
                        blocks_evicted=0,
                        final_meta_size=0,
                    )
                )

    payload = {
        "model": args.model,
        "results": [asdict(r) for r in results],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nresults written to {args.out}")
    print()
    print(
        f"{'strategy':<8} {'budget':<7} {'probe':<6} {'evict_calls':<12} "
        f"{'evicted':<9} {'meta':<6}"
    )
    for r in results:
        print(
            f"{r.strategy:<8} {r.budget_blocks:<7} "
            f"{'OK' if r.probe_ok else 'MISS':<6} {r.eviction_calls:<12} "
            f"{r.blocks_evicted:<9} {r.final_meta_size:<6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
