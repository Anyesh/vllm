"""Needle-in-a-Haystack benchmark for EVOKE on vLLM.

Mirrors `unlearn/scripts/niah_bench.py` (llama.cpp side) workload — same needles,
same haystack generator, same probe + scoring — but drives a vLLM engine. Two
strategies are compared on the GPU active-cache layer:

- `lru`: vLLM default FIFO/LRU eviction.
- `evoke`: `EvokeBlockEvictionPolicy` installed on the BlockPool, source-type
  floors + recency scoring (no attention capture wired yet — the capture path
  needs more debug after the smoke test surfaced the K_full reconstruction
  fallback; see project_vllm_port_state.md).

Budget is controlled via `num_gpu_blocks_override`. A small enough cap forces
the model into eviction territory on the long-context document; eviction
quality is what NIAH measures.

Run:
    ./.venv/bin/python scripts/niah_bench.py
        --model /mnt/c/Applications/llama-cpp/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
        --budgets 512,1024,2048
        --out niah_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
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


TOPICS = [
    "the migratory patterns of the spotted ironwing albatross",
    "fermented tea cultivation in the highland terraces of southwestern China",
    "the geometry of fan vaulting in late Gothic ecclesiastical architecture",
    "polymer chemistry of high-temperature silicone elastomers",
    "the slow erosion of basalt columns along volcanic coastlines",
    "the manufacture of cuneiform tablets in Mesopotamian scribal academies",
    "echolocation strategies of insectivorous bats in dense forest canopy",
    "the typography of medieval illuminated manuscripts and their guild traditions",
    "deep-sea hydrothermal vent communities and chemosynthetic metabolism",
    "the harmonic theory of Baroque keyboard counterpoint",
    "alpine glacier retreat and the formation of proglacial lakes",
    "ceramic glaze formulation in Song dynasty kilns",
]

SENT_TEMPLATES = [
    "{topic} has attracted considerable scholarly interest over the past few decades.",
    "Researchers have documented at least {count} distinct sub-patterns within {topic}.",
    "Recent advances in instrumentation have changed how investigators approach {topic}.",
    "The history of {topic} is intertwined with developments in {discipline}.",
    "Several monographs from the {era} survey early empirical work on {topic}.",
    "Specialised {tool} is now standard equipment in any serious study of {topic}.",
    "Cross-disciplinary collaborations between {discipline} and computational modelling now drive most progress on {topic}.",
]

DISCIPLINES = ["materials science", "ecology", "biophysics", "geochemistry"]
ERAS = ["the 1960s", "the post-war decades", "the 1980s"]
TOOLS = ["mass spectrometry", "acoustic telemetry", "high-resolution microscopy"]

NEEDLES = [
    {
        "id": "password",
        "text": (
            "Important administrative note: the secret password for the "
            "laboratory vault is icarus-pinwheel-43. Personnel are reminded "
            "not to write it on shared notepads."
        ),
        "question": "What is the secret password for the laboratory vault?",
        "expected": "icarus-pinwheel-43",
    },
    {
        "id": "capital",
        "text": (
            "Geographic footnote for the next briefing: the capital city of "
            "the small inland country of Polopia is the historic settlement "
            "of Zithrand, located near the western foothills."
        ),
        "question": "What is the capital city of the country called Polopia?",
        "expected": "Zithrand",
    },
    {
        "id": "amount",
        "text": (
            "Lab record: Dr. Halvorsen's calibration run on the spectrometer "
            "used exactly 47 milligrams of cesium chloride dissolved in "
            "buffered ammonium acetate. Results pending second-pass analysis."
        ),
        "question": (
            "How many milligrams of cesium chloride did Dr. Halvorsen use in "
            "the calibration run?"
        ),
        "expected": "47",
    },
    {
        "id": "code",
        "text": (
            "Operational note for the duty officer: the activation code for "
            "the orbital relay station is BLUE-MOUNTAIN-7-DELTA. Confirm "
            "receipt before the next handover window."
        ),
        "question": "What is the activation code for the orbital relay station?",
        "expected": "BLUE-MOUNTAIN-7-DELTA",
    },
    {
        "id": "date",
        "text": (
            "Archival entry: the Treaty of Vrenholm was signed on the "
            "twenty-third of October, 1786, at a quarter past four in the "
            "afternoon, in the upper hall of the merchants' guild."
        ),
        "question": "On what date and time was the Treaty of Vrenholm signed?",
        "expected": "1786",
    },
]

DEFAULT_DEPTHS = [5, 25, 50, 75, 95]


def build_haystack(n_paragraphs: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    paragraphs: list[str] = []
    for _ in range(n_paragraphs):
        topic = rng.choice(TOPICS)
        n_sentences = rng.randint(4, 7)
        sentences: list[str] = []
        for _ in range(n_sentences):
            template = rng.choice(SENT_TEMPLATES)
            sentences.append(
                template.format(
                    topic=topic,
                    count=rng.randint(3, 24),
                    discipline=rng.choice(DISCIPLINES),
                    tool=rng.choice(TOOLS),
                    era=rng.choice(ERAS),
                )
            )
        paragraphs.append(" ".join(sentences))
    return paragraphs


def make_document(haystack: list[str], needle: dict, depth_pct: int) -> str:
    pos = int(len(haystack) * depth_pct / 100)
    inserted = haystack[:pos] + [needle["text"]] + haystack[pos:]
    return "\n\n".join(inserted)


def build_prompt(document: str, question: str) -> str:
    return (
        "You are a careful research assistant. Read the document below and "
        "answer the user's question exactly, copying the relevant fact "
        "verbatim if it appears.\n\n"
        f"<document>\n{document}\n</document>\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )


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


@dataclass
class Cell:
    strategy: str
    budget_blocks: int
    needle_id: str
    depth: int
    probe_ok: bool
    answer: str
    elapsed_s: float
    num_evictions: int


def load_engine(model: str, num_gpu_blocks_override: int, max_model_len: int) -> LLM:
    kwargs = dict(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        max_model_len=max_model_len,
        dtype="auto",
        attention_backend="FLASH_ATTN",
        num_gpu_blocks_override=num_gpu_blocks_override,
    )
    if model.lower().endswith(".gguf"):
        kwargs["tokenizer"] = "Qwen/Qwen2.5-7B-Instruct"
    return LLM(**kwargs)


def run_strategy(
    strategy: str,
    model: str,
    budget_blocks: int,
    n_paragraphs: int,
    seed: int,
    depths: list[int],
    needles: list[dict],
    max_tokens: int,
    max_model_len: int,
) -> tuple[list[Cell], int]:
    """Load one engine for the (strategy, budget) cell, sweep all (needle,
    depth) pairs through it, return the cells + total cumulative eviction count
    observed on the policy (LRU has none to report, evoke increments on each
    select_eviction_candidates call where blocks are consumed)."""
    print(f"\n=== strategy={strategy} budget_blocks={budget_blocks} ===")
    t0 = time.time()
    llm = load_engine(
        model=model,
        num_gpu_blocks_override=budget_blocks,
        max_model_len=max_model_len,
    )
    print(f"  engine loaded in {time.time() - t0:.1f}s")

    policy = install_evoke_policy(llm) if strategy == "evoke" else None
    initial_meta = len(policy.meta) if policy is not None else 0

    cells: list[Cell] = []
    for needle in needles:
        haystack = build_haystack(n_paragraphs=n_paragraphs, seed=seed)
        for depth in depths:
            document = make_document(haystack, needle, depth)
            prompt = build_prompt(document, needle["question"])
            sampling = SamplingParams(
                max_tokens=max_tokens,
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
            t1 = time.time()
            try:
                outputs = llm.generate([prompt], sampling, use_tqdm=False)
                answer = outputs[0].outputs[0].text
                elapsed = time.time() - t1
                probe_ok = needle["expected"].lower() in answer.lower()
                meta_now = len(policy.meta) if policy is not None else 0
                cells.append(
                    Cell(
                        strategy=strategy,
                        budget_blocks=budget_blocks,
                        needle_id=needle["id"],
                        depth=depth,
                        probe_ok=probe_ok,
                        answer=answer.strip(),
                        elapsed_s=elapsed,
                        num_evictions=max(0, meta_now - initial_meta),
                    )
                )
                tag = "OK" if probe_ok else "MISS"
                print(
                    f"  {needle['id']:>10} depth={depth:>3} "
                    f"[{tag}] {elapsed:.2f}s : "
                    f"{answer.strip()[:80]!r}"
                )
            except Exception as e:
                cells.append(
                    Cell(
                        strategy=strategy,
                        budget_blocks=budget_blocks,
                        needle_id=needle["id"],
                        depth=depth,
                        probe_ok=False,
                        answer=f"ERROR: {e}",
                        elapsed_s=time.time() - t1,
                        num_evictions=0,
                    )
                )
                print(f"  {needle['id']:>10} depth={depth:>3} [ERR] {e}")
    total_meta_growth = len(policy.meta) - initial_meta if policy is not None else 0

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
    return cells, total_meta_growth


def aggregate(cells: list[Cell]) -> dict:
    by_cell: dict[tuple[str, int], list[Cell]] = {}
    for c in cells:
        by_cell.setdefault((c.strategy, c.budget_blocks), []).append(c)
    summary: dict[str, dict[str, float]] = {}
    for (strategy, budget), group in by_cell.items():
        key = f"{strategy}@{budget}"
        n = len(group)
        passes = sum(1 for c in group if c.probe_ok)
        mean_elapsed = sum(c.elapsed_s for c in group) / max(n, 1)
        summary[key] = {
            "n": n,
            "passes": passes,
            "pass_rate": passes / max(n, 1),
            "mean_elapsed_s": mean_elapsed,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "/mnt/c/Applications/llama-cpp/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ),
    )
    parser.add_argument(
        "--strategies",
        default="evoke,lru",
        help="comma-separated strategies to run",
    )
    parser.add_argument(
        "--budgets",
        default="64,128,256",
        help="comma-separated num_gpu_blocks_override values (block=16 tokens)",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--paragraphs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--depths",
        default=",".join(str(d) for d in DEFAULT_DEPTHS),
    )
    parser.add_argument(
        "--needles",
        default=",".join(n["id"] for n in NEEDLES),
    )
    parser.add_argument("--out", default="niah_results.json")
    args = parser.parse_args()

    if args.model.lower().endswith(".gguf"):
        if not Path(args.model).exists():
            print(f"ERROR: model not found: {args.model}", file=sys.stderr)
            return 2

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    needle_ids = {n.strip() for n in args.needles.split(",")}
    needles = [n for n in NEEDLES if n["id"] in needle_ids]

    all_cells: list[Cell] = []
    for strategy in strategies:
        for budget in budgets:
            cells, total_growth = run_strategy(
                strategy=strategy,
                model=args.model,
                budget_blocks=budget,
                n_paragraphs=args.paragraphs,
                seed=args.seed,
                depths=depths,
                needles=needles,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
            )
            print(
                f"  ({strategy}@{budget}) policy.meta growth across cells: "
                f"{total_growth}"
            )
            all_cells.extend(cells)

    summary = aggregate(all_cells)
    result = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "paragraphs": args.paragraphs,
        "seed": args.seed,
        "depths": depths,
        "summary": summary,
        "cells": [asdict(c) for c in all_cells],
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nresults written to {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
