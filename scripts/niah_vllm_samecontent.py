"""Same-content NIAH cell on vLLM for EVOKE evaluation.

Reviewer item #3: produce one end-to-end NIAH cell on vLLM's same-content
gap-fill mode (the EVOKE recovery semantic that vLLM does support: blocks
that were previously offloaded get loaded back into a new request whose
prompt re-sends the same tokens, exercising vLLM's prefix-cache + offload
tier under EVOKE policy governance).

Design:
1. Build N distinct "needle" prompts, each a long haystack with one
   planted fact and one probe question. First turn: send full prompt,
   model generates a brief acknowledgement, blocks for this prompt get
   admitted to the GPU cache, then offloaded to CPU as more requests
   arrive.
2. Send M unrelated filler prompts to push the needle prompts' blocks
   under cache pressure. Filler load determines how many needle blocks
   survive in the offload tier.
3. Probe turn: re-send each needle prompt + a question. vLLM's prefix
   cache scans the prompt's block hashes; those that hit either GPU or
   CPU offload skip recompute. The cache policy (LRU vs EVOKE) is what
   decides which offloaded blocks were evicted vs retained under
   pressure. Score whether the model's answer contains the needle.
4. Compare cache_policy=evoke vs cache_policy=lru on the same prompts.

This exercises a path vLLM natively supports (hash-based prefix-cache
extension across requests). The EVOKE contribution under test is the
*eviction-ordering quality of the offload tier policy* under pressure.
We are not testing similarity-based cross-session recall (which vLLM's
request model does not support, see paper Sec. Future Work).
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
from vllm.config import KVTransferConfig


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


TOPICS = [
    "the migratory patterns of the spotted ironwing albatross",
    "fermented tea cultivation in the highland terraces of southwestern China",
    "the geometry of fan vaulting in late Gothic ecclesiastical architecture",
    "polymer chemistry of high-temperature silicone elastomers",
    "the slow erosion of basalt columns along volcanic coastlines",
    "the manufacture of cuneiform tablets in Mesopotamian scribal academies",
    "echolocation strategies of insectivorous bats in dense forest canopy",
]


def build_haystack(seed: int, n_sentences: int = 18) -> str:
    rng = random.Random(seed)
    sentences = []
    for _ in range(n_sentences):
        topic = rng.choice(TOPICS)
        sentences.append(
            f"{topic} has been studied extensively in the past few decades "
            f"with at least {rng.randint(3, 17)} distinct subfields contributing."
        )
    return " ".join(sentences)


def build_needle_prompt(needle: dict, depth_pct: int, seed: int) -> str:
    haystack_pre = build_haystack(seed=seed, n_sentences=12)
    haystack_post = build_haystack(seed=seed + 1, n_sentences=12)
    pos = max(1, int(len(haystack_pre.split()) * depth_pct / 100))
    pre_words = haystack_pre.split()
    pre_with_needle = (
        " ".join(pre_words[:pos])
        + " "
        + needle["text"]
        + " "
        + " ".join(pre_words[pos:])
    )
    document = pre_with_needle + " " + haystack_post
    return (
        "You are a careful research assistant. Read the document below.\n\n"
        f"<document>\n{document}\n</document>\n\n"
        "Acknowledge in one sentence."
    )


def build_probe_prompt(needle: dict, depth_pct: int, seed: int) -> str:
    haystack_pre = build_haystack(seed=seed, n_sentences=12)
    haystack_post = build_haystack(seed=seed + 1, n_sentences=12)
    pos = max(1, int(len(haystack_pre.split()) * depth_pct / 100))
    pre_words = haystack_pre.split()
    pre_with_needle = (
        " ".join(pre_words[:pos])
        + " "
        + needle["text"]
        + " "
        + " ".join(pre_words[pos:])
    )
    document = pre_with_needle + " " + haystack_post
    return (
        "You are a careful research assistant. Read the document below and "
        "answer the user's question exactly, copying the relevant fact "
        "verbatim if it appears.\n\n"
        f"<document>\n{document}\n</document>\n\n"
        f"User question: {needle['question']}\n\n"
        "Answer:"
    )


def build_filler_prompt(idx: int, n_sentences: int = 80) -> str:
    haystack = build_haystack(seed=10_000 + idx, n_sentences=n_sentences)
    return (
        f"Filler request {idx}. Read the document below and write one short "
        f"sentence summarising the topic.\n\n<document>\n{haystack}\n</document>"
    )


@dataclass
class CellResult:
    cache_policy: str
    needle_id: str
    probe_ok: bool
    answer: str
    probe_elapsed_s: float


def load_engine(
    model: str,
    cache_policy: str,
    cpu_bytes: int,
    max_model_len: int,
    gpu_mem_util: float,
) -> LLM:
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": cpu_bytes,
            "eviction_policy": cache_policy,
        },
    )
    kwargs = dict(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        dtype="auto",
        attention_backend="FLASH_ATTN",
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=True,
    )
    if model.lower().endswith(".gguf"):
        kwargs["tokenizer"] = "Qwen/Qwen2.5-7B-Instruct"
    return LLM(**kwargs)


def run_cell(
    cache_policy: str,
    model: str,
    cpu_bytes: int,
    max_model_len: int,
    gpu_mem_util: float,
    depth_pct: int,
    seed: int,
    num_filler: int,
    ack_tokens: int,
    probe_tokens: int,
) -> list[CellResult]:
    print(f"\n=== cache_policy={cache_policy} ===")
    t0 = time.time()
    llm = load_engine(
        model=model,
        cache_policy=cache_policy,
        cpu_bytes=cpu_bytes,
        max_model_len=max_model_len,
        gpu_mem_util=gpu_mem_util,
    )
    print(f"  engine loaded in {time.time() - t0:.1f}s")

    ack_sampling = SamplingParams(max_tokens=ack_tokens, temperature=0.0)
    probe_sampling = SamplingParams(max_tokens=probe_tokens, temperature=0.0)

    print(f"  phase 1: send {len(NEEDLES)} needle prompts (first turn)")
    for i, needle in enumerate(NEEDLES):
        prompt = build_needle_prompt(needle, depth_pct=depth_pct, seed=seed + i)
        t = time.time()
        llm.generate([prompt], ack_sampling, use_tqdm=False)
        print(f"    {needle['id']:>10} first-turn ack ({time.time() - t:.2f}s)")

    print(f"  phase 2: send {num_filler} filler prompts to pressure cache")
    for i in range(num_filler):
        prompt = build_filler_prompt(i)
        llm.generate([prompt], ack_sampling, use_tqdm=False)
        if (i + 1) % 5 == 0:
            print(f"    filler {i + 1}/{num_filler}")

    print(f"  phase 3: probe {len(NEEDLES)} needles via re-sent prompts")
    cells: list[CellResult] = []
    for i, needle in enumerate(NEEDLES):
        prompt = build_probe_prompt(needle, depth_pct=depth_pct, seed=seed + i)
        t = time.time()
        outputs = llm.generate([prompt], probe_sampling, use_tqdm=False)
        elapsed = time.time() - t
        answer = outputs[0].outputs[0].text.strip()
        probe_ok = needle["expected"].lower() in answer.lower()
        tag = "OK" if probe_ok else "MISS"
        print(f"    {needle['id']:>10} probe [{tag}] {elapsed:.2f}s : {answer[:80]!r}")
        cells.append(
            CellResult(
                cache_policy=cache_policy,
                needle_id=needle["id"],
                probe_ok=probe_ok,
                answer=answer,
                probe_elapsed_s=elapsed,
            )
        )

    try:
        llm.llm_engine.engine_core.engine_core.shutdown()
    except Exception as e:
        print(f"  shutdown warn: {e}")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    return cells


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
        default=256 * 1024 * 1024,
        help="CPU offload tier capacity in bytes; smaller = tighter pressure",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem-util", type=float, default=0.45)
    parser.add_argument("--depth-pct", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-filler",
        type=int,
        default=25,
        help="how many filler prompts to send between phases 1 and 3",
    )
    parser.add_argument("--ack-tokens", type=int, default=24)
    parser.add_argument("--probe-tokens", type=int, default=48)
    parser.add_argument("--out", default="niah_vllm_samecontent.json")
    parser.add_argument("--policies", default="evoke,lru")
    args = parser.parse_args()

    if args.model.lower().endswith(".gguf") and not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 2

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    all_cells: list[CellResult] = []
    for policy in policies:
        cells = run_cell(
            cache_policy=policy,
            model=args.model,
            cpu_bytes=args.cpu_bytes,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
            depth_pct=args.depth_pct,
            seed=args.seed,
            num_filler=args.num_filler,
            ack_tokens=args.ack_tokens,
            probe_tokens=args.probe_tokens,
        )
        all_cells.extend(cells)

    print()
    print(f"{'policy':<8} {'needle':<10} {'probe':<6} {'elapsed':<8}")
    for c in all_cells:
        print(
            f"{c.cache_policy:<8} {c.needle_id:<10} "
            f"{'OK' if c.probe_ok else 'MISS':<6} {c.probe_elapsed_s:.2f}s"
        )

    summary = {}
    for policy in policies:
        policy_cells = [c for c in all_cells if c.cache_policy == policy]
        passes = sum(1 for c in policy_cells if c.probe_ok)
        n = len(policy_cells)
        summary[policy] = {
            "n": n,
            "passes": passes,
            "pass_rate": passes / n if n else 0.0,
        }
    print()
    print("summary:")
    for policy, s in summary.items():
        print(f"  {policy:<8} {s['passes']}/{s['n']} ({100 * s['pass_rate']:.0f}%)")

    payload = {
        "model": args.model,
        "cpu_bytes": args.cpu_bytes,
        "max_model_len": args.max_model_len,
        "depth_pct": args.depth_pct,
        "num_filler": args.num_filler,
        "policies": policies,
        "summary": summary,
        "cells": [asdict(c) for c in all_cells],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nresults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
