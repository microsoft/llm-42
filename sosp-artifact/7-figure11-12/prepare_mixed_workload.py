#!/usr/bin/env python3
"""
Build the mixed_workload dataset: a blend of short, long, and reasoning
prompts for realistic workload benchmarking.

Composition (1024 requests by default):
  - Short (~1/3):     ShareGPT prompts with output_tokens ≤ 100
  - Long  (~1/3):     ArXiv summarization (long input, medium output)
  - Reasoning (~1/3): GSM8K math problems with chain-of-thought prompting

Output files are written to ~/.cache/llm42_bench/ in the same format used by
prepare_shared_datasets.py, so run_balanced_online_benchmark.sh auto-detects
them (dataset name = "mixed_workload").

Usage:
    python prepare_mixed_workload.py
    python prepare_mixed_workload.py --total 2048 --seed 42 --force
    python prepare_mixed_workload.py --short-frac 0.25 --long-frac 0.25 --reasoning-frac 0.50
"""

import argparse
import json
import random
import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "llm42_bench"

COT_SYSTEM = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then give the final answer on a new line "
    "in the format: #### <number>"
)


def _write_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_short_prompts(tokenizer, n: int, seed: int) -> list:
    """Sample ShareGPT prompts with short expected outputs (≤100 tokens)."""
    print(f"  Loading ShareGPT for short prompts ...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
    )
    with open(path) as f:
        raw = json.load(f)

    candidates = []
    for item in raw:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue
        user_msg = convs[0].get("value", convs[0].get("content", "")).strip()
        asst_msg = convs[1].get("value", convs[1].get("content", "")).strip()
        if not user_msg or not asst_msg:
            continue

        prompt_tokens = len(tokenizer.encode(user_msg))
        output_tokens = len(tokenizer.encode(asst_msg))

        if prompt_tokens < 4 or output_tokens < 128 or output_tokens > 256:
            continue

        candidates.append({
            "messages": [{"role": "user", "content": user_msg}],
            "max_tokens": output_tokens,
            "_category": "short",
            "_prompt_tokens": prompt_tokens,
            "_output_tokens": output_tokens,
        })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    print(f"  Short: {len(selected)}/{n} selected from {len(candidates)} candidates")
    return selected


def load_long_prompts(tokenizer, n: int, seed: int) -> list:
    """Sample ArXiv summarization prompts (long input)."""
    print(f"  Loading ArXiv for long prompts ...")
    from datasets import load_dataset
    ds = load_dataset("ccdv/arxiv-summarization", split="test")

    candidates = []
    for item in ds:
        article = item["article"].strip()
        abstract = item["abstract"].strip()
        if not article or not abstract:
            continue

        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
        prompt_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(abstract))

        if prompt_tokens < 10 or output_tokens < 128:
            continue

        candidates.append({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(output_tokens, 128),
            "_category": "long",
            "_prompt_tokens": prompt_tokens,
            "_output_tokens": output_tokens,
        })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    print(f"  Long: {len(selected)}/{n} selected from {len(candidates)} candidates")
    return selected


def load_reasoning_prompts(tokenizer, n: int, seed: int) -> list:
    """Sample GSM8K math problems with chain-of-thought prompting."""
    print(f"  Loading GSM8K for reasoning prompts ...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")

    candidates = []
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        if not question or not answer:
            continue

        prompt = f"{COT_SYSTEM}\n\n{question}"
        prompt_tokens = len(tokenizer.encode(prompt))
        # Use the reference answer length as max_tokens estimate.
        # GSM8K answers include step-by-step reasoning, so they're naturally long.
        output_tokens = len(tokenizer.encode(answer))

        if prompt_tokens < 4 or output_tokens < 4:
            continue

        candidates.append({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(output_tokens * 8, 2048),  # 5-10x longer for chain-of-thought
            "_category": "reasoning",
            "_prompt_tokens": prompt_tokens,
            "_output_tokens": output_tokens,
        })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    print(f"  Reasoning: {len(selected)}/{n} selected from {len(candidates)} candidates")
    return selected


def print_stats(records: list, label: str):
    """Print token distribution statistics."""
    prompt_lens = [r["_prompt_tokens"] for r in records]
    output_lens = [r["_output_tokens"] for r in records]
    max_tokens = [r["max_tokens"] for r in records]

    import numpy as np
    print(f"\n  {label} ({len(records)} requests):")
    print(f"    Input tokens  — min: {min(prompt_lens):>6}  median: {int(np.median(prompt_lens)):>6}"
          f"  mean: {np.mean(prompt_lens):>8.1f}  max: {max(prompt_lens):>6}")
    print(f"    Output tokens — min: {min(output_lens):>6}  median: {int(np.median(output_lens)):>6}"
          f"  mean: {np.mean(output_lens):>8.1f}  max: {max(output_lens):>6}")
    print(f"    max_tokens    — min: {min(max_tokens):>6}  median: {int(np.median(max_tokens)):>6}"
          f"  mean: {np.mean(max_tokens):>8.1f}  max: {max(max_tokens):>6}")

    # Per-category breakdown
    categories = sorted(set(r["_category"] for r in records))
    for cat in categories:
        subset = [r for r in records if r["_category"] == cat]
        pl = [r["_prompt_tokens"] for r in subset]
        ol = [r["_output_tokens"] for r in subset]
        print(f"    [{cat:>9}]  n={len(subset):>4}  "
              f"input: {int(np.median(pl)):>5} (median)  "
              f"output: {int(np.median(ol)):>5} (median)")


def main():
    parser = argparse.ArgumentParser(
        description="Build mixed_workload dataset (short + long + reasoning)",
    )
    parser.add_argument("--total", type=int, default=1024,
                        help="Total number of requests (default: 1024)")
    parser.add_argument("--short-frac", type=float, default=0.45,
                        help="Fraction of short prompts (default: 0.45)")
    parser.add_argument("--long-frac", type=float, default=0.45,
                        help="Fraction of long prompts (default: 0.45)")
    parser.add_argument("--reasoning-frac", type=float, default=0.10,
                        help="Fraction of reasoning prompts (default: 0.10)")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Tokenizer for token counting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    sglang_out = CACHE_DIR / "mixed_workload_sglang.jsonl"
    vllm_out = CACHE_DIR / "mixed_workload_vllm.jsonl"

    if sglang_out.exists() and not args.force:
        ns = sum(1 for _ in open(sglang_out))
        print(f"mixed_workload: already prepared ({ns} prompts). Use --force to re-create.")
        return 0

    # Compute counts per category
    total_frac = args.short_frac + args.long_frac + args.reasoning_frac
    n_short = round(args.total * args.short_frac / total_frac)
    n_reasoning = round(args.total * args.reasoning_frac / total_frac)
    n_long = args.total - n_short - n_reasoning  # remainder to long

    print(f"Building mixed_workload: {args.total} requests "
          f"(short={n_short}, long={n_long}, reasoning={n_reasoning})")

    print(f"\nLoading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    short = load_short_prompts(tokenizer, n_short, args.seed)
    long = load_long_prompts(tokenizer, n_long, args.seed + 1)
    reasoning = load_reasoning_prompts(tokenizer, n_reasoning, args.seed + 2)

    all_records = short + long + reasoning
    rng = random.Random(args.seed + 3)
    rng.shuffle(all_records)

    print_stats(all_records, "mixed_workload")

    # Write sglang format (OpenAI chat)
    sglang_records = []
    for r in all_records:
        sglang_records.append({
            "messages": r["messages"],
            "max_tokens": r["max_tokens"],
        })

    # Write vLLM format
    vllm_records = []
    for r in all_records:
        vllm_records.append({
            "prompt": r["messages"][0]["content"],
            "output_tokens": r["max_tokens"],
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(sglang_out, sglang_records)
    _write_jsonl(vllm_out, vllm_records)

    print(f"\nWrote {len(sglang_records)} prompts:")
    print(f"  sglang: {sglang_out}")
    print(f"  vLLM:   {vllm_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
