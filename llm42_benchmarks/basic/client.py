#!/usr/bin/env python3
"""Basic LLM-42 benchmark client. Sends requests and checks determinism across runs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sglang import bench_serving


def first_mismatch(a, b):
    """Index of first token mismatch; returns length if identical."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else len(a)


def run_once(base_url, request_rate, num_prompts, model, seed,
             output_path, dataset_path="", extra_body=None, warmup=1):
    """Run one benchmark pass. Returns (token_lists, rollback_stats_dict)."""
    args = SimpleNamespace(
        backend="sglang", base_url=base_url, model=model, tokenizer=None,
        dataset_name="sharegpt", dataset_path=dataset_path,
        num_prompts=num_prompts, request_rate=request_rate, max_concurrency=None,
        seed=seed, extra_request_body=extra_body,
        deterministic_ratio=1.0, warmup_requests=warmup,
        output_file=str(output_path), output_details=True,
        output_latencies=str(output_path.with_suffix(".latencies.jsonl")),
        disable_stream=True, disable_tqdm=False, return_logprob=False,
        disable_ignore_eos=False, apply_chat_template=False,
        profile=False, lora_name=None, prompt_suffix="", pd_separated=False,
        flush_cache=False, tokenize_prompt=False, host=None, port=None,
        sharegpt_output_len=None, sharegpt_context_len=None,
        random_input_len=1024, random_output_len=1024, random_range_ratio=0.0,
        random_image_num_images=1, random_image_resolution="1080p",
        use_trace_timestamps=False,
        gsp_num_groups=64, gsp_prompts_per_group=16,
        gsp_system_prompt_len=2048, gsp_question_len=128, gsp_output_len=256,
        mooncake_slowdown_factor=1.0, mooncake_num_rounds=1,
        mooncake_workload="conversation",
    )
    bench_serving.set_global_args(args)
    result = bench_serving.run_benchmark(args)

    tokens = result.get("output_ids", [])
    if not tokens or not any(tokens):
        tok = bench_serving.get_tokenizer(model)
        tokens = [tok.encode(t, add_special_tokens=False) for t in result["generated_texts"]]

    meta = result.get("meta_info", [])
    rb = sum(m.get("llm42_num_rollbacks", 0) for m in meta)
    rb_tok = sum(m.get("llm42_tokens_rolled_back", 0) for m in meta)
    return tokens, {"rollbacks": rb, "tokens_rolled_back": rb_tok}


def main():
    p = argparse.ArgumentParser(description="LLM-42 basic benchmark client")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--num-runs",    type=int, default=2)
    p.add_argument("--request-rate", type=float, default=4.0, help="QPS")
    p.add_argument("--base-url",    default="http://127.0.0.1:30000")
    p.add_argument("--model",       default=None)
    p.add_argument("--dataset-path", default="")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--extra-request-body", default='{"temperature":0}')
    p.add_argument("--output-dir",  type=Path, default=Path("basic_out"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== LLM-42 Benchmark: {args.num_prompts} prompts x "
          f"{args.num_runs} runs @ {args.request_rate} QPS ===\n")

    # Run benchmarks
    all_tokens, all_stats = [], []
    for i in range(args.num_runs):
        print(f"[Run {i}] Starting...")
        tokens, stats = run_once(
            args.base_url, args.request_rate, args.num_prompts, args.model,
            args.seed, args.output_dir / f"run_{i}.jsonl",
            args.dataset_path, args.extra_request_body,
        )
        total = sum(len(t) for t in tokens)
        pct = (stats["tokens_rolled_back"] / total * 100) if total else 0
        print(f"[Run {i}] {len(tokens)} responses, {total} tokens, "
              f"{stats['rollbacks']} rollbacks ({pct:.2f}% rolled back)")
        all_tokens.append(tokens)
        all_stats.append(stats | {"total_tokens": total})

    # Compare pairs
    print(f"\n{'='*50}\nPairwise comparison\n{'='*50}")
    pairwise = []
    for i in range(args.num_runs):
        for j in range(i + 1, args.num_runs):
            deltas = np.array([len(a) - first_mismatch(a, b)
                               for a, b in zip(all_tokens[i], all_tokens[j])])
            n_mm = int(np.sum(deltas > 0))
            frac = float(np.mean(deltas > 0))
            pairwise.append({"a": i, "b": j, "mismatches": n_mm, "fraction": frac})
            print(f"  Run {i} vs {j}: {n_mm}/{len(deltas)} mismatches ({frac:.4f})")

    # Save summary
    summary = {"config": {"num_prompts": args.num_prompts, "num_runs": args.num_runs,
                           "request_rate": args.request_rate, "seed": args.seed},
               "per_run": all_stats, "pairwise": pairwise}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {args.output_dir}/summary.json")

    return 1 if any(p["mismatches"] > 0 for p in pairwise) else 0


if __name__ == "__main__":
    exit(main())
