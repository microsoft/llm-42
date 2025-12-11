#!/usr/bin/env python3
"""
Run 1000 ShareGPT requests twice (sequential vs Poisson arrival) via bench_serving, then
compare token-level outputs and plot the CDF of (output_length - first mismatch index).
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sglang import bench_serving


def first_two_mismatches(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int]:
    """Return first and second mismatch indices; treat length difference as a mismatch.

    If sequences are identical and equal length, both indices equal the length.
    If only a length difference exists, first=len(common_prefix), second=max(len(a), len(b)).
    """
    len_a, len_b = len(a), len(b)
    limit = min(len_a, len_b)
    first = limit
    second = limit
    mismatches = 0
    for i in range(limit):
        if a[i] != b[i]:
            mismatches += 1
            if mismatches == 1:
                first = i
            elif mismatches == 2:
                second = i
                return first, second

    # Handle length difference as a mismatch
    if len_a != len_b:
        mismatches += 1
        if mismatches == 1:
            first = limit
            second = max(len_a, len_b)
        elif mismatches == 2:
            second = limit

    return first, second


def build_base_args(cli_args: argparse.Namespace) -> dict:
    # Populate the fields run_benchmark expects; values mirror bench_serving CLI defaults.
    base_dict = {
        "backend": cli_args.backend,
        "base_url": cli_args.base_url,
        "dataset_name": "sharegpt",
        "dataset_path": cli_args.dataset_path,
        "model": cli_args.model,
        "tokenizer": cli_args.tokenizer,
        "num_prompts": cli_args.num_prompts,
        "sharegpt_output_len": cli_args.sharegpt_output_len,
        "sharegpt_context_len": cli_args.sharegpt_context_len,
        "random_input_len": 1024,
        "random_output_len": 1024,
        "random_range_ratio": 0.0,
        "random_image_num_images": 1,
        "random_image_resolution": "1080p",
        "use_trace_timestamps": False,
        "output_file": None,
        "output_details": True,
        "output_latencies": None,
        "disable_tqdm": False,
        "disable_stream": False,
        "return_logprob": False,
        "seed": cli_args.seed,
        "disable_ignore_eos": False,
        "extra_request_body": cli_args.extra_request_body,
        "deterministic_ratio": cli_args.deterministic_ratio,
        "apply_chat_template": cli_args.apply_chat_template,
        "profile": False,
        "lora_name": None,
        "prompt_suffix": cli_args.prompt_suffix,
        "pd_separated": False,
        "flush_cache": cli_args.flush_cache,
        "warmup_requests": cli_args.warmup_requests,
        "tokenize_prompt": False,
        # generated-shared-prefix defaults
        "gsp_num_groups": 64,
        "gsp_prompts_per_group": 16,
        "gsp_system_prompt_len": 2048,
        "gsp_question_len": 128,
        "gsp_output_len": 256,
        # mooncake defaults
        "mooncake_slowdown_factor": 1.0,
        "mooncake_num_rounds": 1,
        "mooncake_workload": "conversation",
    }
    
    # Only set host/port if base_url is not provided
    if cli_args.base_url is None:
        base_dict["host"] = cli_args.host
        base_dict["port"] = cli_args.port
    else:
        base_dict["host"] = None
        base_dict["port"] = None
    
    return base_dict


def run_once(base: dict, request_rate: float, max_concurrency: int | None, output_file: Path) -> tuple[SimpleNamespace, dict]:
    args = SimpleNamespace(**base)
    args.request_rate = request_rate
    args.max_concurrency = max_concurrency
    args.output_file = str(output_file)
    args.output_latencies = str(output_file.with_suffix(".latencies.jsonl"))
    bench_serving.set_global_args(args)
    result = bench_serving.run_benchmark(args)
    return args, result


def tokenize(texts: List[str], tokenizer) -> List[List[int]]:
    return [tokenizer.encode(t, add_special_tokens=False) for t in texts]


def write_jsonl(path: Path, rows: List[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def plot_cdf(values: List[int], path: Path):
    arr = np.array(values)
    arr.sort()
    cdf = np.arange(1, len(arr) + 1) / len(arr)
    plt.figure(figsize=(6, 4))
    plt.step(arr, cdf, where="post")
    plt.xlabel("output_length - first mismatch index")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare ShareGPT sequential vs Poisson runs")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--base-url", dest="base_url", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--sharegpt-output-len", type=int, default=None)
    parser.add_argument("--sharegpt-context-len", type=int, default=None)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--deterministic-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qps", type=float, default=6.0, help="Poisson arrival rate for the second run")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Optional cap for Poisson run")
    parser.add_argument("--sequential-max-concurrency", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("sharegpt_compare_out"))
    parser.add_argument("--extra-request-body", default=None, help="JSON string passed through to bench_serving")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=1)

    cli_args = parser.parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    base = build_base_args(cli_args)
    seq_file = cli_args.output_dir / "sequential.jsonl"
    poi_file = cli_args.output_dir / "poisson.jsonl"

    seq_args, seq_result = run_once(
        base=base,
        request_rate=float("inf"),
        max_concurrency=cli_args.sequential_max_concurrency,
        output_file=seq_file,
    )
    poi_args, poi_result = run_once(
        base=base,
        request_rate=cli_args.qps,
        max_concurrency=cli_args.max_concurrency,
        output_file=poi_file,
    )

    tokenizer_id = cli_args.tokenizer or seq_args.model
    tokenizer = bench_serving.get_tokenizer(tokenizer_id)

    seq_tokens = tokenize(seq_result["generated_texts"], tokenizer)
    poi_tokens = tokenize(poi_result["generated_texts"], tokenizer)

    mismatches = []
    deltas = []
    for i, (s_tok, p_tok, s_text, p_text) in enumerate(
        zip(seq_tokens, poi_tokens, seq_result["generated_texts"], poi_result["generated_texts"])
    ):
        first_mismatch, second_mismatch = first_two_mismatches(s_tok, p_tok)
        delta = len(s_tok) - first_mismatch
        deltas.append(delta)
        mismatches.append(
            {
                "request_id": i,
                "generated_text_sequential": s_text,
                "generated_token_ids_sequential": s_tok,
                "generated_text_qps_6": p_text,
                "generated_token_ids_qps_6": p_tok,
                "output_length": len(s_tok),
                "first_mismatch_index": first_mismatch,
                "second_mismatch_index": second_mismatch,
                "output_length_minus_mismatch": delta,
            }
        )

    write_jsonl(cli_args.output_dir / "mismatch_per_request.jsonl", mismatches)
    plot_cdf(deltas, cli_args.output_dir / "mismatch_cdf.pdf")

    summary = {
        "sequential_file": str(seq_file),
        "poisson_file": str(poi_file),
        "mismatch_file": str(cli_args.output_dir / "mismatch_per_request.jsonl"),
        "cdf_plot": str(cli_args.output_dir / "mismatch_cdf.pdf"),
        "total_requests": len(mismatches),
        "zero_mismatch_fraction": float(np.mean(np.array(deltas) == 0)),
    }
    with (cli_args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
