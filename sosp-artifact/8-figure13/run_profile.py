#!/usr/bin/env python3
"""
Run the same workload against multiple servers (each with different LLM42 params)
and collect per-config rollback statistics.

Example:
    python run_profile.py \
        --base-urls "http://127.0.0.1:30005,http://127.0.0.1:30006" \
        --profile-configs "ws=64,bs=8;ws=128,bs=16" \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts 500 --select-seed 42 --qps 8 \
        --output-dir runs/my_run
"""

import argparse
import copy
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np

from sglang import bench_serving


@dataclass
class ProfileConfig:
    """A (window_size, verify_batch_size) configuration."""
    window_size: int
    verify_batch_size: int

    @property
    def name(self) -> str:
        return f"ws{self.window_size}_bs{self.verify_batch_size}"

    @classmethod
    def from_string(cls, s: str) -> "ProfileConfig":
        parts = {}
        for part in s.split(","):
            k, v = part.strip().split("=")
            parts[k.strip()] = v.strip()
        return cls(
            window_size=int(parts["ws"]),
            verify_batch_size=int(parts["bs"]),
        )


def build_base_args(cli_args: argparse.Namespace) -> dict:
    """Build base arguments dictionary for bench_serving."""
    return {
        "backend": cli_args.backend,
        "base_url": None,
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
        "disable_stream": True,
        "return_logprob": False,
        "deterministic_seed": 42,
        "seed": cli_args.select_seed,
        "select_seed": cli_args.select_seed,
        "order_seed": cli_args.order_seed,
        "arrival_seed": cli_args.arrival_seed,
        "disable_ignore_eos": not cli_args.ignore_eos,
        "extra_request_body": cli_args.extra_request_body,
        "deterministic_ratio": cli_args.deterministic_ratio,
        "apply_chat_template": False,
        "profile": False,
        "lora_name": None,
        "lora_request_distribution": None,
        "lora_zipf_alpha": 2.0,
        "served_model_name": None,
        "prompt_suffix": "",
        "pd_separated": False,
        "flush_cache": False,
        "warmup_requests": cli_args.warmup_requests,
        "tokenize_prompt": False,
        "host": None,
        "port": None,
        "gsp_num_groups": 64,
        "gsp_prompts_per_group": 16,
        "gsp_system_prompt_len": 2048,
        "gsp_question_len": 128,
        "gsp_output_len": 256,
        "mooncake_slowdown_factor": 1.0,
        "mooncake_num_rounds": 1,
        "mooncake_workload": "conversation",
        "return_routed_experts": False,
        "max_concurrency": None,
        "plot_throughput": False,
        "image_content": "random",
        "image_count": 1,
        "image_format": "png",
        "image_resolution": "1080p",
        "random_image_count": False,
    }


def run_single_profile(
    idx: int,
    base: dict,
    url: str,
    config: ProfileConfig,
    output_dir: Path,
    tokenizer_id: str,
    qps: float,
) -> dict:
    """Run benchmark against one server and collect rollback stats."""
    config_name = config.name
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    output_file = config_dir / "benchmark.jsonl"

    print(f"[{idx+1}] Starting {config_name} on {url} (qps={qps})...", flush=True)

    args = SimpleNamespace(**copy.deepcopy(base))
    args.base_url = url
    args.request_rate = qps
    args.max_concurrency = None
    args.output_file = str(output_file)
    args.output_latencies = str(output_file.with_suffix(".latencies.jsonl"))

    bench_serving.set_global_args(args)
    result = bench_serving.run_benchmark(args)

    # Extract rollback stats from meta_info
    det_num_rollbacks = []
    det_tokens_rolled_back = []
    if "meta_info" in result:
        for meta in result["meta_info"]:
            det_num_rollbacks.append(meta.get("llm42_num_rollbacks", 0))
            det_tokens_rolled_back.append(meta.get("llm42_tokens_rolled_back", 0))

    output_lens = result.get("output_lens", [])
    total_output_tokens = sum(output_lens) if output_lens else 0
    total_tokens_rolled_back = sum(det_tokens_rolled_back)
    total_rollbacks = sum(det_num_rollbacks)

    rollback_stats = {}
    if det_num_rollbacks:
        requests_with_rollbacks = sum(1 for x in det_num_rollbacks if x > 0)
        n = len(det_num_rollbacks)
        rollback_stats = {
            "total_requests": n,
            "total_output_tokens": total_output_tokens,
            "total_rollbacks": total_rollbacks,
            "total_tokens_rolled_back": total_tokens_rolled_back,
            "rollback_pct": (total_tokens_rolled_back / total_output_tokens * 100)
            if total_output_tokens > 0
            else 0.0,
            "avg_rollbacks_per_request": float(np.mean(det_num_rollbacks)),
            "median_rollbacks_per_request": float(np.median(det_num_rollbacks)),
            "max_rollbacks_per_request": int(max(det_num_rollbacks)),
            "p50_rollbacks": float(np.percentile(det_num_rollbacks, 50)),
            "p90_rollbacks": float(np.percentile(det_num_rollbacks, 90)),
            "p99_rollbacks": float(np.percentile(det_num_rollbacks, 99)),
            "requests_with_rollbacks": requests_with_rollbacks,
            "pct_requests_with_rollbacks": requests_with_rollbacks / n * 100,
            "avg_tokens_rolled_back_per_request": float(
                np.mean(det_tokens_rolled_back)
            ),
            "median_tokens_rolled_back_per_request": float(
                np.median(det_tokens_rolled_back)
            ),
            "max_tokens_rolled_back_per_request": int(max(det_tokens_rolled_back)),
            "p50_tokens_rolled_back": float(np.percentile(det_tokens_rolled_back, 50)),
            "p90_tokens_rolled_back": float(np.percentile(det_tokens_rolled_back, 90)),
            "p99_tokens_rolled_back": float(np.percentile(det_tokens_rolled_back, 99)),
        }

    # Latency stats
    ttfts = result.get("ttfts", [])
    latencies = result.get("latencies", [])
    latency_stats = {}
    if latencies:
        latency_stats = {
            "avg_e2e_latency": float(np.mean(latencies)),
            "median_e2e_latency": float(np.median(latencies)),
            "p90_e2e_latency": float(np.percentile(latencies, 90)),
            "p99_e2e_latency": float(np.percentile(latencies, 99)),
        }
    if ttfts:
        latency_stats.update(
            {
                "avg_ttft": float(np.mean(ttfts)),
                "median_ttft": float(np.median(ttfts)),
                "p90_ttft": float(np.percentile(ttfts, 90)),
                "p99_ttft": float(np.percentile(ttfts, 99)),
            }
        )

    profile_result = {
        "config": {
            "window_size": config.window_size,
            "verify_batch_size": config.verify_batch_size,
            "name": config_name,
        },
        "workload": {
            "qps": qps,
            "num_prompts": base["num_prompts"],
            "select_seed": base["select_seed"],
            "order_seed": base["order_seed"],
            "arrival_seed": base["arrival_seed"],
            "deterministic_ratio": base["deterministic_ratio"],
        },
        "rollback_stats": rollback_stats,
        "latency_stats": latency_stats,
        "per_request": {
            "num_rollbacks": det_num_rollbacks,
            "tokens_rolled_back": det_tokens_rolled_back,
        },
    }

    # Save per-config results
    stats_path = config_dir / "rollback_stats.json"
    with open(stats_path, "w") as f:
        json.dump(profile_result, f, indent=2)

    # Save compact client log with throughput stats
    log_path = config_dir / f"log_{config_name}.log"
    log_lines = [
        f"Config: {config_name}",
        "Model: " + str(base.get("model", "unknown")),
        f"QPS: {qps}",
        "Num Prompts: " + str(base["num_prompts"]),
        "Duration (s): {:.2f}".format(result.get("duration", 0)),
        "Completed: " + str(result.get("completed", 0)),
        "Request throughput (req/s): {:.2f}".format(result.get("request_throughput", 0)),
        "Input token throughput (tok/s): {:.2f}".format(result.get("input_throughput", 0)),
        "Output token throughput (tok/s): {:.2f}".format(result.get("output_throughput", 0)),
        "Total token throughput (tok/s): {:.2f}".format(result.get("total_throughput", 0)),
        f"Total output tokens: {total_output_tokens}",
        f"Total rollbacks: {total_rollbacks}",
        f"Total tokens rolled back: {total_tokens_rolled_back}",
        "Rollback pct: {:.4f}%".format(rollback_stats.get("rollback_pct", 0)),
        "Avg E2E latency (s): {:.4f}".format(latency_stats.get("avg_e2e_latency", 0)),
        "P90 E2E latency (s): {:.4f}".format(latency_stats.get("p90_e2e_latency", 0)),
        "P99 E2E latency (s): {:.4f}".format(latency_stats.get("p99_e2e_latency", 0)),
    ]
    with open(log_path, "w") as lf:
        lf.write("\n".join(log_lines) + "\n")


    print(
        f"[{idx+1}] {config_name} done: {total_output_tokens} tokens, "
        f"{total_rollbacks} rollbacks, "
        f"{total_tokens_rolled_back} tokens rolled back "
        f"({rollback_stats.get('rollback_pct', 0):.4f}%)",
        flush=True,
    )
    return profile_result


def main():
    parser = argparse.ArgumentParser(
        description="Profile rollback stats across LLM42 configurations"
    )
    parser.add_argument("--backend", default="sglang")
    parser.add_argument(
        "--base-urls", required=True, help="Comma-separated server URLs"
    )
    parser.add_argument(
        "--profile-configs",
        required=True,
        help="Semicolon-separated profile configs: 'ws=64,bs=8;ws=128,bs=16'",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--sharegpt-output-len", type=int, default=None)
    parser.add_argument("--sharegpt-context-len", type=int, default=None)
    parser.add_argument("--select-seed", type=int, required=True)
    parser.add_argument("--qps", type=float, default=8.0)
    parser.add_argument("--order-seed", type=int, default=132)
    parser.add_argument("--arrival-seed", type=int, default=16)
    parser.add_argument("--deterministic-ratio", type=float, default=1.0)
    parser.add_argument("--extra-request-body", default=None)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs"), help="Run output directory"
    )

    cli_args = parser.parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    base_urls = cli_args.base_urls.split(",")
    configs = [
        ProfileConfig.from_string(c.strip())
        for c in cli_args.profile_configs.split(";")
    ]

    if len(base_urls) < len(configs):
        print(
            f"Error: fewer URLs ({len(base_urls)}) than configs ({len(configs)}). "
            f"Each config needs its own server."
        )
        return 1

    print(f"Profiling {len(configs)} configurations across {len(base_urls)} servers:")
    for config in configs:
        print(f"  {config.name} → window_size={config.window_size}, verify_batch_size={config.verify_batch_size}")
    print()

    base = build_base_args(cli_args)
    tokenizer_id = cli_args.tokenizer or cli_args.model

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=len(configs)) as pool:
        args_list = [
            (
                i,
                base,
                url,
                config,
                cli_args.output_dir,
                tokenizer_id,
                cli_args.qps,
            )
            for i, (url, config) in enumerate(zip(base_urls, configs))
        ]
        results = pool.starmap(run_single_profile, args_list)

    # Print batch summary
    print(f"\n{'='*60}")
    print("Batch Results:")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: (x["config"]["window_size"], x["config"]["verify_batch_size"])):
        cfg = r["config"]
        rs = r["rollback_stats"]
        print(
            f"  {cfg['name']:>12s}: "
            f"{rs.get('total_rollbacks', 0):6d} rollbacks, "
            f"{rs.get('total_tokens_rolled_back', 0):6d} tokens rolled back "
            f"({rs.get('rollback_pct', 0):7.4f}%), "
            f"{rs.get('requests_with_rollbacks', 0):4d}/{rs.get('total_requests', 0)} reqs affected"
        )

    return 0


if __name__ == "__main__":
    exit(main())
