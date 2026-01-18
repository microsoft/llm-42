#!/usr/bin/env python3
"""
Compare outputs across multiple (QPS, order_seed, arrival_seed) configurations.
Uses prompt hash to match outputs across configs with different arrival orders.

Example usage:
    python compare_multi_config_outputs.py \
        --backend sglang \
        --base-urls "http://127.0.0.1:30000,http://127.0.0.1:30001,http://127.0.0.1:30002,http://127.0.0.1:30003" \
        --configs "qps=6,order=40;qps=6,order=242;qps=12,order=34;qps=12,order=123" \
        --select-seed 42 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts 100 \
        --output-dir multi_config_comparison
"""

import argparse
import copy
import json
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from types import SimpleNamespace

from sglang import bench_serving


@dataclass
class ConfigSpec:
    """Specification for a single benchmark configuration."""
    qps: float
    order_seed: int
    arrival_seed: int
    
    @property
    def name(self) -> str:
        return f"qps{self.qps}_o{self.order_seed}_a{self.arrival_seed}"
    
    @classmethod
    def from_string(cls, config_str: str) -> "ConfigSpec":
        """Parse config string like 'qps=6,order=40,arrival=100' or 'qps=6,order=40'"""
        parts = {}
        for part in config_str.split(","):
            key, value = part.strip().split("=")
            parts[key.strip()] = value.strip()
        
        qps = float(parts["qps"])
        order_seed = int(parts["order"])
        # Default arrival_seed to order_seed if not specified
        arrival_seed = int(parts.get("arrival", order_seed))
        
        return cls(qps=qps, order_seed=order_seed, arrival_seed=arrival_seed)


def first_mismatch(a: List[int], b: List[int]) -> int:
    """Return first mismatch index; treat length difference as a mismatch.
    
    If sequences are identical and equal length, return the length.
    """
    len_a, len_b = len(a), len(b)
    limit = min(len_a, len_b)
    
    for i in range(limit):
        if a[i] != b[i]:
            return i
    
    # If we got here, common prefix matches
    if len_a != len_b:
        return limit
    
    return len_a


def build_base_args(cli_args: argparse.Namespace) -> dict:
    """Build base arguments dictionary for bench_serving."""
    base_dict = {
        "backend": cli_args.backend,
        "base_url": None,  # Will be set per run
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
        "deterministic_seed": 42,
        "seed": cli_args.select_seed,  # Base seed (for backward compat)
        "select_seed": cli_args.select_seed,  # Same prompts across all configs
        "order_seed": None,  # Will be set per config
        "arrival_seed": None,  # Will be set per config
        "disable_ignore_eos": not cli_args.ignore_eos,
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
        "host": None,
        "port": None,
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
    return base_dict


def run_once(base: dict, base_url: str, config: ConfigSpec, output_file: Path) -> tuple:
    """Run a single benchmark with the given configuration."""
    args = SimpleNamespace(**copy.deepcopy(base))
    args.base_url = base_url
    args.request_rate = config.qps
    args.order_seed = config.order_seed
    args.arrival_seed = config.arrival_seed
    args.max_concurrency = None
    args.output_file = str(output_file)
    args.output_latencies = str(output_file.with_suffix(".latencies.jsonl"))
    bench_serving.set_global_args(args)
    result = bench_serving.run_benchmark(args)
    return args, result


def run_experiment_process(idx: int, base: dict, url: str, config: ConfigSpec, 
                          output_dir: Path, tokenizer_id: str) -> dict:
    """Run a single experiment in a separate process."""
    config_name = config.name
    print(f"[{idx+1}] Starting {config_name} on {url}...", flush=True)
    output_file = output_dir / f"config_{config_name}.jsonl"
    
    args, result = run_once(
        base=base,
        base_url=url,
        config=config,
        output_file=output_file,
    )
    
    # Use output_ids directly from server
    tokens = result.get("output_ids", [])
    if not tokens or not any(tokens):
        print(f"[{idx+1}] {config_name} output_ids not available, falling back to re-tokenization...", flush=True)
        tokenizer = bench_serving.get_tokenizer(tokenizer_id or args.model)
        tokens = [tokenizer.encode(t, add_special_tokens=False) for t in result["generated_texts"]]
    
    # Get prompt hashes and lengths from benchmark result
    prompt_hashes = result.get("prompt_hashes", [])
    prompt_lens = result.get("input_lens", [None] * len(tokens))
    
    # Extract deterministic rollback stats from meta_info
    det_num_rollbacks = []
    det_tokens_rolled_back = []
    if "meta_info" in result:
        for meta in result["meta_info"]:
            det_num_rollbacks.append(meta.get("det_infer_num_rollbacks", 0))
            det_tokens_rolled_back.append(meta.get("det_infer_tokens_rolled_back", 0))
    
    # Calculate rollback statistics
    rollback_stats = {}
    if det_num_rollbacks:
        rollback_stats = {
            "total_rollbacks": sum(det_num_rollbacks),
            "total_tokens_rolled_back": sum(det_tokens_rolled_back),
            "avg_rollbacks_per_request": float(np.mean(det_num_rollbacks)),
            "max_rollbacks_per_request": max(det_num_rollbacks),
            "requests_with_rollbacks": sum(1 for x in det_num_rollbacks if x > 0),
        }
    
    print(f"[{idx+1}] {config_name} completed: {len(tokens)} responses")
    if rollback_stats:
        print(f"[{idx+1}] {config_name} rollback stats: {rollback_stats['total_rollbacks']} rollbacks, "
              f"{rollback_stats['total_tokens_rolled_back']} tokens rolled back")
    
    return {
        "idx": idx,
        "config": {
            "qps": config.qps,
            "order_seed": config.order_seed,
            "arrival_seed": config.arrival_seed,
            "name": config_name,
        },
        "tokens": tokens,
        "texts": result["generated_texts"],
        "prompt_hashes": prompt_hashes,
        "prompt_lens": prompt_lens,
        "output_lens": result.get("output_lens", [len(t) for t in tokens]),
        "ttfts": result.get("ttfts", []),
        "latencies": result.get("latencies", []),  # e2e latency per request
        "itls": result.get("itls", []),  # inter-token latencies
        "output_file": str(output_file),
        "det_num_rollbacks": det_num_rollbacks,
        "det_tokens_rolled_back": det_tokens_rolled_back,
        "rollback_stats": rollback_stats,
        "is_deterministic": result.get("is_deterministic", []),
    }


def write_per_config_logs(result: dict, output_dir: Path):
    """Write summary and detailed logs for a single config."""
    config_name = result["config"]["name"]
    
    # Get timing data
    ttfts = result.get("ttfts", [])
    latencies = result.get("latencies", [])  # e2e latency
    output_lens = result.get("output_lens", [len(t) for t in result["tokens"]])
    
    # Summary log
    summary_path = output_dir / f"config_{config_name}_summary.log"
    with open(summary_path, "w") as f:
        f.write(f"Config: {config_name}\n")
        f.write(f"QPS: {result['config']['qps']}, Order Seed: {result['config']['order_seed']}, "
                f"Arrival Seed: {result['config']['arrival_seed']}\n")
        f.write("=" * 80 + "\n\n")
        
        num_requests = len(result["tokens"])
        for i in range(num_requests):
            prompt_hash = result["prompt_hashes"][i] if i < len(result["prompt_hashes"]) else "N/A"
            prompt_len = result["prompt_lens"][i] if i < len(result["prompt_lens"]) else 0
            output_len = output_lens[i] if i < len(output_lens) else len(result["tokens"][i])
            ttft = ttfts[i] if i < len(ttfts) else 0
            e2e = latencies[i] if i < len(latencies) else 0
            
            # Calculate tpot (time per output token) = (e2e - ttft) / (output_len - 1)
            if output_len > 1 and e2e > 0:
                tpot = (e2e - ttft) / (output_len - 1)
            else:
                tpot = 0
            
            f.write(f"[REQ {i+1:03d}] hash={prompt_hash} | prompt_len={prompt_len} | output_len={output_len} | "
                   f"ttft={ttft:.3f}s | tpot={tpot:.4f}s | e2e={e2e:.3f}s\n")
    
    # Detailed JSON log
    detailed_path = output_dir / f"config_{config_name}_detailed.json"
    with open(detailed_path, "w") as f:
        requests_data = []
        num_requests = len(result["tokens"])
        for i in range(num_requests):
            output_len = output_lens[i] if i < len(output_lens) else len(result["tokens"][i])
            ttft = ttfts[i] if i < len(ttfts) else 0
            e2e = latencies[i] if i < len(latencies) else 0
            tpot = (e2e - ttft) / (output_len - 1) if output_len > 1 and e2e > 0 else 0
            
            requests_data.append({
                "req_id": i + 1,
                "hash": result["prompt_hashes"][i] if i < len(result["prompt_hashes"]) else None,
                "prompt_len": result["prompt_lens"][i] if i < len(result["prompt_lens"]) else None,
                "output_len": output_len,
                "output_text": result["texts"][i],
                "output_tokens": json.dumps(result["tokens"][i]),  # Compact array as string
                "timing": {
                    "ttft": ttft,
                    "tpot": tpot,
                    "e2e": e2e,
                },
                "rollback": {
                    "num_rollbacks": result["det_num_rollbacks"][i] if i < len(result["det_num_rollbacks"]) else 0,
                    "tokens_rolled_back": result["det_tokens_rolled_back"][i] if i < len(result["det_tokens_rolled_back"]) else 0,
                },
            })
        
        detailed = {
            "config": result["config"],
            "requests": requests_data,
        }
        json.dump(detailed, f, indent=2)


def compare_all_configs(results: List[dict], output_dir: Path, cli_args: argparse.Namespace):
    """Compare outputs across all configs and write unified comparison report."""
    
    compare_det_only = getattr(cli_args, 'compare_deterministic_only', False)
    
    # Build hash -> {config_name: (tokens, text, output_len, prompt_len)} mapping
    hash_to_outputs: Dict[str, Dict[str, Tuple]] = defaultdict(dict)
    # Track which hashes are deterministic (from first config that has it)
    hash_is_deterministic: Dict[str, bool] = {}
    
    for result in results:
        config_name = result["config"]["name"]
        is_det_flags = result.get("is_deterministic", [False] * len(result["prompt_hashes"]))
        for i, (prompt_hash, tokens, text, prompt_len) in enumerate(zip(
            result["prompt_hashes"], result["tokens"], result["texts"], result["prompt_lens"]
        )):
            if prompt_hash:
                is_det = is_det_flags[i] if i < len(is_det_flags) else False
                # Record deterministic status (first occurrence wins)
                if prompt_hash not in hash_is_deterministic:
                    hash_is_deterministic[prompt_hash] = is_det
                hash_to_outputs[prompt_hash][config_name] = (tokens, text, len(tokens), prompt_len)
    
    # Filter to only deterministic prompts if requested
    if compare_det_only:
        original_count = len(hash_to_outputs)
        hash_to_outputs = {h: v for h, v in hash_to_outputs.items() if hash_is_deterministic.get(h, False)}
        filtered_count = len(hash_to_outputs)
        print(f"\nFiltering to deterministic prompts only: {filtered_count}/{original_count} prompts")
    
    # Get all config names
    config_names = [r["config"]["name"] for r in results]
    num_configs = len(config_names)
    
    # Per-prompt analysis - compare ALL configs together (not pairwise)
    prompt_results = []
    for prompt_hash, outputs in hash_to_outputs.items():
        if len(outputs) < 2:
            continue
        
        # Get prompt_len and output_len (should be same across configs if matching)
        first_output = list(outputs.values())[0]
        prompt_len = first_output[3]
        output_len = first_output[2]
        
        # Check if all texts match
        all_texts = [text for (tokens, text, output_len, _) in outputs.values()]
        text_match = len(set(all_texts)) == 1
        
        # Check if all tokens match
        all_tokens = [tuple(tokens) for (tokens, text, output_len, _) in outputs.values()]
        tokens_match = len(set(all_tokens)) == 1
        
        # Overall match if either text OR tokens match
        overall_match = text_match or tokens_match
        
        # Determine status
        if tokens_match:
            status = "match"
        else:
            status = "mismatch"
        
        # Group configs by identical token output (for mismatch details)
        token_groups: Dict[tuple, List[str]] = defaultdict(list)
        text_groups: Dict[str, List[str]] = defaultdict(list)
        for config_name, (tokens, text, ol, _) in outputs.items():
            token_groups[tuple(tokens)].append(config_name)
            text_groups[text].append(config_name)
        
        prompt_results.append({
            "hash": prompt_hash,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "text_match": text_match,
            "tokens_match": tokens_match,
            "overall_match": overall_match,
            "status": status,
            "num_configs": len(outputs),
            "num_token_groups": len(token_groups),
            "num_text_groups": len(text_groups),
            "token_groups": [{"configs": configs, "output_len": len(tokens)} 
                           for tokens, configs in token_groups.items()],
            "text_groups": [{"configs": configs} for text, configs in text_groups.items()],
        })
    
    # Calculate per-config rollback stats
    config_stats = []
    for result in results:
        total_output_tokens = sum(len(t) for t in result["tokens"])
        total_tokens_rolled_back = sum(result.get("det_tokens_rolled_back", []))
        rollback_pct = (total_tokens_rolled_back / total_output_tokens * 100) if total_output_tokens > 0 else 0.0
        
        config_stats.append({
            "config": result["config"],
            "total_output_tokens": total_output_tokens,
            "total_tokens_rolled_back": total_tokens_rolled_back,
            "rollback_pct": rollback_pct,
        })
    
    # Count matches vs mismatches (for tokens, text, and overall)
    num_tokens_match = sum(1 for p in prompt_results if p["tokens_match"])
    num_text_match = sum(1 for p in prompt_results if p["text_match"])
    num_overall_match = sum(1 for p in prompt_results if p["overall_match"])
    num_mismatch = sum(1 for p in prompt_results if not p["overall_match"])
    total = len(prompt_results)
    
    # Write summary log
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Multi-Config Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Num Prompts: {cli_args.num_prompts}\n")
        f.write(f"Select Seed: {cli_args.select_seed}\n")
        f.write(f"Configs: {config_names}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("Per-Config Output & Rollback Stats:\n")
        f.write("-" * 60 + "\n")
        for stat in config_stats:
            cfg = stat["config"]
            f.write(f"QPS {cfg['qps']}, Order Seed = {cfg['order_seed']}, Arrival Seed = {cfg['arrival_seed']}:\n")
            f.write(f"  Total Output Tokens: {stat['total_output_tokens']}\n")
            f.write(f"  Total Tokens Rolled Back: {stat['total_tokens_rolled_back']}\n")
            f.write(f"  Rollback %: {stat['rollback_pct']:.4f}%\n")
        f.write("\n")
        
        # Per-prompt details with text and token comparison
        f.write("-" * 60 + "\n")
        f.write("Per-Prompt Comparison:\n")
        f.write("-" * 60 + "\n")
        for pr in prompt_results:
            n_cfgs = pr["num_configs"]
            text_status = f"✓ ALL MATCH ({n_cfgs}/{n_cfgs} configs)" if pr["text_match"] else f"✗ MISMATCH ({pr['num_text_groups']} groups)"
            tokens_status = f"✓ ALL MATCH ({n_cfgs}/{n_cfgs} configs)" if pr["tokens_match"] else f"✗ MISMATCH ({pr['num_token_groups']} groups)"
            overall_status = "✓ ALL MATCH" if pr["overall_match"] else "✗ MISMATCH"
            
            f.write(f"[hash={pr['hash']}] prompt_len={pr['prompt_len']} | output_len={pr['output_len']} | "
                   f"Text: {text_status} | Tokens: {tokens_status} | {overall_status}\n")
            
            # Show group details for mismatches
            if not pr["tokens_match"]:
                for gi, group in enumerate(pr["token_groups"]):
                    f.write(f"  token_group{gi+1}: {group['configs']} | output_len={group['output_len']}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("TOTAL SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total prompts compared: {total}\n")
        f.write(f"Text Matched: {num_text_match} ({100*num_text_match/total:.2f}%)\n")
        f.write(f"Tokens Matched: {num_tokens_match} ({100*num_tokens_match/total:.2f}%)\n")
        f.write(f"Overall Matched (text OR tokens): {num_overall_match} ({100*num_overall_match/total:.2f}%)\n")
        f.write(f"Mismatched: {num_mismatch} ({100*num_mismatch/total:.2f}%)\n")
    
    # Write detailed JSON
    detailed_path = output_dir / "comparison_detailed.json"
    with open(detailed_path, "w") as f:
        detailed = {
            "select_seed": cli_args.select_seed,
            "configs": [r["config"] for r in results],
            "config_stats": config_stats,
            "prompt_results": prompt_results,
            "summary": {
                "total": total,
                "text_matched": num_text_match,
                "tokens_matched": num_tokens_match,
                "overall_matched": num_overall_match,
                "mismatched": num_mismatch,
                "text_match_rate": num_text_match / total if total > 0 else 0,
                "tokens_match_rate": num_tokens_match / total if total > 0 else 0,
                "overall_match_rate": num_overall_match / total if total > 0 else 0,
            },
        }
        json.dump(detailed, f, indent=2)
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    
    print("\nPer-Config Output & Rollback Stats:")
    for stat in config_stats:
        cfg = stat["config"]
        print(f"  {cfg['name']}: {stat['total_output_tokens']} output tokens, "
              f"{stat['total_tokens_rolled_back']} rolled back ({stat['rollback_pct']:.4f}%)")
    
    print(f"\nTotal: {total} prompts")
    print(f"  Text Matched: {num_text_match} ({100*num_text_match/total:.2f}%)")
    print(f"  Tokens Matched: {num_tokens_match} ({100*num_tokens_match/total:.2f}%)")
    print(f"  Overall Matched (text OR tokens): {num_overall_match} ({100*num_overall_match/total:.2f}%)")
    print(f"  Mismatched: {num_mismatch} ({100*num_mismatch/total:.2f}%)")
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare outputs across multiple (QPS, order_seed, arrival_seed) configurations")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--base-urls", required=True, help="Comma-separated list of server URLs")
    parser.add_argument("--configs", required=True, 
                       help="Semicolon-separated config specs: 'qps=6,order=40;qps=6,order=242;...' "
                            "(arrival defaults to order if not specified)")
    parser.add_argument("--select-seed", type=int, required=True,
                       help="Seed for prompt selection (use same value for all configs to compare same prompts)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--sharegpt-output-len", type=int, default=None)
    parser.add_argument("--sharegpt-context-len", type=int, default=None)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--deterministic-ratio", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("multi_config_comparison"))
    parser.add_argument("--extra-request-body", default=None)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--compare-deterministic-only", action="store_true",
                       help="Only compare prompts that were marked as deterministic")

    cli_args = parser.parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse URLs and configs
    base_urls = cli_args.base_urls.split(',')
    configs = [ConfigSpec.from_string(c.strip()) for c in cli_args.configs.split(';')]
    
    if len(base_urls) < len(configs):
        print(f"Warning: More configs ({len(configs)}) than URLs ({len(base_urls)}). "
              f"Will run in batches.")
    
    print(f"Running {len(configs)} configurations across {len(base_urls)} servers:")
    for config in configs:
        print(f"  {config.name}")
    print(f"Select seed (same prompts): {cli_args.select_seed}")
    print()

    base = build_base_args(cli_args)
    
    # Use 'spawn' to avoid deadlocks with asyncio
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    tokenizer_id = cli_args.tokenizer or cli_args.model
    all_results = []
    
    # Run in batches if more configs than servers
    batch_size = len(base_urls)
    num_batches = (len(configs) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(configs))
        batch_configs = configs[start_idx:end_idx]
        batch_urls = base_urls[:len(batch_configs)]
        
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
        print(f"Running configs: {[c.name for c in batch_configs]}")
        
        with mp.Pool(processes=len(batch_configs)) as pool:
            args_list = [
                (i + start_idx, base, url, config, cli_args.output_dir, tokenizer_id)
                for i, (url, config) in enumerate(zip(batch_urls, batch_configs))
            ]
            batch_results = pool.starmap(run_experiment_process, args_list)
        
        all_results.extend(batch_results)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
    
    # Sort results by original index
    all_results.sort(key=lambda x: x["idx"])
    
    # Write per-config logs
    for result in all_results:
        write_per_config_logs(result, cli_args.output_dir)
    
    # Compare all configs
    compare_all_configs(all_results, cli_args.output_dir, cli_args)
    
    return 0


if __name__ == "__main__":
    exit(main())
