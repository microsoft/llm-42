#!/usr/bin/env python3
"""
Compare outputs across different determinism modes and mismatch percentages.
This benchmark compares:
  - default: Non-deterministic baseline
  - global: Global deterministic mode
  - detinfer_512_0pct: DetInfer with 0% mismatches (no rollback)
  - detinfer_512_5pct: DetInfer with 5% mismatches (forced rollback)
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sglang import bench_serving


def first_mismatch(a: Sequence[int], b: Sequence[int]) -> int:
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
    base_dict = {
        "backend": cli_args.backend,
        "base_url": None,  # Will be set per run
        "dataset_name": cli_args.dataset_name,
        "dataset_path": cli_args.dataset_path,
        "model": cli_args.model,
        "tokenizer": cli_args.tokenizer,
        "num_prompts": cli_args.num_prompts,
        "sharegpt_output_len": cli_args.sharegpt_output_len,
        "sharegpt_context_len": cli_args.sharegpt_context_len,
        "random_input_len": cli_args.random_input_len,
        "random_output_len": cli_args.random_output_len,
        "random_range_ratio": cli_args.random_range_ratio,
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


def run_once(base: dict, base_url: str, qps: float, output_file: Path) -> tuple:
    from types import SimpleNamespace
    import copy
    
    # Make a deep copy to avoid shared state issues
    base_copy = copy.deepcopy(base)
    args = SimpleNamespace(**base_copy)
    args.base_url = base_url
    args.request_rate = qps
    args.max_concurrency = None
    args.output_file = str(output_file)
    args.output_latencies = str(output_file.with_suffix(".latencies.jsonl"))
    bench_serving.set_global_args(args)
    result = bench_serving.run_benchmark(args)
    return args, result


def run_experiment_process(idx: int, base: dict, url: str, qps: float, config_name: str,
                          output_dir: Path, tokenizer_id: str) -> dict:
    """Run a single experiment in a separate process."""
    import copy
    import sys
    
    print(f"[{idx+1}] Starting config={config_name} (QPS={qps}) on {url}...", flush=True)
    output_file = output_dir / f"config_{config_name}.jsonl"
    
    args, result = run_once(
        base=base,
        base_url=url,
        qps=qps,
        output_file=output_file,
    )
    
    print(f"[{idx+1}] config={config_name} benchmark completed, tokenizing outputs...", flush=True)
    
    # Tokenize outputs
    tokenizer = bench_serving.get_tokenizer(tokenizer_id or args.model)
    print(f"[{idx+1}] config={config_name} tokenizer loaded", flush=True)
    tokens = tokenize(result["generated_texts"], tokenizer)
    print(f"[{idx+1}] config={config_name} tokenization done", flush=True)
    
    # Get prompt lengths from benchmark result
    prompt_lens = result.get("input_lens", [None] * len(tokens))
    output_lens = result.get("output_lens", [len(t) for t in tokens])
    
    # Read latency data from .latencies.jsonl file (written by bench_serving)
    # This file has per-request: ttft_ms, tpot_ms, e2e_latency_ms, output_len (all in ms)
    latencies_file = output_file.with_suffix(".latencies.jsonl")
    ttfts_ms = []
    tpots_ms = []
    e2e_latencies_ms = []
    
    if latencies_file.exists():
        with open(latencies_file) as f:
            for line in f:
                row = json.loads(line)
                ttfts_ms.append(row.get("ttft_ms", 0))
                tpots_ms.append(row.get("tpot_ms", 0))
                e2e_latencies_ms.append(row.get("e2e_latency_ms", 0))
        print(f"[{idx+1}] config={config_name} loaded {len(ttfts_ms)} latency records from {latencies_file.name}", flush=True)
    else:
        print(f"[{idx+1}] config={config_name} WARNING: latencies file not found: {latencies_file}", flush=True)
    
    # Extract deterministic rollback stats from meta_info (if available)
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
            "avg_rollbacks_per_request": np.mean(det_num_rollbacks),
            "max_rollbacks_per_request": max(det_num_rollbacks),
            "avg_tokens_rolled_back_per_request": np.mean(det_tokens_rolled_back),
            "max_tokens_rolled_back_per_request": max(det_tokens_rolled_back),
            "requests_with_rollbacks": sum(1 for x in det_num_rollbacks if x > 0),
        }
    
    # Calculate latency statistics (data is already in ms)
    latency_stats = {
        "mean_ttft_ms": float(np.mean(ttfts_ms)) if ttfts_ms else None,
        "p50_ttft_ms": float(np.percentile(ttfts_ms, 50)) if ttfts_ms else None,
        "p90_ttft_ms": float(np.percentile(ttfts_ms, 90)) if ttfts_ms else None,
        "p99_ttft_ms": float(np.percentile(ttfts_ms, 99)) if ttfts_ms else None,
        "mean_tpot_ms": float(np.mean(tpots_ms)) if tpots_ms else None,
        "p50_tpot_ms": float(np.percentile(tpots_ms, 50)) if tpots_ms else None,
        "p90_tpot_ms": float(np.percentile(tpots_ms, 90)) if tpots_ms else None,
        "p99_tpot_ms": float(np.percentile(tpots_ms, 99)) if tpots_ms else None,
        "mean_e2e_ms": float(np.mean(e2e_latencies_ms)) if e2e_latencies_ms else None,
        "p50_e2e_ms": float(np.percentile(e2e_latencies_ms, 50)) if e2e_latencies_ms else None,
        "p90_e2e_ms": float(np.percentile(e2e_latencies_ms, 90)) if e2e_latencies_ms else None,
        "p99_e2e_ms": float(np.percentile(e2e_latencies_ms, 99)) if e2e_latencies_ms else None,
    }
    
    print(f"[{idx+1}] config={config_name} completed: {len(tokens)} responses")
    if latency_stats['mean_ttft_ms'] is not None and latency_stats['mean_tpot_ms'] is not None and latency_stats['mean_e2e_ms'] is not None:
        print(f"[{idx+1}] config={config_name} latency: TTFT={latency_stats['mean_ttft_ms']:.1f}ms, "
              f"TPOT={latency_stats['mean_tpot_ms']:.2f}ms, E2E={latency_stats['mean_e2e_ms']:.1f}ms")
    if rollback_stats:
        print(f"[{idx+1}] config={config_name} rollback stats: {rollback_stats['total_rollbacks']} rollbacks, "
              f"{rollback_stats['total_tokens_rolled_back']} tokens rolled back")
    
    # Return serializable data (not the full result object)
    return {
        "idx": idx,
        "config_name": config_name,
        "qps": qps,
        "tokens": tokens,
        "texts": result["generated_texts"],
        "prompt_lens": prompt_lens,
        "output_lens": output_lens,
        "ttfts_ms": ttfts_ms,
        "tpots_ms": tpots_ms,
        "e2e_latencies_ms": e2e_latencies_ms,
        "output_file": str(output_file),
        "det_num_rollbacks": det_num_rollbacks,
        "det_tokens_rolled_back": det_tokens_rolled_back,
        "rollback_stats": rollback_stats,
        "latency_stats": latency_stats,
    }


def tokenize(texts: List[str], tokenizer) -> List[List[int]]:
    return [tokenizer.encode(t, add_special_tokens=False) for t in texts]


def write_jsonl(path: Path, rows: List[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def plot_mismatch_heatmap(mismatch_matrix: np.ndarray, config_names: List[str], output_path: Path):
    """Plot heatmap showing fraction of mismatches between each pair of configs."""
    plt.figure(figsize=(10, 8))
    im = plt.imshow(mismatch_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='Mismatch Fraction')
    
    # Shorten config names for display
    short_names = [name.replace("sglang_", "").replace("detinfer_", "det_") for name in config_names]
    plt.xticks(range(len(config_names)), short_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(config_names)), short_names, fontsize=10)
    
    # Add text annotations
    for i in range(len(config_names)):
        for j in range(len(config_names)):
            text = plt.text(j, i, f'{mismatch_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.title('Pairwise Output Mismatch Fractions')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare outputs across different determinism modes and mismatch percentages")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--base-urls", required=True, help="Comma-separated list of server URLs")
    parser.add_argument("--config-names", required=True, help="Comma-separated list of config names")
    parser.add_argument("--qps", type=float, default=12, help="QPS to use for all servers (default: 12)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-name", default="random", help="Dataset type: sharegpt, random")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--num-prompts", type=int, default=4096)
    parser.add_argument("--sharegpt-output-len", type=int, default=None)
    parser.add_argument("--sharegpt-context-len", type=int, default=None)
    parser.add_argument("--random-input-len", type=int, default=512, help="Input token length for random dataset")
    parser.add_argument("--random-output-len", type=int, default=1024, help="Output token length for random dataset")
    parser.add_argument("--random-range-ratio", type=float, default=0.0, help="Ratio to vary random lengths (0.0 = fixed)")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--deterministic-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("verification_percentage"))
    parser.add_argument("--extra-request-body", default=None)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS tokens and continue generation to max length")

    cli_args = parser.parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse URLs and config names
    base_urls = cli_args.base_urls.split(',')
    config_names = cli_args.config_names.split(',')
    qps = cli_args.qps
    
    if len(base_urls) != len(config_names):
        print(f"Error: Number of URLs ({len(base_urls)}) must match number of config names ({len(config_names)})")
        return 1
    
    print(f"Running {len(base_urls)} experiments at QPS={qps}:")
    for url, config_name in zip(base_urls, config_names):
        print(f"  config={config_name} on {url}")
    print()

    base = build_base_args(cli_args)
    
    # Run all experiments in parallel using multiprocessing (separate processes)
    import multiprocessing as mp
    
    # CRITICAL: Use 'spawn' instead of 'fork' (Linux default) to avoid deadlocks
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print("Launching all experiments in parallel processes (spawn mode)...")
    print()
    
    # Prepare arguments for each process
    tokenizer_id = cli_args.tokenizer or cli_args.model
    
    # Use multiprocessing Pool to run experiments in parallel
    with mp.Pool(processes=len(base_urls)) as pool:
        args_list = [
            (i, base, url, qps, config_name, cli_args.output_dir, tokenizer_id)
            for i, (url, config_name) in enumerate(zip(base_urls, config_names))
        ]
        
        # Launch all experiments
        process_results = pool.starmap(run_experiment_process, args_list)
    
    print()
    print("All experiments completed!")
    print()
    
    # Sort results by index to maintain order
    process_results.sort(key=lambda x: x["idx"])
    
    # Collect per-config stats for summary
    config_stats = []
    for r in process_results:
        stats = {
            "config_name": r["config_name"],
            "num_requests": len(r["tokens"]),
            **r.get("latency_stats", {}),
            **r.get("rollback_stats", {}),
        }
        config_stats.append(stats)
    
    # Save per-config latency data for plotting
    latency_data = {
        "config_names": config_names,
        "ttfts_ms": {r["config_name"]: r["ttfts_ms"] for r in process_results},
        "tpots_ms": {r["config_name"]: r["tpots_ms"] for r in process_results},
        "e2e_latencies_ms": {r["config_name"]: r["e2e_latencies_ms"] for r in process_results},
    }
    
    latency_file = cli_args.output_dir / "latency_data.json"
    with latency_file.open("w") as f:
        json.dump(latency_data, f, indent=2)
    
    # Extract tokenized outputs in correct order
    tokenized_outputs = [
        (r["config_name"], r["tokens"], r["texts"], r["prompt_lens"], 
         r.get("det_num_rollbacks", []), r.get("det_tokens_rolled_back", []))
        for r in process_results
    ]
    
    # Compare all pairs
    num_runs = len(config_names)
    mismatch_matrix = np.zeros((num_runs, num_runs))
    
    pairwise_details = []
    
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            config_i, tokens_i, texts_i, prompt_lens_i, det_rollbacks_i, det_tokens_rb_i = tokenized_outputs[i]
            config_j, tokens_j, texts_j, prompt_lens_j, det_rollbacks_j, det_tokens_rb_j = tokenized_outputs[j]
            
            mismatches = []
            for req_id, (tok_i, tok_j, text_i, text_j, plen_i, plen_j) in enumerate(zip(tokens_i, tokens_j, texts_i, texts_j, prompt_lens_i, prompt_lens_j)):
                first_mm = first_mismatch(tok_i, tok_j)
                delta = len(tok_i) - first_mm
                output_len_i = len(tok_i)
                output_len_j = len(tok_j)
                
                mismatches.append({
                    "request_id": req_id,
                    "prompt_len": plen_i,
                    f"output_len_{config_i}": output_len_i,
                    f"output_len_{config_j}": output_len_j,
                    f"{config_i}_tokens": tok_i,
                    f"{config_j}_tokens": tok_j,
                    f"{config_i}_text": text_i,
                    f"{config_j}_text": text_j,
                    "first_mismatch_index": first_mm,
                    "output_length": output_len_i,
                    "delta": delta,
                })
            
            # Calculate mismatch fraction and delta statistics
            deltas = [m["delta"] for m in mismatches]
            delta_array = np.array(deltas)
            mismatch_frac = 1.0 - np.mean(delta_array == 0)
            num_delta_gt_64 = np.sum(delta_array > 64)
            num_delta_gt_128 = np.sum(delta_array > 128)
            mismatch_matrix[i, j] = mismatch_frac
            mismatch_matrix[j, i] = mismatch_frac
            
            # Save pairwise comparison (full details)
            pair_file = cli_args.output_dir / f"compare_{config_i}_vs_{config_j}.jsonl"
            write_jsonl(pair_file, mismatches)
            
            pairwise_details.append({
                "config_1": config_i,
                "config_2": config_j,
                "mismatch_fraction": float(mismatch_frac),
                "zero_mismatch_fraction": float(np.mean(delta_array == 0)),
                "num_delta_gt_64": int(num_delta_gt_64),
                "num_delta_gt_128": int(num_delta_gt_128),
                "comparison_file": str(pair_file),
            })
            
            print(f"{config_i} vs {config_j}: {mismatch_frac:.2%} mismatch rate, "
                  f"{num_delta_gt_64} deltas > 64, {num_delta_gt_128} deltas > 128")
    
    # Plot heatmap
    heatmap_path = cli_args.output_dir / "mismatch_heatmap.pdf"
    plot_mismatch_heatmap(mismatch_matrix, config_names, heatmap_path)
    
    # Save summary
    summary = {
        "config_names": config_names,
        "qps": qps,
        "base_urls": base_urls,
        "num_prompts": cli_args.num_prompts,
        "random_input_len": cli_args.random_input_len,
        "random_output_len": cli_args.random_output_len,
        "seed": cli_args.seed,
        "config_stats": config_stats,
        "pairwise_comparisons": pairwise_details,
        "heatmap_plot": str(heatmap_path),
        "latency_data_file": str(latency_file),
    }
    
    summary_file = cli_args.output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"{'='*50}")
    print(json.dumps(summary, indent=2))
    print(f"\nAll results saved to: {cli_args.output_dir}")
    print(f"\nTo plot latency comparison:")
    print(f"  python plot_latency.py --results-dir {cli_args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
