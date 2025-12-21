#!/usr/bin/env python3
"""
Compare outputs across multiple QPS runs (no sequential baseline).
Each server runs at a different QPS, and we compare their outputs token-by-token.
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


def run_experiment_process(idx: int, base: dict, url: str, qps: float, output_dir: Path, 
                          tokenizer_id: str) -> dict:
    """Run a single experiment in a separate process."""
    import copy
    import sys
    
    print(f"[{idx+1}] Starting QPS={qps} on {url}...", flush=True)
    output_file = output_dir / f"qps_{qps}.jsonl"
    
    args, result = run_once(
        base=base,
        base_url=url,
        qps=qps,
        output_file=output_file,
    )
    
    print(f"[{idx+1}] QPS={qps} benchmark completed, tokenizing outputs...", flush=True)
    
    # Tokenize outputs
    tokenizer = bench_serving.get_tokenizer(tokenizer_id or args.model)
    print(f"[{idx+1}] QPS={qps} tokenizer loaded", flush=True)
    tokens = tokenize(result["generated_texts"], tokenizer)
    print(f"[{idx+1}] QPS={qps} tokenization done", flush=True)
    
    # Get prompt lengths from benchmark result
    prompt_lens = result.get("input_lens", [None] * len(tokens))
    
    # Extract deterministic rollback stats from meta_info (if available)
    det_num_rollbacks = []
    det_tokens_rolled_back = []
    if "meta_info" in result:
        for meta in result["meta_info"]:
            det_num_rollbacks.append(meta.get("det_num_rollbacks", 0))
            det_tokens_rolled_back.append(meta.get("det_tokens_rolled_back", 0))
    
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
    
    print(f"[{idx+1}] QPS={qps} completed: {len(tokens)} responses")
    if rollback_stats:
        print(f"[{idx+1}] QPS={qps} rollback stats: {rollback_stats['total_rollbacks']} rollbacks, "
              f"{rollback_stats['total_tokens_rolled_back']} tokens rolled back")
    
    # Return serializable data (not the full result object)
    return {
        "idx": idx,
        "qps": qps,
        "tokens": tokens,
        "texts": result["generated_texts"],
        "prompt_lens": prompt_lens,
        "output_file": str(output_file),
        "det_num_rollbacks": det_num_rollbacks,
        "det_tokens_rolled_back": det_tokens_rolled_back,
        "rollback_stats": rollback_stats,
    }


def tokenize(texts: List[str], tokenizer) -> List[List[int]]:
    return [tokenizer.encode(t, add_special_tokens=False) for t in texts]


def write_jsonl(path: Path, rows: List[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def plot_mismatch_heatmap(mismatch_matrix: np.ndarray, qps_values: List[float], output_path: Path):
    """Plot heatmap showing fraction of mismatches between each pair of QPS values."""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mismatch_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='Mismatch Fraction')
    
    labels = [f"QPS {qps}" for qps in qps_values]
    plt.xticks(range(len(qps_values)), labels, rotation=45, ha='right')
    plt.yticks(range(len(qps_values)), labels)
    
    # Add text annotations
    for i in range(len(qps_values)):
        for j in range(len(qps_values)):
            text = plt.text(j, i, f'{mismatch_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.title('Pairwise Output Mismatch Fractions')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare outputs across multiple QPS runs")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--base-urls", required=True, help="Comma-separated list of server URLs")
    parser.add_argument("--qps-values", required=True, help="Comma-separated list of QPS values")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--sharegpt-output-len", type=int, default=None)
    parser.add_argument("--sharegpt-context-len", type=int, default=None)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--deterministic-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("multi_qps_comparison"))
    parser.add_argument("--extra-request-body", default=None)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS tokens and continue generation to max length")

    cli_args = parser.parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse URLs and QPS values
    base_urls = cli_args.base_urls.split(',')
    qps_values = [float(x) for x in cli_args.qps_values.split(',')]
    
    if len(base_urls) != len(qps_values):
        print(f"Error: Number of URLs ({len(base_urls)}) must match number of QPS values ({len(qps_values)})")
        return 1
    
    print(f"Running {len(base_urls)} experiments:")
    for url, qps in zip(base_urls, qps_values):
        print(f"  QPS {qps} on {url}")
    print()

    base = build_base_args(cli_args)
    
    # Run all experiments in parallel using multiprocessing (separate processes)
    # Each process has its own memory space, avoiding global variable conflicts
    # NOTE: All experiments use the same seed, so they will:
    #   1. Sample the same prompts from the dataset in the same order
    #   2. Generate requests deterministically
    #   3. This ensures fair comparison across different QPS values
    
    import multiprocessing as mp
    
    # CRITICAL: Use 'spawn' instead of 'fork' (Linux default) to avoid deadlocks
    # fork + asyncio.run() inside bench_serving causes the child processes to hang
    # because the forked process inherits a broken event loop state
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
        # Prepare arguments: (idx, base, url, qps, output_dir, tokenizer_id)
        args_list = [
            (i, base, url, qps, cli_args.output_dir, tokenizer_id)
            for i, (url, qps) in enumerate(zip(base_urls, qps_values))
        ]
        
        # Launch all experiments
        process_results = pool.starmap(run_experiment_process, args_list)
    
    print()
    print("All experiments completed!")
    print()
    
    # Sort results by index to maintain order
    process_results.sort(key=lambda x: x["idx"])
    
    # Collect per-QPS rollback stats for summary
    qps_rollback_stats = []
    for r in process_results:
        if r.get("rollback_stats"):
            qps_rollback_stats.append({
                "qps": r["qps"],
                **r["rollback_stats"]
            })
    
    # Extract tokenized outputs in correct order
    tokenized_outputs = [
        (r["qps"], r["tokens"], r["texts"], r["prompt_lens"], 
         r.get("det_num_rollbacks", []), r.get("det_tokens_rolled_back", []))
        for r in process_results
    ]
    
    # Compare all pairs
    num_runs = len(qps_values)
    mismatch_matrix = np.zeros((num_runs, num_runs))
    
    pairwise_details = []
    
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            qps_i, tokens_i, texts_i, prompt_lens_i, det_rollbacks_i, det_tokens_rb_i = tokenized_outputs[i]
            qps_j, tokens_j, texts_j, prompt_lens_j, det_rollbacks_j, det_tokens_rb_j = tokenized_outputs[j]
            
            mismatches = []
            for req_id, (tok_i, tok_j, text_i, text_j, plen_i, plen_j) in enumerate(zip(tokens_i, tokens_j, texts_i, texts_j, prompt_lens_i, prompt_lens_j)):
                first_mm = first_mismatch(tok_i, tok_j)
                delta = len(tok_i) - first_mm
                output_len_i = len(tok_i)
                output_len_j = len(tok_j)
                
                mismatches.append({
                    "request_id": req_id,
                    "prompt_len": plen_i,  # Should be same for both QPS with same seed
                    f"output_len_qps_{qps_i}": output_len_i,
                    f"output_len_qps_{qps_j}": output_len_j,
                    f"qps_{qps_i}_tokens": tok_i,
                    f"qps_{qps_j}_tokens": tok_j,
                    f"qps_{qps_i}_text": text_i,
                    f"qps_{qps_j}_text": text_j,
                    "first_mismatch_index": first_mm,
                    "output_length": output_len_i,  # Kept for backward compatibility
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
            pair_file = cli_args.output_dir / f"compare_qps_{qps_i}_vs_{qps_j}.jsonl"
            write_jsonl(pair_file, mismatches)
            
            # Save summary file with just key metrics
            summary_mismatches = [
                {
                    "request_id": m["request_id"],
                    "prompt_len": m["prompt_len"],
                    f"output_len_qps_{qps_i}": m[f"output_len_qps_{qps_i}"],
                    f"output_len_qps_{qps_j}": m[f"output_len_qps_{qps_j}"],
                    f"det_num_rollbacks_qps_{qps_i}": det_rollbacks_i[m["request_id"]] if det_rollbacks_i and m["request_id"] < len(det_rollbacks_i) else 0,
                    f"det_tokens_rolled_back_qps_{qps_i}": det_tokens_rb_i[m["request_id"]] if det_tokens_rb_i and m["request_id"] < len(det_tokens_rb_i) else 0,
                    f"det_num_rollbacks_qps_{qps_j}": det_rollbacks_j[m["request_id"]] if det_rollbacks_j and m["request_id"] < len(det_rollbacks_j) else 0,
                    f"det_tokens_rolled_back_qps_{qps_j}": det_tokens_rb_j[m["request_id"]] if det_tokens_rb_j and m["request_id"] < len(det_tokens_rb_j) else 0,
                    "first_mismatch_index": m["first_mismatch_index"],
                    "output_length": m["output_length"],
                    "delta": m["delta"],
                }
                for m in mismatches
            ]
            summary_file = cli_args.output_dir / f"compare_qps_{qps_i}_vs_{qps_j}_summary.jsonl"
            write_jsonl(summary_file, summary_mismatches)
            
            pairwise_details.append({
                "qps_1": qps_i,
                "qps_2": qps_j,
                "mismatch_fraction": float(mismatch_frac),
                "zero_mismatch_fraction": float(np.mean(delta_array == 0)),
                "num_delta_gt_64": int(num_delta_gt_64),
                "num_delta_gt_128": int(num_delta_gt_128),
                "comparison_file": str(pair_file),
            })
            
            print(f"QPS {qps_i} vs QPS {qps_j}: {mismatch_frac:.2%} mismatch rate, "
                  f"{num_delta_gt_64} deltas > 64, {num_delta_gt_128} deltas > 128")
    
    # Plot heatmap
    heatmap_path = cli_args.output_dir / "mismatch_heatmap.pdf"
    plot_mismatch_heatmap(mismatch_matrix, qps_values, heatmap_path)
    
    # Save summary
    summary = {
        "qps_values": qps_values,
        "base_urls": base_urls,
        "num_prompts": cli_args.num_prompts,
        "seed": cli_args.seed,
        "qps_rollback_stats": qps_rollback_stats,
        "pairwise_comparisons": pairwise_details,
        "heatmap_plot": str(heatmap_path),
    }
    
    summary_file = cli_args.output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"{'='*50}")
    print(json.dumps(summary, indent=2))
    print(f"\nAll results saved to: {cli_args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
