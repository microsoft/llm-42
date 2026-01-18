#!/usr/bin/env python3
"""
Compare outputs from existing benchmark output files.
Use this when you have interrupted runs or want to compare pre-existing results.

Example usage:
    python compare_existing_outputs.py \
        --input-dir /path/to/results/reqs_92812 \
        --output-dir comparison_results

    # Or specify specific files:
    python compare_existing_outputs.py \
        --input-files config_qps4.0_o132_a16.jsonl config_qps5.0_o134_a18.jsonl \
        --output-dir comparison_results
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # Will use fallback for mean calculation


def load_jsonl_result(filepath: Path) -> dict:
    """Load a single JSONL result file (one JSON object per line, we take the first)."""
    with open(filepath, 'r') as f:
        for line in f:
            return json.loads(line)
    raise ValueError(f"Empty file: {filepath}")


def parse_config_from_filename(filename: str) -> dict:
    """Parse config info from filename like 'config_qps4.0_o132_a16.jsonl'."""
    # Pattern: config_qps{qps}_o{order_seed}_a{arrival_seed}.jsonl
    pattern = r"config_qps([\d.]+)_o(\d+)_a(\d+)"
    match = re.search(pattern, filename)
    if match:
        return {
            "qps": float(match.group(1)),
            "order_seed": int(match.group(2)),
            "arrival_seed": int(match.group(3)),
            "name": f"qps{match.group(1)}_o{match.group(2)}_a{match.group(3)}",
        }
    # Fallback: just use filename
    return {
        "qps": 0.0,
        "order_seed": 0,
        "arrival_seed": 0,
        "name": Path(filename).stem,
    }


def load_config_result(filepath: Path, deterministic_only: bool = False) -> dict:
    """Load a config result file and extract relevant data.
    
    Args:
        filepath: Path to the jsonl file
        deterministic_only: If True, only include requests marked as deterministic
    """
    data = load_jsonl_result(filepath)
    config = parse_config_from_filename(filepath.name)
    
    # Extract data matching the format expected by compare_all_configs
    tokens = data.get("output_ids", [])
    texts = data.get("generated_texts", [])
    prompt_hashes = data.get("prompt_hashes", [])
    prompt_lens = data.get("input_lens", [])
    output_lens = data.get("output_lens", [len(t) for t in tokens])
    ttfts = data.get("ttfts", [])
    latencies = data.get("latencies", [])
    itls = data.get("itls", [])
    is_deterministic = data.get("is_deterministic", [True] * len(tokens))
    
    # Filter to deterministic requests only if requested
    if deterministic_only:
        indices = [i for i, det in enumerate(is_deterministic) if det]
        tokens = [tokens[i] for i in indices]
        texts = [texts[i] for i in indices]
        prompt_hashes = [prompt_hashes[i] for i in indices] if prompt_hashes else []
        prompt_lens = [prompt_lens[i] for i in indices] if prompt_lens else []
        output_lens = [output_lens[i] for i in indices] if output_lens else []
        ttfts = [ttfts[i] for i in indices] if ttfts else []
        latencies = [latencies[i] for i in indices] if latencies else []
        itls = [itls[i] for i in indices] if itls else []
        # Update meta_info extraction to use filtered indices
        meta_info_filtered = [data["meta_info"][i] for i in indices] if "meta_info" in data else []
    else:
        indices = list(range(len(tokens)))
        meta_info_filtered = data.get("meta_info", [])
    
    # Extract rollback info from meta_info
    det_num_rollbacks = []
    det_tokens_rolled_back = []
    for meta in meta_info_filtered:
        det_num_rollbacks.append(meta.get("det_infer_num_rollbacks", 0))
        det_tokens_rolled_back.append(meta.get("det_infer_tokens_rolled_back", 0))
    
    # Calculate rollback statistics
    rollback_stats = {}
    if det_num_rollbacks:
        avg_rollbacks = sum(det_num_rollbacks) / len(det_num_rollbacks) if det_num_rollbacks else 0
        rollback_stats = {
            "total_rollbacks": sum(det_num_rollbacks),
            "total_tokens_rolled_back": sum(det_tokens_rolled_back),
            "avg_rollbacks_per_request": float(avg_rollbacks),
            "max_rollbacks_per_request": max(det_num_rollbacks) if det_num_rollbacks else 0,
            "requests_with_rollbacks": sum(1 for x in det_num_rollbacks if x > 0),
        }
    
    # Count deterministic vs non-deterministic
    is_deterministic_all = data.get("is_deterministic", [True] * len(data.get("output_ids", [])))
    num_deterministic = sum(1 for d in is_deterministic_all if d)
    num_non_deterministic = len(is_deterministic_all) - num_deterministic
    
    return {
        "config": config,
        "tokens": tokens,
        "texts": texts,
        "prompt_hashes": prompt_hashes,
        "prompt_lens": prompt_lens,
        "output_lens": output_lens,
        "ttfts": ttfts,
        "latencies": latencies,
        "itls": itls,
        "det_num_rollbacks": det_num_rollbacks,
        "det_tokens_rolled_back": det_tokens_rolled_back,
        "rollback_stats": rollback_stats,
        "raw_stats": {
            "completed": data.get("completed", len(data.get("output_ids", []))),
            "num_deterministic": num_deterministic,
            "num_non_deterministic": num_non_deterministic,
            "filtered_count": len(tokens),
            "duration": data.get("duration", 0),
            "request_throughput": data.get("request_throughput", 0),
            "output_throughput": data.get("output_throughput", 0),
            "mean_e2e_latency_ms": data.get("mean_e2e_latency_ms", 0),
            "p99_e2e_latency_ms": data.get("p99_e2e_latency_ms", 0),
            "mean_ttft_ms": data.get("mean_ttft_ms", 0),
            "p99_ttft_ms": data.get("p99_ttft_ms", 0),
            "mean_tpot_ms": data.get("mean_tpot_ms", 0),
            "p99_tpot_ms": data.get("p99_tpot_ms", 0),
        },
        "output_file": str(filepath),
    }


def write_per_config_logs(result: dict, output_dir: Path):
    """Write summary and detailed logs for a single config."""
    config_name = result["config"]["name"]
    
    ttfts = result.get("ttfts", [])
    latencies = result.get("latencies", [])
    output_lens = result.get("output_lens", [len(t) for t in result["tokens"]])
    
    # Summary log
    summary_path = output_dir / f"config_{config_name}_summary.log"
    with open(summary_path, "w") as f:
        f.write(f"Config: {config_name}\n")
        f.write(f"QPS: {result['config']['qps']}, Order Seed: {result['config']['order_seed']}, "
                f"Arrival Seed: {result['config']['arrival_seed']}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write raw stats
        f.write("Benchmark Stats:\n")
        for k, v in result.get("raw_stats", {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        num_requests = len(result["tokens"])
        for i in range(min(num_requests, 100)):  # Limit to first 100 for summary
            prompt_hash = result["prompt_hashes"][i] if i < len(result["prompt_hashes"]) else "N/A"
            prompt_len = result["prompt_lens"][i] if i < len(result["prompt_lens"]) else 0
            output_len = output_lens[i] if i < len(output_lens) else len(result["tokens"][i])
            ttft = ttfts[i] if i < len(ttfts) else 0
            e2e = latencies[i] if i < len(latencies) else 0
            
            if output_len > 1 and e2e > 0:
                tpot = (e2e - ttft) / (output_len - 1)
            else:
                tpot = 0
            
            f.write(f"[REQ {i+1:03d}] hash={prompt_hash} | prompt_len={prompt_len} | output_len={output_len} | "
                   f"ttft={ttft:.3f}s | tpot={tpot:.4f}s | e2e={e2e:.3f}s\n")
        
        if num_requests > 100:
            f.write(f"\n... and {num_requests - 100} more requests\n")


def compare_all_configs(results: List[dict], output_dir: Path, cli_args: argparse.Namespace):
    """Compare outputs across all configs and write unified comparison report."""
    
    # Build hash -> {config_name: (tokens, text, output_len, prompt_len)} mapping
    hash_to_outputs: Dict[str, Dict[str, Tuple]] = defaultdict(dict)
    
    for result in results:
        config_name = result["config"]["name"]
        for i, (prompt_hash, tokens, text, prompt_len) in enumerate(zip(
            result["prompt_hashes"], result["tokens"], result["texts"], result["prompt_lens"]
        )):
            if prompt_hash:
                hash_to_outputs[prompt_hash][config_name] = (tokens, text, len(tokens), prompt_len)
    
    # Get all config names
    config_names = [r["config"]["name"] for r in results]
    num_configs = len(config_names)
    
    # Per-prompt analysis
    prompt_results = []
    for prompt_hash, outputs in hash_to_outputs.items():
        if len(outputs) < 2:
            continue
        
        first_output = list(outputs.values())[0]
        prompt_len = first_output[3]
        output_len = first_output[2]
        
        all_texts = [text for (tokens, text, output_len, _) in outputs.values()]
        text_match = len(set(all_texts)) == 1
        
        all_tokens = [tuple(tokens) for (tokens, text, output_len, _) in outputs.values()]
        tokens_match = len(set(all_tokens)) == 1
        
        overall_match = text_match or tokens_match
        
        if tokens_match:
            status = "match"
        else:
            status = "mismatch"
        
        # Group configs by identical token output
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
            "raw_stats": result.get("raw_stats", {}),
            "total_output_tokens": total_output_tokens,
            "total_tokens_rolled_back": total_tokens_rolled_back,
            "rollback_pct": rollback_pct,
        })
    
    # Count matches vs mismatches
    num_tokens_match = sum(1 for p in prompt_results if p["tokens_match"])
    num_text_match = sum(1 for p in prompt_results if p["text_match"])
    num_overall_match = sum(1 for p in prompt_results if p["overall_match"])
    num_mismatch = sum(1 for p in prompt_results if not p["overall_match"])
    total = len(prompt_results)
    
    # Write summary log
    deterministic_only = getattr(cli_args, 'deterministic_only', False)
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Multi-Config Comparison Summary (from existing files)\n")
        if deterministic_only:
            f.write("*** DETERMINISTIC REQUESTS ONLY ***\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Input Directory: {cli_args.input_dir}\n")
        f.write(f"Deterministic Only: {deterministic_only}\n")
        f.write(f"Configs: {config_names}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Per-Config Benchmark Stats:\n")
        f.write("-" * 80 + "\n")
        for stat in config_stats:
            cfg = stat["config"]
            raw = stat.get("raw_stats", {})
            f.write(f"\n{cfg['name']}:\n")
            f.write(f"  QPS: {cfg['qps']}, Order Seed: {cfg['order_seed']}, Arrival Seed: {cfg['arrival_seed']}\n")
            f.write(f"  Total Completed: {raw.get('completed', 'N/A')}\n")
            f.write(f"  Deterministic: {raw.get('num_deterministic', 'N/A')}, Non-Det: {raw.get('num_non_deterministic', 'N/A')}\n")
            if deterministic_only:
                f.write(f"  Filtered Count (det only): {raw.get('filtered_count', 'N/A')}\n")
            f.write(f"  Duration: {raw.get('duration', 0):.2f}s\n")
            f.write(f"  Request Throughput: {raw.get('request_throughput', 0):.2f} req/s\n")
            f.write(f"  Output Throughput: {raw.get('output_throughput', 0):.2f} tok/s\n")
            f.write(f"  Mean E2E Latency: {raw.get('mean_e2e_latency_ms', 0):.2f} ms\n")
            f.write(f"  P99 E2E Latency: {raw.get('p99_e2e_latency_ms', 0):.2f} ms\n")
            f.write(f"  Mean TTFT: {raw.get('mean_ttft_ms', 0):.2f} ms\n")
            f.write(f"  Mean TPOT: {raw.get('mean_tpot_ms', 0):.2f} ms\n")
            f.write(f"  Total Output Tokens: {stat['total_output_tokens']}\n")
            f.write(f"  Total Tokens Rolled Back: {stat['total_tokens_rolled_back']}\n")
            f.write(f"  Rollback %: {stat['rollback_pct']:.4f}%\n")
        f.write("\n")
        
        # Per-prompt details
        f.write("-" * 80 + "\n")
        f.write("Per-Prompt Comparison:\n")
        f.write("-" * 80 + "\n")
        
        # Show only mismatches in detail
        mismatches = [pr for pr in prompt_results if not pr["overall_match"]]
        if mismatches:
            f.write(f"\nMismatched prompts ({len(mismatches)}):\n")
            for pr in mismatches[:50]:  # Limit output
                n_cfgs = pr["num_configs"]
                f.write(f"\n[hash={pr['hash']}] prompt_len={pr['prompt_len']} | output_len={pr['output_len']}\n")
                for gi, group in enumerate(pr["token_groups"]):
                    f.write(f"  token_group{gi+1}: {group['configs']} | output_len={group['output_len']}\n")
            if len(mismatches) > 50:
                f.write(f"\n... and {len(mismatches) - 50} more mismatches\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOTAL SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total prompts compared: {total}\n")
        if total > 0:
            f.write(f"Text Matched: {num_text_match} ({100*num_text_match/total:.2f}%)\n")
            f.write(f"Tokens Matched: {num_tokens_match} ({100*num_tokens_match/total:.2f}%)\n")
            f.write(f"Overall Matched (text OR tokens): {num_overall_match} ({100*num_overall_match/total:.2f}%)\n")
            f.write(f"Mismatched: {num_mismatch} ({100*num_mismatch/total:.2f}%)\n")
    
    # Write detailed JSON
    detailed_path = output_dir / "comparison_detailed.json"
    with open(detailed_path, "w") as f:
        detailed = {
            "input_dir": str(cli_args.input_dir),
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
    
    print("\nPer-Config Stats:")
    for stat in config_stats:
        cfg = stat["config"]
        raw = stat.get("raw_stats", {})
        print(f"  {cfg['name']}: {raw.get('completed', 'N/A')} reqs, "
              f"{raw.get('output_throughput', 0):.1f} tok/s, "
              f"rollback {stat['rollback_pct']:.4f}%")
    
    if total > 0:
        print(f"\nTotal: {total} prompts compared")
        print(f"  Text Matched: {num_text_match} ({100*num_text_match/total:.2f}%)")
        print(f"  Tokens Matched: {num_tokens_match} ({100*num_tokens_match/total:.2f}%)")
        print(f"  Overall Matched (text OR tokens): {num_overall_match} ({100*num_overall_match/total:.2f}%)")
        print(f"  Mismatched: {num_mismatch} ({100*num_mismatch/total:.2f}%)")
    else:
        print("\nNo prompts to compare (need at least 2 configs with matching prompt hashes)")
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare outputs from existing benchmark output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all config files in a directory:
  python compare_existing_outputs.py --input-dir /path/to/results/reqs_92812

  # Compare specific files:
  python compare_existing_outputs.py --input-files file1.jsonl file2.jsonl --output-dir comparison

  # Specify file pattern:
  python compare_existing_outputs.py --input-dir /path/to/results --pattern "config_qps*.jsonl"
"""
    )
    parser.add_argument("--input-dir", type=Path, 
                        help="Directory containing config_*.jsonl files")
    parser.add_argument("--input-files", nargs="+", type=Path,
                        help="Specific JSONL files to compare")
    parser.add_argument("--pattern", default="config_qps*.jsonl",
                        help="Glob pattern for finding config files (default: config_qps*.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for comparison results (default: input-dir)")
    parser.add_argument("--write-per-config", action="store_true",
                        help="Write per-config summary logs")
    parser.add_argument("--deterministic-only", action="store_true",
                        help="Only compare deterministic requests (filter by is_deterministic field)")

    cli_args = parser.parse_args()
    
    # Determine input files
    if cli_args.input_files:
        input_files = cli_args.input_files
    elif cli_args.input_dir:
        # Find all matching files, exclude latency files
        input_files = sorted([
            f for f in cli_args.input_dir.glob(cli_args.pattern)
            if not f.name.endswith(".latencies.jsonl")
        ])
    else:
        parser.error("Either --input-dir or --input-files must be specified")
    
    if len(input_files) < 2:
        print(f"Error: Need at least 2 config files to compare, found {len(input_files)}")
        return 1
    
    # Set output directory
    if cli_args.output_dir:
        output_dir = cli_args.output_dir
    elif cli_args.input_dir:
        output_dir = cli_args.input_dir
    else:
        output_dir = input_files[0].parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_args.input_dir = cli_args.input_dir or input_files[0].parent
    
    print(f"Loading {len(input_files)} config files...")
    for f in input_files:
        print(f"  - {f.name}")
    print()
    
    # Load all results
    deterministic_only = getattr(cli_args, 'deterministic_only', False)
    if deterministic_only:
        print("Filtering to DETERMINISTIC requests only\n")
    
    results = []
    for i, filepath in enumerate(input_files):
        print(f"[{i+1}/{len(input_files)}] Loading {filepath.name}...", end=" ", flush=True)
        try:
            result = load_config_result(filepath, deterministic_only=deterministic_only)
            print(f"OK ({len(result['tokens'])} requests)")
            results.append(result)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
    
    if len(results) < 2:
        print(f"Error: Need at least 2 valid config files to compare, loaded {len(results)}")
        return 1
    
    # Write per-config logs if requested
    if cli_args.write_per_config:
        print("\nWriting per-config logs...")
        for result in results:
            write_per_config_logs(result, output_dir)
    
    # Compare all configs
    print("\nComparing configs...")
    compare_all_configs(results, output_dir, cli_args)
    
    return 0


if __name__ == "__main__":
    exit(main())
