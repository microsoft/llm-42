#!/usr/bin/env python3
"""Parse benchmark results to extract rollback statistics for LaTeX table."""

import json
import os
from collections import defaultdict
from pathlib import Path

# Results directories
RESULTS_DIRS = [
    "results_arxiv_n4096_20260105_162427",
    "results_sharegpt_n4096_20260105_194742",
    "results_random_in512_out256_n4096_20260105_200726",
    "results_random_in1024_out256_n4096_20260105_201955",
    "results_random_in1024_out512_n4096_20260105_203743",
    "results_random_in2048_out256_n4096_20260105_210410",
    "results_random_in2048_out512_n4096_20260105_213253",
    "results_random_in4096_out512_n4096_20260105_221320",
]

# Map directory names to dataset config labels
DATASET_MAP = {
    "arxiv": "ArXiv",
    "sharegpt": "ShareGPT",
    "random_in512_out256": "Random (512/256)",
    "random_in1024_out256": "Random (1024/256)",
    "random_in1024_out512": "Random (1024/512)",
    "random_in2048_out256": "Random (2048/256)",
    "random_in2048_out512": "Random (2048/512)",
    "random_in4096_out512": "Random (4096/512)",
}

# Deterministic ratios we care about
DET_RATIOS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def extract_dataset_name(dir_name):
    """Extract dataset name from directory name."""
    # results_arxiv_n4096_20260105_162427 -> arxiv
    # results_random_in512_out256_n4096_20260105_200726 -> random_in512_out256
    import re
    match = re.match(r'results_(.+)_n\d+_\d+', dir_name)
    if match:
        return match.group(1)
    return dir_name


# LLM42 configurations to extract
LLM42_CONFIGS = ["llm42_ws_32_bs_16", "llm42_ws_64_bs_8"]


def parse_benchmark_file(filepath, config_filter=None):
    """Parse a benchmark_results.jsonl file and extract rollback stats per det_ratio.
    
    Args:
        filepath: Path to the benchmark_results.jsonl file
        config_filter: If set, only include results from this config (e.g., "llm42_ws_32_bs_16")
    """
    results = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                det_ratio = data.get("deterministic_ratio", None)
                rollback_stats = data.get("rollback_stats", {})
                config_name = data.get("config_name", "")
                
                # Skip baseline configs (non_deterministic, global_deterministic)
                # These don't have LLM-42 rollback behavior we want to measure
                if "non_deterministic" in config_name or "global_deterministic" in config_name:
                    continue
                
                # Filter by config if specified
                if config_filter and config_name != config_filter:
                    continue
                
                if det_ratio is not None and rollback_stats:
                    total_rollbacks = rollback_stats.get("total_rollbacks", 0)
                    total_tokens = rollback_stats.get("total_tokens_rolled_back", 0)
                    
                    # Store by det_ratio (keep the first/best one if duplicates)
                    if det_ratio not in results:
                        results[det_ratio] = {
                            "total_rollbacks": total_rollbacks,
                            "total_tokens": total_tokens,
                            "config_name": config_name,
                        }
            except json.JSONDecodeError:
                continue
    
    return results


def main():
    base_dir = Path(__file__).parent
    
    # Dataset order for display
    dataset_order = [
        "ArXiv",
        "ShareGPT", 
        "Random (512/256)",
        "Random (1024/256)",
        "Random (1024/512)",
        "Random (2048/256)",
        "Random (2048/512)",
        "Random (4096/512)",
    ]
    
    # Collect data for each LLM42 config
    all_data_by_config = {}
    
    for llm42_config in LLM42_CONFIGS:
        all_data = {}
        
        for dir_name in RESULTS_DIRS:
            dir_path = base_dir / dir_name
            if not dir_path.exists():
                print(f"Warning: {dir_path} does not exist")
                continue
            
            results_file = dir_path / "benchmark_results.jsonl"
            if not results_file.exists():
                print(f"Warning: {results_file} does not exist")
                continue
            
            dataset_key = extract_dataset_name(dir_name)
            dataset_label = DATASET_MAP.get(dataset_key, dataset_key)
            
            results = parse_benchmark_file(results_file, config_filter=llm42_config)
            if results:
                all_data[dataset_label] = results
        
        all_data_by_config[llm42_config] = all_data
    
    # Print summary tables for each config
    for llm42_config in LLM42_CONFIGS:
        all_data = all_data_by_config[llm42_config]
        
        print("\n" + "=" * 100)
        print(f"ROLLBACK STATISTICS - {llm42_config}")
        print("Format: rollbacks / recomputed_tokens")
        print("=" * 100)
        
        # Header row
        header = f"{'Dataset Config':<25}"
        for ratio in DET_RATIOS:
            pct = int(ratio * 100)
            header += f"  {pct:>12}%"
        print(header)
        print("-" * 100)
        
        for dataset in dataset_order:
            if dataset not in all_data:
                continue
            
            row = f"{dataset:<25}"
            for ratio in DET_RATIOS:
                if ratio in all_data[dataset]:
                    total_roll = all_data[dataset][ratio]["total_rollbacks"]
                    total_tok = all_data[dataset][ratio]["total_tokens"]
                    row += f"  {total_roll:>5}/{total_tok:<6}"
                else:
                    row += f"  {'--':>12}"
            print(row)
    
    # Print LaTeX table
    print("\n" + "=" * 100)
    print("LATEX TABLE (copy-paste ready)")
    print("Format: rollbacks/recomputed_tokens per cell")
    print("=" * 100)
    
    # Use ws_32_bs_16 for the table (or change as needed)
    primary_config = "llm42_ws_32_bs_16"
    all_data = all_data_by_config[primary_config]
    
    print(f"\n% Using config: {primary_config}")
    for dataset in dataset_order:
        if dataset not in all_data:
            continue
        
        latex_row = f"{dataset:<25} "
        for ratio in DET_RATIOS:
            if ratio in all_data[dataset]:
                total_roll = all_data[dataset][ratio]["total_rollbacks"]
                total_tok = all_data[dataset][ratio]["total_tokens"]
                latex_row += f"& {total_roll}/{total_tok} "
            else:
                latex_row += "& -- "
        latex_row += r"\\"
        print(latex_row)


if __name__ == "__main__":
    main()
