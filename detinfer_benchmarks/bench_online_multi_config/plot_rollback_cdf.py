#!/usr/bin/env python3
"""
Plot CDF of rollbacks and recomputed tokens from multi-config comparison output.
Generates PDF plots:
  1. CDF of num_rollbacks per request (one line per config)
  2. CDF of recomputed tokens per request (one line per config)
  3. CDF of recompute ratio per request (one line per config)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_rollback_data(output_dir: Path) -> Dict[str, Dict[str, List]]:
    """
    Load per-request rollback data from summary.jsonl files.
    
    Returns:
        Dict mapping config_name -> {"num_rollbacks": [...], "tokens_rolled_back": [...], 
                                     "output_len": [...], "recompute_ratio": [...]}
    """
    config_data: Dict[str, Dict[str, List]] = {}
    
    # Find all summary files
    summary_files = list(output_dir.glob("compare_*_summary.jsonl"))
    
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {output_dir}")
    
    for summary_file in summary_files:
        with open(summary_file) as f:
            rows = [json.loads(line) for line in f]
        
        if not rows:
            continue
        
        # Extract config names from the first row's keys
        first_row = rows[0]
        config_names = set()
        for key in first_row.keys():
            if key.startswith("det_num_rollbacks_"):
                config_name = key.replace("det_num_rollbacks_", "")
                config_names.add(config_name)
        
        # Extract per-request data for each config
        for config_name in config_names:
            if config_name in config_data:
                continue  # Already loaded
            
            config_data[config_name] = {
                "num_rollbacks": [], 
                "tokens_rolled_back": [],
                "output_len": [],
                "recompute_ratio": []
            }
            
            num_rollbacks_key = f"det_num_rollbacks_{config_name}"
            tokens_rb_key = f"det_tokens_rolled_back_{config_name}"
            output_len_key = f"output_len_{config_name}"
            
            for row in rows:
                num_rb = row.get(num_rollbacks_key, 0)
                tokens_rb = row.get(tokens_rb_key, 0)
                output_len = row.get(output_len_key, 0)
                
                config_data[config_name]["num_rollbacks"].append(num_rb)
                config_data[config_name]["tokens_rolled_back"].append(tokens_rb)
                config_data[config_name]["output_len"].append(output_len)
                
                # Compute ratio (avoid division by zero)
                ratio = tokens_rb / output_len if output_len > 0 else 0.0
                config_data[config_name]["recompute_ratio"].append(ratio)
    
    return config_data


def compute_cdf(data: List) -> tuple:
    """Compute CDF from data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def plot_cdf(config_data: Dict[str, Dict[str, List]], 
             metric: str, 
             output_path: Path,
             title: str,
             xlabel: str):
    """Plot CDF for a given metric across all config values."""
    plt.figure(figsize=(10, 6))
    
    # Sort config names for consistent legend order
    sorted_configs = sorted(config_data.keys())
    
    # Use a color map for distinct colors
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown']
    
    for config_name, color in zip(sorted_configs, colors):
        data = config_data[config_name][metric]
        if not data:
            continue
        
        x, y = compute_cdf(data)
        # Shorten config name for legend
        short_name = config_name.replace("sglang_", "").replace("detinfer_", "det_")
        plt.step(x, y, where='post', label=short_name, color=color, linewidth=2)
    
    plt.xlabel(xlabel, fontsize=22, fontweight='bold')
    plt.ylabel("CDF", fontsize=22, fontweight='bold')
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def print_summary_stats(config_data: Dict[str, Dict[str, List]]):
    """Print summary statistics for each config."""
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for config_name in sorted(config_data.keys()):
        data = config_data[config_name]
        num_rb = np.array(data["num_rollbacks"])
        tokens_rb = np.array(data["tokens_rolled_back"])
        output_len = np.array(data["output_len"])
        recompute_ratio = np.array(data["recompute_ratio"])
        
        print(f"\nconfig={config_name}:")
        print(f"  Requests: {len(num_rb)}")
        print(f"  Num Rollbacks:")
        print(f"    Mean: {np.mean(num_rb):.2f}, Median: {np.median(num_rb):.0f}, "
              f"Max: {np.max(num_rb)}, P99: {np.percentile(num_rb, 99):.0f}")
        print(f"    Requests with rollbacks: {np.sum(num_rb > 0)} ({100*np.mean(num_rb > 0):.1f}%)")
        print(f"  Recomputed Tokens:")
        print(f"    Mean: {np.mean(tokens_rb):.2f}, Median: {np.median(tokens_rb):.0f}, "
              f"Max: {np.max(tokens_rb)}, P99: {np.percentile(tokens_rb, 99):.0f}")
        print(f"    Total: {np.sum(tokens_rb)}")
        print(f"  Output Length:")
        print(f"    Mean: {np.mean(output_len):.2f}, Median: {np.median(output_len):.0f}, "
              f"Max: {np.max(output_len)}, P99: {np.percentile(output_len, 99):.0f}")
        print(f"  Recompute Ratio (Recomputed/Output):")
        print(f"    Mean: {np.mean(recompute_ratio):.4f}, Median: {np.median(recompute_ratio):.4f}, "
              f"Max: {np.max(recompute_ratio):.4f}, P99: {np.percentile(recompute_ratio, 99):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF of rollbacks and recomputed tokens from multi-config comparison"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Directory containing compare_*_summary.jsonl files"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output PDF files (default: sharegpt_reqs{num_prompts})"
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Output directory does not exist: {args.output_dir}")
        return 1
    
    # Load data
    print(f"Loading rollback data from: {args.output_dir}")
    config_data = load_rollback_data(args.output_dir)
    
    if not config_data:
        print("Error: No rollback data found")
        return 1
    
    # Determine num_prompts from data
    first_config = next(iter(config_data.keys()))
    num_prompts = len(config_data[first_config]["num_rollbacks"])
    
    # Set default prefix if not provided
    if args.prefix is None:
        args.prefix = f"sharegpt_reqs{num_prompts}"
    
    print(f"Found data for configs: {sorted(config_data.keys())}")
    
    # Print summary statistics
    print_summary_stats(config_data)
    
    # Plot CDF of num_rollbacks
    rollbacks_pdf = args.output_dir / f"{args.prefix}_num_rollbacks_cdf.pdf"
    plot_cdf(
        config_data,
        metric="num_rollbacks",
        output_path=rollbacks_pdf,
        title="CDF of Number of Rollbacks per Request",
        xlabel="Number of Rollbacks"
    )
    
    # Plot CDF of tokens_rolled_back (recomputed tokens)
    tokens_pdf = args.output_dir / f"{args.prefix}_recomputed_tokens_cdf.pdf"
    plot_cdf(
        config_data,
        metric="tokens_rolled_back",
        output_path=tokens_pdf,
        title="CDF of Recomputed Tokens per Request",
        xlabel="Recomputed Tokens"
    )
    
    # Plot CDF of recompute ratio (recomputed tokens / output length)
    ratio_pdf = args.output_dir / f"{args.prefix}_recompute_ratio_cdf.pdf"
    plot_cdf(
        config_data,
        metric="recompute_ratio",
        output_path=ratio_pdf,
        title="CDF of Recompute Ratio per Request",
        xlabel="Recompute Ratio (Recomputed Tokens / Output Length)"
    )
    
    print(f"\nPlots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
