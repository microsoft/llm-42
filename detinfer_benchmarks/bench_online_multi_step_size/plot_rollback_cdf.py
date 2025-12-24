#!/usr/bin/env python3
"""
Plot CDF of rollbacks and recomputed tokens from multi-step-size comparison output.
Generates two PDF plots:
  1. CDF of num_rollbacks per request (one line per step_size)
  2. CDF of recomputed tokens per request (one line per step_size)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_rollback_data(output_dir: Path) -> Dict[int, Dict[str, List]]:
    """
    Load per-request rollback data from summary.jsonl files.
    
    Returns:
        Dict mapping step_size -> {"num_rollbacks": [...], "tokens_rolled_back": [...], 
                                   "output_len": [...], "recompute_ratio": [...]}
    """
    step_size_data: Dict[int, Dict[str, List]] = {}
    
    # Find all summary files
    summary_files = list(output_dir.glob("compare_step_*_summary.jsonl"))
    
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {output_dir}")
    
    for summary_file in summary_files:
        with open(summary_file) as f:
            rows = [json.loads(line) for line in f]
        
        if not rows:
            continue
        
        # Extract step_size values from the first row's keys
        first_row = rows[0]
        step_sizes = set()
        for key in first_row.keys():
            if key.startswith("det_num_rollbacks_step_"):
                step_str = key.replace("det_num_rollbacks_step_", "")
                step_sizes.add(int(step_str))
        
        # Extract per-request data for each step_size
        for step_size in step_sizes:
            if step_size not in step_size_data:
                step_size_data[step_size] = {
                    "num_rollbacks": [], 
                    "tokens_rolled_back": [],
                    "output_len": [],
                    "recompute_ratio": []
                }
            
            num_rollbacks_key = f"det_num_rollbacks_step_{step_size}"
            tokens_rb_key = f"det_tokens_rolled_back_step_{step_size}"
            output_len_key = f"output_len_step_{step_size}"
            
            # Only add if we haven't collected data for this step_size yet
            # (avoid duplicates from multiple pairwise comparison files)
            if len(step_size_data[step_size]["num_rollbacks"]) == 0:
                for row in rows:
                    num_rb = row.get(num_rollbacks_key, 0)
                    tokens_rb = row.get(tokens_rb_key, 0)
                    output_len = row.get(output_len_key, 0)
                    
                    step_size_data[step_size]["num_rollbacks"].append(num_rb)
                    step_size_data[step_size]["tokens_rolled_back"].append(tokens_rb)
                    step_size_data[step_size]["output_len"].append(output_len)
                    
                    # Compute ratio (avoid division by zero)
                    ratio = tokens_rb / output_len if output_len > 0 else 0.0
                    step_size_data[step_size]["recompute_ratio"].append(ratio)
    
    return step_size_data


def compute_cdf(data: List[int]) -> tuple:
    """Compute CDF from data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def plot_cdf(step_size_data: Dict[int, Dict[str, List]], 
             metric: str, 
             output_path: Path,
             title: str,
             xlabel: str):
    """Plot CDF for a given metric across all step_size values."""
    plt.figure(figsize=(8, 6))
    
    # Sort step_size values for consistent legend order
    sorted_step_sizes = sorted(step_size_data.keys())
    
    # Use a color map for distinct colors
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
    
    for step_size, color in zip(sorted_step_sizes, colors):
        data = step_size_data[step_size][metric]
        if not data:
            continue
        
        x, y = compute_cdf(data)
        plt.step(x, y, where='post', label=f"step_size={step_size}", color=color, linewidth=2)
    
    plt.xlabel(xlabel, fontsize=22, fontweight='bold')
    plt.ylabel("CDF", fontsize=22, fontweight='bold')
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='lower right', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def print_summary_stats(step_size_data: Dict[int, Dict[str, List]]):
    """Print summary statistics for each step_size."""
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for step_size in sorted(step_size_data.keys()):
        data = step_size_data[step_size]
        num_rb = np.array(data["num_rollbacks"])
        tokens_rb = np.array(data["tokens_rolled_back"])
        output_len = np.array(data["output_len"])
        recompute_ratio = np.array(data["recompute_ratio"])
        
        print(f"\nstep_size={step_size}:")
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
        description="Plot CDF of rollbacks and recomputed tokens from multi-step-size comparison"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Directory containing compare_step_*_summary.jsonl files"
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
    step_size_data = load_rollback_data(args.output_dir)
    
    if not step_size_data:
        print("Error: No rollback data found")
        return 1
    
    # Determine num_prompts from data
    first_step_size = next(iter(step_size_data.keys()))
    num_prompts = len(step_size_data[first_step_size]["num_rollbacks"])
    
    # Set default prefix if not provided
    if args.prefix is None:
        args.prefix = f"sharegpt_reqs{num_prompts}"
    
    print(f"Found data for step_size values: {sorted(step_size_data.keys())}")
    
    # Print summary statistics
    print_summary_stats(step_size_data)
    
    # Plot CDF of num_rollbacks
    rollbacks_pdf = args.output_dir / f"{args.prefix}_num_rollbacks_cdf.pdf"
    plot_cdf(
        step_size_data,
        metric="num_rollbacks",
        output_path=rollbacks_pdf,
        title="CDF of Number of Rollbacks per Request",
        xlabel="Number of Rollbacks"
    )
    
    # Plot CDF of tokens_rolled_back (recomputed tokens)
    tokens_pdf = args.output_dir / f"{args.prefix}_recomputed_tokens_cdf.pdf"
    plot_cdf(
        step_size_data,
        metric="tokens_rolled_back",
        output_path=tokens_pdf,
        title="CDF of Recomputed Tokens per Request",
        xlabel="Recomputed Tokens"
    )
    
    # Plot CDF of recompute ratio (recomputed tokens / output length)
    ratio_pdf = args.output_dir / f"{args.prefix}_recompute_ratio_cdf.pdf"
    plot_cdf(
        step_size_data,
        metric="recompute_ratio",
        output_path=ratio_pdf,
        title="CDF of Recompute Ratio per Request",
        xlabel="Recompute Ratio (Recomputed Tokens / Output Length)"
    )
    
    print(f"\nPlots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
