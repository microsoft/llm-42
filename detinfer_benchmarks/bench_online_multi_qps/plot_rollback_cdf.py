#!/usr/bin/env python3
"""
Plot CDF of rollbacks and recomputed tokens from multi-QPS comparison output.
Generates two PDF plots:
  1. CDF of num_rollbacks per request (one line per QPS)
  2. CDF of recomputed tokens per request (one line per QPS)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_rollback_data(output_dir: Path) -> Dict[float, Dict[str, List[int]]]:
    """
    Load per-request rollback data from summary.jsonl files.
    
    Returns:
        Dict mapping QPS -> {"num_rollbacks": [...], "tokens_rolled_back": [...]}
    """
    qps_data: Dict[float, Dict[str, List[int]]] = {}
    
    # Find all summary files
    summary_files = list(output_dir.glob("compare_qps_*_summary.jsonl"))
    
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {output_dir}")
    
    for summary_file in summary_files:
        with open(summary_file) as f:
            rows = [json.loads(line) for line in f]
        
        if not rows:
            continue
        
        # Extract QPS values from the first row's keys
        first_row = rows[0]
        qps_values = set()
        for key in first_row.keys():
            if key.startswith("det_num_rollbacks_qps_"):
                qps_str = key.replace("det_num_rollbacks_qps_", "")
                qps_values.add(float(qps_str))
        
        # Extract per-request data for each QPS
        for qps in qps_values:
            if qps not in qps_data:
                qps_data[qps] = {"num_rollbacks": [], "tokens_rolled_back": []}
            
            num_rollbacks_key = f"det_num_rollbacks_qps_{qps}"
            tokens_rb_key = f"det_tokens_rolled_back_qps_{qps}"
            
            # Only add if we haven't collected data for this QPS yet
            # (avoid duplicates from multiple pairwise comparison files)
            if len(qps_data[qps]["num_rollbacks"]) == 0:
                for row in rows:
                    qps_data[qps]["num_rollbacks"].append(row.get(num_rollbacks_key, 0))
                    qps_data[qps]["tokens_rolled_back"].append(row.get(tokens_rb_key, 0))
    
    return qps_data


def compute_cdf(data: List[int]) -> tuple:
    """Compute CDF from data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def plot_cdf(qps_data: Dict[float, Dict[str, List[int]]], 
             metric: str, 
             output_path: Path,
             title: str,
             xlabel: str):
    """Plot CDF for a given metric across all QPS values."""
    plt.figure(figsize=(8, 6))
    
    # Sort QPS values for consistent legend order
    sorted_qps = sorted(qps_data.keys())
    
    # Use a color map for distinct colors
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
    
    for qps, color in zip(sorted_qps, colors):
        data = qps_data[qps][metric]
        if not data:
            continue
        
        x, y = compute_cdf(data)
        plt.step(x, y, where='post', label=f"QPS {qps}", color=color, linewidth=2)
    
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


def print_summary_stats(qps_data: Dict[float, Dict[str, List[int]]]):
    """Print summary statistics for each QPS."""
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for qps in sorted(qps_data.keys()):
        data = qps_data[qps]
        num_rb = np.array(data["num_rollbacks"])
        tokens_rb = np.array(data["tokens_rolled_back"])
        
        print(f"\nQPS {qps}:")
        print(f"  Requests: {len(num_rb)}")
        print(f"  Num Rollbacks:")
        print(f"    Mean: {np.mean(num_rb):.2f}, Median: {np.median(num_rb):.0f}, "
              f"Max: {np.max(num_rb)}, P99: {np.percentile(num_rb, 99):.0f}")
        print(f"    Requests with rollbacks: {np.sum(num_rb > 0)} ({100*np.mean(num_rb > 0):.1f}%)")
        print(f"  Recomputed Tokens:")
        print(f"    Mean: {np.mean(tokens_rb):.2f}, Median: {np.median(tokens_rb):.0f}, "
              f"Max: {np.max(tokens_rb)}, P99: {np.percentile(tokens_rb, 99):.0f}")
        print(f"    Total: {np.sum(tokens_rb)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF of rollbacks and recomputed tokens from multi-QPS comparison"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Directory containing compare_qps_*_summary.jsonl files"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rollback",
        help="Prefix for output PDF files (default: rollback)"
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Output directory does not exist: {args.output_dir}")
        return 1
    
    # Load data
    print(f"Loading rollback data from: {args.output_dir}")
    qps_data = load_rollback_data(args.output_dir)
    
    if not qps_data:
        print("Error: No rollback data found")
        return 1
    
    print(f"Found data for QPS values: {sorted(qps_data.keys())}")
    
    # Print summary statistics
    print_summary_stats(qps_data)
    
    # Plot CDF of num_rollbacks
    rollbacks_pdf = args.output_dir / f"{args.prefix}_num_rollbacks_cdf.pdf"
    plot_cdf(
        qps_data,
        metric="num_rollbacks",
        output_path=rollbacks_pdf,
        title="CDF of Number of Rollbacks per Request",
        xlabel="Number of Rollbacks"
    )
    
    # Plot CDF of tokens_rolled_back (recomputed tokens)
    tokens_pdf = args.output_dir / f"{args.prefix}_recomputed_tokens_cdf.pdf"
    plot_cdf(
        qps_data,
        metric="tokens_rolled_back",
        output_path=tokens_pdf,
        title="CDF of Recomputed Tokens per Request",
        xlabel="Recomputed Tokens"
    )
    
    print(f"\nPlots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
