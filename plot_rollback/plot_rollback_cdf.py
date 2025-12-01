#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
Plot CDF of per-request rollback statistics for different det_step_sizes.

Usage:
    # Plot from result JSON files:
    python plot_rollback_cdf.py --results results_step10.json results_step20.json results_step50.json results_step100.json
    
    # Or from log files directly:
    python plot_rollback_cdf.py --logs server_step10.log server_step20.log --step-sizes 10 20
    
    # Specify output:
    python plot_rollback_cdf.py --results *.json --output rollback_cdf.png
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


@dataclass
class StepSizeData:
    """Data for one det_step_size configuration."""
    step_size: int
    rollbacks_per_request: List[int]
    tokens_per_request: List[int]
    
    @property
    def num_requests(self) -> int:
        return len(self.rollbacks_per_request)


def parse_log_file(log_file: str) -> Tuple[List[int], List[int]]:
    """
    Parse server log file for per-request rollback stats.
    
    Returns:
        (rollbacks_per_request, tokens_per_request) lists
    """
    pattern = re.compile(
        r'Det Rollback Stats\(rid=([^)]+)\): rollbacks=(\d+), tokens_rolled_back=(\d+)'
    )
    
    rollbacks = []
    tokens = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if match := pattern.search(line):
                rollbacks.append(int(match.group(2)))
                tokens.append(int(match.group(3)))
    
    return rollbacks, tokens


def load_results_json(json_file: str) -> Tuple[int, List[int], List[int]]:
    """
    Load results from JSON file.
    
    Returns:
        (step_size, rollbacks_per_request, tokens_per_request)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Try to extract step_size from config or filename
    step_size = None
    if "config" in data:
        step_size = data["config"].get("step_size") or data["config"].get("min_det_step_size")
    
    if step_size is None:
        # Try to extract from filename (e.g., results_step10.json)
        match = re.search(r'step[_-]?(\d+)', json_file)
        if match:
            step_size = int(match.group(1))
        else:
            step_size = 0  # Unknown
    
    rollbacks = []
    tokens = []
    
    if "per_request_stats" in data:
        for stat in data["per_request_stats"]:
            rollbacks.append(stat.get("num_rollbacks", 0))
            tokens.append(stat.get("tokens_rolled_back", 0))
    
    return step_size, rollbacks, tokens


def compute_cdf(values: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CDF from a list of values.
    
    Returns:
        (x_values, cdf_values) where cdf_values[i] = P(X <= x_values[i])
    """
    if not values:
        return np.array([]), np.array([])
    
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    
    return sorted_values, cdf


def plot_cdf(
    data_list: List[StepSizeData],
    metric: str = "rollbacks",  # "rollbacks" or "tokens"
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot CDF of rollback metrics for different step sizes.
    
    Args:
        data_list: List of StepSizeData objects
        metric: "rollbacks" for rollbacks per request, "tokens" for tokens rolled back
        output_file: Output file path (if None, displays plot)
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a colormap for different step sizes
    colors = cm.viridis(np.linspace(0.2, 0.8, len(data_list)))
    
    for data, color in zip(sorted(data_list, key=lambda d: d.step_size), colors):
        if metric == "rollbacks":
            values = data.rollbacks_per_request
            xlabel = "Rollbacks per Request"
        else:
            values = data.tokens_per_request
            xlabel = "Tokens Rolled Back per Request"
        
        if not values:
            continue
        
        x, cdf = compute_cdf(values)
        
        label = f"step_size={data.step_size} (n={data.num_requests})"
        ax.step(x, cdf, where='post', label=label, color=color, linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("CDF (Cumulative Probability)", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        metric_name = "Rollbacks" if metric == "rollbacks" else "Tokens Rolled Back"
        ax.set_title(f"CDF of {metric_name} per Request by det_step_size", fontsize=14)
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_combined(
    data_list: List[StepSizeData],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    Plot both rollbacks and tokens CDFs side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = cm.viridis(np.linspace(0.2, 0.8, len(data_list)))
    sorted_data = sorted(data_list, key=lambda d: d.step_size)
    
    # Plot rollbacks CDF
    for data, color in zip(sorted_data, colors):
        if not data.rollbacks_per_request:
            continue
        x, cdf = compute_cdf(data.rollbacks_per_request)
        label = f"step_size={data.step_size}"
        ax1.step(x, cdf, where='post', label=label, color=color, linewidth=2)
    
    ax1.set_xlabel("Rollbacks per Request", fontsize=12)
    ax1.set_ylabel("CDF", fontsize=12)
    ax1.set_title("CDF of Rollbacks per Request", fontsize=13)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot tokens CDF
    for data, color in zip(sorted_data, colors):
        if not data.tokens_per_request:
            continue
        x, cdf = compute_cdf(data.tokens_per_request)
        label = f"step_size={data.step_size}"
        ax2.step(x, cdf, where='post', label=label, color=color, linewidth=2)
    
    ax2.set_xlabel("Tokens Rolled Back per Request", fontsize=12)
    ax2.set_ylabel("CDF", fontsize=12)
    ax2.set_title("CDF of Tokens Rolled Back per Request", fontsize=13)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved combined plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_summary_bars(
    data_list: List[StepSizeData],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot bar charts of average rollbacks and tokens by step size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    sorted_data = sorted(data_list, key=lambda d: d.step_size)
    step_sizes = [d.step_size for d in sorted_data]
    
    # Average rollbacks
    avg_rollbacks = [np.mean(d.rollbacks_per_request) if d.rollbacks_per_request else 0 
                    for d in sorted_data]
    std_rollbacks = [np.std(d.rollbacks_per_request) if d.rollbacks_per_request else 0 
                    for d in sorted_data]
    
    x = np.arange(len(step_sizes))
    bars1 = ax1.bar(x, avg_rollbacks, yerr=std_rollbacks, capsize=5, 
                    color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel("det_step_size", fontsize=12)
    ax1.set_ylabel("Avg Rollbacks per Request", fontsize=12)
    ax1.set_title("Average Rollbacks per Request", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(step_sizes)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, avg_rollbacks):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Average tokens rolled back
    avg_tokens = [np.mean(d.tokens_per_request) if d.tokens_per_request else 0 
                 for d in sorted_data]
    std_tokens = [np.std(d.tokens_per_request) if d.tokens_per_request else 0 
                 for d in sorted_data]
    
    bars2 = ax2.bar(x, avg_tokens, yerr=std_tokens, capsize=5,
                    color='coral', alpha=0.8, edgecolor='black')
    ax2.set_xlabel("det_step_size", fontsize=12)
    ax2.set_ylabel("Avg Tokens Rolled Back per Request", fontsize=12)
    ax2.set_title("Average Tokens Rolled Back per Request", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(step_sizes)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, avg_tokens):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def print_statistics(data_list: List[StepSizeData]):
    """Print summary statistics for each step size."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    header = f"{'Step Size':>10} | {'Requests':>10} | {'Avg Rollbacks':>14} | {'Avg Tokens':>12} | {'P50 Rollbacks':>14} | {'P99 Rollbacks':>14}"
    print(header)
    print("-" * len(header))
    
    for data in sorted(data_list, key=lambda d: d.step_size):
        n = data.num_requests
        if n == 0:
            continue
        
        avg_rb = np.mean(data.rollbacks_per_request) if data.rollbacks_per_request else 0
        avg_tok = np.mean(data.tokens_per_request) if data.tokens_per_request else 0
        p50_rb = np.percentile(data.rollbacks_per_request, 50) if data.rollbacks_per_request else 0
        p99_rb = np.percentile(data.rollbacks_per_request, 99) if data.rollbacks_per_request else 0
        
        print(f"{data.step_size:>10} | {n:>10} | {avg_rb:>14.2f} | {avg_tok:>12.1f} | {p50_rb:>14.1f} | {p99_rb:>14.1f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF of per-request rollback statistics"
    )
    
    # Input options
    parser.add_argument("--results", nargs="+", type=str, default=[],
                        help="Result JSON files from bench_per_request_rollbacks.py")
    parser.add_argument("--logs", nargs="+", type=str, default=[],
                        help="Server log files to parse directly")
    parser.add_argument("--step-sizes", nargs="+", type=int, default=[],
                        help="Step sizes corresponding to --logs files (required if using --logs)")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file prefix (will generate _cdf.png, _summary.png)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for plots")
    
    # Plot options
    parser.add_argument("--plot-type", choices=["cdf", "combined", "summary", "all"], 
                        default="all", help="Type of plot to generate")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots (only save)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.results and not args.logs:
        print("Error: Must provide either --results or --logs", file=sys.stderr)
        sys.exit(1)
    
    if args.logs and len(args.logs) != len(args.step_sizes):
        print("Error: --step-sizes must have same length as --logs", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    data_list = []
    
    # Load from JSON results files
    for json_file in args.results:
        if not Path(json_file).exists():
            print(f"Warning: File not found: {json_file}", file=sys.stderr)
            continue
        
        step_size, rollbacks, tokens = load_results_json(json_file)
        if rollbacks:
            data_list.append(StepSizeData(
                step_size=step_size,
                rollbacks_per_request=rollbacks,
                tokens_per_request=tokens,
            ))
            print(f"Loaded {len(rollbacks)} requests from {json_file} (step_size={step_size})")
    
    # Load from log files
    for log_file, step_size in zip(args.logs, args.step_sizes):
        if not Path(log_file).exists():
            print(f"Warning: File not found: {log_file}", file=sys.stderr)
            continue
        
        rollbacks, tokens = parse_log_file(log_file)
        if rollbacks:
            data_list.append(StepSizeData(
                step_size=step_size,
                rollbacks_per_request=rollbacks,
                tokens_per_request=tokens,
            ))
            print(f"Loaded {len(rollbacks)} requests from {log_file} (step_size={step_size})")
    
    if not data_list:
        print("Error: No data loaded", file=sys.stderr)
        sys.exit(1)
    
    # Print statistics
    print_statistics(data_list)
    
    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_prefix = args.output or "rollback"
    
    if args.plot_type in ["cdf", "all"]:
        # CDF of rollbacks
        out_file = output_dir / f"{output_prefix}_cdf_rollbacks.png" if args.output or args.no_show else None
        plot_cdf(data_list, metric="rollbacks", output_file=str(out_file) if out_file else None)
        
        # CDF of tokens
        out_file = output_dir / f"{output_prefix}_cdf_tokens.png" if args.output or args.no_show else None
        plot_cdf(data_list, metric="tokens", output_file=str(out_file) if out_file else None)
    
    if args.plot_type in ["combined", "all"]:
        out_file = output_dir / f"{output_prefix}_cdf_combined.png" if args.output or args.no_show else None
        plot_combined(data_list, output_file=str(out_file) if out_file else None)
    
    if args.plot_type in ["summary", "all"]:
        out_file = output_dir / f"{output_prefix}_summary.png" if args.output or args.no_show else None
        plot_summary_bars(data_list, output_file=str(out_file) if out_file else None)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
