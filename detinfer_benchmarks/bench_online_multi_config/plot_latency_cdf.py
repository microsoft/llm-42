#!/usr/bin/env python3
"""
Plot CDF of latency metrics from multi-config comparison output.
Generates PDF plots for:
  1. CDF of TTFT (Time To First Token) per request
  2. CDF of TPOT (Time Per Output Token) per request
  3. CDF of E2E (End-to-End latency) per request
  4. CDF of Output Length per request
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_latency_data(output_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Load per-request latency data from .latencies.jsonl files.
    
    Returns:
        Dict mapping config_name -> {"ttft": [...], "tpot": [...], "e2e": [...], "output_len": [...]}
    """
    config_data: Dict[str, Dict[str, List[float]]] = {}
    
    # Find all latency files (format: config_*.latencies.jsonl)
    latency_files = list(output_dir.glob("config_*.latencies.jsonl"))
    
    if not latency_files:
        raise FileNotFoundError(f"No latency files found in {output_dir}")
    
    for latency_file in latency_files:
        # Extract config_name from filename (e.g., config_sglang_non_deterministic.latencies.jsonl)
        filename = latency_file.stem  # config_sglang_non_deterministic.latencies
        config_name = filename.replace("config_", "").replace(".latencies", "")
        
        if config_name in config_data:
            continue  # Already loaded
        
        config_data[config_name] = {
            "ttft": [],
            "tpot": [],
            "e2e": [],
            "output_len": []
        }
        
        with open(latency_file) as f:
            for line in f:
                row = json.loads(line)
                
                # Extract latency metrics (values are in milliseconds)
                ttft_ms = row.get("ttft_ms", row.get("ttft", 0))
                tpot_ms = row.get("tpot_ms", row.get("tpot", 0))
                e2e_ms = row.get("e2e_latency_ms", row.get("e2e_latency", row.get("latency", 0)))
                output_len = row.get("output_len", row.get("output_tokens", row.get("completion_tokens", 0)))
                
                config_data[config_name]["ttft"].append(ttft_ms)
                config_data[config_name]["tpot"].append(tpot_ms)
                config_data[config_name]["e2e"].append(e2e_ms)
                config_data[config_name]["output_len"].append(output_len)
    
    return config_data


def compute_cdf(data: List[float]) -> tuple:
    """Compute CDF from data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def plot_cdf(config_data: Dict[str, Dict[str, List[float]]], 
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
    if metric in ["ttft", "tpot"]:
        plt.xscale("log")
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


def print_summary_stats(config_data: Dict[str, Dict[str, List[float]]]):
    """Print summary statistics for each config."""
    print("\n" + "=" * 80)
    print("Latency Summary Statistics")
    print("=" * 80)
    
    for config_name in sorted(config_data.keys()):
        data = config_data[config_name]
        ttft = np.array(data["ttft"])  # already in ms
        tpot = np.array(data["tpot"])  # already in ms
        e2e = np.array(data["e2e"])    # already in ms
        output_len = np.array(data["output_len"])
        
        print(f"\nconfig={config_name}:")
        print(f"  Requests: {len(ttft)}")
        print(f"  TTFT (ms):")
        print(f"    Mean: {np.mean(ttft):.2f}, Median: {np.median(ttft):.2f}, "
              f"P90: {np.percentile(ttft, 90):.2f}, P99: {np.percentile(ttft, 99):.2f}")
        print(f"  TPOT (ms):")
        print(f"    Mean: {np.mean(tpot):.2f}, Median: {np.median(tpot):.2f}, "
              f"P90: {np.percentile(tpot, 90):.2f}, P99: {np.percentile(tpot, 99):.2f}")
        print(f"  E2E Latency (ms):")
        print(f"    Mean: {np.mean(e2e):.2f}, Median: {np.median(e2e):.2f}, "
              f"P90: {np.percentile(e2e, 90):.2f}, P99: {np.percentile(e2e, 99):.2f}")
        print(f"  Output Length:")
        print(f"    Mean: {np.mean(output_len):.2f}, Median: {np.median(output_len):.0f}, "
              f"P90: {np.percentile(output_len, 90):.0f}, P99: {np.percentile(output_len, 99):.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF of latency metrics from multi-config comparison"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Directory containing config_*.latencies.jsonl files"
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
    print(f"Loading latency data from: {args.output_dir}")
    config_data = load_latency_data(args.output_dir)
    
    if not config_data:
        print("Error: No latency data found")
        return 1
    
    # Determine num_prompts from data
    first_config = next(iter(config_data.keys()))
    num_prompts = len(config_data[first_config]["ttft"])
    
    # Set default prefix if not provided
    if args.prefix is None:
        args.prefix = f"sharegpt_reqs{num_prompts}"
    
    print(f"Found data for configs: {sorted(config_data.keys())}")
    
    # Print summary statistics
    print_summary_stats(config_data)
    
    # Plot CDF of TTFT
    ttft_pdf = args.output_dir / f"{args.prefix}_ttft_cdf.pdf"
    plot_cdf(
        config_data,
        metric="ttft",
        output_path=ttft_pdf,
        title="CDF of Time To First Token (TTFT)",
        xlabel="TTFT (ms)"
    )
    
    # Plot CDF of TPOT
    tpot_pdf = args.output_dir / f"{args.prefix}_tpot_cdf.pdf"
    plot_cdf(
        config_data,
        metric="tpot",
        output_path=tpot_pdf,
        title="CDF of Time Per Output Token (TPOT)",
        xlabel="TPOT (ms)"
    )
    
    # Plot CDF of E2E latency
    e2e_pdf = args.output_dir / f"{args.prefix}_e2e_cdf.pdf"
    plot_cdf(
        config_data,
        metric="e2e",
        output_path=e2e_pdf,
        title="CDF of End-to-End Latency",
        xlabel="E2E Latency (ms)"
    )
    
    # Plot CDF of output length
    output_len_pdf = args.output_dir / f"{args.prefix}_output_len_cdf.pdf"
    plot_cdf(
        config_data,
        metric="output_len",
        output_path=output_len_pdf,
        title="CDF of Output Length",
        xlabel="Output Length (tokens)"
    )
    
    print(f"\nPlots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
