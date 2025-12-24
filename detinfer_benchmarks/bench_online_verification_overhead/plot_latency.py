#!/usr/bin/env python3
"""
Plot latency comparison (TTFT, TPOT, E2E) across different step sizes.
Uses consistent font sizes across all plots.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set consistent font sizes
FONT_SIZE = 14
TITLE_SIZE = 16
TICK_SIZE = 12
LEGEND_SIZE = 12

# Configure matplotlib
matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.titlesize': TITLE_SIZE,
})


def get_short_name(config_name: str) -> str:
    """Convert config name to short display name."""
    return config_name.replace("detinfer_step_size_", "step=")


def plot_latency_bars(
    config_names: List[str],
    latency_stats: Dict[str, Dict[str, float]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    """Plot bar chart for a latency metric with mean and percentiles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    short_names = [get_short_name(name) for name in config_names]
    x = np.arange(len(config_names))
    width = 0.2
    
    # Extract metrics
    means = [latency_stats[name].get(f"mean_{metric}_ms", 0) or 0 for name in config_names]
    p50s = [latency_stats[name].get(f"p50_{metric}_ms", 0) or 0 for name in config_names]
    p90s = [latency_stats[name].get(f"p90_{metric}_ms", 0) or 0 for name in config_names]
    p99s = [latency_stats[name].get(f"p99_{metric}_ms", 0) or 0 for name in config_names]
    
    # Plot bars
    bars1 = ax.bar(x - 1.5*width, means, width, label='Mean', color='#2196F3')
    bars2 = ax.bar(x - 0.5*width, p50s, width, label='P50', color='#4CAF50')
    bars3 = ax.bar(x + 0.5*width, p90s, width, label='P90', color='#FF9800')
    bars4 = ax.bar(x + 1.5*width, p99s, width, label='P99', color='#F44336')
    
    ax.set_xlabel('Step Size')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8, rotation=45)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_cdf(
    config_names: List[str],
    latencies: Dict[str, List[float]],
    xlabel: str,
    title: str,
    output_path: Path,
):
    """Plot CDF for latency distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    for i, config_name in enumerate(config_names):
        data = latencies.get(config_name, [])
        if not data:
            continue
        
        # Data is already in ms
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        short_name = get_short_name(config_name)
        ax.plot(sorted_data, cdf, label=short_name, color=colors[i], linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_metrics_combined(
    config_names: List[str],
    latency_stats: Dict[str, Dict[str, float]],
    output_path: Path,
):
    """Plot all three metrics (TTFT, TPOT, E2E) in a single figure with subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    short_names = [get_short_name(name) for name in config_names]
    x = np.arange(len(config_names))
    width = 0.2
    
    metrics = [
        ('ttft', 'TTFT (ms)', 'Time to First Token'),
        ('tpot', 'TPOT (ms)', 'Time per Output Token'),
        ('e2e', 'E2E Latency (ms)', 'End-to-End Latency'),
    ]
    
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        means = [latency_stats[name].get(f"mean_{metric}_ms", 0) or 0 for name in config_names]
        p50s = [latency_stats[name].get(f"p50_{metric}_ms", 0) or 0 for name in config_names]
        p90s = [latency_stats[name].get(f"p90_{metric}_ms", 0) or 0 for name in config_names]
        p99s = [latency_stats[name].get(f"p99_{metric}_ms", 0) or 0 for name in config_names]
        
        bars1 = ax.bar(x - 1.5*width, means, width, label='Mean', color='#2196F3')
        bars2 = ax.bar(x - 0.5*width, p50s, width, label='P50', color='#4CAF50')
        bars3 = ax.bar(x + 0.5*width, p90s, width, label='P90', color='#FF9800')
        bars4 = ax.bar(x + 1.5*width, p99s, width, label='P99', color='#F44336')
        
        ax.set_xlabel('Step Size')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mean_comparison(
    config_names: List[str],
    latency_stats: Dict[str, Dict[str, float]],
    output_path: Path,
):
    """Plot mean latencies for all metrics in a single grouped bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    short_names = [get_short_name(name) for name in config_names]
    x = np.arange(len(config_names))
    width = 0.25
    
    ttfts = [latency_stats[name].get("mean_ttft_ms", 0) or 0 for name in config_names]
    tpots = [latency_stats[name].get("mean_tpot_ms", 0) or 0 for name in config_names]
    e2es = [latency_stats[name].get("mean_e2e_ms", 0) or 0 for name in config_names]
    
    # Scale TPOT for visibility (it's usually much smaller)
    tpot_scale = 10
    tpots_scaled = [t * tpot_scale for t in tpots]
    
    bars1 = ax.bar(x - width, ttfts, width, label='TTFT', color='#2196F3')
    bars2 = ax.bar(x, tpots_scaled, width, label=f'TPOT (×{tpot_scale})', color='#4CAF50')
    bars3 = ax.bar(x + width, e2es, width, label='E2E', color='#FF9800')
    
    ax.set_xlabel('Step Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Mean Latency Comparison Across Step Sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars, values in [(bars1, ttfts), (bars2, tpots), (bars3, e2es)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot latency comparison for verification overhead")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory containing benchmark results")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory for plots (default: same as results-dir)")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summary
    summary_file = args.results_dir / "summary.json"
    if not summary_file.exists():
        print(f"Error: summary.json not found in {args.results_dir}")
        return 1
    
    with summary_file.open() as f:
        summary = json.load(f)
    
    config_names = summary["config_names"]
    
    # Build latency_stats dict from config_stats
    latency_stats = {}
    for stats in summary.get("config_stats", []):
        name = stats["config_name"]
        latency_stats[name] = stats
    
    if not latency_stats:
        print("Error: No latency stats found in summary")
        return 1
    
    # Load raw latency data for CDFs
    latency_data_file = args.results_dir / "latency_data.json"
    latency_data = {}
    if latency_data_file.exists():
        with latency_data_file.open() as f:
            latency_data = json.load(f)
    
    print(f"Plotting latency comparison for {len(config_names)} configurations...")
    print(f"Configs: {config_names}")
    print()
    
    # Plot individual metric bar charts
    plot_latency_bars(
        config_names, latency_stats,
        "ttft", "TTFT (ms)", "Time to First Token (TTFT)",
        output_dir / "ttft_bars.pdf"
    )
    
    plot_latency_bars(
        config_names, latency_stats,
        "tpot", "TPOT (ms)", "Time per Output Token (TPOT)",
        output_dir / "tpot_bars.pdf"
    )
    
    plot_latency_bars(
        config_names, latency_stats,
        "e2e", "E2E Latency (ms)", "End-to-End Latency",
        output_dir / "e2e_bars.pdf"
    )
    
    # Plot combined figure
    plot_all_metrics_combined(
        config_names, latency_stats,
        output_dir / "latency_combined.pdf"
    )
    
    # Plot mean comparison
    plot_mean_comparison(
        config_names, latency_stats,
        output_dir / "latency_mean_comparison.pdf"
    )
    
    # Plot CDFs if raw data available
    if latency_data:
        if "ttfts_ms" in latency_data:
            plot_latency_cdf(
                config_names, latency_data["ttfts_ms"],
                "TTFT (ms)", "TTFT Distribution (CDF)",
                output_dir / "ttft_cdf.pdf"
            )
        
        if "tpots_ms" in latency_data:
            plot_latency_cdf(
                config_names, latency_data["tpots_ms"],
                "TPOT (ms)", "TPOT Distribution (CDF)",
                output_dir / "tpot_cdf.pdf"
            )
        
        if "e2e_latencies_ms" in latency_data:
            plot_latency_cdf(
                config_names, latency_data["e2e_latencies_ms"],
                "E2E Latency (ms)", "E2E Latency Distribution (CDF)",
                output_dir / "e2e_cdf.pdf"
            )
    
    print()
    print(f"All plots saved to: {output_dir}")
    
    # Print summary table
    print()
    print("=" * 80)
    print("Latency Summary (ms)")
    print("=" * 80)
    print(f"{'Config':<25} {'TTFT':>10} {'TPOT':>10} {'E2E':>12}")
    print("-" * 80)
    for name in config_names:
        stats = latency_stats.get(name, {})
        ttft = stats.get("mean_ttft_ms", 0) or 0
        tpot = stats.get("mean_tpot_ms", 0) or 0
        e2e = stats.get("mean_e2e_ms", 0) or 0
        short_name = get_short_name(name)
        print(f"{short_name:<25} {ttft:>10.2f} {tpot:>10.2f} {e2e:>12.2f}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
