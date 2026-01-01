#!/usr/bin/env python3
"""
Plot throughput comparison across different server configurations.

This script parses benchmark results and creates bar charts comparing throughput
across different server configurations, similar to plot_prefill_batch_sizes.py.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: Path) -> list:
    """Load benchmark results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_short_config_name(config_name: str) -> str:
    """Get shortened config name for display."""
    mapping = {
        'sglang_non_deterministic': 'Non-Det',
        'sglang_global_deterministic': 'Global-Det',
        'detinfer_ws_32_bs_16': 'DetInfer-ws32-bs16',
        'detinfer_ws_16_bs_32': 'DetInfer-ws16-bs32',
    }
    return mapping.get(config_name, config_name)


def get_config_color(config_name: str) -> str:
    """Get color for each configuration."""
    mapping = {
        'sglang_non_deterministic': 'tab:blue',
        'sglang_global_deterministic': 'tab:orange',
        'detinfer_ws_32_bs_16': 'tab:green',
        'detinfer_ws_16_bs_32': 'tab:red',
    }
    return mapping.get(config_name, 'tab:gray')


def get_config_hatch(config_name: str) -> str:
    """Get hatch pattern for each configuration."""
    mapping = {
        'sglang_non_deterministic': '',
        'sglang_global_deterministic': '//',
        'detinfer_ws_32_bs_16': '\\\\',
        'detinfer_ws_16_bs_32': 'xx',
    }
    return mapping.get(config_name, '')


def plot_throughput_bars(
    data: dict,
    output_path: Path,
    metric: str = 'output_throughput',
    title_suffix: str = '',
):
    """
    Plot grouped bar graph of throughput.
    
    data structure:
    {
        'det_ratio': {
            'config_name': throughput_value
        }
    }
    """
    # Define the order of configs we want
    server_config_order = [
        'sglang_non_deterministic',
        'sglang_global_deterministic',
        'detinfer_ws_32_bs_16',
        'detinfer_ws_16_bs_32'
    ]
    
    # Filter to only configs present in data
    all_configs = set()
    for det_ratio, config_data in data.items():
        all_configs.update(config_data.keys())
    server_config_order = [c for c in server_config_order if c in all_configs]
    
    det_ratios = sorted(data.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Number of groups (det ratios) and bars per group (server configs)
    n_ratios = len(det_ratios)
    n_servers = len(server_config_order)
    
    # Bar width and positions
    bar_width = 0.2
    group_width = bar_width * n_servers
    
    # X positions for each ratio group
    x_positions = np.arange(n_ratios) * (group_width + 0.3)
    
    # Plot bars for each server config
    for i, server_config in enumerate(server_config_order):
        throughputs = []
        for det_ratio in det_ratios:
            if server_config in data[det_ratio]:
                throughputs.append(data[det_ratio][server_config])
            else:
                throughputs.append(0)
        
        x = x_positions + i * bar_width
        color = get_config_color(server_config)
        hatch = get_config_hatch(server_config)
        label = get_short_config_name(server_config)
        
        bars = ax.bar(x, throughputs, bar_width, 
                      color=color, label=label, edgecolor='black', 
                      linewidth=1.0, alpha=0.9, hatch=hatch)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, throughputs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9,
                       fontweight='bold')
    
    # Customize plot
    metric_label = 'Output Throughput (tokens/s)' if metric == 'output_throughput' else 'Total Throughput (tokens/s)'
    ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
    ax.set_xlabel('Deterministic Request Ratio', fontsize=18, fontweight='bold')
    title = f'Throughput Comparison{title_suffix}'
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Set x-ticks at center of each group
    group_centers = x_positions + (n_servers - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f'{r:.0%}' if r <= 1 else f'{r}' for r in det_ratios], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, frameon=True)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {pdf_path}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {png_path}")
    
    plt.close()


def plot_throughput_by_ratio_line(
    data: dict,
    output_path: Path,
    metric: str = 'output_throughput',
    title_suffix: str = '',
):
    """
    Plot line graph of throughput vs deterministic ratio.
    
    data structure:
    {
        'det_ratio': {
            'config_name': throughput_value
        }
    }
    """
    # Define the order of configs we want
    server_config_order = [
        'sglang_non_deterministic',
        'sglang_global_deterministic',
        'detinfer_ws_32_bs_16',
        'detinfer_ws_16_bs_32'
    ]
    
    # Filter to only configs present in data
    all_configs = set()
    for det_ratio, config_data in data.items():
        all_configs.update(config_data.keys())
    server_config_order = [c for c in server_config_order if c in all_configs]
    
    det_ratios = sorted(data.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for i, server_config in enumerate(server_config_order):
        throughputs = []
        ratios_for_config = []
        for det_ratio in det_ratios:
            if server_config in data[det_ratio]:
                throughputs.append(data[det_ratio][server_config])
                ratios_for_config.append(det_ratio)
        
        if throughputs:
            color = get_config_color(server_config)
            label = get_short_config_name(server_config)
            marker = markers[i % len(markers)]
            ax.plot(ratios_for_config, throughputs, marker=marker, 
                   color=color, label=label, linewidth=2, markersize=8)
    
    # Customize plot
    metric_label = 'Output Throughput (tokens/s)' if metric == 'output_throughput' else 'Total Throughput (tokens/s)'
    ax.set_ylabel(metric_label, fontsize=16, fontweight='bold')
    ax.set_xlabel('Deterministic Request Ratio', fontsize=16, fontweight='bold')
    title = f'Throughput vs Deterministic Ratio{title_suffix}'
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='best', fontsize=11, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved line plot to {pdf_path}")
    
    plt.close()


def plot_stacked_throughput(
    data: dict,
    output_path: Path,
    title_suffix: str = '',
):
    """
    Plot stacked bar graph similar to plot_prefill_batch_sizes style.
    Each bar represents a deterministic ratio, stacked by config throughput.
    
    data structure:
    {
        'det_ratio': {
            'config_name': throughput_value
        }
    }
    """
    server_config_order = [
        'sglang_non_deterministic',
        'sglang_global_deterministic',
        'detinfer_ws_32_bs_16',
        'detinfer_ws_16_bs_32'
    ]
    
    # Filter to only configs present in data
    all_configs = set()
    for det_ratio, config_data in data.items():
        all_configs.update(config_data.keys())
    server_config_order = [c for c in server_config_order if c in all_configs]
    
    det_ratios = sorted(data.keys())
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    hatches = ['', '//', '\\\\', 'xx']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(det_ratios))
    bar_width = 0.6
    
    bottoms = np.zeros(len(det_ratios))
    
    for i, server_config in enumerate(server_config_order):
        heights = []
        for det_ratio in det_ratios:
            if server_config in data[det_ratio]:
                heights.append(data[det_ratio][server_config])
            else:
                heights.append(0)
        
        heights = np.array(heights)
        color = colors[i % len(colors)]
        hatch = hatches[i % len(hatches)]
        label = get_short_config_name(server_config)
        
        ax.bar(x, heights, bar_width, bottom=bottoms,
               color=color, label=label, edgecolor='black',
               linewidth=1.0, alpha=0.9, hatch=hatch)
        bottoms += heights
    
    # Customize plot
    ax.set_ylabel('Throughput (tokens/s)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Deterministic Request Ratio', fontsize=18, fontweight='bold')
    title = f'Stacked Throughput Comparison{title_suffix}'
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.0%}' if r <= 1 else f'{r}' for r in det_ratios], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), 
              ncol=len(server_config_order), fontsize=12, frameon=False)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved stacked plot to {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot offline throughput comparison")
    parser.add_argument("--results-file", type=Path, required=True,
                       help="Path to benchmark results JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                       help="Output directory for plots")
    parser.add_argument("--metric", type=str, default="output_throughput",
                       choices=["output_throughput", "total_throughput"],
                       help="Throughput metric to plot")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} benchmark results")
    
    if not results:
        print("Error: No results found")
        return 1
    
    # Organize data by deterministic ratio and config
    data = defaultdict(dict)
    for r in results:
        det_ratio = r.get('deterministic_ratio', 0)
        config_name = r.get('config_name', 'unknown')
        throughput = r.get(args.metric, r.get('output_throughput', 0))
        data[det_ratio][config_name] = throughput
    
    # Get metadata for title
    sample = results[0]
    input_len = sample.get('input_len', 'N/A')
    output_len = sample.get('output_len', 'N/A')
    title_suffix = f'\n(Input={input_len}, Output={output_len})'
    
    # Print summary
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'Det Ratio':<12} {'Config':<30} {args.metric:<20}")
    print("-"*80)
    for det_ratio in sorted(data.keys()):
        for config_name, throughput in sorted(data[det_ratio].items()):
            print(f"{det_ratio:<12.2f} {config_name:<30} {throughput:<20.2f}")
    print("="*80 + "\n")
    
    # Generate plots
    print("Generating plots...")
    
    # Grouped bar chart
    plot_throughput_bars(
        data,
        args.output_dir / "throughput_bars",
        metric=args.metric,
        title_suffix=title_suffix
    )
    
    # Line chart
    plot_throughput_by_ratio_line(
        data,
        args.output_dir / "throughput_line",
        metric=args.metric,
        title_suffix=title_suffix
    )
    
    # Save raw data as CSV
    csv_path = args.output_dir / "throughput_results.csv"
    with open(csv_path, 'w') as f:
        f.write("deterministic_ratio,config_name,throughput\n")
        for det_ratio in sorted(data.keys()):
            for config_name, throughput in sorted(data[det_ratio].items()):
                f.write(f"{det_ratio},{config_name},{throughput}\n")
    print(f"Saved CSV to {csv_path}")
    
    print(f"\nAll plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
