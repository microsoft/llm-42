#!/usr/bin/env python3
"""
Plot stacked bar graph of prefill batch size distribution across different configurations.

This script parses server logs to count prefill batch sizes (1, 2, 3, >=4) and creates
a stacked bar chart comparing different server configurations across dataset configs.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def parse_prefill_batch_sizes(log_file: Path) -> dict:
    """Parse a server log file and count prefill batch sizes."""
    counts = defaultdict(int)
    
    if not log_file.exists():
        print(f"Warning: Log file not found: {log_file}")
        return counts
    
    # Pattern to match: "Prefill batch. #new-seq: N"
    pattern = re.compile(r"Prefill batch\. #new-seq: (\d+)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                batch_size = int(match.group(1))
                counts[batch_size] += 1
    
    return counts


def load_counts_from_csv(csv_file: Path) -> dict:
    """Load prefill batch counts from CSV file."""
    config_data = {}
    
    if not csv_file.exists():
        print(f"Warning: CSV file not found: {csv_file}")
        return config_data
    
    with open(csv_file, 'r') as f:
        header = f.readline().strip()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                config_name = parts[0]
                config_data[config_name] = {
                    '1': int(parts[1]),
                    '2': int(parts[2]),
                    '3': int(parts[3]),
                    '≥4': int(parts[4])
                }
    
    return config_data


def get_config_name_from_log(log_file: Path) -> str:
    """Extract config name from log filename."""
    # Expected format: server_gpuN_portXXXXX_CONFIG_NAME.log
    name = log_file.stem
    parts = name.split('_')
    if len(parts) >= 4:
        # Join everything after gpuN_portXXXXX
        return '_'.join(parts[3:])
    return name


def categorize_batch_sizes(counts: dict) -> dict:
    """Categorize batch sizes into 1, 2, 3, >=4."""
    categorized = {
        '1': counts.get(1, 0),
        '2': counts.get(2, 0),
        '3': counts.get(3, 0),
        '≥4': sum(v for k, v in counts.items() if k >= 4)
    }
    return categorized


def parse_dataset_config_from_dir(dir_name: str) -> str:
    """Extract dataset config name from directory name."""
    # Expected formats:
    # results_sharegpt_qpsX_nY
    # results_random_in1024_out1_qpsX_nY
    if 'sharegpt' in dir_name:
        return 'ShareGPT'
    elif 'random_in1024_out1_' in dir_name:
        return 'Random\n(in=1024, out=1)'
    elif 'random_in1024_out129' in dir_name:
        return 'Random\n(in=1024, out=129)'
    elif 'random_in1024_out257' in dir_name:
        return 'Random\n(in=1024, out=257)'
    else:
        return dir_name


def get_short_config_name(config_name: str) -> str:
    """Get shortened config name for display."""
    mapping = {
        'sglang_non_deterministic': 'Non-Det',
        'sglang_global_deterministic': 'Global-Det',
        'detinfer_step_size_64': 'DetInfer-64',
        'detinfer_step_size_128': 'DetInfer-128',
    }
    return mapping.get(config_name, config_name)


def plot_stacked_bars(data: dict, output_path: Path, qps: int = None):
    """
    Plot stacked bar graph.
    
    data structure:
    {
        'dataset_config': {
            'server_config': {'1': count, '2': count, '3': count, '≥4': count}
        }
    }
    """
    # Define the order of configs we want
    server_config_order = [
        'sglang_non_deterministic',
        'sglang_global_deterministic', 
        'detinfer_step_size_64',
        'detinfer_step_size_128'
    ]
    
    dataset_configs = list(data.keys())
    categories = ['1', '2', '3', '≥4']
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    hatches = ['', '//', '\\\\', 'xx']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Number of groups (dataset configs) and bars per group (server configs)
    n_datasets = len(dataset_configs)
    n_servers = len(server_config_order)
    
    # Bar width and positions
    bar_width = 0.22
    group_width = bar_width * n_servers + 0.15
    
    # X positions for each dataset group
    x_positions = np.arange(n_datasets) * (group_width + 0.4)
    
    # Plot bars for each server config within each dataset
    for i, server_config in enumerate(server_config_order):
        # Collect data for this server config across all datasets
        bottoms = np.zeros(n_datasets)
        
        for cat_idx, category in enumerate(categories):
            heights = []
            for dataset_config in dataset_configs:
                if server_config in data[dataset_config]:
                    heights.append(data[dataset_config][server_config].get(category, 0))
                else:
                    heights.append(0)
            
            heights = np.array(heights)
            x = x_positions + i * bar_width
            
            label = f'Batch={category}' if i == 0 else None
            ax.bar(x, heights, bar_width, bottom=bottoms, 
                   color=colors[cat_idx], label=label, edgecolor='black', linewidth=1.0, alpha=0.9,
                   hatch=hatches[cat_idx])
            bottoms += heights
    
    # Add server config labels inside each bar
    for i, server_config in enumerate(server_config_order):
        for j, dataset_config in enumerate(dataset_configs):
            x = x_positions[j] + i * bar_width
            # Calculate total bar height
            total_height = 0
            if server_config in data[dataset_config]:
                total_height = sum(data[dataset_config][server_config].values())
            # Position text inside bar, near the bottom
            if total_height > 0:
                ax.text(x, total_height * 0.02, get_short_config_name(server_config), 
                       ha='center', va='bottom', fontsize=14, rotation=90,
                       color='white', fontweight='bold')
    
    # Customize plot
    ax.set_ylabel('Number of Prefill Batches', fontsize=22, fontweight='bold')
    title = 'Prefill Batch Size Distribution'
    if qps is not None:
        title += f' (QPS={qps})'
    ax.set_title(title, fontsize=24, pad=30)
    
    # Set x-ticks at center of each group - position as dataset labels
    group_centers = x_positions + (n_servers - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(dataset_configs, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    
    # Legend above plot, 4 columns
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=16, frameon=False)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    plt.ylim(0, ax.get_ylim()[1]*1.05)
    # Adjust bottom margin to fit rotated labels
    plt.subplots_adjust(bottom=0.2)
    
    plt.tight_layout()
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=1200, bbox_inches='tight')
    print(f"Saved plot to {pdf_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot prefill batch size distribution")
    parser.add_argument("--results-dirs", nargs='+', required=True,
                       help="List of result directories to process")
    parser.add_argument("--output", type=Path, default=Path("prefill_batch_sizes.pdf"),
                       help="Output plot filename")
    parser.add_argument("--qps", type=int, default=None,
                       help="QPS value to display in title")
    
    args = parser.parse_args()
    
    # Collect data from all result directories
    all_data = {}
    
    for results_dir in args.results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: Results directory not found: {results_path}")
            continue
        
        dataset_config = parse_dataset_config_from_dir(results_path.name)
        
        # Try CSV file first (preferred), then fall back to log parsing
        csv_file = results_path / "prefill_batch_counts.csv"
        
        if csv_file.exists():
            print(f"\nProcessing: {dataset_config} (from CSV)")
            all_data[dataset_config] = load_counts_from_csv(csv_file)
            
            for config_name, counts in all_data[dataset_config].items():
                total = sum(counts.values())
                if total > 0:
                    print(f"  {config_name}: total={total}, "
                          f"bs=1: {counts['1']} ({counts['1']/total*100:.1f}%), "
                          f"bs=2: {counts['2']} ({counts['2']/total*100:.1f}%), "
                          f"bs=3: {counts['3']} ({counts['3']/total*100:.1f}%), "
                          f"bs≥4: {counts['≥4']} ({counts['≥4']/total*100:.1f}%)")
        else:
            # Fall back to parsing log files
            log_dir = results_path / "server_logs_multi_config"
            
            if not log_dir.exists():
                print(f"Warning: Neither CSV nor log directory found in: {results_path}")
                continue
            
            print(f"\nProcessing: {dataset_config} (from logs)")
            all_data[dataset_config] = {}
            
            for log_file in sorted(log_dir.glob("server_gpu*.log")):
                config_name = get_config_name_from_log(log_file)
                counts = parse_prefill_batch_sizes(log_file)
                categorized = categorize_batch_sizes(counts)
                
                all_data[dataset_config][config_name] = categorized
                
                total = sum(categorized.values())
                if total > 0:
                    print(f"  {config_name}: total={total}, "
                          f"bs=1: {categorized['1']} ({categorized['1']/total*100:.1f}%), "
                          f"bs=2: {categorized['2']} ({categorized['2']/total*100:.1f}%), "
                          f"bs=3: {categorized['3']} ({categorized['3']/total*100:.1f}%), "
                          f"bs≥4: {categorized['≥4']} ({categorized['≥4']/total*100:.1f}%)")
    
    if not all_data:
        print("Error: No data found to plot")
        return 1
    
    # Print summary table
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    
    # Plot the data
    plot_stacked_bars(all_data, args.output, qps=args.qps)
    
    return 0


if __name__ == "__main__":
    exit(main())
