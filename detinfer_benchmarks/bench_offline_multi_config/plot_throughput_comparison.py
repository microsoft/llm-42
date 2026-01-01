#!/usr/bin/env python3
"""
Plot throughput comparison across different server configurations and datasets.

This script parses benchmark results from multiple directories and creates
bar charts comparing throughput, similar to plot_prefill_batch_sizes.py style.
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
    if not filepath.exists():
        print(f"Warning: Results file not found: {filepath}")
        return results
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
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


def parse_dataset_config_from_dir(dir_name: str) -> str:
    """Extract dataset config name from directory name."""
    if 'sharegpt' in dir_name:
        return 'ShareGPT'
    # Varying output lengths (fixed input=1024)
    elif 'random_in1024_out1_' in dir_name:
        return 'Random\n(in=1024, out=1)'
    elif 'random_in1024_out257' in dir_name:
        return 'Random\n(in=1024, out=257)'
    elif 'random_in1024_out513' in dir_name:
        return 'Random\n(in=1024, out=513)'
    elif 'random_in1024_out1025' in dir_name:
        return 'Random\n(in=1024, out=1025)'
    # Varying input lengths (fixed output=257)
    elif 'random_in512_out257' in dir_name:
        return 'Random\n(in=512, out=257)'
    elif 'random_in2048_out257' in dir_name:
        return 'Random\n(in=2048, out=257)'
    elif 'random_in4096_out257' in dir_name:
        return 'Random\n(in=4096, out=257)'
    else:
        return dir_name


def plot_throughput_by_dataset(
    data: dict,
    output_path: Path,
    metric: str = 'output_throughput',
    det_ratio: float = None,
):
    """
    Plot grouped bar graph of throughput across datasets.
    
    data structure:
    {
        'dataset_config': {
            'server_config': throughput_value
        }
    }
    """
    # Define the order of configs we want
    server_config_order = [
        'sglang_non_deterministic',
        'sglang_global_deterministic',
        'detinfer_ws_32_bs_16',
        'detinfer_ws_16_bs_32',
    ]
    
    # Filter to only configs present in data
    all_configs = set()
    for dataset, config_data in data.items():
        all_configs.update(config_data.keys())
    server_config_order = [c for c in server_config_order if c in all_configs]
    
    dataset_configs = list(data.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Number of groups (datasets) and bars per group (server configs)
    n_datasets = len(dataset_configs)
    n_servers = len(server_config_order)
    
    # Bar width and positions
    bar_width = 0.2
    group_width = bar_width * n_servers + 0.15
    
    # X positions for each dataset group
    x_positions = np.arange(n_datasets) * (group_width + 0.4)
    
    # Plot bars for each server config within each dataset
    for i, server_config in enumerate(server_config_order):
        throughputs = []
        for dataset_config in dataset_configs:
            if server_config in data[dataset_config]:
                throughputs.append(data[dataset_config][server_config])
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
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9, rotation=90,
                       fontweight='bold')
    
    # Customize plot
    metric_label = 'Output Throughput (tokens/s)' if metric == 'output_throughput' else 'Total Throughput (tokens/s)'
    ax.set_ylabel(metric_label, fontsize=22, fontweight='bold')
    
    title = 'Throughput Comparison by Dataset'
    if det_ratio is not None:
        title += f' (Det Ratio={det_ratio:.0%})'
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    # Set x-ticks at center of each group
    group_centers = x_positions + (n_servers - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(dataset_configs, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Legend above plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), 
              ncol=len(server_config_order), fontsize=14, frameon=False)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=1200, bbox_inches='tight')
    print(f"Saved plot to {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot throughput comparison across datasets")
    parser.add_argument("--results-dirs", nargs='+', required=True,
                       help="List of result directories to process")
    parser.add_argument("--output", type=Path, default=Path("throughput_comparison.pdf"),
                       help="Output plot filename (base name, will generate _output.pdf and _total.pdf)")
    parser.add_argument("--det-ratio", type=float, default=1.0,
                       help="Deterministic ratio to plot (default: 1.0)")
    
    args = parser.parse_args()
    
    # Collect data from all result directories (for both metrics)
    all_data_output = {}
    all_data_total = {}
    
    for results_dir in args.results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: Results directory not found: {results_path}")
            continue
        
        dataset_config = parse_dataset_config_from_dir(results_path.name)
        
        # Load results from JSONL file
        results_file = results_path / "benchmark_results.jsonl"
        results = load_results(results_file)
        
        if not results:
            print(f"Warning: No results found in {results_path}")
            continue
        
        print(f"\nProcessing: {dataset_config}")
        all_data_output[dataset_config] = {}
        all_data_total[dataset_config] = {}
        
        # Filter by deterministic ratio and organize by config
        for r in results:
            det_ratio = r.get('deterministic_ratio', 0)
            if abs(det_ratio - args.det_ratio) < 0.01:  # Match det ratio
                config_name = r.get('config_name', 'unknown')
                output_tp = r.get('output_throughput', 0)
                # Calculate total_throughput from input + output tokens and duration
                total_input = r.get('total_input_tokens', 0)
                total_output = r.get('total_output_tokens', 0)
                duration = r.get('duration', 1)
                total_tp = (total_input + total_output) / duration if duration > 0 else 0
                
                all_data_output[dataset_config][config_name] = output_tp
                all_data_total[dataset_config][config_name] = total_tp
                print(f"  {config_name}: output={output_tp:.2f}, total={total_tp:.2f} tokens/s")
    
    if not all_data_output:
        print("Error: No data found to plot")
        return 1
    
    # Print summary table
    print("\n" + "="*80)
    print(f"Summary Table (Det Ratio = {args.det_ratio:.0%})")
    print("="*80)
    
    # Generate output filename base
    output_base = args.output.with_suffix('')
    
    # Plot output throughput
    output_pdf = Path(f"{output_base}_output.pdf")
    plot_throughput_by_dataset(all_data_output, output_pdf, metric='output_throughput', det_ratio=args.det_ratio)
    
    # Plot total throughput
    total_pdf = Path(f"{output_base}_total.pdf")
    plot_throughput_by_dataset(all_data_total, total_pdf, metric='total_throughput', det_ratio=args.det_ratio)
    
    # Save as CSV (with both metrics)
    csv_path = args.output.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("dataset,config_name,output_throughput,total_throughput\n")
        for dataset in all_data_output.keys():
            dataset_clean = dataset.replace('\n', ' ')
            for config_name in all_data_output[dataset].keys():
                output_tp = all_data_output[dataset].get(config_name, 0)
                total_tp = all_data_total[dataset].get(config_name, 0)
                f.write(f"{dataset_clean},{config_name},{output_tp},{total_tp}\n")
    print(f"Saved CSV to {csv_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
