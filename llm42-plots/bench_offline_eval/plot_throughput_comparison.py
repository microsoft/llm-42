#!/usr/bin/env python3
"""
Plot throughput comparison across different server configurations and datasets.

Creates two plots (one per detinfer config):
- Each plot has grouped bars by dataset config
- 7 bars per group: Non-Det, Global-Det, DetInfer@0.02, @0.05, @0.1, @0.2, @0.5, @1.0
- Uses hatching, tab colors, unfilled bars
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


def parse_dataset_config_from_dir(dir_name: str) -> str:
    """Extract dataset config name from directory name."""
    if 'arxiv' in dir_name:
        return 'ArXiv'
    if 'sharegpt' in dir_name:
        return 'ShareGPT'
    elif 'random_in512_out256' in dir_name:
        return 'in=512\nout=256'
    # Varying output lengths (fixed input=1024)
    elif 'random_in1024_out1_' in dir_name:
        return 'in=1024\nout=1'
    elif 'random_in1024_out256' in dir_name:
        return 'in=1024\nout=256'
    elif 'random_in1024_out512' in dir_name:
        return 'in=1024\nout=512'
    elif 'random_in1024_out1024' in dir_name:
        return 'in=1024\nout=1024'
    # Varying input lengths (fixed output)
    elif 'random_in512_out256' in dir_name:
        return 'in=512\nout=256'
    elif 'random_in2048_out256' in dir_name:
        return 'in=2048\nout=256'
    elif 'random_in2048_out512' in dir_name:
        return 'in=2048\nout=512'
    elif 'random_in4096_out256' in dir_name:
        return 'in=4096\nout=256'
    elif 'random_in4096_out512' in dir_name:
        return 'in=4096\nout=512'
    else:
        return dir_name


# Bar labels and their properties
BAR_CONFIGS = [
    ('non_det', 'Non-Deterministic', 'tab:green', '|||'),
    ('global_det', 'Global-Deterministic', 'tab:red', '////'),
    ('detinfer_0.02', 'LLM-42\n@2%', 'tab:purple', '\\\\\\\\'),
    ('detinfer_0.05', 'LLM-42\n@5%', 'tab:purple', '+++'),
    ('detinfer_0.1', 'LLM-42\n@10%', 'tab:purple', '///'),
    ('detinfer_0.2', 'LLM-42\n@20%', 'tab:purple', '---'),
    ('detinfer_0.5', 'LLM-42\n@50%', 'tab:purple', '...'),
    ('detinfer_1.0', 'LLM-42\n@100%', 'tab:purple', 'xxxx'),
]


def plot_throughput_comparison(
    data: dict,
    output_path: Path,
    detinfer_config: str,
    title_suffix: str = "",
):
    """
    Plot grouped bar graph of total throughput across datasets.
    
    data structure:
    {
        'dataset_config': {
            'bar_key': throughput_value  # e.g., 'non_det', 'global_det', 'detinfer_0.02', etc.
        }
    }
    """
    dataset_configs = list(data.keys())
    n_datasets = len(dataset_configs)
    n_bars = len(BAR_CONFIGS)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(18, n_datasets * 3), 6))
    
    # Bar width and positions - scale based on figure size
    bar_width = 0.20
    group_width = bar_width * n_bars
    
    # X positions for each dataset group
    x_positions = np.arange(n_datasets) * (group_width + 0.25)
    
    # Get non-det throughputs for speedup calculation
    non_det_throughputs = []
    for dataset_config in dataset_configs:
        if 'non_det' in data[dataset_config]:
            non_det_throughputs.append(data[dataset_config]['non_det'])
        else:
            non_det_throughputs.append(0)
    
    # Plot bars for each config
    for i, (bar_key, label, color, hatch) in enumerate(BAR_CONFIGS):
        throughputs = []
        for dataset_config in dataset_configs:
            if bar_key in data[dataset_config]:
                throughputs.append(data[dataset_config][bar_key])
            else:
                throughputs.append(0)
        
        x = x_positions + i * bar_width
        
        # Unfilled bars with edge color and hatch
        bars = ax.bar(x, throughputs, bar_width,
                      facecolor='none',  # Unfilled
                      edgecolor=color,
                      linewidth=3.0,
                      alpha=0.8,
                      hatch=hatch,
                      label=label if i == 0 or BAR_CONFIGS[i-1][2] != color else None)  # Only label first of each color
        
        # Add value labels on top of bars with speedup (skip non-det)
        if bar_key != 'non_det':
            for bar, val, non_det_val in zip(bars, throughputs, non_det_throughputs):
                if val > 0:
                    # Calculate speedup relative to non-det
                    if non_det_val > 0:
                        speedup = val / non_det_val
                        speedup_str = f'{speedup:.2f}x'
                    else:
                        speedup_str = ''
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                           f'{speedup_str}', ha='center', va='bottom', fontsize=16,rotation=90)
    # ax.set_ylim(ymax=ax.get_ylim()[1] * 1.1)  
    # Customize plot
    ax.set_ylabel('Throughput (tokens/s)', fontsize=20, fontweight='bold')
    
    title = f'Throughput by Dataset ({title_suffix})'
    # ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Set x-ticks at center of each group
    group_centers = x_positions + (n_bars - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(dataset_configs, fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    for bar_key, label, color, hatch in BAR_CONFIGS:
        legend_elements.append(
            Patch(facecolor='none', edgecolor=color, hatch=hatch, 
                  linewidth=2.0, label=label.replace('\n', ' '))
        )
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=4, fontsize=20, frameon=False)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Keep only x-axis and y-axis lines (remove top and right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)
    
    # Reduce x-axis margins to minimize gap between y-axis and first bar
    ax.set_xlim(left=-bar_width, right=x_positions[-1] + n_bars * bar_width)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot throughput comparison across datasets")
    parser.add_argument("--results-dirs", nargs='+', required=True,
                       help="List of result directories to process")
    parser.add_argument("--output", type=Path, default=Path("throughput_comparison"),
                       help="Output plot filename base (will generate _ws32bs16.pdf and _ws16bs32.pdf)")
    
    args = parser.parse_args()
    
    # Data structure: {detinfer_config: {dataset: {bar_key: throughput}}}
    # detinfer_config is 'ws_32_bs_16' or 'ws_64_bs_8'
    all_data = {
        'ws_32_bs_16': defaultdict(dict),
        'ws_64_bs_8': defaultdict(dict),
    }
    
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
        
        for r in results:
            config_name = r.get('config_name', 'unknown')
            det_ratio = r.get('deterministic_ratio', 0)
            
            # Calculate total throughput
            total_input = r.get('total_input_tokens', 0)
            total_output = r.get('total_output_tokens', 0)
            duration = r.get('duration', 1)
            total_tp = (total_input + total_output) / duration if duration > 0 else 0
            
            # Determine bar key
            if config_name == 'sglang_non_deterministic':
                bar_key = 'non_det'
                # Add to both detinfer plots
                for di_cfg in all_data.keys():
                    all_data[di_cfg][dataset_config][bar_key] = total_tp
            elif config_name == 'sglang_global_deterministic':
                bar_key = 'global_det'
                for di_cfg in all_data.keys():
                    all_data[di_cfg][dataset_config][bar_key] = total_tp
            elif 'detinfer' in config_name:
                # Format ratio consistently (ensure trailing zero for whole numbers)
                if det_ratio == int(det_ratio):
                    bar_key = f'detinfer_{int(det_ratio)}.0'
                else:
                    bar_key = f'detinfer_{det_ratio}'
                # Determine which detinfer config
                if 'ws_32_bs_16' in config_name or 'ws32' in config_name:
                    all_data['ws_32_bs_16'][dataset_config][bar_key] = total_tp
                elif 'ws_64_bs_8' in config_name or 'ws64' in config_name:
                    all_data['ws_64_bs_8'][dataset_config][bar_key] = total_tp
            
            print(f"  {config_name} det={det_ratio}: total={total_tp:.2f} tokens/s")
    
    if not any(all_data[k] for k in all_data):
        print("Error: No data found to plot")
        return 1
    
    # Generate plots
    output_base = args.output
    
    for di_cfg, data in all_data.items():
        if not data:
            continue
        
        # Sort datasets by a consistent order
        sorted_data = dict(sorted(data.items()))
        
        output_file = Path(f"{output_base}_{di_cfg.replace('_', '')}.pdf")
        title_suffix = f"DetInfer {di_cfg.replace('_', '-')}"
        plot_throughput_comparison(sorted_data, output_file, di_cfg, title_suffix)
    
    # Save CSV with all data
    csv_path = args.output.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("dataset,bar_key,detinfer_config,total_throughput\n")
        for di_cfg, data in all_data.items():
            for dataset in data:
                dataset_clean = dataset.replace('\n', ' ')
                for bar_key, tp in data[dataset].items():
                    f.write(f"{dataset_clean},{bar_key},{di_cfg},{tp}\n")
    print(f"\nSaved CSV to {csv_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
