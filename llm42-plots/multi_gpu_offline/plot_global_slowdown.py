#!/usr/bin/env python3
"""
Plot throughput comparison between Non-Deterministic and Global-Deterministic.

Creates a plot with:
- Selected dataset configs: (1024, 256), (1024, 512), (2048, 256), (2048, 512)
- 2 bars per group: Non-Det, Global-Det
- Annotates slowdown (Global-Det relative to Non-Det)
- Uses hatching, tab colors, unfilled bars

Usage:
    python plot_global_slowdown.py --results-dirs \\
        /path/to/results_random_in1024_out256 \\
        /path/to/results_random_in1024_out512 \\
        /path/to/results_random_in2048_out256 \\
        /path/to/results_random_in2048_out512 \\
        --output global_slowdown.pdf

Example:
    python plot_global_slowdown.py --results-dirs \\
        ./results_random_in1024_out256_n4096_20260105_201955 \\
        ./results_random_in1024_out512_n4096_20260105_203743 \\
        ./results_random_in2048_out256_n4096_20260105_210410 \\
        ./results_random_in2048_out512_n4096_20260105_213253 \\
        --output global_slowdown.pdf
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


def parse_dataset_config_from_dir(dir_name: str) -> tuple:
    """
    Extract dataset config name from directory name.
    Returns (display_name, sort_key) or (None, None) if not in selected configs.
    """
    # Selected configs: (1024, 256), (1024, 512), (2048, 256), (2048, 512)
    if 'random_in1024_out256' in dir_name:
        return ('in=1024\nout=256', (1024, 256))
    elif 'random_in1024_out512' in dir_name:
        return ('in=1024\nout=512', (1024, 512))
    elif 'random_in2048_out256' in dir_name:
        return ('in=2048\nout=256', (2048, 256))
    elif 'random_in2048_out512' in dir_name:
        return ('in=2048\nout=512', (2048, 512))
    else:
        return (None, None)


# Bar labels and their properties (only Non-Det and Global-Det)
BAR_CONFIGS = [
    ('non_det', 'Non-Deterministic', 'tab:blue', '---'),
    ('global_det', 'Global-Deterministic', 'tab:red', '////'),
]


def plot_throughput_comparison(
    data: dict,
    output_path: Path,
    title_suffix: str = "",
):
    """
    Plot grouped bar graph of total throughput across datasets.
    
    data structure:
    {
        'dataset_config': {
            'bar_key': throughput_value  # 'non_det' or 'global_det'
        }
    }
    """
    dataset_configs = list(data.keys())
    n_datasets = len(dataset_configs)
    n_bars = len(BAR_CONFIGS)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n_datasets * 2.5), 6))
    
    # Bar width and positions
    bar_width = 0.30
    group_width = bar_width * n_bars
    
    # X positions for each dataset group
    x_positions = np.arange(n_datasets) * (group_width + 0.3)
    
    # Get non-det throughputs for slowdown calculation
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
                      label=label)
        
        # Add slowdown labels on top of global_det bars
        if bar_key == 'global_det':
            for bar, val, non_det_val in zip(bars, throughputs, non_det_throughputs):
                if val > 0 and non_det_val > 0:
                    # Calculate slowdown (non_det / global_det, since global_det is slower)
                    slowdown = non_det_val / val
                    slowdown_str = f'{slowdown:.2f}x\nslowdown'
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                           slowdown_str, ha='center', va='bottom', fontsize=20, rotation=90, fontweight='bold')
    
    # Customize plot
    ax.set_ylabel('Throughput (tokens/s)', fontsize=24, fontweight='bold')
    
    if title_suffix:
        title = f'Throughput Comparison ({title_suffix})'
    else:
        title = 'Non-Deterministic vs Global-Deterministic Throughput'
    # ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Set x-ticks at center of each group
    group_centers = x_positions + (n_bars - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(dataset_configs, fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelsize=20)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    for bar_key, label, color, hatch in BAR_CONFIGS:
        legend_elements.append(
            Patch(facecolor='none', edgecolor=color, hatch=hatch, 
                  linewidth=2.0, label=label)
        )
    
    ax.legend(handles=legend_elements, loc='best', 
              ncol=2, fontsize=20, frameon=False)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Keep only x-axis and y-axis lines (remove top and right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.20)
    
    # Reduce x-axis margins
    ax.set_xlim(left=-bar_width, right=x_positions[-1] + n_bars * bar_width + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot throughput comparison (Non-Det vs Global-Det)")
    parser.add_argument("--results-dirs", nargs='+', required=True,
                       help="List of result directories to process")
    parser.add_argument("--output", type=Path, default=Path("global_slowdown.pdf"),
                       help="Output plot filename")
    
    args = parser.parse_args()
    
    # Data structure: {dataset: {bar_key: throughput}}
    # Also track sort keys for ordering
    data = {}
    sort_keys = {}
    
    for results_dir in args.results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: Results directory not found: {results_path}")
            continue
        
        dataset_config, sort_key = parse_dataset_config_from_dir(results_path.name)
        
        # Skip if not in our selected configs
        if dataset_config is None:
            print(f"Skipping: {results_path.name} (not in selected configs)")
            continue
        
        # Load results from JSONL file
        results_file = results_path / "benchmark_results.jsonl"
        results = load_results(results_file)
        
        if not results:
            print(f"Warning: No results found in {results_path}")
            continue
        
        print(f"\nProcessing: {dataset_config.replace(chr(10), ', ')}")
        
        if dataset_config not in data:
            data[dataset_config] = {}
            sort_keys[dataset_config] = sort_key
        
        for r in results:
            config_name = r.get('config_name', 'unknown')
            
            # Calculate total throughput
            total_input = r.get('total_input_tokens', 0)
            total_output = r.get('total_output_tokens', 0)
            duration = r.get('duration', 1)
            total_tp = (total_input + total_output) / duration if duration > 0 else 0
            
            # Determine bar key (only non_det and global_det)
            if config_name == 'sglang_non_deterministic':
                bar_key = 'non_det'
                data[dataset_config][bar_key] = total_tp
                print(f"  {config_name}: {total_tp:.2f} tokens/s")
            elif config_name == 'sglang_global_deterministic':
                bar_key = 'global_det'
                data[dataset_config][bar_key] = total_tp
                print(f"  {config_name}: {total_tp:.2f} tokens/s")
    
    if not data:
        print("Error: No data found to plot")
        return 1
    
    # Sort datasets by (input_len, output_len)
    sorted_datasets = sorted(data.keys(), key=lambda x: sort_keys[x])
    sorted_data = {k: data[k] for k in sorted_datasets}
    
    # Generate plot
    plot_throughput_comparison(sorted_data, args.output)
    
    # Save CSV with all data
    csv_path = args.output.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("dataset,bar_key,total_throughput,slowdown\n")
        for dataset in sorted_data:
            dataset_clean = dataset.replace('\n', ' ')
            non_det_tp = sorted_data[dataset].get('non_det', 0)
            for bar_key, tp in sorted_data[dataset].items():
                if bar_key == 'global_det' and non_det_tp > 0:
                    slowdown = non_det_tp / tp
                else:
                    slowdown = 1.0
                f.write(f"{dataset_clean},{bar_key},{tp},{slowdown:.4f}\n")
    print(f"\nSaved CSV to {csv_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
