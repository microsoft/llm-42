#!/usr/bin/env python3
"""
Plot output throughput vs batch size for global-det vs detinfer (LLM-42).
Creates a line graph showing how throughput scales with batch size.
"""

import argparse
import json
from pathlib import Path
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


def plot_throughput_vs_batchsize(results: list, output_path: Path):
    """
    Plot line graph of output throughput vs batch size.
    """
    # Organize data by config
    data = {}  # config_name -> {batch_size: throughput}
    
    for r in results:
        config = r.get('config_name', 'unknown')
        batch_size = r.get('batch_size', r.get('num_prompts', 0))
        throughput = r.get('output_throughput', 0)
        
        if config not in data:
            data[config] = {}
        data[config][batch_size] = throughput
    
    if not data:
        print("No data to plot!")
        return
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    
    # Config styles
    styles = {
        'global_det': {
            'label': 'Global Deterministic',
            'color': 'tab:red',
            'marker': 's',
            'linestyle': '-',
        },
        'detinfer': {
            'label': 'LLM42',
            'color': 'tab:purple',
            'marker': 'o',
            'linestyle': '-',
        },
        'non_det': {
            'label': 'Non-Deterministic',
            'color': 'tab:green',
            'marker': '^',
            'linestyle': '--',
        },
    }
    
    # Plot each config
    for config_name, batch_data in sorted(data.items()):
        batch_sizes = sorted(batch_data.keys())
        throughputs = [batch_data[bs] for bs in batch_sizes]
        
        style = styles.get(config_name, {
            'label': config_name,
            'color': 'tab:blue',
            'marker': 'x',
            'linestyle': '-',
        })
        
        plt.plot(batch_sizes, throughputs,
                 label=style['label'],
                 color=style['color'],
                 marker=style['marker'],
                 linestyle=style['linestyle'],
                 linewidth=2,
                 markersize=8)
    
    # Formatting
    plt.xlabel('Batch Size (Number of Prompts)', fontsize=24)
    plt.ylabel('Output Throughput (tokens/sec)', fontsize=24)
    
    # Use log scale for x-axis since batch sizes are powers of 2
    plt.xscale('log', base=2)
    
    # Set x-ticks to actual batch sizes
    all_batch_sizes = sorted(set(
        bs for batch_data in data.values() for bs in batch_data.keys()
    ))
    plt.xticks(all_batch_sizes, [str(bs) for bs in all_batch_sizes], fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20, loc='best')
    
    # Add some padding
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    print(f"Plot saved to: {output_path.with_suffix('.png')}")
    
    # Also save data as CSV
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write('config,batch_size,output_throughput\n')
        for config_name, batch_data in sorted(data.items()):
            for batch_size in sorted(batch_data.keys()):
                f.write(f'{config_name},{batch_size},{batch_data[batch_size]:.2f}\n')
    print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot throughput vs batch size')
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input JSONL file with benchmark results')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output plot file (PDF)')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.parent / 'throughput_vs_batchsize.pdf'
    
    results = load_results(args.input)
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} results")
    plot_throughput_vs_batchsize(results, args.output)


if __name__ == '__main__':
    main()
