#!/usr/bin/env python3
"""
Plot output throughput vs batch size for global-det vs llm42 (LLM-42).
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
    Plot bar graph of output throughput vs batch size.
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
    plt.figure(figsize=(9, 6))
    
    # Config styles - plot order determines z-order (first = back)
    # non_det should be widest (back), then global_det, then llm42 (front)
    styles = {
        'non_det': {
            'label': 'SGLang non-deterministic',
            'color': 'tab:blue',
            'hatch': '',
            'width': 1.0,
            'zorder': 1,
        },
        'global_det': {
            'label': 'SGLang deterministic',
            'color': 'tab:red',
            'hatch': '',
            'width': 1.0,
            'zorder': 3,
        },
        'llm42': {
            'label': 'LLM-42',
            'color': 'tab:green',
            'hatch': '',
            'width': 1.0,
            'zorder': 2,
        },
    }
    
    # Get all batch sizes
    all_batch_sizes = sorted(set(
        bs for batch_data in data.values() for bs in batch_data.keys()
    ))
    
    # Bar plot settings
    x = np.arange(len(all_batch_sizes))
    
    # Plot bars for each config (overlapped - same position, decreasing widths)
    # Plot in order: non_det (widest, back), global_det, llm42 (narrowest, front)
    plot_order = ['non_det', 'global_det', 'llm42']
    for config_name in plot_order:
        if config_name not in data:
            continue
        batch_data = data[config_name]
        throughputs = [batch_data.get(bs, 0) for bs in all_batch_sizes]
        
        style = styles.get(config_name, {
            'label': config_name,
            'color': 'tab:gray',
            'hatch': '',
            'width': 0.8,
            'zorder': 1,
        })
        
        width = style['width']
        bars = plt.bar(x, throughputs,
                       width=width,
                       label=style['label'],
                       color=style['color'],
                       hatch=style['hatch'],
                       edgecolor='black',
                       linewidth=1.5,
                       alpha=0.6,
                       zorder=style['zorder'])
    
    # Formatting
    plt.xlabel('Batch Size', fontsize=22, fontweight='bold')
    plt.ylabel('Decode Throughput\n(tokens/sec)', fontsize=22, fontweight='bold')
    
    # Set x-ticks at center of each bar group
    plt.xticks(x, [str(bs) for bs in all_batch_sizes], fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set axis limits
    plt.xlim(-0.5, len(all_batch_sizes) - 0.5)
    
    # Set y-axis limit based on max throughput
    max_throughput = max(
        t for batch_data in data.values() for t in batch_data.values()
    )
    plt.ylim(bottom=0, top=max_throughput * 1.3)
    
    # plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=18, loc='upper left', ncol=2)
    
    # Add some padding
    plt.tight_layout()
    
    # Save PDF only
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
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
