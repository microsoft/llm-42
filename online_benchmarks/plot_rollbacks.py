#!/usr/bin/env python3
"""
Generate publication-quality PDF plots for rollback metrics.

Two types of plots:
1. Per-dataset plots: Separate PDFs for ShareGPT and Arxiv (each with their own QPS)
2. Cross-dataset plots: Combined plots with both datasets (can use different QPS per dataset)

Each plot shows 4 deterministic percentages (1%, 5%, 10%, 100%) as line configurations.

X-axis: Step size
Y-axis: Metric value (rollbacks or tokens recomputed)

Usage:
    # Generate per-dataset plots only
    python plot_rollbacks_publication.py data.jsonl --output-dir ./plots

    # Generate cross-dataset plots with specified QPS
    python plot_rollbacks_publication.py data.jsonl --cross-dataset --sharegpt-qps 6 --arxiv-qps 4
"""

import argparse
import json
import re
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.8',
    'text.usetex': False,
})

# Constants
DATASETS = ['sharegpt', 'arxiv']
DATASET_LABELS = {'sharegpt': 'ShareGPT', 'arxiv': 'Arxiv'}
METRIC_LABELS = {
    'rollbacks': 'Number of Rollbacks',
    'tokens_recomputed': 'Tokens Recomputed'
}
COLORS = {
    0.0:  '#808080',  # Gray (baseline)
    0.01: '#1f77b4',  # Blue
    0.05: '#2ca02c',  # Green
    0.10: '#ff7f0e',  # Orange
    1.0:  '#d62728',  # Red
}
MARKERS = {
    0.0:  'x',  # X (baseline)
    0.01: 'o',  # Circle
    0.05: 's',  # Square
    0.10: '^',  # Triangle up
    1.0:  'D',  # Diamond
}
LINESTYLES = {
    'sharegpt': '-',   # Solid
    'arxiv': '--',     # Dashed
}


def load_rollback_data(filepath: str) -> list:
    """Load rollback metrics from JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def extract_step_size(config_name: str) -> int:
    """Extract step size from config name like 'det_infer_step64'."""
    match = re.search(r'step(\d+)', config_name)
    return int(match.group(1)) if match else None


def organize_data(data: list) -> dict:
    """
    Organize data by dataset, det_ratio, rate, and step_size.
    
    Returns: {dataset: {det_ratio: {rate: {step_size: {rollbacks, tokens_recomputed}}}}}
    """
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for record in data:
        step_size = extract_step_size(record.get('config', ''))
        if step_size is None:
            continue
        
        dataset = record.get('dataset', '')
        det_ratio = record.get('det_ratio', 0)
        rate = record.get('rate', 0)
        
        organized[dataset][det_ratio][rate][step_size] = {
            'rollbacks': record.get('num_rollbacks', 0),
            'tokens_recomputed': record.get('tokens_recomputed', 0)
        }
    
    return organized


def get_rates_per_dataset(organized: dict) -> dict:
    """Get unique QPS rates for each dataset."""
    rates_per_dataset = defaultdict(set)
    for dataset in organized:
        for det_ratio in organized[dataset]:
            for rate in organized[dataset][det_ratio]:
                rates_per_dataset[dataset].add(rate)
    return {d: sorted(r) for d, r in rates_per_dataset.items()}


def format_det_ratio(det_ratio: float) -> str:
    """Format det_ratio as percentage string."""
    pct = det_ratio * 100
    return "100%" if pct == 100 else f"{pct:.0f}%"


def setup_axes(ax, step_sizes: list):
    """Configure axes with log scale and grid."""
    if len(step_sizes) > 0 and max(step_sizes) / min(step_sizes) > 10:
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(step_sizes))
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    else:
        ax.set_xticks(sorted(step_sizes))
    
    ax.set_ylim(bottom=0)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)


def plot_line(ax, x_vals, y_vals, det_ratio, label, dataset='sharegpt', use_linestyle=False):
    """Plot a single line with consistent styling."""
    linestyle = LINESTYLES[dataset] if use_linestyle else '-'
    ax.plot(x_vals, y_vals,
            color=COLORS[det_ratio],
            linestyle=linestyle,
            marker=MARKERS[det_ratio],
            label=label,
            markerfacecolor='white',
            markeredgewidth=1.5,
            markeredgecolor=COLORS[det_ratio])


def create_per_dataset_plots(organized: dict, step_sizes: list, 
                              det_ratios: list, metric: str, output_dir: str):
    """
    Create separate plots for each dataset.
    Each dataset gets its own PDF for each QPS rate.
    """
    rates_per_dataset = get_rates_per_dataset(organized)
    
    for dataset in DATASETS:
        if dataset not in rates_per_dataset:
            continue
        
        for rate in rates_per_dataset[dataset]:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for det_ratio in det_ratios:
                x_vals, y_vals = [], []
                
                for step_size in step_sizes:
                    if (det_ratio in organized.get(dataset, {}) and
                        rate in organized[dataset][det_ratio] and
                        step_size in organized[dataset][det_ratio][rate]):
                        
                        record = organized[dataset][det_ratio][rate][step_size]
                        x_vals.append(step_size)
                        y_vals.append(record[metric])
                
                if x_vals:
                    sorted_data = sorted(zip(x_vals, y_vals))
                    x_vals, y_vals = zip(*sorted_data)
                    label = f"{format_det_ratio(det_ratio)} Deterministic"
                    plot_line(ax, x_vals, y_vals, det_ratio, label, dataset)
            
            ax.set_xlabel('Step Size')
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f'{DATASET_LABELS[dataset]} - {METRIC_LABELS[metric]} (QPS = {rate})')
            
            setup_axes(ax, step_sizes)
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                      ncol=len(det_ratios), frameon=True, fancybox=True)
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f'{metric}_{dataset}_qps{rate}.pdf')
            plt.savefig(output_path, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")


def create_cross_dataset_plots(organized: dict, step_sizes: list, det_ratios: list,
                                metric: str, output_dir: str,
                                sharegpt_qps: int = None, arxiv_qps: int = None):
    """
    Create a single plot combining both ShareGPT and Arxiv datasets.
    Each dataset can use a different QPS rate.
    """
    rates_per_dataset = get_rates_per_dataset(organized)
    
    # Determine QPS for each dataset
    dataset_qps = {}
    for dataset, qps_arg in [('sharegpt', sharegpt_qps), ('arxiv', arxiv_qps)]:
        if dataset in rates_per_dataset:
            available_rates = rates_per_dataset[dataset]
            if qps_arg is not None and qps_arg in available_rates:
                dataset_qps[dataset] = qps_arg
            else:
                dataset_qps[dataset] = available_rates[0]
    
    if not dataset_qps:
        print("No data available for cross-dataset plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for dataset in DATASETS:
        if dataset not in dataset_qps:
            continue
        rate = dataset_qps[dataset]
        
        for det_ratio in det_ratios:
            x_vals, y_vals = [], []
            
            for step_size in step_sizes:
                if (det_ratio in organized.get(dataset, {}) and
                    rate in organized[dataset][det_ratio] and
                    step_size in organized[dataset][det_ratio][rate]):
                    
                    record = organized[dataset][det_ratio][rate][step_size]
                    x_vals.append(step_size)
                    y_vals.append(record[metric])
            
            if x_vals:
                sorted_data = sorted(zip(x_vals, y_vals))
                x_vals, y_vals = zip(*sorted_data)
                label = f"{DATASET_LABELS[dataset]} QPS={rate} ({format_det_ratio(det_ratio)} det)"
                plot_line(ax, x_vals, y_vals, det_ratio, label, dataset, use_linestyle=True)
    
    ax.set_xlabel('Step Size')
    ax.set_ylabel(METRIC_LABELS[metric])
    
    qps_info = [f"{DATASET_LABELS[d]} QPS={dataset_qps[d]}" for d in DATASETS if d in dataset_qps]
    ax.set_title(f'{METRIC_LABELS[metric]} vs Step Size\n({", ".join(qps_info)})')
    
    setup_axes(ax, step_sizes)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
              ncol=1, frameon=True, fancybox=True, borderaxespad=0)
    
    plt.tight_layout()
    
    sharegpt_q = dataset_qps.get('sharegpt', 0)
    arxiv_q = dataset_qps.get('arxiv', 0)
    output_path = os.path.join(output_dir, f'{metric}_cross_dataset_sharegpt{sharegpt_q}_arxiv{arxiv_q}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality PDF plots for rollback metrics')
    parser.add_argument('input_file', help='Path to rollback_metrics.jsonl file')
    parser.add_argument('--output-dir', '-o', default='./pdfs', 
                        help='Output directory for PDF plots')
    parser.add_argument('--cross-dataset', action='store_true',
                        help='Create cross-dataset plots (ShareGPT + Arxiv on same plot)')
    parser.add_argument('--sharegpt-qps', type=int, default=None,
                        help='QPS rate for ShareGPT in cross-dataset plots')
    parser.add_argument('--arxiv-qps', type=int, default=None,
                        help='QPS rate for Arxiv in cross-dataset plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    print(f"Loading rollback data from: {args.input_file}")
    data = load_rollback_data(args.input_file)
    print(f"Loaded {len(data)} records")
    
    if not data:
        print("No data to plot")
        return 1
    
    organized = organize_data(data)
    
    if not organized:
        print("No det_infer data found to plot")
        return 1
    
    # Extract unique values
    all_det_ratios, all_step_sizes = set(), set()
    for dataset in organized:
        for det_ratio in organized[dataset]:
            all_det_ratios.add(det_ratio)
            for rate in organized[dataset][det_ratio]:
                for step_size in organized[dataset][det_ratio][rate]:
                    all_step_sizes.add(step_size)
    
    # Filter out 0.0 (baseline) and keep only 1%, 5%, 10%, 100%
    det_ratios = sorted([d for d in all_det_ratios if d != 0.0])
    step_sizes = sorted(all_step_sizes)
    
    print(f"Found det_ratios: {det_ratios}")
    print(f"Found step_sizes: {step_sizes}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always create per-dataset plots
    print("\nGenerating per-dataset rollback plots...")
    create_per_dataset_plots(organized, step_sizes, det_ratios, 'rollbacks', args.output_dir)
    
    print("\nGenerating per-dataset tokens recomputed plots...")
    create_per_dataset_plots(organized, step_sizes, det_ratios, 'tokens_recomputed', args.output_dir)
    
    # Optionally create cross-dataset plots
    if args.cross_dataset:
        print("\nGenerating cross-dataset plots...")
        create_cross_dataset_plots(organized, step_sizes, det_ratios, 'rollbacks',
                                    args.output_dir, args.sharegpt_qps, args.arxiv_qps)
        create_cross_dataset_plots(organized, step_sizes, det_ratios, 'tokens_recomputed',
                                    args.output_dir, args.sharegpt_qps, args.arxiv_qps)
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
