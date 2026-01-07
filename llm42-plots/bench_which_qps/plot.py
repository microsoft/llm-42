#!/usr/bin/env python3
"""
Plot CDF of TTFT and E2E latency for QPS comparison benchmark.
Supports both ShareGPT and ArXiv datasets.

Usage:
    python plot.py --results-file results_*/benchmark_results.jsonl
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set global style for aesthetics
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors for different QPS values (ShareGPT)
SHAREGPT_QPS_COLORS = {
    10: 'tab:blue',
    12: 'tab:orange',
    14: 'tab:green',
    16: 'tab:red',
}

# Colors for different QPS values (ArXiv)
ARXIV_QPS_COLORS = {
    0.6: 'tab:blue',
    0.8: 'tab:orange',
    1.0: 'tab:green',
    1.2: 'tab:red',
}


def load_results(filepath: Path) -> list:
    """Load benchmark results from JSONL file."""
    results = []
    if not filepath.exists():
        print(f"Error: Results file not found: {filepath}")
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


def extract_metrics(results: list) -> dict:
    """
    Extract per-request latency data for each (dataset, qps) combination.
    
    Returns:
        dict mapping dataset -> {qps -> {'e2e_latency': [...], 'ttft': [...]}}
    """
    per_dataset_qps_data = {'sharegpt': {}, 'arxiv': {}}
    
    for result in results:
        config_name = result.get('config_name', '')
        dataset = result.get('dataset_name', '')
        
        # Parse dataset and qps from config_name (e.g., "sharegpt_qps_10" or "arxiv_qps_0.6")
        if '_qps_' in config_name:
            parts = config_name.split('_qps_')
            if len(parts) == 2:
                dataset = parts[0]
                try:
                    qps = float(parts[1])
                except ValueError:
                    continue
            else:
                continue
        else:
            qps = result.get('qps')
            if qps is None:
                continue
        
        if dataset not in per_dataset_qps_data:
            per_dataset_qps_data[dataset] = {}
        
        # Extract latencies and ttfts
        latencies = result.get('latencies', [])
        ttfts = result.get('ttfts', [])
        
        per_dataset_qps_data[dataset][qps] = {
            'e2e_latency': [x * 1000 for x in latencies if x is not None],  # Convert to ms
            'ttft': [x * 1000 for x in ttfts if x is not None],  # Convert to ms
        }
        
        # Also store summary stats
        per_dataset_qps_data[dataset][qps]['p50_e2e_latency_ms'] = result.get('p50_e2e_latency_ms')
        per_dataset_qps_data[dataset][qps]['p99_e2e_latency_ms'] = result.get('p99_e2e_latency_ms')
        per_dataset_qps_data[dataset][qps]['p50_ttft_ms'] = result.get('p50_ttft_ms')
        per_dataset_qps_data[dataset][qps]['p99_ttft_ms'] = result.get('p99_ttft_ms')
        per_dataset_qps_data[dataset][qps]['throughput'] = result.get('output_throughput')
    
    return per_dataset_qps_data


def compute_cdf(data: list) -> tuple:
    """Compute CDF from a list of values."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def plot_cdf_ttft(per_qps_data: dict, output_path: Path, qps_values: list = None, dataset: str = 'sharegpt'):
    """Plot CDF of TTFT for each QPS value."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if qps_values is None:
        qps_values = sorted(per_qps_data.keys())
    
    qps_colors = SHAREGPT_QPS_COLORS if dataset == 'sharegpt' else ARXIV_QPS_COLORS
    
    for qps in qps_values:
        if qps not in per_qps_data:
            continue
        
        data = per_qps_data[qps].get('ttft', [])
        if not data:
            continue
        
        x_vals, y_vals = compute_cdf(data)
        color = qps_colors.get(qps, 'tab:gray')
        label = f'QPS={qps}'
        
        ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
    
    # Axis labels (font size 24)
    ax.set_xlabel('Time to First Token (ms)', fontsize=24, fontweight='bold')
    ax.set_ylabel('CDF', fontsize=24, fontweight='bold')
    
    # Tick font size (20)
    ax.tick_params(axis='both', labelsize=20)
    
    # Legend
    ax.legend(fontsize=18, loc='lower right')
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_cdf_e2e(per_qps_data: dict, output_path: Path, qps_values: list = None, dataset: str = 'sharegpt'):
    """Plot CDF of E2E latency for each QPS value."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if qps_values is None:
        qps_values = sorted(per_qps_data.keys())
    
    qps_colors = SHAREGPT_QPS_COLORS if dataset == 'sharegpt' else ARXIV_QPS_COLORS
    
    for qps in qps_values:
        if qps not in per_qps_data:
            continue
        
        data = per_qps_data[qps].get('e2e_latency', [])
        if not data:
            continue
        
        x_vals, y_vals = compute_cdf(data)
        color = qps_colors.get(qps, 'tab:gray')
        label = f'QPS={qps}'
        
        ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
    
    # Axis labels (font size 24)
    ax.set_xlabel('E2E Latency (ms)', fontsize=24, fontweight='bold')
    ax.set_ylabel('CDF', fontsize=24, fontweight='bold')
    
    # Tick font size (20)
    ax.tick_params(axis='both', labelsize=20)
    
    # Legend
    ax.legend(fontsize=18, loc='lower right')
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_both_cdfs(per_qps_data: dict, output_path: Path, qps_values: list = None, dataset: str = 'sharegpt'):
    """Plot both TTFT and E2E latency CDFs side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    if qps_values is None:
        qps_values = sorted(per_qps_data.keys())
    
    qps_colors = SHAREGPT_QPS_COLORS if dataset == 'sharegpt' else ARXIV_QPS_COLORS
    
    # Plot TTFT CDF (left)
    ax1 = axes[0]
    for qps in qps_values:
        if qps not in per_qps_data:
            continue
        data = per_qps_data[qps].get('ttft', [])
        if not data:
            continue
        x_vals, y_vals = compute_cdf(data)
        color = qps_colors.get(qps, 'tab:gray')
        label = f'QPS={qps}'
        ax1.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Time to First Token (ms)', fontsize=22, fontweight='bold')
    ax1.set_ylabel('CDF', fontsize=22, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=20)
    ax1.legend(fontsize=16, loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot E2E latency CDF (right)
    ax2 = axes[1]
    for qps in qps_values:
        if qps not in per_qps_data:
            continue
        data = per_qps_data[qps].get('e2e_latency', [])
        if not data:
            continue
        x_vals, y_vals = compute_cdf(data)
        color = qps_colors.get(qps, 'tab:gray')
        label = f'QPS={qps}'
        ax2.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
    
    ax2.set_xlabel('E2E Latency (ms)', fontsize=22, fontweight='bold')
    ax2.set_ylabel('CDF', fontsize=22, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=20)
    ax2.legend(fontsize=16, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def print_summary(per_qps_data: dict, dataset: str):
    """Print summary statistics for each QPS value."""
    print(f"\n=== {dataset.upper()} Summary Statistics ===")
    print(f"{'QPS':<8} {'P50 TTFT':<12} {'P99 TTFT':<12} {'P50 E2E':<12} {'P99 E2E':<12} {'Throughput':<12}")
    print("-" * 68)
    
    for qps in sorted(per_qps_data.keys()):
        data = per_qps_data[qps]
        p50_ttft = data.get('p50_ttft_ms', 'N/A')
        p99_ttft = data.get('p99_ttft_ms', 'N/A')
        p50_e2e = data.get('p50_e2e_latency_ms', 'N/A')
        p99_e2e = data.get('p99_e2e_latency_ms', 'N/A')
        throughput = data.get('throughput', 'N/A')
        
        p50_ttft_str = f"{p50_ttft:.2f} ms" if isinstance(p50_ttft, (int, float)) else str(p50_ttft)
        p99_ttft_str = f"{p99_ttft:.2f} ms" if isinstance(p99_ttft, (int, float)) else str(p99_ttft)
        p50_e2e_str = f"{p50_e2e:.2f} ms" if isinstance(p50_e2e, (int, float)) else str(p50_e2e)
        p99_e2e_str = f"{p99_e2e:.2f} ms" if isinstance(p99_e2e, (int, float)) else str(p99_e2e)
        throughput_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else str(throughput)
        
        print(f"{qps:<8} {p50_ttft_str:<12} {p99_ttft_str:<12} {p50_e2e_str:<12} {p99_e2e_str:<12} {throughput_str:<12}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot CDF of TTFT and E2E latency for QPS comparison benchmark'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to benchmark_results.jsonl file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as results file)'
    )
    args = parser.parse_args()
    
    results_path = Path(args.results_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plot subdirectory
    plot_dir = output_dir / 'plot'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    print(f"Loaded {len(results)} result entries")
    
    if not results:
        print("No results found. Exiting.")
        return
    
    # Extract metrics per dataset
    per_dataset_qps_data = extract_metrics(results)
    
    # Process each dataset
    for dataset in ['sharegpt', 'arxiv']:
        per_qps_data = per_dataset_qps_data.get(dataset, {})
        if not per_qps_data:
            print(f"\nNo data found for {dataset} dataset. Skipping...")
            continue
        
        print(f"\nFound {dataset} data for QPS values: {sorted(per_qps_data.keys())}")
        
        # Print summary
        print_summary(per_qps_data, dataset)
        
        # Generate individual plots
        plot_cdf_ttft(
            per_qps_data,
            plot_dir / f'cdf_ttft_{dataset}.png',
            dataset=dataset,
        )
        
        plot_cdf_e2e(
            per_qps_data,
            plot_dir / f'cdf_e2e_{dataset}.png',
            dataset=dataset,
        )
        
        # Generate combined plot
        plot_both_cdfs(
            per_qps_data,
            plot_dir / f'cdf_both_{dataset}.png',
            dataset=dataset,
        )
        
        # Also save as PDF for paper quality
        plot_cdf_ttft(
            per_qps_data,
            plot_dir / f'cdf_ttft_{dataset}.pdf',
            dataset=dataset,
        )
        
        plot_cdf_e2e(
            per_qps_data,
            plot_dir / f'cdf_e2e_{dataset}.pdf',
            dataset=dataset,
        )
        
        plot_both_cdfs(
            per_qps_data,
            plot_dir / f'cdf_both_{dataset}.pdf',
            dataset=dataset,
        )
    
    print(f"\nAll plots saved to: {plot_dir}")


if __name__ == '__main__':
    main()
