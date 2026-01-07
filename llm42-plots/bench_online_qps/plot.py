#!/usr/bin/env python3
"""
Plot CDF and generate tables for Online QPS Benchmark.

For each QPS value:
  - CDF plots of TTFT and E2E latency
  - Table with P50, P75, P90, P99 for TTFT and E2E

Usage:
    python plot.py --results-file results_*/benchmark_results.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set global style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True

# Colors for different configurations
CONFIG_COLORS = {
    'default': 'tab:green',
    'global': 'tab:red',
}

# Colors for different detinfer ratios
RATIO_COLORS = {
    0.02: 'tab:blue',
    0.05: 'tab:orange',
    0.1: 'tab:purple',
    0.2: 'tab:brown',
    0.5: 'tab:pink',
    1.0: 'tab:cyan',
}

# Line styles for different ratios
RATIO_LINESTYLES = {
    0.02: '-',
    0.05: '-',
    0.1: '-',
    0.2: '-',
    0.5: '-',
    1.0: '-',
}

RATIO_ALPHAS = {
    0.02: 1.0,
    0.05: 1.0,
    0.1: 1.0,
    0.2: 1.0,
    0.5: 1.0,
    1.0: 1.0,
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
    Extract metrics organized by dataset -> qps -> (server_config, ratio).
    
    Returns:
        dict: {dataset: {qps: {(server_config, ratio): {...metrics...}}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    for result in results:
        dataset = result.get('dataset_name', '')
        qps = result.get('qps')
        server_config = result.get('server_config', '')
        det_ratio = result.get('deterministic_ratio', 1.0)
        
        if not dataset or qps is None or not server_config:
            continue
        
        # Extract latencies and ttfts
        latencies = result.get('latencies', [])
        ttfts = result.get('ttfts', [])
        
        data[dataset][qps][(server_config, det_ratio)] = {
            'e2e_latency': [x * 1000 for x in latencies if x is not None],  # Convert to ms
            'ttft': [x * 1000 for x in ttfts if x is not None],  # Convert to ms
            'p50_e2e_latency_ms': result.get('p50_e2e_latency_ms'),
            'p75_e2e_latency_ms': result.get('p75_e2e_latency_ms'),
            'p90_e2e_latency_ms': result.get('p90_e2e_latency_ms'),
            'p99_e2e_latency_ms': result.get('p99_e2e_latency_ms'),
            'p50_ttft_ms': result.get('p50_ttft_ms'),
            'p75_ttft_ms': result.get('p75_ttft_ms'),
            'p90_ttft_ms': result.get('p90_ttft_ms'),
            'p99_ttft_ms': result.get('p99_ttft_ms'),
            'throughput': result.get('output_throughput'),
        }
    
    return data


def compute_cdf(data: list) -> tuple:
    """Compute CDF from a list of values."""
    if not data:
        return np.array([]), np.array([])
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def get_label(server_config: str, ratio: float) -> str:
    """Generate label for legend."""
    if server_config == 'default':
        return 'Non-deterministic'
    elif server_config == 'global':
        return 'Global-deterministic'
    else:
        # For detinfer configs, show as LLM-42 @X%
        pct = int(ratio * 100)
        return f"LLM-42 @{pct}%"


def plot_cdf_for_qps(
    qps_data: dict,
    dataset: str,
    qps: float,
    output_dir: Path,
):
    """Plot separate CDF plots for each detinfer config, with default, global, and all ratios."""
    
    # Identify detinfer configs
    detinfer_configs = set()
    for (server_config, ratio) in qps_data.keys():
        if server_config.startswith('detinfer_'):
            detinfer_configs.add(server_config)
    
    # For each detinfer config, create separate TTFT and E2E plots
    for detinfer_config in sorted(detinfer_configs):
        # Extract ws and bs from config name (e.g., detinfer_ws64_bs8 -> ws64_bs8)
        config_suffix = detinfer_config.replace('detinfer_', '')
        
        # Collect data for this plot: default, global, and all ratios of this detinfer
        plot_configs = []
        
        # Add default (ratio 1.0)
        if ('default', 1.0) in qps_data:
            plot_configs.append(('default', 1.0))
        
        # Add global (ratio 1.0)
        if ('global', 1.0) in qps_data:
            plot_configs.append(('global', 1.0))
        
        # Add all ratios for this detinfer config
        for (server_config, ratio) in sorted(qps_data.keys(), key=lambda x: x[1]):
            if server_config == detinfer_config:
                plot_configs.append((server_config, ratio))
        
        # --- Plot TTFT CDF ---
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for (server_config, ratio) in plot_configs:
            metrics = qps_data[(server_config, ratio)]
            data = metrics.get('ttft', [])
            if not data:
                continue
            
            x_vals, y_vals = compute_cdf(data)
            # Use CONFIG_COLORS for default/global, RATIO_COLORS for detinfer
            if server_config in CONFIG_COLORS:
                color = CONFIG_COLORS[server_config]
            else:
                color = RATIO_COLORS.get(ratio, 'tab:gray')
            linestyle = RATIO_LINESTYLES.get(ratio, '-')
            alpha = RATIO_ALPHAS.get(ratio, 1.0)
            label = get_label(server_config, ratio)
            
            ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, 
                    linewidth=2, alpha=alpha, label=label)
        
        ax.set_xlabel('Time to First Token (ms)', fontsize=24, fontweight='bold')
        ax.set_ylabel('CDF', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(fontsize=20, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        ttft_path = output_dir / f'cdf_ttft_{dataset}_qps_{qps}_{config_suffix}.pdf'
        plt.savefig(ttft_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {ttft_path}")
        
        # --- Plot E2E latency CDF ---
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for (server_config, ratio) in plot_configs:
            metrics = qps_data[(server_config, ratio)]
            data = metrics.get('e2e_latency', [])
            if not data:
                continue
            
            x_vals, y_vals = compute_cdf(data)
            # Use CONFIG_COLORS for default/global, RATIO_COLORS for detinfer
            if server_config in CONFIG_COLORS:
                color = CONFIG_COLORS[server_config]
            else:
                color = RATIO_COLORS.get(ratio, 'tab:gray')
            linestyle = RATIO_LINESTYLES.get(ratio, '-')
            alpha = RATIO_ALPHAS.get(ratio, 1.0)
            label = get_label(server_config, ratio)
            
            ax.plot(x_vals, y_vals, color=color, linestyle=linestyle,
                    linewidth=2, alpha=alpha, label=label)
        
        ax.set_xlabel('E2E Latency (ms)', fontsize=24, fontweight='bold')
        ax.set_ylabel('CDF', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(fontsize=20, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        e2e_path = output_dir / f'cdf_e2e_{dataset}_qps_{qps}_{config_suffix}.pdf'
        plt.savefig(e2e_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {e2e_path}")


def generate_table_for_qps(
    qps_data: dict,
    dataset: str,
    qps: float,
    output_dir: Path,
):
    """Generate table with P50, P75, P90, P99 for TTFT and E2E, per detinfer config."""
    
    # Identify detinfer configs
    detinfer_configs = set()
    for (server_config, ratio) in qps_data.keys():
        if server_config.startswith('detinfer_'):
            detinfer_configs.add(server_config)
    
    # For each detinfer config, create a separate table
    for detinfer_config in sorted(detinfer_configs):
        config_suffix = detinfer_config.replace('detinfer_', '')
        rows = []
        
        # Collect configs: default, global, and all ratios of this detinfer
        plot_configs = []
        if ('default', 1.0) in qps_data:
            plot_configs.append(('default', 1.0))
        if ('global', 1.0) in qps_data:
            plot_configs.append(('global', 1.0))
        for (server_config, ratio) in sorted(qps_data.keys(), key=lambda x: x[1]):
            if server_config == detinfer_config:
                plot_configs.append((server_config, ratio))
        
        for (server_config, ratio) in plot_configs:
            metrics = qps_data[(server_config, ratio)]
            
            # Compute percentiles from raw data if not available
            ttft_data = metrics.get('ttft', [])
            e2e_data = metrics.get('e2e_latency', [])
            
            if ttft_data:
                p50_ttft = np.percentile(ttft_data, 50)
                p75_ttft = np.percentile(ttft_data, 75)
                p90_ttft = np.percentile(ttft_data, 90)
                p99_ttft = np.percentile(ttft_data, 99)
            else:
                p50_ttft = metrics.get('p50_ttft_ms', np.nan)
                p75_ttft = metrics.get('p75_ttft_ms', np.nan)
                p90_ttft = metrics.get('p90_ttft_ms', np.nan)
                p99_ttft = metrics.get('p99_ttft_ms', np.nan)
            
            if e2e_data:
                p50_e2e = np.percentile(e2e_data, 50)
                p75_e2e = np.percentile(e2e_data, 75)
                p90_e2e = np.percentile(e2e_data, 90)
                p99_e2e = np.percentile(e2e_data, 99)
            else:
                p50_e2e = metrics.get('p50_e2e_latency_ms', np.nan)
                p75_e2e = metrics.get('p75_e2e_latency_ms', np.nan)
                p90_e2e = metrics.get('p90_e2e_latency_ms', np.nan)
                p99_e2e = metrics.get('p99_e2e_latency_ms', np.nan)
            
            label = get_label(server_config, ratio)
            rows.append({
                'Config': label,
                'TTFT P50': p50_ttft,
                'TTFT P75': p75_ttft,
                'TTFT P90': p90_ttft,
                'TTFT P99': p99_ttft,
                'E2E P50': p50_e2e,
                'E2E P75': p75_e2e,
                'E2E P90': p90_e2e,
                'E2E P99': p99_e2e,
            })
        
        df = pd.DataFrame(rows)
        
        # Save as CSV
        csv_path = output_dir / f'table_{dataset}_qps_{qps}_{config_suffix}.csv'
        df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"Saved: {csv_path}")
        
        # Print to console
        print(f"\n=== {dataset.upper()} QPS={qps} ({config_suffix}) ===")
        print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


def generate_latex_table(
    qps_data: dict,
    dataset: str,
    qps: float,
    output_dir: Path,
):
    """Generate LaTeX table, per detinfer config."""
    
    # Identify detinfer configs
    detinfer_configs = set()
    for (server_config, ratio) in qps_data.keys():
        if server_config.startswith('detinfer_'):
            detinfer_configs.add(server_config)
    
    # For each detinfer config, create a separate table
    for detinfer_config in sorted(detinfer_configs):
        config_suffix = detinfer_config.replace('detinfer_', '')
        rows = []
        
        # Collect configs: default, global, and all ratios of this detinfer
        plot_configs = []
        if ('default', 1.0) in qps_data:
            plot_configs.append(('default', 1.0))
        if ('global', 1.0) in qps_data:
            plot_configs.append(('global', 1.0))
        for (server_config, ratio) in sorted(qps_data.keys(), key=lambda x: x[1]):
            if server_config == detinfer_config:
                plot_configs.append((server_config, ratio))
        
        for (server_config, ratio) in plot_configs:
            metrics = qps_data[(server_config, ratio)]
            ttft_data = metrics.get('ttft', [])
            e2e_data = metrics.get('e2e_latency', [])
            
            if ttft_data:
                p50_ttft = np.percentile(ttft_data, 50)
                p75_ttft = np.percentile(ttft_data, 75)
                p90_ttft = np.percentile(ttft_data, 90)
                p99_ttft = np.percentile(ttft_data, 99)
            else:
                p50_ttft = p75_ttft = p90_ttft = p99_ttft = np.nan
            
            if e2e_data:
                p50_e2e = np.percentile(e2e_data, 50)
                p75_e2e = np.percentile(e2e_data, 75)
                p90_e2e = np.percentile(e2e_data, 90)
                p99_e2e = np.percentile(e2e_data, 99)
            else:
                p50_e2e = p75_e2e = p90_e2e = p99_e2e = np.nan
            
            label = get_label(server_config, ratio)
            rows.append(f"{label} & {p50_ttft:.1f} & {p75_ttft:.1f} & {p90_ttft:.1f} & {p99_ttft:.1f} & {p50_e2e:.1f} & {p75_e2e:.1f} & {p90_e2e:.1f} & {p99_e2e:.1f} \\\\")
        
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{dataset.upper()} QPS={qps} ({config_suffix}) Latency Percentiles (ms)}}
\\label{{tab:{dataset}_qps{qps}_{config_suffix}}}
\\begin{{tabular}}{{l|cccc|cccc}}
\\toprule
& \\multicolumn{{4}}{{c|}}{{TTFT}} & \\multicolumn{{4}}{{c}}{{E2E}} \\\\
Config & P50 & P75 & P90 & P99 & P50 & P75 & P90 & P99 \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        
        latex_path = output_dir / f'table_{dataset}_qps_{qps}_{config_suffix}.tex'
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"Saved: {latex_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot CDF and generate tables for Online QPS Benchmark'
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
    
    # Extract metrics
    data = extract_metrics(results)
    
    # Generate plots and tables for each dataset and QPS
    for dataset in sorted(data.keys()):
        print(f"\n{'='*50}")
        print(f"Processing {dataset.upper()}")
        print(f"{'='*50}")
        
        for qps in sorted(data[dataset].keys()):
            qps_data = data[dataset][qps]
            print(f"\nQPS={qps}: {len(qps_data)} configurations")
            
            # Plot CDFs
            plot_cdf_for_qps(qps_data, dataset, qps, plot_dir)
            
            # Generate tables
            generate_table_for_qps(qps_data, dataset, qps, plot_dir)
            generate_latex_table(qps_data, dataset, qps, plot_dir)
    
    print(f"\nAll plots and tables saved to: {plot_dir}")


if __name__ == '__main__':
    main()
