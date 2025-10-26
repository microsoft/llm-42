#!/usr/bin/env python3

"""
Plot Etalon Benchmark Results
Compares performance metrics across different deterministic modes
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install matplotlib numpy")
    sys.exit(1)


def load_summary(result_dir: Path) -> Optional[Dict]:
    """Load performance summary from perf_metrics.csv
    
    CSV structure:
    Row 0: Headers (Mean, P50, P90, P99) - describing column statistics
    Row 1: Number of Prompt Tokens  
    Row 2: Number of Output Tokens
    Row 3: Number of Total Tokens
    Row 4: TTFT (seconds)
    Row 5: TPOT (seconds)
    Row 6: TBT (seconds)
    Row 7: End-to-End Latency (seconds)
    Row 8: Normalized E2E Latency
    Row 9: Output Throughput (tokens/sec)
    
    Returns dict with keys matching expected format:
    - request_mean_ttft_ms
    - request_mean_tpot_ms
    - request_output_throughput_token_per_s
    - request_mean_e2e_latency_ms
    """
    try:
        import csv
        perf_metrics_path = result_dir / 'perf_metrics.csv'
        if not perf_metrics_path.exists():
            return None
        
        with open(perf_metrics_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) < 10:  # Need at least 10 rows for all metrics (header + 9 metric rows)
                return None
            
            # Extract mean values (first column, index 0)
            # Rows are 0-indexed: header at 0, metrics start at 1
            ttft_s = float(rows[4][0])  # Row 4 = TTFT mean in seconds
            tpot_s = float(rows[5][0])  # Row 5 = TPOT mean in seconds
            e2e_latency_s = float(rows[7][0])  # Row 7 = E2E latency mean in seconds
            throughput = float(rows[9][0])  # Row 9 = Output throughput in tokens/sec
            
            return {
                'request_mean_ttft_ms': ttft_s * 1000,  # Convert to ms
                'request_mean_tpot_ms': tpot_s * 1000,  # Convert to ms
                'request_output_throughput_token_per_s': throughput,
                'request_mean_e2e_latency_ms': e2e_latency_s * 1000,  # Convert to ms
            }
    except Exception as e:
        print(f"Warning: Failed to load summary from {result_dir}: {e}")
        return None


def load_raw_data(result_dir: Path) -> Optional[Dict]:
    """Load raw data files for CDF plotting from CSV files
    
    CSV format: index,cdf,<Metric Name>
    The third column contains the actual metric values in seconds.
    """
    try:
        import csv
        data = {}
        
        # Load TTFT data from CSV (column: "Time to First Token")
        ttft_file = result_dir / 'ttft.csv'
        if ttft_file.exists():
            with open(ttft_file, 'r') as f:
                reader = csv.DictReader(f)
                ttft_values = []
                for row in reader:
                    # Get the third column (metric value in seconds)
                    metric_col = list(row.keys())[2] if len(row.keys()) > 2 else None
                    if metric_col and row[metric_col]:
                        ttft_values.append(float(row[metric_col]) * 1000)  # Convert to ms
                if ttft_values:
                    data['ttft'] = ttft_values
        
        # Load TPOT data from CSV (column: "Time per Output Token")
        tpot_file = result_dir / 'tpot.csv'
        if tpot_file.exists():
            with open(tpot_file, 'r') as f:
                reader = csv.DictReader(f)
                tpot_values = []
                for row in reader:
                    metric_col = list(row.keys())[2] if len(row.keys()) > 2 else None
                    if metric_col and row[metric_col]:
                        tpot_values.append(float(row[metric_col]) * 1000)  # Convert to ms
                if tpot_values:
                    data['tpot'] = tpot_values
        
        # Load TBOT data from CSV (column: "Time Between Tokens")
        tbot_file = result_dir / 'tbt.csv'  # TBT = Time Between Tokens
        if tbot_file.exists():
            with open(tbot_file, 'r') as f:
                reader = csv.DictReader(f)
                tbot_values = []
                for row in reader:
                    metric_col = list(row.keys())[2] if len(row.keys()) > 2 else None
                    if metric_col and row[metric_col]:
                        tbot_values.append(float(row[metric_col]) * 1000)  # Convert to ms
                if tbot_values:
                    data['tbot'] = tbot_values
        
        # Load E2E latency data from CSV (column: "End-to-End Latency")
        e2e_file = result_dir / 'end_to_end_latency.csv'
        if e2e_file.exists():
            with open(e2e_file, 'r') as f:
                reader = csv.DictReader(f)
                e2e_values = []
                for row in reader:
                    metric_col = list(row.keys())[2] if len(row.keys()) > 2 else None
                    if metric_col and row[metric_col]:
                        e2e_values.append(float(row[metric_col]) * 1000)  # Convert to ms
                if e2e_values:
                    data['e2e_latency'] = e2e_values
        
        return data if data else None
    except Exception as e:
        print(f"Warning: Failed to load raw data from {result_dir}: {e}")
        return None


def extract_metrics(summary: Dict) -> Dict:
    """Extract key metrics from summary"""
    metrics = {}
    
    # Request metrics
    if 'request_output_throughput_token_per_s' in summary:
        metrics['throughput'] = summary['request_output_throughput_token_per_s']
    
    if 'number_of_completed_requests' in summary:
        metrics['completed_requests'] = summary['number_of_completed_requests']
    
    # Latency metrics (mean)
    if 'request_mean_ttft_ms' in summary:
        metrics['ttft_mean'] = summary['request_mean_ttft_ms']
    
    if 'request_mean_tpot_ms' in summary:
        metrics['tpot_mean'] = summary['request_mean_tpot_ms']
    
    if 'request_mean_e2e_latency_ms' in summary:
        metrics['e2e_latency_mean'] = summary['request_mean_e2e_latency_ms']
    
    # Latency metrics (percentiles)
    if 'request_p50_ttft_ms' in summary:
        metrics['ttft_p50'] = summary['request_p50_ttft_ms']
    
    if 'request_p99_ttft_ms' in summary:
        metrics['ttft_p99'] = summary['request_p99_ttft_ms']
    
    if 'request_p50_tpot_ms' in summary:
        metrics['tpot_p50'] = summary['request_p50_tpot_ms']
    
    if 'request_p99_tpot_ms' in summary:
        metrics['tpot_p99'] = summary['request_p99_tpot_ms']
    
    return metrics


def plot_cdf_comparison(raw_data: Dict[str, Dict], output_dir: Path, mode_labels: Dict, colors: Dict):
    """Plot CDF comparisons for TTFT, TPOT, TBOT, and E2E latency"""
    
    metrics = {
        'ttft': 'Time to First Token (ms)',
        'tpot': 'Time per Output Token (ms)', 
        'tbot': 'Time Between Tokens (ms)',
        'e2e_latency': 'End-to-End Latency (ms)'
    }
    
    for metric_key, metric_name in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mode, data in raw_data.items():
            if metric_key in data:
                values = np.array(data[metric_key])
                # Sort values for CDF
                sorted_values = np.sort(values)
                # Calculate CDF (percentiles)
                percentiles = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
                
                label = mode_labels.get(mode, mode)
                color = colors.get(mode, None)
                ax.plot(sorted_values, percentiles, label=label, color=color, linewidth=2)
        
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_ylabel('Percentile (%)', fontsize=12)
        ax.set_title(f'{metric_name} - CDF Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)
        
        # Save plot as PDF
        output_file_pdf = output_dir / f'{metric_key}_cdf_comparison.pdf'
        plt.savefig(output_file_pdf, bbox_inches='tight')
        print(f"Saved: {output_file_pdf}")
        
        plt.close()


def plot_comparison(results: Dict[str, Dict], raw_data: Dict[str, Dict], output_dir: Path):
    """Create CDF comparison plots - one plot per metric with 2 subplots each"""
    
    # Define mode groups
    thinking_machine_modes = ['baseline_nondet', 'det_mode_1', 'det_mode_129', 'det_mode_65']
    cutlass_modes = ['baseline_nondet', 'det_mode_2', 'det_mode_130', 'det_mode_66']
    
    # Colors for different modes
    colors = {
        'baseline_nondet': '#e74c3c',
        'det_mode_1': '#3498db',
        'det_mode_2': '#9b59b6',
        'det_mode_129': '#2ecc71',
        'det_mode_130': '#1abc9c',
        'det_mode_65': '#f39c12',
        'det_mode_66': '#d35400',
    }
    
    mode_labels = {
        'baseline_nondet': 'Baseline (Non-Det)',
        'det_mode_1': 'Det Mode 1 (Full Det)',
        'det_mode_2': 'Det Mode 2 (Cutlass Det)',
        'det_mode_129': 'Det Mode 129 (Non-Det Attn)',
        'det_mode_130': 'Det Mode 130 (Cutlass:: Non-Det Attn)',
        'det_mode_65': 'Det Mode 65 (Non-Det RMSNorm)',
        'det_mode_66': 'Det Mode 66 (Cutlass:: Non-Det RMSNorm)',
    }
    
    line_styles = {
        'baseline_nondet': '-',
        'det_mode_1': '--',
        'det_mode_2': '-.',
        'det_mode_129': '-.',
        'det_mode_130': ':',
        'det_mode_65': ':',
        'det_mode_66': (0, (3, 1, 1, 1)),
    }
    
    def plot_cdf(ax, data_dict, metric_name, title, xlabel, modes_to_plot):
        """Plot CDF for a specific metric"""
        for mode in modes_to_plot:
            if mode in data_dict and metric_name in data_dict[mode]:
                data = np.array(data_dict[mode][metric_name])
                if len(data) > 0:
                    # Sort data
                    sorted_data = np.sort(data)
                    # Calculate CDF
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    ax.plot(sorted_data, y, 
                           label=mode_labels.get(mode, mode),
                           color=colors.get(mode, '#95a5a6'),
                           linestyle=line_styles.get(mode, '-'),
                           linewidth=2)
        
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=11)
        ax.set_ylabel('CDF', fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_ylim([0, 1.05])
    
    # Define metrics to plot
    metrics = [
        ('ttft', 'Time to First Token (TTFT)', 'Latency (ms)'),
        ('tpot', 'Time Per Output Token (TPOT)', 'Latency (ms)'),
        ('tbot', 'Time Between Output Tokens (TBOT)', 'Latency (ms)'),
        ('e2e_latency', 'End-to-End Latency', 'Latency (ms)')
    ]
    
    # Create one plot per metric, each with 2 subplots
    for metric_key, metric_name, xlabel in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'SGLang Deterministic Mode - {metric_name} Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Thinking Machine modes
        plot_cdf(ax1, raw_data, metric_key,
                 'Thinking Machine', xlabel, thinking_machine_modes)
        
        # 2. Cutlass-based modes
        plot_cdf(ax2, raw_data, metric_key,
                 'Cutlass-Based', xlabel, cutlass_modes)
        
        plt.tight_layout()
        
        # Save plot as PDF
        output_file_pdf = output_dir / f'{metric_key}_cdf_comparison.pdf'
        plt.savefig(output_file_pdf, bbox_inches='tight')
        print(f"✓ {metric_name} CDF plot saved to: {output_file_pdf}")
        
        plt.close()
    
    # Also create a summary bar chart for quick comparison
    if results:  # Only create bar charts if we have summary data
        plot_summary_bars(results, output_dir, mode_labels, colors)


def plot_summary_bars(results: Dict[str, Dict], output_dir: Path, mode_labels: Dict, colors: Dict):
    """Create summary bar charts for mean metrics"""
    
    modes = list(results.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SGLang Deterministic Mode Performance Summary (Mean Values)', fontsize=16, fontweight='bold')
    
    # 1. Throughput
    ax = axes[0, 0]
    throughputs = [results[mode]['throughput'] for mode in modes if 'throughput' in results[mode]]
    mode_names = [mode_labels.get(mode, mode) for mode in modes if 'throughput' in results[mode]]
    bars = ax.bar(range(len(throughputs)), throughputs, 
                   color=[colors.get(mode, '#95a5a6') for mode in modes if 'throughput' in results[mode]])
    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Tokens/Second', fontweight='bold')
    ax.set_title('Output Throughput', fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels(mode_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, throughputs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. TTFT Mean
    ax = axes[0, 1]
    ttft_means = [results[mode]['ttft_mean'] for mode in modes if 'ttft_mean' in results[mode]]
    mode_names = [mode_labels.get(mode, mode) for mode in modes if 'ttft_mean' in results[mode]]
    bars = ax.bar(range(len(ttft_means)), ttft_means,
                   color=[colors.get(mode, '#95a5a6') for mode in modes if 'ttft_mean' in results[mode]])
    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Milliseconds', fontweight='bold')
    ax.set_title('Time to First Token (Mean)', fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels(mode_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, ttft_means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. TPOT Mean
    ax = axes[1, 0]
    tpot_means = [results[mode]['tpot_mean'] for mode in modes if 'tpot_mean' in results[mode]]
    mode_names = [mode_labels.get(mode, mode) for mode in modes if 'tpot_mean' in results[mode]]
    bars = ax.bar(range(len(tpot_means)), tpot_means,
                   color=[colors.get(mode, '#95a5a6') for mode in modes if 'tpot_mean' in results[mode]])
    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Milliseconds', fontweight='bold')
    ax.set_title('Time Per Output Token (Mean)', fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels(mode_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, tpot_means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. E2E Latency Mean
    ax = axes[1, 1]
    e2e_means = [results[mode]['e2e_latency_mean'] for mode in modes if 'e2e_latency_mean' in results[mode]]
    mode_names = [mode_labels.get(mode, mode) for mode in modes if 'e2e_latency_mean' in results[mode]]
    bars = ax.bar(range(len(e2e_means)), e2e_means,
                   color=[colors.get(mode, '#95a5a6') for mode in modes if 'e2e_latency_mean' in results[mode]])
    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Milliseconds', fontweight='bold')
    ax.set_title('End-to-End Latency (Mean)', fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels(mode_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, e2e_means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file_pdf = output_dir / 'performance_summary.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Summary plot saved to: {output_file_pdf}")
    
    plt.close()


def print_summary_table(results: Dict[str, Dict]):
    """Print a summary table of results"""
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    
    print(f"\n{'Mode':<25} {'Throughput':<15} {'TTFT (ms)':<15} {'TPOT (ms)':<15}")
    print(f"{'':25} {'(tokens/s)':<15} {'Mean':<15} {'Mean':<15}")
    print("-"*80)
    
    for mode, metrics in results.items():
        throughput = metrics.get('throughput', 'N/A')
        ttft = metrics.get('ttft_mean', 'N/A')
        tpot = metrics.get('tpot_mean', 'N/A')
        
        throughput_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else throughput
        ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else ttft
        tpot_str = f"{tpot:.3f}" if isinstance(tpot, (int, float)) else tpot
        
        print(f"{mode:<25} {throughput_str:<15} {ttft_str:<15} {tpot_str:<15}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot etalon benchmark results')
    parser.add_argument('--input-dir', type=str, default='etalon_results_automated',
                        help='Input directory containing benchmark results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as input-dir)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    print(f"Loading results from: {input_dir}")
    
    # Load both summary and raw data
    results = {}
    raw_data = {}
    mode_dirs = ['baseline_nondet', 'det_mode_1', 'det_mode_129', 'det_mode_65', 'det_mode_2', 'det_mode_130', 'det_mode_66']
    
    for mode_dir in mode_dirs:
        mode_path = input_dir / mode_dir
        
        if mode_path.exists():
            print(f"Loading: {mode_dir}")
            
            # Load summary for bar charts
            summary = load_summary(mode_path)
            if summary:
                results[mode_dir] = extract_metrics(summary)
                print(f"  ✓ Loaded summary metrics")
            
            # Load raw data for CDFs
            raw = load_raw_data(mode_path)
            if raw:
                raw_data[mode_dir] = raw
                print(f"  ✓ Loaded {len(raw)} raw metrics")
            else:
                print(f"  ✗ No raw data found")
        else:
            print(f"Warning: Directory not found: {mode_dir}")
    
    if not raw_data:
        print("Error: No raw data found to plot")
        sys.exit(1)
    
    print(f"\nLoaded raw data for {len(raw_data)} modes")
    
    # Create comparison plots (4 plots, each with 2 subplots)
    print("\nCreating comparison plots...")
    plot_comparison(results, raw_data, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
