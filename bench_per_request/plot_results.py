#!/usr/bin/env python3

"""
Plot Component Test Results
Compares performance metrics across different component deterministic modes:
- Baseline (Non-deterministic)
- Matmul Deterministic Only (TM)
- Matmul Deterministic Only (Cutlass)
- Attention Deterministic Only
- RMSNorm Deterministic Only
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
        
        # Load TBOT data from CSV (column: "Time Between Output Tokens")
        tbot_file = result_dir / 'tbt.csv'
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
    return {
        'throughput': summary.get('request_output_throughput_token_per_s'),
        'ttft_mean': summary.get('request_mean_ttft_ms'),
        'tpot_mean': summary.get('request_mean_tpot_ms'),
        'e2e_mean': summary.get('request_mean_e2e_latency_ms'),
    }


def plot_cdf_comparison(raw_data: Dict[str, Dict], output_dir: Path):
    """Create CDF comparison plots for each metric"""
    
    # Component mode definitions
    mode_labels = {
        'baseline_nondet': 'Baseline (Non-Det)',
        'det_mode_66': 'Mode 66 (BI: vllm-rms + cutlass)',
        'det_mode_257': 'Mode 257 (BI: native-rms + TM)',
        'det_mode_578_temp0_1pct': 'Mode 578 (1% temp=0)',
        'det_mode_578_temp0_2pct': 'Mode 578 (2% temp=0)',
        'det_mode_578_temp0_5pct': 'Mode 578 (5% temp=0)',
        'det_mode_578_temp0_10pct': 'Mode 578 (10% temp=0)',
        'det_mode_578_temp0_50pct': 'Mode 578 (50% temp=0)',
        'det_mode_578_temp0_100pct': 'Mode 578 (100% temp=0)',
    }
    
    # Colors for different component tests
    colors = {
        'baseline_nondet': '#e74c3c',      # Red - baseline
        'det_mode_66': '#3498db',          # Blue - mode 66
        'det_mode_257': '#9b59b6',         # Purple - mode 257
        'det_mode_578_temp0_1pct': '#95a5a6',    # Light gray
        'det_mode_578_temp0_2pct': '#7f8c8d',    # Gray
        'det_mode_578_temp0_5pct': '#2ecc71',    # Green
        'det_mode_578_temp0_10pct': '#27ae60',   # Dark green
        'det_mode_578_temp0_50pct': '#f39c12',   # Orange
        'det_mode_578_temp0_100pct': '#e67e22',  # Dark orange
    }
    
    line_styles = {
        'baseline_nondet': '-',
        'det_mode_66': '--',
        'det_mode_257': '-.',
        'det_mode_578_temp0_1pct': ':',
        'det_mode_578_temp0_2pct': (0, (3, 1, 1, 1)),
        'det_mode_578_temp0_5pct': (0, (5, 2, 1, 2)),
        'det_mode_578_temp0_10pct': '--',
        'det_mode_578_temp0_50pct': '-.',
        'det_mode_578_temp0_100pct': '-',
    }
    
    # Define metrics to plot
    metrics = [
        ('ttft', 'Time to First Token (TTFT)', 'Latency (ms)'),
        ('tpot', 'Time Per Output Token (TPOT)', 'Latency (ms)'),
        ('tbot', 'Time Between Output Tokens (TBOT)', 'Latency (ms)'),
        ('e2e_latency', 'End-to-End Latency', 'Latency (ms)')
    ]
    
    mode_order = ['baseline_nondet', 'det_mode_66', 'det_mode_257',
                  'det_mode_578_temp0_1pct', 'det_mode_578_temp0_2pct', 'det_mode_578_temp0_5pct',
<<<<<<< HEAD
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct'
                  ]
=======
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct']
>>>>>>> 4f54bab06 (per request -- cuda graphs is pending, so decode is not working correctly)
    
    # Create one plot per metric
    for metric_key, metric_name, xlabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for mode in mode_order:
            if mode in raw_data and metric_key in raw_data[mode]:
                data = np.array(raw_data[mode][metric_key])
                if len(data) > 0:
                    # Sort data
                    sorted_data = np.sort(data)
                    # Calculate CDF
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    ax.plot(sorted_data, y, 
                           label=mode_labels.get(mode, mode),
                           color=colors.get(mode, '#95a5a6'),
                           linestyle=line_styles.get(mode, '-'),
                           linewidth=2.5)
        
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=12)
        ax.set_ylabel('CDF', fontweight='bold', fontsize=12)
        ax.set_title(f'Component Test - {metric_name} CDF', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot as PDF
        output_file_pdf = output_dir / f'{metric_key}_cdf_component_comparison.pdf'
        plt.savefig(output_file_pdf, bbox_inches='tight', dpi=150)
        print(f"✓ Saved: {output_file_pdf}")
        
        plt.close()


def plot_bar_comparison(results: Dict[str, Dict], output_dir: Path):
    """Create bar chart comparison of mean metrics"""
    
    mode_labels = {
        'baseline_nondet': 'Baseline\n(Non-Det)',
        'det_mode_66': 'Mode 66\n(BI: vllm+cutlass)',
        'det_mode_257': 'Mode 257\n(BI: native+TM)',
        'det_mode_578_temp0_1pct': 'Mode 578\n(1% temp=0)',
        'det_mode_578_temp0_2pct': 'Mode 578\n(2% temp=0)',
        'det_mode_578_temp0_5pct': 'Mode 578\n(5% temp=0)',
        'det_mode_578_temp0_10pct': 'Mode 578\n(10% temp=0)',
        'det_mode_578_temp0_50pct': 'Mode 578\n(50% temp=0)',
        'det_mode_578_temp0_100pct': 'Mode 578\n(100% temp=0)',
    }
    
    colors = {
        'baseline_nondet': '#e74c3c',
        'det_mode_66': '#3498db',
        'det_mode_257': '#9b59b6',
        'det_mode_578_temp0_1pct': '#95a5a6',
        'det_mode_578_temp0_2pct': '#7f8c8d',
        'det_mode_578_temp0_5pct': '#2ecc71',
        'det_mode_578_temp0_10pct': '#27ae60',
        'det_mode_578_temp0_50pct': '#f39c12',
        'det_mode_578_temp0_100pct': '#e67e22',
    }
    
    mode_order = ['baseline_nondet', 'det_mode_66', 'det_mode_257',
                  'det_mode_578_temp0_1pct', 'det_mode_578_temp0_2pct', 'det_mode_578_temp0_5pct',
<<<<<<< HEAD
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct'
                  ]
=======
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct']
>>>>>>> 4f54bab06 (per request -- cuda graphs is pending, so decode is not working correctly)
    
    # Filter to available modes and prepare data
    available_modes = [m for m in mode_order if m in results]
    mode_names = [mode_labels[m] for m in available_modes]
    bar_colors = [colors[m] for m in available_modes]
    
    # Extract metrics
    throughput_vals = [results[m].get('throughput', 0) for m in available_modes]
    ttft_vals = [results[m].get('ttft_mean', 0) for m in available_modes]
    tpot_vals = [results[m].get('tpot_mean', 0) for m in available_modes]
    e2e_vals = [results[m].get('e2e_mean', 0) for m in available_modes]
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Component Test - Performance Summary (Mean Values)', 
                 fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(available_modes))
    
    # 1. Throughput
    bars1 = ax1.bar(x_pos, throughput_vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Throughput (tokens/s)', fontweight='bold', fontsize=11)
    ax1.set_title('Output Throughput', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(mode_names, fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, throughput_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. TTFT
    bars2 = ax2.bar(x_pos, ttft_vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('TTFT (ms)', fontweight='bold', fontsize=11)
    ax2.set_title('Time to First Token', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(mode_names, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, ttft_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. TPOT
    bars3 = ax3.bar(x_pos, tpot_vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('TPOT (ms)', fontweight='bold', fontsize=11)
    ax3.set_title('Time Per Output Token', fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(mode_names, fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, tpot_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. E2E Latency
    bars4 = ax4.bar(x_pos, e2e_vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('E2E Latency (ms)', fontweight='bold', fontsize=11)
    ax4.set_title('End-to-End Latency', fontweight='bold', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(mode_names, fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, e2e_vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file_pdf = output_dir / 'component_performance_summary.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', dpi=150)
    print(f"✓ Summary plot saved to: {output_file_pdf}")
    
    plt.close()


def plot_overhead_comparison(results: Dict[str, Dict], output_dir: Path):
    """Create overhead comparison chart relative to baseline"""
    
    if 'baseline_nondet' not in results:
        print("Warning: Baseline results not found, skipping overhead plot")
        return
    
    baseline = results['baseline_nondet']
    baseline_throughput = baseline.get('throughput', 1)
    
    mode_labels = {
        'det_mode_66': 'Mode 66\n(BI: vllm+cutlass)',
        'det_mode_257': 'Mode 257\n(BI: native+TM)',
        'det_mode_578_temp0_1pct': 'Mode 578\n(1% temp=0)',
        'det_mode_578_temp0_2pct': 'Mode 578\n(2% temp=0)',
        'det_mode_578_temp0_5pct': 'Mode 578\n(5% temp=0)',
        'det_mode_578_temp0_10pct': 'Mode 578\n(10% temp=0)',
        'det_mode_578_temp0_50pct': 'Mode 578\n(50% temp=0)',
        'det_mode_578_temp0_100pct': 'Mode 578\n(100% temp=0)',
    }
    
    colors = {
        'det_mode_66': '#3498db',
        'det_mode_257': '#9b59b6',
        'det_mode_578_temp0_1pct': '#95a5a6',
        'det_mode_578_temp0_2pct': '#7f8c8d',
        'det_mode_578_temp0_5pct': '#2ecc71',
        'det_mode_578_temp0_10pct': '#27ae60',
        'det_mode_578_temp0_50pct': '#f39c12',
        'det_mode_578_temp0_100pct': '#e67e22',
    }
    
    mode_order = ['det_mode_66', 'det_mode_257',
                  'det_mode_578_temp0_1pct', 'det_mode_578_temp0_2pct', 'det_mode_578_temp0_5pct',
<<<<<<< HEAD
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct'
                  ]
=======
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct']
>>>>>>> 4f54bab06 (per request -- cuda graphs is pending, so decode is not working correctly)
    available_modes = [m for m in mode_order if m in results]
    
    # Calculate overhead percentages
    throughput_overhead = []
    for mode in available_modes:
        mode_throughput = results[mode].get('throughput', 0)
        if baseline_throughput > 0:
            overhead = ((baseline_throughput - mode_throughput) / baseline_throughput) * 100
            throughput_overhead.append(overhead)
        else:
            throughput_overhead.append(0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x_pos = np.arange(len(available_modes))
    mode_names = [mode_labels[m] for m in available_modes]
    bar_colors = [colors[m] for m in available_modes]
    
    bars = ax.bar(x_pos, throughput_overhead, color=bar_colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Throughput Overhead (%)', fontweight='bold', fontsize=12)
    ax.set_title('Component Deterministic Overhead vs Baseline\n(Lower is Better)', 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_names, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Baseline (0% overhead)')
    
    # Add value labels on bars
    for bar, val in zip(bars, throughput_overhead):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    output_file_pdf = output_dir / 'component_overhead_comparison.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', dpi=150)
    print(f"✓ Overhead plot saved to: {output_file_pdf}")
    
    plt.close()


def print_summary_table(results: Dict[str, Dict]):
    """Print a summary table of results"""
    print("\n" + "="*100)
    print("Component Test Performance Summary")
    print("="*100)
    
    print(f"\n{'Component Mode':<30} {'Throughput':<15} {'TTFT (ms)':<15} {'TPOT (ms)':<15} {'E2E (ms)':<15}")
    print(f"{'':30} {'(tokens/s)':<15} {'Mean':<15} {'Mean':<15} {'Mean':<15}")
    print("-"*100)
    
    mode_labels = {
        'baseline_nondet': 'Baseline (Non-Det)',
        'det_mode_66': 'Mode 66 (BI: vllm-rms + cutlass)',
        'det_mode_257': 'Mode 257 (BI: native-rms + TM)',
        'det_mode_578_temp0_1pct': 'Mode 578 (1% temp=0)',
        'det_mode_578_temp0_2pct': 'Mode 578 (2% temp=0)',
        'det_mode_578_temp0_5pct': 'Mode 578 (5% temp=0)',
        'det_mode_578_temp0_10pct': 'Mode 578 (10% temp=0)',
        'det_mode_578_temp0_50pct': 'Mode 578 (50% temp=0)',
        'det_mode_578_temp0_100pct': 'Mode 578 (100% temp=0)',
    }
    
    mode_order = ['baseline_nondet', 'det_mode_66', 'det_mode_257',
                  'det_mode_578_temp0_1pct', 'det_mode_578_temp0_2pct', 'det_mode_578_temp0_5pct',
<<<<<<< HEAD
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct'
                  ]
=======
                  'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct']
>>>>>>> 4f54bab06 (per request -- cuda graphs is pending, so decode is not working correctly)
    
    for mode in mode_order:
        if mode not in results:
            continue
        
        metrics = results[mode]
        label = mode_labels.get(mode, mode)
        throughput = metrics.get('throughput', 'N/A')
        ttft = metrics.get('ttft_mean', 'N/A')
        tpot = metrics.get('tpot_mean', 'N/A')
        e2e = metrics.get('e2e_mean', 'N/A')
        
        throughput_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else throughput
        ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else ttft
        tpot_str = f"{tpot:.3f}" if isinstance(tpot, (int, float)) else tpot
        e2e_str = f"{e2e:.1f}" if isinstance(e2e, (int, float)) else e2e
        
        print(f"{label:<30} {throughput_str:<15} {ttft_str:<15} {tpot_str:<15} {e2e_str:<15}")
    
    print("="*100)
    
    # Print overhead analysis if baseline exists
    if 'baseline_nondet' in results:
        print("\nOverhead Analysis (relative to baseline):")
        print("-"*100)
        baseline_throughput = results['baseline_nondet'].get('throughput', 0)
        
        for mode in ['det_mode_66', 'det_mode_257',
                     'det_mode_578_temp0_1pct', 'det_mode_578_temp0_2pct', 'det_mode_578_temp0_5pct',
<<<<<<< HEAD
                     'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct'
                    ]:
=======
                     'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct']:
>>>>>>> 4f54bab06 (per request -- cuda graphs is pending, so decode is not working correctly)
            if mode not in results:
                continue
            
            label = mode_labels.get(mode, mode)
            mode_throughput = results[mode].get('throughput', 0)
            
            if baseline_throughput > 0:
                overhead = ((baseline_throughput - mode_throughput) / baseline_throughput) * 100
                print(f"{label:<30} Throughput overhead: {overhead:>6.2f}%")
        
        print("="*100)
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Plot component test benchmark results')
    parser.add_argument('--input-dir', type=str, default='etalon_results_component_tests',
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
    print(f"Output directory: {output_dir}")
    print()
    
    # Load both summary and raw data
    results = {}
    raw_data = {}
    mode_dirs = ['baseline_nondet', 'det_mode_66', 'det_mode_257',
                 'det_mode_578_temp0_1pct', 'det_mode_578_temp0_2pct', 'det_mode_578_temp0_5pct',
<<<<<<< HEAD
                 'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct'
                 ]
=======
                 'det_mode_578_temp0_10pct', 'det_mode_578_temp0_50pct', 'det_mode_578_temp0_100pct']
>>>>>>> 4f54bab06 (per request -- cuda graphs is pending, so decode is not working correctly)
    
    for mode_dir in mode_dirs:
        mode_path = input_dir / mode_dir
        
        if mode_path.exists():
            print(f"Loading: {mode_dir}")
            
            # Load summary for bar charts
            summary = load_summary(mode_path)
            if summary:
                results[mode_dir] = extract_metrics(summary)
                print(f"  ✓ Loaded summary metrics")
            else:
                print(f"  ✗ No summary found")
            
            # Load raw data for CDFs
            raw = load_raw_data(mode_path)
            if raw:
                raw_data[mode_dir] = raw
                print(f"  ✓ Loaded {len(raw)} raw metrics")
            else:
                print(f"  ✗ No raw data found")
        else:
            print(f"Warning: Directory not found: {mode_dir}")
    
    if not results and not raw_data:
        print("\nError: No data found to plot")
        sys.exit(1)
    
    print(f"\nLoaded data for {len(results)} modes")
    
    # Print summary table
    if results:
        print_summary_table(results)
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("Creating plots...")
    
    if raw_data:
        print("\n1. Creating CDF comparison plots...")
        plot_cdf_comparison(raw_data, output_dir)
    
    if results:
        print("\n2. Creating bar chart comparison...")
        plot_bar_comparison(results, output_dir)
        
        print("\n3. Creating overhead comparison...")
        plot_overhead_comparison(results, output_dir)
    
    print("\n" + "="*100)
    print("✓ All plots created successfully!")
    print("="*100)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - ttft_cdf_component_comparison.pdf")
    print("  - tpot_cdf_component_comparison.pdf")
    print("  - tbot_cdf_component_comparison.pdf")
    print("  - e2e_latency_cdf_component_comparison.pdf")
    print("  - component_performance_summary.pdf")
    print("  - component_overhead_comparison.pdf")
    print()


if __name__ == '__main__':
    main()
