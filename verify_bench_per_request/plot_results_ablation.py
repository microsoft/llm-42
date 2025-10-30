#!/usr/bin/env python3

"""
Plot Mixed Temperature Test Results
Compares performance metrics across different deterministic modes with mixed temperatures:
- Baseline (Non-deterministic)
- Mode 66 (batch-invariant: vllm-rmsnorm + cutlass)
- Mode 257 (batch-invariant: native-rmsnorm + TM)
- Mode 578 (temperature-based switching)
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

# Colors for different modes
colors = {
    'baseline_nondet': '#e74c3c',      # Red - baseline
    'det_mode_66': '#3498db',          # Blue - mode 66
    'det_mode_257': '#9b59b6',         # Purple - mode 257
    'det_mode_578': '#2ecc71',         # Green - mode 578
}

mode_order = [
    'baseline_nondet',
    'det_mode_66',
    'det_mode_2',
    'det_mode_257',
    'det_mode_578'
]

sorted_configs = ['baseline_nondet@100%', 'det_mode_257@100%', 'det_mode_2@100%',
                  'det_mode_66@100%', 'det_mode_578@5%']

pct_names = ['pct_5', 'pct_100']

# Fixed colors for batch-invariant modes (darker shades)
fixed_colors = {
    'baseline': '#c0392b',  # Darker red
    '66': '#2471a3',        # Darker blue
    '257': '#7d3c98',       # Darker purple
    '2': '#ba4a00',         # Dark orange
}

# Diverse colors for mode 578 - using distinct colors for better differentiation
mode_578_colors = [
    '#1e8449',  # Dark green
    '#d68910',  # Dark gold
    '#641e16',  # Very dark red (100%)
]

def get_config_label(config: str) -> str:
    """Convert config key to display label"""
    if 'baseline_nondet' in config:
        return 'SGLang (Non-deterministic)'
    elif 'det_mode_66' in config:
        return 'DetInfer (GEMM + RMSNorm)'
    elif 'det_mode_257' in config:
        return 'SGLang (Deterministic)'
    elif 'det_mode_578' in config:
        # Extract percentage
        pct = config.split('@')[1].replace('%', '').strip()
        return f'DetInfer (GEMM  + RMSNorm + {pct}% deterministic)'
    elif 'det_mode_2' in config:
        return 'DetInfer (Only GEMM)'
    else:
        return config.replace('det_mode_', 'Mode ').replace('@', ' @ ')


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

def plot_unified_cdf_comparison(raw_data: Dict[str, Dict], output_dir: Path):
    """Create unified CDF plots comparing all modes across percentages"""

    # Assign colors and styles with darker colors for better visibility
    config_colors = {}
    config_styles = {}

    # Sort configs to assign 578 colors by percentage
    #sorted_configs = sorted(raw_data.keys())
    mode_578_configs = [c for c in sorted_configs if '578' in c and '@100%' not in c]

    mode_578_color_map = {}
    if len(mode_578_configs) > 0:
        for idx, config in enumerate(sorted(mode_578_configs, key=lambda x: int(x.split('@')[1].replace('%', '')))):
            mode_578_color_map[config] = mode_578_colors[idx]

    # Assign colors and styles
    for config in sorted_configs:
        # Assign colors
        if '@100%' in config:
            if 'baseline' in config:
                config_colors[config] = fixed_colors['baseline']
                config_styles[config] = '-'  # Solid
            elif '66' in config:
                config_colors[config] = fixed_colors['66']
                config_styles[config] = '--'  # Dashed
            elif '257' in config:
                config_colors[config] = fixed_colors['257']
                config_styles[config] = '-'  # Dash-dot
            elif '2' in config:
                config_colors[config] = fixed_colors['2']  # Dark orange for 100%
                config_styles[config] = ':'  # Solid
            elif '578' in config:
                config_colors[config] = mode_578_colors[-1]  # Dark orange for 100%
                config_styles[config] = '-'  # Solid
        else:
            config_colors[config] = mode_578_color_map.get(config, '#2ecc71')
            config_styles[config] = '-.'  # Dotted for mode 578 variants

    print(config_colors)

    # Metrics to plot
    metrics = [
        ('ttft', 'Time to First Token (TTFT)', 'TTFT (ms)'),
        ('tpot', 'Time Per Output Token (TPOT)', 'TPOT (ms)'),
        ('tbot', 'Time Between Tokens (TBoT)', 'TBoT (ms)'),
        ('e2e_latency', 'End-to-End Latency', 'E2E Latency (ms)'),
    ]

    for metric_key, metric_name, xlabel in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot each configuration
        for config in sorted_configs:
            if metric_key in raw_data[config]:
                data = np.array(raw_data[config][metric_key])
                if len(data) > 0:
                    sorted_data = np.sort(data)
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

                    ax.plot(sorted_data, y,
                           label=get_config_label(config),
                           color=config_colors[config],
                           linestyle=config_styles[config],
                           linewidth=1.5)

        ax.set_xlabel(xlabel, fontweight='bold', fontsize=16)
        ax.set_ylabel('CDF', fontweight='bold', fontsize=16)
        #ax.set_title(f'Unified Comparison - {metric_name} CDF', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=14, ncol=1)
        ax.set_ylim([0, 1.05])

        # set tick font sizes for x and y axes
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()

        output_file_pdf = output_dir / f'unified_{metric_key}_cdf.png'
        plt.savefig(output_file_pdf, bbox_inches='tight', dpi=150)
        print(f"  ✓ Saved: {output_file_pdf.name}")

        plt.close()


def plot_unified_bar_comparison(results: Dict[str, Dict], output_dir: Path):
    """Create unified bar chart comparing all modes"""

    # Sort configurations: batch-invariant first, then mode 578 by percentage
    def sort_key(config):
        if '@100%' in config:
            if 'baseline' in config:
                return (0, 0)
            elif '66' in config:
                return (0, 1)
            elif '257' in config:
                return (0, 2)
            elif '578' in config:
                return (1, 100)  # Mode 578 @ 100%
            else:
                return (2, 0)  # Unknown config
        else:
            # Extract percentage number for mode 578
            pct = int(config.split('@')[1].replace('%', ''))
            return (1, pct)

    sorted_configs = sorted(results.keys(), key=sort_key)

    # Generate colors with darker shades for better visibility
    fixed_colors = {
        'baseline': '#c0392b',  # Darker red
        '66': '#2471a3',        # Darker blue
        '257': '#7d3c98',       # Darker purple
    }

    # Diverse colors for mode 578 - using distinct colors for better differentiation
    mode_578_colors = [
        '#117864',  # Dark teal (0%)
        '#1e8449',  # Dark green
        '#d68910',  # Dark gold
        '#ca6f1e',  # Dark orange
        '#ba4a00',  # Very dark orange
        '#922b21',  # Dark crimson
        '#641e16',  # Very dark red (100%)
    ]

    # Build color list
    colors_list = []
    mode_578_configs = [c for c in sorted_configs if '578' in c and '@100%' not in c]
    mode_578_sorted = sorted(mode_578_configs, key=lambda x: int(x.split('@')[1].replace('%', '')))

    for config in sorted_configs:
        if '@100%' in config:
            if 'baseline' in config:
                colors_list.append(fixed_colors['baseline'])
            elif '66' in config:
                colors_list.append(fixed_colors['66'])
            elif '257' in config:
                colors_list.append(fixed_colors['257'])
            elif '578' in config:
                colors_list.append(mode_578_colors[-1])  # Dark orange for 100%
            else:
                colors_list.append('#95a5a6')  # Gray for unknown
        else:
            # Assign gradient color based on position
            idx = mode_578_sorted.index(config)
            color_idx = int(idx * (len(mode_578_colors) - 1) / max(1, len(mode_578_sorted) - 1))
            colors_list.append(mode_578_colors[color_idx])

    # Prepare labels using the proper naming convention
    config_labels = [get_config_label(c).replace(' ', '\n', 1) for c in sorted_configs]

    # Extract metrics
    throughput_vals = [results[c].get('throughput', 0) for c in sorted_configs]
    ttft_vals = [results[c].get('ttft_mean', 0) for c in sorted_configs]
    tpot_vals = [results[c].get('tpot_mean', 0) for c in sorted_configs]
    e2e_vals = [results[c].get('e2e_mean', 0) for c in sorted_configs]

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Unified Comparison - Performance Summary (All Configurations)',
                 fontsize=16, fontweight='bold')

    x_pos = np.arange(len(sorted_configs))
    bar_width = 0.8

    # 1. Throughput
    bars1 = ax1.bar(x_pos, throughput_vals, bar_width, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Throughput (tokens/s)', fontweight='bold', fontsize=11)
    ax1.set_title('Output Throughput', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_labels, fontsize=8, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, throughput_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # 2. TTFT
    bars2 = ax2.bar(x_pos, ttft_vals, bar_width, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('TTFT (ms)', fontweight='bold', fontsize=11)
    ax2.set_title('Time to First Token', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(config_labels, fontsize=8, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, ttft_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # 3. TPOT
    bars3 = ax3.bar(x_pos, tpot_vals, bar_width, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('TPOT (ms)', fontweight='bold', fontsize=11)
    ax3.set_title('Time Per Output Token', fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(config_labels, fontsize=8, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, tpot_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # 4. E2E Latency
    bars4 = ax4.bar(x_pos, e2e_vals, bar_width, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('E2E Latency (ms)', fontweight='bold', fontsize=11)
    ax4.set_title('End-to-End Latency', fontweight='bold', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(config_labels, fontsize=8, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, e2e_vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    output_file = output_dir / 'unified_performance_summary.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"  ✓ Saved: {output_file.name}")

    plt.close()


def plot_unified_overhead_comparison(results: Dict[str, Dict], output_dir: Path):
    """Create overhead comparison plot relative to baseline"""

    # Find baseline
    baseline_key = None
    for key in results.keys():
        if 'baseline' in key:
            baseline_key = key
            break

    if not baseline_key:
        print("  ⚠ No baseline found, skipping overhead plot")
        return

    baseline_throughput = results[baseline_key].get('throughput', 0)
    if baseline_throughput == 0:
        print("  ⚠ Baseline throughput is 0, skipping overhead plot")
        return

    # Calculate overheads
    configs = []
    overheads = []
    colors_list = []

    # Diverse colors for mode 578 - using distinct colors for better differentiation
    mode_578_colors = [
        '#117864',  # Dark teal (0%)
        '#1e8449',  # Dark green
        '#d68910',  # Dark gold
        '#ca6f1e',  # Dark orange
        '#ba4a00',  # Very dark orange
        '#922b21',  # Dark crimson
        '#641e16',  # Very dark red (100%)
    ]

    # First pass: collect mode 578 configs
    mode_578_configs = []
    for config in sorted(results.keys()):
        if config != baseline_key and '578' in config and '@100%' not in config:
            mode_578_configs.append(config)
    mode_578_sorted = sorted(mode_578_configs, key=lambda x: int(x.split('@')[1].replace('%', '')))

    # Second pass: build data and colors
    for config in sorted(results.keys()):
        if config == baseline_key:
            continue

        config_throughput = results[config].get('throughput', 0)
        overhead = ((baseline_throughput - config_throughput) / baseline_throughput) * 100

        configs.append(get_config_label(config).replace(' ', '\n', 1))
        overheads.append(overhead)

        if '@100%' in config:
            if '66' in config:
                colors_list.append('#2471a3')  # Darker blue
            elif '257' in config:
                colors_list.append('#7d3c98')  # Darker purple
            elif '578' in config:
                colors_list.append(mode_578_colors[-1])  # Very dark orange for 100%
            else:
                colors_list.append('#7f8c8d')  # Darker gray
        else:
            # Assign gradient color based on position
            idx = mode_578_sorted.index(config)
            color_idx = int(idx * (len(mode_578_colors) - 1) / max(1, len(mode_578_sorted) - 1))
            colors_list.append(mode_578_colors[color_idx])

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, overheads, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Baseline')
    ax.set_ylabel('Throughput Overhead (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'Unified Overhead Analysis (Relative to {baseline_key.replace("@", " @ ")})',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=9, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, overheads):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)

    plt.tight_layout()

    output_file = output_dir / 'unified_overhead_comparison.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"  ✓ Saved: {output_file.name}")

    plt.close()


def print_unified_summary_table(results: Dict[str, Dict]):
    """Print unified summary table for all configurations"""

    print("\n" + "="*120)
    print("UNIFIED SUMMARY TABLE - ALL CONFIGURATIONS")
    print("="*120)

    print(f"\n{'Configuration':<40} {'Throughput':<15} {'TTFT (ms)':<15} {'TPOT (ms)':<15} {'E2E (ms)':<15}")
    print(f"{'':40} {'(tokens/s)':<15} {'Mean':<15} {'Mean':<15} {'Mean':<15}")
    print("-"*120)

    # Sort configurations
    def sort_key(config):
        if '@100%' in config:
            if 'baseline' in config:
                return (0, 0)
            elif '66' in config:
                return (0, 1)
            elif '257' in config:
                return (0, 2)
            elif '578' in config:
                return (1, 100)  # Mode 578 @ 100%
            else:
                return (2, 0)  # Unknown config
        else:
            pct = int(config.split('@')[1].replace('%', ''))
            return (1, pct)

    for config in sorted(results.keys(), key=sort_key):
        metrics = results[config]

        label = get_config_label(config)
        throughput = metrics.get('throughput', 'N/A')
        ttft = metrics.get('ttft_mean', 'N/A')
        tpot = metrics.get('tpot_mean', 'N/A')
        e2e = metrics.get('e2e_mean', 'N/A')

        throughput_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else throughput
        ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else ttft
        tpot_str = f"{tpot:.3f}" if isinstance(tpot, (int, float)) else tpot
        e2e_str = f"{e2e:.1f}" if isinstance(e2e, (int, float)) else e2e

        print(f"{label:<40} {throughput_str:<15} {ttft_str:<15} {tpot_str:<15} {e2e_str:<15}")

    print("="*120)

    # Print overhead analysis
    baseline_key = None
    for key in results.keys():
        if 'baseline' in key:
            baseline_key = key
            break

    if baseline_key:
        print("\nOverhead Analysis (relative to baseline):")
        print("-"*120)
        baseline_throughput = results[baseline_key].get('throughput', 0)

        for config in sorted(results.keys(), key=sort_key):
            if config == baseline_key:
                continue

            label = get_config_label(config)
            config_throughput = results[config].get('throughput', 0)

            if baseline_throughput > 0:
                overhead = ((baseline_throughput - config_throughput) / baseline_throughput) * 100
                print(f"{label:<40} Throughput overhead: {overhead:>6.2f}%")

        print("="*120)

    print()


def main():
    parser = argparse.ArgumentParser(description='Plot mixed temperature benchmark results')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing benchmark results (e.g., llama-3.1-8b_arxiv_pct_0_1_5_10)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as input-dir)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path('ablation')

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    print(f"Loading results from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Detect if this is new multi-percentage format or old single-percentage format
    # New format: llama-3.1-8b_arxiv_pct_0_1_5_10/pct_0/baseline_nondet/
    # Old format: pct_10_random/baseline_nondet/

    mode_dirs = mode_order#['baseline_nondet', 'det_mode_66', 'det_mode_257', 'det_mode_578']

    # Check if we have pct_X subdirectories (new format)
    pct_subdirs =  [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('pct_') and d.name in pct_names]

    if pct_subdirs:
        # New multi-percentage format
        print("Detected multi-percentage format")
        print(f"Found {len(pct_subdirs)} percentage directories")
        print()
        # Second, create unified cross-comparison plots
        print("\n" + "="*100)
        print("STEP 2: Creating unified cross-comparison plots")
        print("="*100)
        print()

        # Load batch-invariant modes from pct_100
        pct_100_dir = input_dir / 'pct_100'
        unified_results = {}
        unified_raw_data = {}

        if pct_100_dir.exists():
            print("Loading batch-invariant modes from pct_100...")
            for mode_dir in ['baseline_nondet', 'det_mode_66', 'det_mode_257', 'det_mode_2']:
                mode_path = pct_100_dir / mode_dir
                if mode_path.exists():
                    # Use mode name with @100% suffix for clarity
                    unified_key = f"{mode_dir}@100%"

                    summary = load_summary(mode_path)
                    if summary:
                        unified_results[unified_key] = extract_metrics(summary)
                        print(f"  ✓ Loaded {mode_dir} @ 100%")

                    raw = load_raw_data(mode_path)
                    if raw:
                        unified_raw_data[unified_key] = raw

        # Load mode 578 from all percentage directories
        print("\nLoading mode 578 from all percentages...")
        mode_578_dirs = [d for d in pct_subdirs if (d / 'det_mode_578').exists()]

        for pct_dir in sorted(mode_578_dirs):
            pct_name = pct_dir.name
            pct_value = pct_name.replace('pct_', '')
            mode_path = pct_dir / 'det_mode_578'

            if mode_path.exists():
                unified_key = f"det_mode_578@{pct_value}%"

                summary = load_summary(mode_path)
                if summary:
                    unified_results[unified_key] = extract_metrics(summary)
                    print(f"  ✓ Loaded mode 578 @ {pct_value}%")

                raw = load_raw_data(mode_path)
                if raw:
                    unified_raw_data[unified_key] = raw

        # Create unified plots
        if unified_results or unified_raw_data:
            print(f"\nCreating unified comparison plots...")
            print(f"Total configurations loaded: {len(unified_results)}")

            # Create unified output directory
            unified_output_dir = output_dir / 'unified_comparison'
            unified_output_dir.mkdir(parents=True, exist_ok=True)

            if unified_raw_data:
                print("\n1. Creating unified CDF comparison plots...")
                plot_unified_cdf_comparison(unified_raw_data, unified_output_dir)

            if unified_results:
                print("\n2. Creating unified bar chart comparison...")
                plot_unified_bar_comparison(unified_results, unified_output_dir)

                print("\n3. Creating unified overhead comparison...")
                plot_unified_overhead_comparison(unified_results, unified_output_dir)

                print("\n4. Printing unified summary table...")
                print_unified_summary_table(unified_results)

            print(f"\n✓ Unified plots created in: {unified_output_dir}")
        else:
            print("\nWarning: No data loaded for unified comparison")

        print("\n" + "="*100)
        print("✓ All plots created successfully!")
        print("="*100)
        print(f"\nPlots saved to: {output_dir}")
        print("\nGenerated files:")
        print("\nPer-percentage plots (in pct_X/ subdirectories):")
        print("  - ttft_cdf_comparison.png")
        print("  - tpot_cdf_comparison.png")
        print("  - tbot_cdf_comparison.png")
        print("  - e2e_latency_cdf_comparison.png")
        print("  - performance_summary.png")
        print("  - overhead_comparison.png")
        print("\nUnified comparison plots (in unified_comparison/ subdirectory):")
        print("  - unified_ttft_cdf.png")
        print("  - unified_tpot_cdf.png")
        print("  - unified_tbot_cdf.png")
        print("  - unified_e2e_latency_cdf.png")
        print("  - unified_performance_summary.png")
        print("  - unified_overhead_comparison.png")
        print("\n💡 The unified plots show all configurations together:")
        print("   - Baseline, Mode 66, Mode 257 @ 100%")
        print("   - Mode 578 @ all tested percentages")
        print()

    else:
        # Old single-percentage format
        print("Detected single-percentage format (legacy)")
        print()

        results = {}
        raw_data = {}

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
        print("  - ttft_cdf_comparison.png")
        print("  - tpot_cdf_comparison.png")
        print("  - tbot_cdf_comparison.png")
        print("  - e2e_latency_cdf_comparison.png")
        print("  - performance_summary.png")
        print("  - overhead_comparison.png")
        print()


if __name__ == '__main__':
    main()
