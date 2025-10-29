#!/usr/bin/env python3

"""
Plot Comprehensive Test Results
Compares performance metrics across:
- Baseline (Non-deterministic)
- Mode 66 (batch-invariant: vllm-rmsnorm + cutlass)
- Mode 257 (batch-invariant: native-rmsnorm + TM)
- Mode 578 with varying temperature percentages: 0%, 1%, 2%, 5%, 10%, 50%, 100%
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install matplotlib numpy")
    sys.exit(1)


def load_summary(result_dir: Path) -> Optional[Dict]:
    """Load performance summary from perf_metrics.csv"""
    try:
        import csv
        perf_metrics_path = result_dir / 'perf_metrics.csv'
        if not perf_metrics_path.exists():
            return None
        
        with open(perf_metrics_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) < 10:
                return None
            
            # Extract mean values (first column, index 0)
            ttft_s = float(rows[4][0])  # Row 4 = TTFT mean in seconds
            tpot_s = float(rows[5][0])  # Row 5 = TPOT mean in seconds
            e2e_latency_s = float(rows[7][0])  # Row 7 = E2E latency mean in seconds
            throughput = float(rows[9][0])  # Row 9 = Output throughput in tokens/sec
            
            return {
                'request_mean_ttft_ms': ttft_s * 1000,
                'request_mean_tpot_ms': tpot_s * 1000,
                'request_output_throughput_token_per_s': throughput,
                'request_mean_e2e_latency_ms': e2e_latency_s * 1000,
            }
    except Exception as e:
        print(f"Warning: Failed to load summary from {result_dir}: {e}")
        return None


def parse_run_name(run_name: str) -> Tuple[str, int]:
    """Parse run name to extract mode and temperature percentage
    
    Examples:
        baseline_10pct -> ('baseline', 10)
        mode66_10pct -> ('66', 10)
        mode578_5pct -> ('578', 5)
    """
    parts = run_name.split('_')
    if parts[0] == 'baseline':
        mode = 'baseline'
    else:
        # Extract mode number from 'modeXXX'
        mode = parts[0].replace('mode', '')
    
    # Extract percentage from 'XXpct'
    temp_pct = int(parts[1].replace('pct', ''))
    
    return mode, temp_pct


def get_mode_label(mode: str, temp_pct: int) -> str:
    """Get display label for a mode configuration"""
    if mode == 'baseline':
        return "Baseline (sglang-non-deter)"
    elif mode == '66':
        return "Ours-deter"
    elif mode == '257':
        return "Sglang-deter"
    elif mode == '578':
        return f"Ours-per-request-{temp_pct}pct"
    else:
        return f"Mode {mode} ({temp_pct}%)"


def get_mode_color(mode: str) -> str:
    """Get color for a mode"""
    color_map = {
        'baseline': '#FF6B6B',  # Red
        '66': '#4ECDC4',        # Teal
        '257': '#45B7D1',       # Blue
        '578': '#95E1D3',       # Light teal/green
    }
    return color_map.get(mode, '#999999')


def get_mode_marker(mode: str) -> str:
    """Get marker style for a mode"""
    marker_map = {
        'baseline': 'o',
        '66': 's',
        '257': '^',
        '578': 'D',
    }
    return marker_map.get(mode, 'x')


def plot_mode_578_progression(results: Dict[str, Dict], output_dir: Path):
    """Plot Mode 578 metrics as temperature percentage varies"""
    
    # Filter for mode 578 only
    mode_578_results = {}
    for run_name, data in results.items():
        mode, temp_pct = parse_run_name(run_name)
        if mode == '578':
            mode_578_results[temp_pct] = data
    
    if not mode_578_results:
        print("Warning: No Mode 578 results found for progression plot")
        return
    
    # Sort by temperature percentage
    temp_pcts = sorted(mode_578_results.keys())
    
    # Extract metrics
    ttft_values = [mode_578_results[pct]['request_mean_ttft_ms'] for pct in temp_pcts]
    tpot_values = [mode_578_results[pct]['request_mean_tpot_ms'] for pct in temp_pcts]
    e2e_values = [mode_578_results[pct]['request_mean_e2e_latency_ms'] for pct in temp_pcts]
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Ours-per-request: Performance vs Temperature=0 Percentage', fontsize=16, fontweight='bold')
    
    # TTFT
    axes[0].plot(temp_pcts, ttft_values, marker='D', linewidth=2, markersize=8, color=get_mode_color('578'))
    axes[0].set_xlabel('Temperature=0 Percentage (%)', fontsize=12)
    axes[0].set_ylabel('TTFT (ms)', fontsize=12)
    axes[0].set_title('Time to First Token', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(temp_pcts)
    
    # TPOT
    axes[1].plot(temp_pcts, tpot_values, marker='D', linewidth=2, markersize=8, color=get_mode_color('578'))
    axes[1].set_xlabel('Temperature=0 Percentage (%)', fontsize=12)
    axes[1].set_ylabel('TPOT (ms)', fontsize=12)
    axes[1].set_title('Time Per Output Token', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(temp_pcts)
    
    # E2E Latency
    axes[2].plot(temp_pcts, e2e_values, marker='D', linewidth=2, markersize=8, color=get_mode_color('578'))
    axes[2].set_xlabel('Temperature=0 Percentage (%)', fontsize=12)
    axes[2].set_ylabel('E2E Latency (ms)', fontsize=12)
    axes[2].set_title('End-to-End Latency', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(temp_pcts)
    
    plt.tight_layout()
    
    output_file = output_dir / 'mode_578_progression.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Mode 578 progression plot: {output_file}")
    plt.close()


def plot_mode_comparison(results: Dict[str, Dict], output_dir: Path):
    """Plot comparison across all modes at 100% temperature"""
    
    # Filter for 100% temperature results
    comparison_results = {}
    for run_name, data in results.items():
        mode, temp_pct = parse_run_name(run_name)
        if temp_pct == 100:
            comparison_results[mode] = data
    
    if not comparison_results:
        print("Warning: No 100% temperature results found for mode comparison")
        return
    
    # Desired mode order
    mode_order = ['baseline', '66', '257', '578']
    modes = [m for m in mode_order if m in comparison_results]
    
    if not modes:
        print("Warning: No valid modes found for comparison")
        return
    
    labels = [get_mode_label(m, 100) for m in modes]
    colors = [get_mode_color(m) for m in modes]
    
    # Extract metrics
    ttft_values = [comparison_results[m]['request_mean_ttft_ms'] for m in modes]
    tpot_values = [comparison_results[m]['request_mean_tpot_ms'] for m in modes]
    e2e_values = [comparison_results[m]['request_mean_e2e_latency_ms'] for m in modes]
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Mode Comparison at 100% Temperature=0', fontsize=16, fontweight='bold')
    
    x = np.arange(len(labels))
    width = 0.6
    
    # TTFT
    axes[0].bar(x, ttft_values, width, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('TTFT (ms)', fontsize=12)
    axes[0].set_title('Time to First Token', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(ttft_values):
        axes[0].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    # TPOT
    axes[1].bar(x, tpot_values, width, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('TPOT (ms)', fontsize=12)
    axes[1].set_title('Time Per Output Token', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(tpot_values):
        axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # E2E Latency
    axes[2].bar(x, e2e_values, width, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('E2E Latency (ms)', fontsize=12)
    axes[2].set_title('End-to-End Latency', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(e2e_values):
        axes[2].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = output_dir / 'mode_comparison_100pct.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mode comparison plot: {output_file}")
    plt.close()


def plot_all_modes_scatter(results: Dict[str, Dict], output_dir: Path):
    """Create scatter plots showing all modes and temperature percentages"""
    
    # Organize data by mode
    mode_data = {}
    for run_name, data in results.items():
        mode, temp_pct = parse_run_name(run_name)
        if mode not in mode_data:
            mode_data[mode] = {'temp_pcts': [], 'data': []}
        mode_data[mode]['temp_pcts'].append(temp_pct)
        mode_data[mode]['data'].append(data)
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('All Modes: Performance vs Temperature=0 Percentage', fontsize=16, fontweight='bold')
    
    # Plot each mode
    for mode, mode_info in mode_data.items():
        temp_pcts = mode_info['temp_pcts']
        data_list = mode_info['data']
        
        # Sort by temperature percentage
        sorted_indices = np.argsort(temp_pcts)
        temp_pcts_sorted = [temp_pcts[i] for i in sorted_indices]
        data_sorted = [data_list[i] for i in sorted_indices]
        
        # Extract metrics
        ttft_values = [d['request_mean_ttft_ms'] for d in data_sorted]
        tpot_values = [d['request_mean_tpot_ms'] for d in data_sorted]
        e2e_values = [d['request_mean_e2e_latency_ms'] for d in data_sorted]
        
        color = get_mode_color(mode)
        marker = get_mode_marker(mode)
        
        if mode == 'baseline':
            label = 'Baseline (sglang-non-deter)'
        elif mode == '66':
            label = 'Ours-deter'
        elif mode == '257':
            label = 'Sglang-deter'
        elif mode == '578':
            label = 'Ours-per-request'
        else:
            label = f'Mode {mode}'
        
        # TTFT
        axes[0].plot(temp_pcts_sorted, ttft_values, marker=marker, linewidth=2, 
                       markersize=8, color=color, label=label, alpha=0.8)
        
        # TPOT
        axes[1].plot(temp_pcts_sorted, tpot_values, marker=marker, linewidth=2,
                       markersize=8, color=color, label=label, alpha=0.8)
        
        # E2E Latency
        axes[2].plot(temp_pcts_sorted, e2e_values, marker=marker, linewidth=2,
                       markersize=8, color=color, label=label, alpha=0.8)
    
    # Configure TTFT subplot
    axes[0].set_xlabel('Temperature=0 Percentage (%)', fontsize=12)
    axes[0].set_ylabel('TTFT (ms)', fontsize=12)
    axes[0].set_title('Time to First Token', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    
    # Configure TPOT subplot
    axes[1].set_xlabel('Temperature=0 Percentage (%)', fontsize=12)
    axes[1].set_ylabel('TPOT (ms)', fontsize=12)
    axes[1].set_title('Time Per Output Token', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    
    # Configure E2E Latency subplot
    axes[2].set_xlabel('Temperature=0 Percentage (%)', fontsize=12)
    axes[2].set_ylabel('E2E Latency (ms)', fontsize=12)
    axes[2].set_title('End-to-End Latency', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    output_file = output_dir / 'all_modes_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved all modes comparison plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot comprehensive test results'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing test results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as input-dir)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Comprehensive Test Results Plotting")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all result directories
    result_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not result_dirs:
        print("Error: No result directories found")
        sys.exit(1)
    
    print(f"Found {len(result_dirs)} result directories")
    
    # Load all results
    results = {}
    for result_dir in result_dirs:
        run_name = result_dir.name
        summary = load_summary(result_dir)
        if summary:
            results[run_name] = summary
            mode, temp_pct = parse_run_name(run_name)
            print(f"  ✓ Loaded: {get_mode_label(mode, temp_pct)}")
        else:
            print(f"  ⚠ Failed to load: {run_name}")
    
    if not results:
        print("Error: No valid results found")
        sys.exit(1)
    
    print()
    print(f"Successfully loaded {len(results)} results")
    print()
    
    # Generate plots
    print("Generating plots...")
    print()
    
    # 1. Mode 578 progression
    print("1. Ours-per-request progression plot...")
    plot_mode_578_progression(results, output_dir)
    
    # 2. Mode comparison at 100%
    print("2. Mode comparison at 100% temperature...")
    plot_mode_comparison(results, output_dir)
    
    # 3. All modes scatter plot
    print("3. All modes comparison plot...")
    plot_all_modes_scatter(results, output_dir)
    
    print()
    print("=" * 60)
    print("Plotting completed!")
    print("=" * 60)
    print(f"Plots saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
