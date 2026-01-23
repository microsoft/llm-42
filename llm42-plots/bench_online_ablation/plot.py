#!/usr/bin/env python3
"""
Plot heatmaps for DetInfer Matrix Ablation: Window Size vs Batch Size.

Creates two 6x6 heatmaps:
  1. P99 E2E Latency (ms) - lower is better
  2. Recompute Ratio (total_tokens_rolled_back / total_output_tokens)

Usage:
    python plot_matrix_heatmap.py --results-file results_matrix_ablation_*/benchmark_results.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Set global style for aesthetics
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True

# Custom aesthetic colormaps
# Latency: Deep indigo (low/good) -> Soft violet -> Warm orange (high/bad)  
LATENCY_COLORS = ['#4f46e5', '#818cf8', '#c4b5fd', '#fcd34d', '#f97316']
LATENCY_CMAP = LinearSegmentedColormap.from_list('latency_aesthetic', LATENCY_COLORS, N=256)

# Recompute: Deep indigo (low/good) -> Soft violet -> Warm orange (high/bad)  
RECOMPUTE_COLORS = ['#4f46e5', '#818cf8', '#c4b5fd', '#fcd34d', '#f97316']
RECOMPUTE_CMAP = LinearSegmentedColormap.from_list('recompute_aesthetic', RECOMPUTE_COLORS, N=256)

# Invalid cell color (subtle light gray)
INVALID_COLOR = '#f5f5f5'


# Configuration
WINDOW_SIZES = [16, 32, 64, 128, 256, 512]
BATCH_SIZES = [32, 16, 8, 4, 2, 1]

# Colors for CDF plots (one per config) - using tab colors
CDF_COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]


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
    Extract P99 E2E latency and recompute ratio for each (window_size, batch_size) pair.
    Also extract per-request rollback data for CDF plots.
    
    Returns:
        dict with keys 'p99_latency', 'recompute_ratio', and 'per_request_data'
    """
    p99_latency = {}
    recompute_ratio = {}
    per_request_data = {}  # (ws, bs) -> {'rollbacks': [...], 'tokens_rolled_back': [...]}
    
    for result in results:
        config_name = result.get('config_name', '')
        if not config_name.startswith('detinfer_ws_'):
            continue
        
        ws = result.get('window_size')
        bs = result.get('batch_size')
        
        if ws is None or bs is None:
            # Try to parse from config_name
            import re
            match = re.match(r'detinfer_ws_(\d+)_bs_(\d+)', config_name)
            if match:
                ws = int(match.group(1))
                bs = int(match.group(2))
            else:
                continue
        
        # Extract P99 E2E latency (ms) - direct field name
        p99 = result.get('p99_e2e_latency_ms')
        if p99 is not None:
            p99_latency[(ws, bs)] = p99 / 1000.0  # Convert to seconds
        
        # Extract recompute ratio
        rollback_stats = result.get('rollback_stats', {})
        total_rolled_back = rollback_stats.get('total_tokens_rolled_back', 0)
        total_output = rollback_stats.get('total_output_tokens', 0)
        
        if total_output > 0:
            recompute_ratio[(ws, bs)] = total_rolled_back / total_output
        else:
            # Fall back to total_output_tokens from result
            total_output = result.get('total_output_tokens', 0)
            if total_output > 0:
                recompute_ratio[(ws, bs)] = total_rolled_back / total_output
            else:
                recompute_ratio[(ws, bs)] = 0
        
        # Extract per-request rollback data for CDF plots
        # First try the new fields (per_request_rollbacks, per_request_tokens_rolled_back)
        per_req_rollbacks = result.get('per_request_rollbacks', [])
        per_req_tokens = result.get('per_request_tokens_rolled_back', [])
        per_req_verification_windows = result.get('per_request_verification_windows', [])
        
        # Compute rollback rate per request: rollbacks / verification_windows
        rollback_rates = []
        if per_req_rollbacks and per_req_verification_windows:
            for rb, vw in zip(per_req_rollbacks, per_req_verification_windows):
                if vw > 0:
                    rollback_rates.append(rb / vw)
                else:
                    rollback_rates.append(0.0)
        
        # Also extract latencies and ttfts for CDF plots
        latencies = result.get('latencies', [])
        ttfts = result.get('ttfts', [])
        
        per_request_data[(ws, bs)] = {
            'rollbacks': per_req_rollbacks if per_req_rollbacks else [],
            'tokens_rolled_back': per_req_tokens if per_req_tokens else [],
            'verification_windows': per_req_verification_windows if per_req_verification_windows else [],
            'rollback_rate': rollback_rates,  # rollbacks / verification_windows per request
            'e2e_latency': [x * 1000 for x in latencies if x is not None],  # Convert to ms
            'ttft': [x * 1000 for x in ttfts if x is not None],  # Convert to ms
        }
    
    return {
        'p99_latency': p99_latency,
        'recompute_ratio': recompute_ratio,
        'per_request_data': per_request_data,
    }


def build_matrix(data: dict, window_sizes: list, batch_sizes: list) -> np.ndarray:
    """
    Build a 2D matrix from (ws, bs) -> value mapping.
    
    Rows: batch sizes (top to bottom: small to large)
    Cols: window sizes (left to right: small to large)
    
    Cells where ws * bs > 512 are set to np.nan (invalid configs).
    """
    matrix = np.zeros((len(batch_sizes), len(window_sizes)))
    
    for i, bs in enumerate(batch_sizes):
        for j, ws in enumerate(window_sizes):
            # Only valid if ws * bs <= 512
            if ws * bs > 512:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = data.get((ws, bs), np.nan)
    
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    window_sizes: list,
    batch_sizes: list,
    title: str,
    cbar_label: str,
    output_path: Path,
    cmap = 'viridis',
    fmt: str = '.2f',
    vmin: float = None,
    vmax: float = None,
    reverse_cmap: bool = False,
):
    """Plot a single heatmap with annotations. Invalid cells (NaN) shown as light gray."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Handle colormap - support both string names and colormap objects
    if isinstance(cmap, str):
        if reverse_cmap:
            cmap_obj = plt.get_cmap(cmap).reversed()
        else:
            cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap.reversed() if reverse_cmap else cmap
    
    # Set bad (NaN) color to subtle light gray
    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(color=INVALID_COLOR)
    
    # Create masked array for NaN handling
    masked_matrix = np.ma.masked_invalid(matrix)
    
    # Create heatmap
    im = ax.imshow(
        masked_matrix,
        cmap=cmap_obj,
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
    )
    
    # Add colorbar with rounded edges
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=22, fontweight='bold')
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(window_sizes)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(window_sizes, fontsize=20, fontweight='medium')
    ax.set_yticklabels(batch_sizes, fontsize=20, fontweight='medium')
    
    # Labels
    ax.set_xlabel('Window Size', fontsize=22, fontweight='bold')
    ax.set_ylabel('Batch Size', fontsize=22, fontweight='bold')
    # No title
    
    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(window_sizes)):
            val = matrix[i, j]
            bs, ws = batch_sizes[i], window_sizes[j]
            
            if np.isnan(val):
                # Invalid cell (ws * bs > 512) - leave blank
                text = ''
                text_color = 'white'
            elif fmt == '.2%':
                text = f'{val:.2%}'
                # Determine text color based on background
                valid_vals = matrix[~np.isnan(matrix)]
                norm_val = (val - valid_vals.min()) / (valid_vals.max() - valid_vals.min() + 1e-10)
                if reverse_cmap:
                    norm_val = 1 - norm_val
                text_color = 'white' if norm_val < 0.5 else 'black'
            elif fmt == '.1f':
                text = f'{val:.1f}'
                valid_vals = matrix[~np.isnan(matrix)]
                norm_val = (val - valid_vals.min()) / (valid_vals.max() - valid_vals.min() + 1e-10)
                if reverse_cmap:
                    norm_val = 1 - norm_val
                text_color = 'white' if norm_val < 0.5 else 'black'
            elif fmt == '.0f':
                text = f'{val:.0f}'
                valid_vals = matrix[~np.isnan(matrix)]
                norm_val = (val - valid_vals.min()) / (valid_vals.max() - valid_vals.min() + 1e-10)
                if reverse_cmap:
                    norm_val = 1 - norm_val
                text_color = 'white' if norm_val < 0.5 else 'black'
            else:
                text = f'{val:{fmt}}'
                valid_vals = matrix[~np.isnan(matrix)]
                norm_val = (val - valid_vals.min()) / (valid_vals.max() - valid_vals.min() + 1e-10)
                if reverse_cmap:
                    norm_val = 1 - norm_val
                text_color = 'white' if norm_val < 0.5 else 'black'
            
            ax.text(j, i, text, ha='center', va='center', color=text_color, 
                    fontsize=17, fontweight='semibold')
    
    # Subtle grid
    ax.set_xticks(np.arange(len(window_sizes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(batch_sizes) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='#e5e5e5', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_both_heatmaps(
    p99_latency_matrix: np.ndarray,
    recompute_matrix: np.ndarray,
    window_sizes: list,
    batch_sizes: list,
    output_path: Path,
):
    """Plot both heatmaps side by side. Invalid cells (ws*bs > 512) shown as light gray."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Create masked arrays
    lat_masked = np.ma.masked_invalid(p99_latency_matrix)
    rc_masked = np.ma.masked_invalid(recompute_matrix)
    
    # P99 Latency heatmap (left) - custom aesthetic colormap
    ax1 = axes[0]
    cmap1 = LATENCY_CMAP.copy()
    cmap1.set_bad(color=INVALID_COLOR)
    im1 = ax1.imshow(lat_masked, cmap=cmap1, aspect='auto')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel('P99 E2E Latency (s)', rotation=-90, va="bottom", fontsize=22, fontweight='bold')
    cbar1.ax.tick_params(labelsize=20)
    cbar1.outline.set_visible(False)
    
    ax1.set_xticks(np.arange(len(window_sizes)))
    ax1.set_yticks(np.arange(len(batch_sizes)))
    ax1.set_xticklabels(window_sizes, fontsize=20, fontweight='medium')
    ax1.set_yticklabels(batch_sizes, fontsize=20, fontweight='medium')
    ax1.set_xlabel('Window Size', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Batch Size', fontsize=22, fontweight='bold')
    # No title for subplot
    
    # Add annotations for P99 latency
    lat_valid = p99_latency_matrix[~np.isnan(p99_latency_matrix)]
    if len(lat_valid) > 0:
        lat_min, lat_max = lat_valid.min(), lat_valid.max()
    else:
        lat_min, lat_max = 0, 1  # Fallback for empty data
    for i in range(len(batch_sizes)):
        for j in range(len(window_sizes)):
            val = p99_latency_matrix[i, j]
            if np.isnan(val):
                text = ''
                text_color = 'white'
            else:
                text = f'{val:.0f}'
                norm_val = (val - lat_min) / (lat_max - lat_min + 1e-10)
                text_color = 'black' if norm_val < 0.5 else 'white'
            ax1.text(j, i, text, ha='center', va='center', color=text_color, 
                     fontsize=17, fontweight='semibold')
    
    ax1.set_xticks(np.arange(len(window_sizes) + 1) - 0.5, minor=True)
    ax1.set_yticks(np.arange(len(batch_sizes) + 1) - 0.5, minor=True)
    ax1.grid(which='minor', color='#e5e5e5', linestyle='-', linewidth=1.5)
    ax1.tick_params(which='minor', bottom=False, left=False)
    
    # Recompute ratio heatmap (right) - custom aesthetic colormap
    ax2 = axes[1]
    cmap2 = RECOMPUTE_CMAP.copy()
    cmap2.set_bad(color=INVALID_COLOR)
    im2 = ax2.imshow(rc_masked, cmap=cmap2, aspect='auto', vmin=0)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.set_ylabel('Recompute Ratio', rotation=-90, va="bottom", fontsize=22, fontweight='bold')
    cbar2.ax.tick_params(labelsize=20)
    cbar2.outline.set_visible(False)
    
    ax2.set_xticks(np.arange(len(window_sizes)))
    ax2.set_yticks(np.arange(len(batch_sizes)))
    ax2.set_xticklabels(window_sizes, fontsize=20, fontweight='medium')
    ax2.set_yticklabels(batch_sizes, fontsize=20, fontweight='medium')
    ax2.set_xlabel('Window Size', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Batch Size', fontsize=22, fontweight='bold')
    # No title for subplot
    
    # Add annotations for recompute ratio
    rc_valid = recompute_matrix[~np.isnan(recompute_matrix)]
    rc_min, rc_max = 0, rc_valid.max() if len(rc_valid) > 0 else 1
    for i in range(len(batch_sizes)):
        for j in range(len(window_sizes)):
            val = recompute_matrix[i, j]
            if np.isnan(val):
                text = ''
                text_color = 'white'
            else:
                text = f'{val:.2%}'
                norm_val = (val - rc_min) / (rc_max - rc_min + 1e-10)
                text_color = 'black' if norm_val < 0.5 else 'white'
            ax2.text(j, i, text, ha='center', va='center', color=text_color, 
                     fontsize=17, fontweight='semibold')
    
    ax2.set_xticks(np.arange(len(window_sizes) + 1) - 0.5, minor=True)
    ax2.set_yticks(np.arange(len(batch_sizes) + 1) - 0.5, minor=True)
    ax2.grid(which='minor', color='#e5e5e5', linestyle='-', linewidth=1.5)
    ax2.tick_params(which='minor', bottom=False, left=False)
    
    # No suptitle
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def compute_cdf(data: list) -> tuple:
    """Compute CDF from a list of values."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def plot_cdf(
    per_request_data: dict,
    metric_key: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    configs_to_plot: list = None,
):
    """
    Plot CDF for a given metric across multiple configurations.
    
    Args:
        per_request_data: dict mapping (ws, bs) -> {'rollbacks': [...], 'tokens_rolled_back': [...]}
        metric_key: 'rollbacks' or 'tokens_rolled_back'
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        output_path: Path to save the plot
        configs_to_plot: List of (ws, bs) tuples to plot. If None, plot all valid configs.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Filter valid configs (ws * bs <= 512)
    if configs_to_plot is None:
        configs_to_plot = [
            (ws, bs) for ws in WINDOW_SIZES for bs in BATCH_SIZES
            if ws * bs <= 512 and (ws, bs) in per_request_data
        ]
    
    # Sort by window size for consistent legend order
    configs_to_plot = sorted(configs_to_plot, key=lambda x: (x[0], x[1]))
    
    for i, (ws, bs) in enumerate(configs_to_plot):
        if (ws, bs) not in per_request_data:
            continue
        
        data = per_request_data[(ws, bs)].get(metric_key, [])
        if not data:
            continue
        
        x_vals, y_vals = compute_cdf(data)
        color = CDF_COLORS[i % len(CDF_COLORS)]
        label = f'Window Size={ws}' if bs == 1 else f'Window Size={ws}, Batch Size={bs}'
        
        ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
    
    # Axis labels (font size 24)
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    
    # Tick font size (22)
    ax.tick_params(axis='both', labelsize=22)
    
    # Legend (only if multiple configs)
    if len(configs_to_plot) > 1:
        ax.legend(fontsize=20, loc='best')
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_cdf_rollbacks(per_request_data: dict, output_path: Path, configs_to_plot: list = None):
    """Plot CDF of rollbacks per request."""
    plot_cdf(
        per_request_data=per_request_data,
        metric_key='rollbacks',
        xlabel='Number of Rollbacks per Request',
        ylabel='CDF',
        output_path=output_path,
        configs_to_plot=configs_to_plot,
    )


def plot_cdf_tokens_rolled_back(per_request_data: dict, output_path: Path, configs_to_plot: list = None):
    """Plot CDF of tokens rolled back (recomputed) per request."""
    plot_cdf(
        per_request_data=per_request_data,
        metric_key='tokens_rolled_back',
        xlabel='Recomputed Tokens per Request',
        ylabel='CDF',
        output_path=output_path,
        configs_to_plot=configs_to_plot,
    )


def plot_cdf_e2e_latency(per_request_data: dict, output_path: Path, configs_to_plot: list = None):
    """Plot CDF of E2E latency per request."""
    plot_cdf(
        per_request_data=per_request_data,
        metric_key='e2e_latency',
        xlabel='E2E Latency (ms)',
        ylabel='CDF',
        output_path=output_path,
        configs_to_plot=configs_to_plot,
    )


def plot_cdf_ttft(per_request_data: dict, output_path: Path, configs_to_plot: list = None):
    """Plot CDF of TTFT per request."""
    plot_cdf(
        per_request_data=per_request_data,
        metric_key='ttft',
        xlabel='Time to First Token (ms)',
        ylabel='CDF',
        output_path=output_path,
        configs_to_plot=configs_to_plot,
    )


def plot_both_cdfs(per_request_data: dict, output_path: Path, configs_to_plot: list = None):
    """Plot both rollbacks and tokens rolled back CDFs side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Filter valid configs
    if configs_to_plot is None:
        configs_to_plot = [
            (ws, bs) for ws in WINDOW_SIZES for bs in BATCH_SIZES
            if ws * bs <= 512 and (ws, bs) in per_request_data
        ]
    
    configs_to_plot = sorted(configs_to_plot, key=lambda x: (x[0], x[1]))
    
    # Plot rollbacks CDF
    ax1 = axes[0]
    for i, (ws, bs) in enumerate(configs_to_plot):
        if (ws, bs) not in per_request_data:
            continue
        data = per_request_data[(ws, bs)].get('rollbacks', [])
        if not data:
            continue
        x_vals, y_vals = compute_cdf(data)
        color = CDF_COLORS[i % len(CDF_COLORS)]
        label = f'WS={ws}' if bs == 1 else f'WS={ws}, BS={bs}'
        ax1.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Number of Rollbacks per Request', fontsize=22, fontweight='bold')
    ax1.set_ylabel('CDF', fontsize=22, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    if len(configs_to_plot) > 1:
        ax1.legend(fontsize=20, loc='best')
    
    # Plot tokens rolled back CDF
    ax2 = axes[1]
    for i, (ws, bs) in enumerate(configs_to_plot):
        if (ws, bs) not in per_request_data:
            continue
        data = per_request_data[(ws, bs)].get('tokens_rolled_back', [])
        if not data:
            continue
        x_vals, y_vals = compute_cdf(data)
        color = CDF_COLORS[i % len(CDF_COLORS)]
        label = f'WS={ws}' if bs == 1 else f'WS={ws}, BS={bs}'
        ax2.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
    
    ax2.set_xlabel('Tokens Recomputed per Request', fontsize=22, fontweight='bold')
    ax2.set_ylabel('CDF', fontsize=22, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    if len(configs_to_plot) > 1:
        ax2.legend(fontsize=14, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_recompute_ratio_bar_bs1(
    recompute_ratio: dict,
    window_sizes: list,
    output_path: Path,
):
    """
    Plot bar chart of recompute ratio for batch_size=1 with varying window sizes.
    
    Args:
        recompute_ratio: dict mapping (ws, bs) -> recompute_ratio value
        window_sizes: list of window sizes to plot
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data for BS=1 only
    ws_values = []
    ratios = []
    for ws in window_sizes:
        if (ws, 1) in recompute_ratio:
            ws_values.append(ws)
            ratios.append(recompute_ratio[(ws, 1)] * 100)  # Convert to percentage
    
    if not ws_values:
        print(f"No data found for batch_size=1. Skipping bar plot.")
        plt.close()
        return
    
    # Create bar positions with equal spacing
    x_indices = range(len(ws_values))
    x_labels = [str(ws) for ws in ws_values]
    
    # Plot bars with hatch pattern, no fill, purple edge and hatch
    bars = ax.bar(
        x_indices,
        ratios,
        facecolor='none',
        edgecolor='tab:purple',
        hatch='////',
        linewidth=2,
        width=0.5,
    )
    
    # Add value labels on top of bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.annotate(
            f'{ratio:.2f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold',
        )
    
    # Axis labels (font size 24)
    ax.set_xlabel('Window Size', fontsize=24, fontweight='bold')
    ax.set_ylabel('Recompute Ratio (%)', fontsize=24, fontweight='bold')
    
    # Tick font size (20)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)
    
    # Grid (y-axis only)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot heatmaps for DetInfer matrix ablation results'
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
    metrics = extract_metrics(results)
    print(f"Found P99 latency data for {len(metrics['p99_latency'])} configs")
    print(f"Found recompute ratio data for {len(metrics['recompute_ratio'])} configs")
    print(f"Found per-request data for {len(metrics['per_request_data'])} configs")
    
    # Build matrices
    p99_latency_matrix = build_matrix(metrics['p99_latency'], WINDOW_SIZES, BATCH_SIZES)
    recompute_matrix = build_matrix(metrics['recompute_ratio'], WINDOW_SIZES, BATCH_SIZES)
    
    # Print summary
    print("\n=== P99 E2E Latency Matrix (ms) ===")
    print(f"Window\\Batch\t{BATCH_SIZES}")
    for i, ws in enumerate(WINDOW_SIZES):
        row = [f"{p99_latency_matrix[i, j]:.2f}" if not np.isnan(p99_latency_matrix[i, j]) else "N/A" 
               for j in range(len(BATCH_SIZES))]
        print(f"{ws}\t\t{row}")
    
    print("\n=== Recompute Ratio Matrix ===")
    print(f"Window\\Batch\t{BATCH_SIZES}")
    for i, ws in enumerate(WINDOW_SIZES):
        row = [f"{recompute_matrix[i, j]:.2%}" if not np.isnan(recompute_matrix[i, j]) else "N/A" 
               for j in range(len(BATCH_SIZES))]
        print(f"{ws}\t\t{row}")
    
    # Plot individual heatmaps
    plot_heatmap(
        p99_latency_matrix,
        WINDOW_SIZES,
        BATCH_SIZES,
        title='',
        cbar_label='P99 E2E Latency (s)',
        output_path=plot_dir / 'heatmap_p99_latency.pdf',
        cmap=LATENCY_CMAP,
        fmt='.2f',
    )
    
    plot_heatmap(
        recompute_matrix,
        WINDOW_SIZES,
        BATCH_SIZES,
        title='',
        cbar_label='Recompute Overhead (%)',
        output_path=plot_dir / 'heatmap_recompute_ratio.pdf',
        cmap=RECOMPUTE_CMAP,
        fmt='.2%',
    )
    
    # Plot combined heatmaps
    plot_both_heatmaps(
        p99_latency_matrix,
        recompute_matrix,
        WINDOW_SIZES,
        BATCH_SIZES,
        output_path=plot_dir / 'heatmap_combined.pdf',
    )
    
    # Plot CDF plots for rollback rate and tokens recomputed
    if metrics['per_request_data']:
        print("\n=== Generating CDF plots ===")
        
        # List of (ws, bs) configs to plot CDFs for - window sizes 32, 64, 128, 256 with BS=1
        cdf_configs = [(32, 1), (64, 1), (128, 1), (256, 1)]
        
        # CDF of rollback rate (rollbacks / verification_windows per request)
        plot_cdf(
            metrics['per_request_data'],
            metric_key='rollback_rate',
            xlabel='Rollbacks per Verification Window',
            ylabel='CDF',
            output_path=plot_dir / 'cdf_rollback_rate.pdf',
            configs_to_plot=cdf_configs,
        )
        
        plot_cdf_tokens_rolled_back(
            metrics['per_request_data'],
            output_path=plot_dir / 'cdf_tokens_recomputed.pdf',
            configs_to_plot=cdf_configs,
        )
        
        # Individual CDF plots - latency
        plot_cdf_e2e_latency(
            metrics['per_request_data'],
            output_path=plot_dir / 'cdf_e2e_latency.pdf',
            configs_to_plot=cdf_configs,
        )
        
        plot_cdf_ttft(
            metrics['per_request_data'],
            output_path=plot_dir / 'cdf_ttft.pdf',
            configs_to_plot=cdf_configs,
        )
        
        # Combined CDF plot
        plot_both_cdfs(
            metrics['per_request_data'],
            output_path=plot_dir / 'cdf_combined.pdf',
            configs_to_plot=cdf_configs,
        )
    else:
        print("\nNo per-request data available for CDF plots.")
        print("Re-run the benchmark with the updated run_matrix_ablation.sh to collect per-request rollback data.")
    
    # Plot bar chart of recompute ratio for BS=1 with window sizes 32, 64, 128, 256
    print("\n=== Generating bar plot (BS=1) ===")
    bar_window_sizes = [32, 64, 128, 256]
    plot_recompute_ratio_bar_bs1(
        metrics['recompute_ratio'],
        bar_window_sizes,
        output_path=plot_dir / 'bar_recompute_ratio_bs1.pdf',
    )
    
    # Save raw data as CSV for further analysis
    import csv
    
    # P99 Latency CSV
    p99_latency_csv_path = output_dir / 'p99_latency_matrix.csv'
    with open(p99_latency_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: empty cell + window sizes
        writer.writerow(['batch_size'] + WINDOW_SIZES)
        # Data rows
        for i, bs in enumerate(BATCH_SIZES):
            row = [bs] + [p99_latency_matrix[i, j] if not np.isnan(p99_latency_matrix[i, j]) else '' 
                          for j in range(len(WINDOW_SIZES))]
            writer.writerow(row)
    print(f"Saved: {p99_latency_csv_path}")
    
    # Recompute ratio CSV
    recompute_csv_path = output_dir / 'recompute_ratio_matrix.csv'
    with open(recompute_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: empty cell + window sizes
        writer.writerow(['batch_size'] + WINDOW_SIZES)
        # Data rows
        for i, bs in enumerate(BATCH_SIZES):
            row = [bs] + [recompute_matrix[i, j] if not np.isnan(recompute_matrix[i, j]) else '' 
                          for j in range(len(WINDOW_SIZES))]
            writer.writerow(row)
    print(f"Saved: {recompute_csv_path}")


if __name__ == '__main__':
    main()
