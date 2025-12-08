#!/usr/bin/env python3
"""
Plot rollback metrics from online benchmarks.

Creates a grid of subplots:
- One row per deterministic ratio (1%, 5%, 10%, 100%)
- Two columns: ShareGPT (left) and Arxiv (right)
- X-axis: Step size
- Y-axis (left): Number of rollbacks
- Y-axis (right): Tokens recomputed

Usage:
    python plot_rollbacks.py rollback_metrics.jsonl --output-dir ./plots
"""

import argparse
import json
import re
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    if match:
        return int(match.group(1))
    return None


def organize_data(data: list) -> dict:
    """
    Organize data by:
    - dataset (sharegpt, arxiv)
    - det_ratio (0.01, 0.05, 0.1, 1.0)
    - rate (QPS)
    - step_size
    
    Returns nested dict: {dataset: {det_ratio: {rate: {step_size: {rollbacks, tokens_recomputed}}}}}
    """
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for record in data:
        config_name = record.get('config_name', '')
        
        # Only include det_infer configs (which have step sizes)
        step_size = extract_step_size(config_name)
        if step_size is None:
            continue
            
        dataset = record.get('dataset', '')
        det_ratio = record.get('det_ratio', 0)
        rate = record.get('rate', 0)
        rollbacks = record.get('rollbacks', 0)
        tokens_recomputed = record.get('tokens_recomputed', 0)
        
        organized[dataset][det_ratio][rate][step_size] = {
            'rollbacks': rollbacks,
            'tokens_recomputed': tokens_recomputed
        }
    
    return organized


def plot_rollbacks(data: list, output_dir: str):
    """Create rollback plots: one row per det_ratio, two columns (sharegpt, arxiv)."""
    
    organized = organize_data(data)
    
    if not organized:
        print("No det_infer data found to plot")
        return
    
    # Get unique datasets and det_ratios
    datasets = sorted([d for d in organized.keys() if d])
    all_det_ratios = set()
    all_rates = set()
    all_step_sizes = set()
    
    for dataset in datasets:
        for det_ratio in organized[dataset]:
            all_det_ratios.add(det_ratio)
            for rate in organized[dataset][det_ratio]:
                all_rates.add(rate)
                for step_size in organized[dataset][det_ratio][rate]:
                    all_step_sizes.add(step_size)
    
    det_ratios = sorted(all_det_ratios)
    rates = sorted(all_rates)
    step_sizes = sorted(all_step_sizes)
    
    if not det_ratios or not step_sizes:
        print("Insufficient data for plotting")
        return
    
    # Ensure both datasets exist
    if 'sharegpt' not in datasets:
        datasets.insert(0, 'sharegpt')
    if 'arxiv' not in datasets:
        datasets.append('arxiv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create one plot per QPS rate
    for rate in rates:
        fig, axes = plt.subplots(
            len(det_ratios), 2,
            figsize=(14, 4 * len(det_ratios)),
            squeeze=False
        )
        
        fig.suptitle(f'Rollback Metrics by Step Size (QPS={rate})', fontsize=14, fontweight='bold')
        
        for row_idx, det_ratio in enumerate(det_ratios):
            for col_idx, dataset in enumerate(['sharegpt', 'arxiv']):
                ax = axes[row_idx, col_idx]
                ax2 = ax.twinx()
                
                # Get data for this combination
                rollbacks_data = []
                tokens_data = []
                valid_step_sizes = []
                
                for step_size in step_sizes:
                    if (dataset in organized and 
                        det_ratio in organized[dataset] and
                        rate in organized[dataset][det_ratio] and
                        step_size in organized[dataset][det_ratio][rate]):
                        
                        record = organized[dataset][det_ratio][rate][step_size]
                        rollbacks_data.append(record['rollbacks'])
                        tokens_data.append(record['tokens_recomputed'])
                        valid_step_sizes.append(step_size)
                
                if valid_step_sizes:
                    x = np.arange(len(valid_step_sizes))
                    width = 0.35
                    
                    # Plot rollbacks on left y-axis
                    bars1 = ax.bar(x - width/2, rollbacks_data, width, 
                                   label='Rollbacks', color='steelblue', alpha=0.8)
                    
                    # Plot tokens recomputed on right y-axis
                    bars2 = ax2.bar(x + width/2, tokens_data, width,
                                    label='Tokens Recomputed', color='coral', alpha=0.8)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(valid_step_sizes)
                    
                    # Style
                    ax.set_xlabel('Step Size')
                    ax.set_ylabel('Rollbacks', color='steelblue')
                    ax2.set_ylabel('Tokens Recomputed', color='coral')
                    ax.tick_params(axis='y', labelcolor='steelblue')
                    ax2.tick_params(axis='y', labelcolor='coral')
                    
                    # Add value labels on bars
                    for bar, val in zip(bars1, rollbacks_data):
                        if val > 0:
                            ax.annotate(f'{val:.0f}',
                                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                       ha='center', va='bottom', fontsize=8, color='steelblue')
                    
                    for bar, val in zip(bars2, tokens_data):
                        if val > 0:
                            ax2.annotate(f'{val:.0f}',
                                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                        ha='center', va='bottom', fontsize=8, color='coral')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='gray')
                
                # Title
                det_pct = det_ratio * 100
                ax.set_title(f'{dataset.upper()} - {det_pct:.0f}% Deterministic')
                
                # Legend (only on first subplot)
                if row_idx == 0 and col_idx == 0:
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'rollbacks_qps{rate}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    # Also create a summary plot aggregating across all QPS rates
    create_summary_plot(organized, det_ratios, step_sizes, rates, output_dir)


def create_summary_plot(organized: dict, det_ratios: list, step_sizes: list, rates: list, output_dir: str):
    """Create summary plot with aggregated metrics across all QPS rates."""
    
    fig, axes = plt.subplots(
        len(det_ratios), 2,
        figsize=(14, 4 * len(det_ratios)),
        squeeze=False
    )
    
    fig.suptitle('Rollback Metrics by Step Size (Aggregated Across All QPS)', fontsize=14, fontweight='bold')
    
    for row_idx, det_ratio in enumerate(det_ratios):
        for col_idx, dataset in enumerate(['sharegpt', 'arxiv']):
            ax = axes[row_idx, col_idx]
            ax2 = ax.twinx()
            
            # Aggregate across all rates
            rollbacks_sum = []
            tokens_sum = []
            valid_step_sizes = []
            
            for step_size in step_sizes:
                total_rollbacks = 0
                total_tokens = 0
                found = False
                
                for rate in rates:
                    if (dataset in organized and 
                        det_ratio in organized[dataset] and
                        rate in organized[dataset][det_ratio] and
                        step_size in organized[dataset][det_ratio][rate]):
                        
                        record = organized[dataset][det_ratio][rate][step_size]
                        total_rollbacks += record['rollbacks']
                        total_tokens += record['tokens_recomputed']
                        found = True
                
                if found:
                    rollbacks_sum.append(total_rollbacks)
                    tokens_sum.append(total_tokens)
                    valid_step_sizes.append(step_size)
            
            if valid_step_sizes:
                x = np.arange(len(valid_step_sizes))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, rollbacks_sum, width, 
                               label='Total Rollbacks', color='steelblue', alpha=0.8)
                bars2 = ax2.bar(x + width/2, tokens_sum, width,
                                label='Total Tokens Recomputed', color='coral', alpha=0.8)
                
                ax.set_xticks(x)
                ax.set_xticklabels(valid_step_sizes)
                ax.set_xlabel('Step Size')
                ax.set_ylabel('Rollbacks', color='steelblue')
                ax2.set_ylabel('Tokens Recomputed', color='coral')
                ax.tick_params(axis='y', labelcolor='steelblue')
                ax2.tick_params(axis='y', labelcolor='coral')
                
                # Add value labels
                for bar, val in zip(bars1, rollbacks_sum):
                    if val > 0:
                        ax.annotate(f'{val:.0f}',
                                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=8, color='steelblue')
                
                for bar, val in zip(bars2, tokens_sum):
                    if val > 0:
                        ax2.annotate(f'{val:.0f}',
                                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                    ha='center', va='bottom', fontsize=8, color='coral')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='gray')
            
            det_pct = det_ratio * 100
            ax.set_title(f'{dataset.upper()} - {det_pct:.0f}% Deterministic')
            
            if row_idx == 0 and col_idx == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rollbacks_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot rollback metrics from online benchmarks')
    parser.add_argument('input_file', help='Path to rollback_metrics.jsonl file')
    parser.add_argument('--output-dir', '-o', default='./plots', help='Output directory for plots')
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
    
    plot_rollbacks(data, args.output_dir)
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
