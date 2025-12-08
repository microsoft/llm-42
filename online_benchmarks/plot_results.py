#!/usr/bin/env python3
"""Plot online benchmark results: TTFT, TPOT, E2E latency, and throughput across configurations."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> list[dict]:
    """Load JSONL results."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def group_by(results: list[dict], key: str) -> dict[str, list[dict]]:
    """Group results by key."""
    grouped = defaultdict(list)
    for r in results:
        grouped[r.get(key, 'unknown')].append(r)
    return grouped


def plot_latency_bars(results: list[dict], output_dir: str):
    """Plot TTFT, TPOT, E2E as bar chart subplots, grouped by (dataset, rate) and det_ratio.
    Creates one figure per (det_ratio, metric) with a grid of subplots."""
    metrics = [
        ('median_ttft_ms', 'TTFT', 'ms'),
        ('median_tpot_ms', 'TPOT', 'ms'),
        ('median_e2e_latency_ms', 'E2E Latency', 'ms'),
    ]
    
    # Group by det_ratio first
    ratio_grouped = group_by(results, 'det_ratio')
    
    for det_ratio, ratio_results in sorted(ratio_grouped.items(), key=lambda x: float(x[0]) if x[0] else 0):
        grouped = group_by(ratio_results, 'config_name')
        
        # Get all unique datasets and request rates
        datasets = sorted(set(r.get('dataset', 'unknown') for r in ratio_results))
        rates = sorted(set(float(r.get('rate', 0)) for r in ratio_results if r.get('rate') != 'inf'))
        
        # Get all config names
        config_names = sorted(grouped.keys())
        n_configs = len(config_names)
        
        if not datasets or not rates:
            continue
        
        # Create one figure per metric
        for metric_key, metric_name, metric_unit in metrics:
            # Create subplot grid: rows = datasets, cols = rates
            n_rows = len(datasets)
            n_cols = len(rates)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), 
                                      squeeze=False)
            
            colors = plt.cm.tab10(np.linspace(0, 1, n_configs))
            
            for i, dataset in enumerate(datasets):
                for j, rate in enumerate(rates):
                    ax = axes[i, j]
                    
                    # Get metric value for each config at this (dataset, rate)
                    values = []
                    for config_name in config_names:
                        config_results = grouped[config_name]
                        matching = [r for r in config_results 
                                   if r.get('dataset') == dataset and float(r.get('rate', 0)) == rate]
                        if matching:
                            values.append(matching[0].get(metric_key, 0) or 0)
                        else:
                            values.append(0)
                    
                    # Create bar chart
                    x = np.arange(n_configs)
                    bars = ax.bar(x, values, color=colors, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, values):
                        if val > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                                   f'{val:.1f}', ha='center', va='bottom', fontsize=7)
                    
                    ax.set_title(f'{dataset}, rate={rate:.0f}', fontsize=10)
                    ax.set_xticks(x)
                    ax.set_xticklabels([])  # Hide x-tick labels, use legend instead
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add y-label only for leftmost column
                    if j == 0:
                        ax.set_ylabel(f'{metric_name} ({metric_unit})', fontsize=9)
            
            # Create legend with config names
            legend_labels = config_names
            fig.legend(bars, legend_labels, loc='lower center', ncol=min(4, n_configs), 
                       fontsize=9, bbox_to_anchor=(0.5, 0.02))
            
            fig.suptitle(f'{metric_name} by Configuration (det_ratio = {det_ratio})', 
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 0.96])
            
            # Sanitize metric name for filename
            metric_filename = metric_key.replace('median_', '').replace('_ms', '')
            filepath = os.path.join(output_dir, f'{metric_filename}_bars_detratio{det_ratio}.pdf')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filepath}")


def plot_total_throughput_bars(results: list[dict], output_dir: str):
    """Plot total throughput as bar chart subplots, grouped by (dataset, rate) and det_ratio.
    Creates one figure per det_ratio with a grid of subplots."""
    # Group by det_ratio first
    ratio_grouped = group_by(results, 'det_ratio')
    
    for det_ratio, ratio_results in sorted(ratio_grouped.items(), key=lambda x: float(x[0]) if x[0] else 0):
        grouped = group_by(ratio_results, 'config_name')
        
        # Get all unique datasets and request rates
        datasets = sorted(set(r.get('dataset', 'unknown') for r in ratio_results))
        rates = sorted(set(float(r.get('rate', 0)) for r in ratio_results if r.get('rate') != 'inf'))
        
        # Get all config names
        config_names = sorted(grouped.keys())
        n_configs = len(config_names)
        
        if not datasets or not rates:
            continue
        
        # Create subplot grid: rows = datasets, cols = rates
        n_rows = len(datasets)
        n_cols = len(rates)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), 
                                  squeeze=False)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_configs))
        
        for i, dataset in enumerate(datasets):
            for j, rate in enumerate(rates):
                ax = axes[i, j]
                
                # Get throughput for each config at this (dataset, rate)
                throughputs = []
                for config_name in config_names:
                    config_results = grouped[config_name]
                    matching = [r for r in config_results 
                               if r.get('dataset') == dataset and float(r.get('rate', 0)) == rate]
                    if matching:
                        throughputs.append(matching[0].get('output_throughput', 0))
                    else:
                        throughputs.append(0)
                
                # Create bar chart
                x = np.arange(n_configs)
                bars = ax.bar(x, throughputs, color=colors, alpha=0.8)
                
                # Add value labels on bars
                for bar, val in zip(bars, throughputs):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                               f'{val:.0f}', ha='center', va='bottom', fontsize=7)
                
                ax.set_title(f'{dataset}, rate={rate:.0f}', fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels([])  # Hide x-tick labels, use legend instead
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add y-label only for leftmost column
                if j == 0:
                    ax.set_ylabel('Throughput (tok/s)', fontsize=9)
        
        # Create legend with config names
        legend_labels = config_names
        fig.legend(bars, legend_labels, loc='lower center', ncol=min(4, n_configs), 
                   fontsize=9, bbox_to_anchor=(0.5, 0.02))
        
        fig.suptitle(f'Total Throughput by Configuration (det_ratio = {det_ratio})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        
        filepath = os.path.join(output_dir, f'total_throughput_bars_detratio{det_ratio}.pdf')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")


def generate_summary(results: list[dict], output_dir: str):
    """Generate text summary."""
    lines = ["=" * 60, "ONLINE BENCHMARK SUMMARY", "=" * 60, ""]
    
    by_config = group_by(results, 'config_name')
    for config, config_data in sorted(by_config.items()):
        lines.append(f"Configuration: {config}")
        lines.append("-" * 40)
        
        by_dataset = group_by(config_data, 'dataset')
        for dataset, ds_data in by_dataset.items():
            lines.append(f"  Dataset: {dataset}")
            by_det = group_by(ds_data, 'det_ratio')
            
            for det, det_data in sorted(by_det.items(), key=lambda x: float(x[0]) if x[0] else 0):
                ttft = np.mean([r.get('median_ttft_ms', 0) for r in det_data if r.get('median_ttft_ms')])
                tpot = np.mean([r.get('median_tpot_ms', 0) for r in det_data if r.get('median_tpot_ms')])
                e2e = np.mean([r.get('median_e2e_latency_ms', 0) for r in det_data if r.get('median_e2e_latency_ms')])
                tput = np.mean([r.get('output_throughput', 0) for r in det_data if r.get('output_throughput')])
                lines.append(f"    det={det}: TTFT={ttft:.1f}ms, TPOT={tpot:.1f}ms, E2E={e2e:.1f}ms, tput={tput:.1f} tok/s")
        lines.append("")
    
    summary = "\n".join(lines)
    print(summary)
    Path(f'{output_dir}/summary.txt').write_text(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', help='JSONL results file')
    parser.add_argument('--output-dir', default='plots')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.results_file)
    
    if not results:
        print(f"No results in {args.results_file}")
        return
    
    print(f"Loaded {len(results)} results")
    plot_latency_bars(results, args.output_dir)
    plot_total_throughput_bars(results, args.output_dir)
    generate_summary(results, args.output_dir)


if __name__ == '__main__':
    main()
