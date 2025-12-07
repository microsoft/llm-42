#!/usr/bin/env python3
"""Plot online benchmark results: TTFT, TPOT, E2E latency."""

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


def plot_metrics_vs_rate(results: list[dict], output_dir: str):
    """Plot TTFT, TPOT, E2E vs request rate."""
    by_dataset = group_by(results, 'dataset')
    
    metrics = [
        ('median_ttft_ms', 'TTFT'),
        ('median_tpot_ms', 'TPOT'),
        ('median_e2e_latency_ms', 'E2E Latency'),
    ]
    
    for dataset, data in by_dataset.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        by_det = group_by(data, 'det_ratio')
        
        for ax, (metric, title) in zip(axes, metrics):
            for det_ratio, det_data in sorted(by_det.items(), key=lambda x: float(x[0])):
                sorted_data = sorted(det_data, key=lambda x: float(x.get('rate', 0)) if x.get('rate') != 'inf' else 1e9)
                rates = [float(r['rate']) for r in sorted_data if r.get('rate') != 'inf']
                vals = [r.get(metric, 0) for r in sorted_data if r.get('rate') != 'inf']
                if rates:
                    ax.plot(rates, vals, 'o-', label=f'det={det_ratio}', linewidth=2, markersize=6)
            
            ax.set_xlabel('Request Rate (req/s)')
            ax.set_ylabel(f'{title} (ms)')
            ax.set_title(f'{title} - {dataset}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/latency_{dataset}.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/latency_{dataset}.png")


def plot_throughput(results: list[dict], output_dir: str):
    """Plot throughput vs request rate."""
    by_dataset = group_by(results, 'dataset')
    
    fig, axes = plt.subplots(1, len(by_dataset), figsize=(6*len(by_dataset), 5))
    if len(by_dataset) == 1:
        axes = [axes]
    
    for ax, (dataset, data) in zip(axes, by_dataset.items()):
        by_det = group_by(data, 'det_ratio')
        
        for det_ratio, det_data in sorted(by_det.items(), key=lambda x: float(x[0])):
            sorted_data = sorted(det_data, key=lambda x: float(x.get('rate', 0)) if x.get('rate') != 'inf' else 1e9)
            rates = [float(r['rate']) for r in sorted_data if r.get('rate') != 'inf']
            vals = [r.get('output_throughput', 0) for r in sorted_data if r.get('rate') != 'inf']
            if rates:
                ax.plot(rates, vals, 's-', label=f'det={det_ratio}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Request Rate (req/s)')
        ax.set_ylabel('Throughput (tokens/s)')
        ax.set_title(f'Throughput - {dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/throughput.png")


def plot_comparison_bars(results: list[dict], output_dir: str):
    """Bar chart comparing det vs non-det at fixed rate."""
    # Pick rate=8 for comparison
    target_rate = '8'
    filtered = [r for r in results if str(r.get('rate')) == target_rate]
    if not filtered:
        return
    
    by_dataset = group_by(filtered, 'dataset')
    datasets = list(by_dataset.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [('median_ttft_ms', 'TTFT'), ('median_tpot_ms', 'TPOT'), ('median_e2e_latency_ms', 'E2E')]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    for ax, (metric, title) in zip(axes, metrics):
        det0_vals, det1_vals = [], []
        for ds in datasets:
            ds_data = by_dataset[ds]
            det0_vals.append(np.mean([r.get(metric, 0) for r in ds_data if r.get('det_ratio') == 0.0]))
            det1_vals.append(np.mean([r.get(metric, 0) for r in ds_data if r.get('det_ratio') == 1.0]))
        
        ax.bar(x - width/2, det0_vals, width, label='det=0.0', color='steelblue')
        ax.bar(x + width/2, det1_vals, width, label='det=1.0', color='coral')
        ax.set_ylabel(f'{title} (ms)')
        ax.set_title(f'{title} @ rate={target_rate}')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/comparison.png")


def generate_summary(results: list[dict], output_dir: str):
    """Generate text summary."""
    lines = ["=" * 50, "ONLINE BENCHMARK SUMMARY", "=" * 50, ""]
    
    by_dataset = group_by(results, 'dataset')
    for dataset, data in by_dataset.items():
        lines.append(f"Dataset: {dataset}")
        by_det = group_by(data, 'det_ratio')
        
        for det, det_data in sorted(by_det.items(), key=lambda x: float(x[0])):
            ttft = np.mean([r.get('median_ttft_ms', 0) for r in det_data])
            tpot = np.mean([r.get('median_tpot_ms', 0) for r in det_data])
            e2e = np.mean([r.get('median_e2e_latency_ms', 0) for r in det_data])
            tput = np.mean([r.get('output_throughput', 0) for r in det_data])
            lines.append(f"  det={det}: TTFT={ttft:.1f}ms, TPOT={tpot:.1f}ms, E2E={e2e:.1f}ms, throughput={tput:.1f} tok/s")
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
    plot_metrics_vs_rate(results, args.output_dir)
    plot_throughput(results, args.output_dir)
    plot_comparison_bars(results, args.output_dir)
    generate_summary(results, args.output_dir)


if __name__ == '__main__':
    main()
