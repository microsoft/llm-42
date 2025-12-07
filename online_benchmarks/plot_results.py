#!/usr/bin/env python3
"""Plot online benchmark results: TTFT, TPOT, E2E latency across configurations."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Color scheme for configurations
CONFIG_COLORS = {
    'baseline': 'steelblue',
    'global_det': 'coral',
    'det_infer': 'forestgreen',
}
CONFIG_MARKERS = {
    'baseline': 'o',
    'global_det': 's',
    'det_infer': '^',
}


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


def plot_metrics_vs_rate_by_config(results: list[dict], output_dir: str):
    """Plot TTFT, TPOT, E2E vs request rate, comparing configurations."""
    by_dataset = group_by(results, 'dataset')
    
    metrics = [
        ('median_ttft_ms', 'TTFT'),
        ('median_tpot_ms', 'TPOT'),
        ('median_e2e_latency_ms', 'E2E Latency'),
    ]
    
    for dataset, data in by_dataset.items():
        by_det = group_by(data, 'det_ratio')
        
        for det_ratio, det_data in sorted(by_det.items(), key=lambda x: float(x[0]) if x[0] else 0):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            by_config = group_by(det_data, 'config_name')
            
            for ax, (metric, title) in zip(axes, metrics):
                for config_name, config_data in sorted(by_config.items()):
                    sorted_data = sorted(config_data, key=lambda x: float(x.get('rate', 0)) if x.get('rate') != 'inf' else 1e9)
                    rates = [float(r['rate']) for r in sorted_data if r.get('rate') != 'inf']
                    vals = [r.get(metric, 0) for r in sorted_data if r.get('rate') != 'inf']
                    if rates:
                        color = CONFIG_COLORS.get(config_name, 'gray')
                        marker = CONFIG_MARKERS.get(config_name, 'o')
                        ax.plot(rates, vals, f'{marker}-', label=config_name, color=color, linewidth=2, markersize=6)
                
                ax.set_xlabel('Request Rate (req/s)')
                ax.set_ylabel(f'{title} (ms)')
                ax.set_title(f'{title}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
            
            plt.suptitle(f'{dataset} - det_ratio={det_ratio}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/latency_{dataset}_det{det_ratio}.png', dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/latency_{dataset}_det{det_ratio}.png")


def plot_throughput_by_config(results: list[dict], output_dir: str):
    """Plot throughput vs request rate, comparing configurations."""
    by_dataset = group_by(results, 'dataset')
    
    for dataset, data in by_dataset.items():
        by_det = group_by(data, 'det_ratio')
        
        for det_ratio, det_data in sorted(by_det.items(), key=lambda x: float(x[0]) if x[0] else 0):
            fig, ax = plt.subplots(figsize=(8, 6))
            by_config = group_by(det_data, 'config_name')
            
            for config_name, config_data in sorted(by_config.items()):
                sorted_data = sorted(config_data, key=lambda x: float(x.get('rate', 0)) if x.get('rate') != 'inf' else 1e9)
                rates = [float(r['rate']) for r in sorted_data if r.get('rate') != 'inf']
                vals = [r.get('output_throughput', 0) for r in sorted_data if r.get('rate') != 'inf']
                if rates:
                    color = CONFIG_COLORS.get(config_name, 'gray')
                    marker = CONFIG_MARKERS.get(config_name, 'o')
                    ax.plot(rates, vals, f'{marker}-', label=config_name, color=color, linewidth=2, markersize=6)
            
            ax.set_xlabel('Request Rate (req/s)')
            ax.set_ylabel('Throughput (tokens/s)')
            ax.set_title(f'Throughput - {dataset} (det_ratio={det_ratio})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/throughput_{dataset}_det{det_ratio}.png', dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/throughput_{dataset}_det{det_ratio}.png")


def plot_config_comparison_bars(results: list[dict], output_dir: str):
    """Bar chart comparing configurations at fixed rate."""
    target_rate = '8'
    filtered = [r for r in results if str(r.get('rate')) == target_rate]
    if not filtered:
        return
    
    by_det = group_by(filtered, 'det_ratio')
    
    for det_ratio, det_data in sorted(by_det.items(), key=lambda x: float(x[0]) if x[0] else 0):
        by_config = group_by(det_data, 'config_name')
        configs = sorted(by_config.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        metrics = [('median_ttft_ms', 'TTFT'), ('median_tpot_ms', 'TPOT'), ('median_e2e_latency_ms', 'E2E')]
        
        x = np.arange(len(configs))
        
        for ax, (metric, title) in zip(axes, metrics):
            vals = []
            colors = []
            for config in configs:
                config_data = by_config[config]
                vals.append(np.mean([r.get(metric, 0) for r in config_data if r.get(metric)]))
                colors.append(CONFIG_COLORS.get(config, 'gray'))
            
            ax.bar(x, vals, color=colors)
            ax.set_ylabel(f'{title} (ms)')
            ax.set_title(f'{title}')
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=15)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, v in enumerate(vals):
                if v > 0:
                    ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Configuration Comparison @ rate={target_rate}, det_ratio={det_ratio}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/config_comparison_det{det_ratio}.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/config_comparison_det{det_ratio}.png")


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
    plot_metrics_vs_rate_by_config(results, args.output_dir)
    plot_throughput_by_config(results, args.output_dir)
    plot_config_comparison_bars(results, args.output_dir)
    generate_summary(results, args.output_dir)


if __name__ == '__main__':
    main()
