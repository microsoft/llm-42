#!/usr/bin/env python3
"""
Plot offline throughput benchmark results.

Usage:
    python plot_results.py <results_file.jsonl> --output-dir <output_dir>
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def group_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by configuration name."""
    grouped = defaultdict(list)
    for r in results:
        grouped[r['config_name']].append(r)
    return grouped


def get_config_display_name(config_name: str) -> str:
    """Convert config name to display-friendly name."""
    name_map = {
        'default': 'Default (Baseline)',
        'det_inference_2': 'Deterministic Inference (Mode 2)',
    }
    if config_name in name_map:
        return name_map[config_name]
    if config_name.startswith('det_infer_3_step'):
        step_size = config_name.replace('det_infer_3_step', '')
        return f'Det-Infer Mode 3 (step={step_size})'
    return config_name


def plot_throughput_by_input_len(results: List[Dict], output_dir: str):
    """Plot output throughput vs input length for each configuration."""
    grouped = group_results(results)
    
    # Group by output_len for separate plots
    output_lens = sorted(set(r['output_len'] for r in results))
    
    for output_len in output_lens:
        plt.figure(figsize=(12, 8))
        
        for config_name, config_results in sorted(grouped.items()):
            # Filter by output_len
            filtered = [r for r in config_results if r['output_len'] == output_len]
            if not filtered:
                continue
            
            # Sort by input_len
            filtered.sort(key=lambda x: x['input_len'])
            input_lens = [r['input_len'] for r in filtered]
            throughputs = [r['output_throughput'] for r in filtered]
            
            label = get_config_display_name(config_name)
            marker = 'o' if 'default' in config_name else ('s' if 'det_inference' in config_name else '^')
            plt.plot(input_lens, throughputs, marker=marker, label=label, linewidth=2, markersize=8)
        
        plt.xlabel('Input Length (tokens)', fontsize=12)
        plt.ylabel('Output Throughput (tokens/s)', fontsize=12)
        plt.title(f'Output Throughput vs Input Length\n(Output Length = {output_len} tokens)', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, f'throughput_by_input_len_out{output_len}.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved: {filepath}")


def plot_throughput_by_output_len(results: List[Dict], output_dir: str):
    """Plot output throughput vs output length for each configuration."""
    grouped = group_results(results)
    
    # Group by input_len for separate plots
    input_lens = sorted(set(r['input_len'] for r in results))
    
    for input_len in input_lens:
        plt.figure(figsize=(12, 8))
        
        for config_name, config_results in sorted(grouped.items()):
            # Filter by input_len
            filtered = [r for r in config_results if r['input_len'] == input_len]
            if not filtered:
                continue
            
            # Sort by output_len
            filtered.sort(key=lambda x: x['output_len'])
            output_lens = [r['output_len'] for r in filtered]
            throughputs = [r['output_throughput'] for r in filtered]
            
            label = get_config_display_name(config_name)
            marker = 'o' if 'default' in config_name else ('s' if 'det_inference' in config_name else '^')
            plt.plot(output_lens, throughputs, marker=marker, label=label, linewidth=2, markersize=8)
        
        plt.xlabel('Output Length (tokens)', fontsize=12)
        plt.ylabel('Output Throughput (tokens/s)', fontsize=12)
        plt.title(f'Output Throughput vs Output Length\n(Input Length = {input_len} tokens)', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, f'throughput_by_output_len_in{input_len}.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved: {filepath}")


def plot_throughput_heatmaps(results: List[Dict], output_dir: str):
    """Create heatmaps showing throughput for each configuration."""
    grouped = group_results(results)
    
    input_lens = sorted(set(r['input_len'] for r in results))
    output_lens = sorted(set(r['output_len'] for r in results))
    
    for config_name, config_results in grouped.items():
        # Build 2D array
        throughput_matrix = np.zeros((len(input_lens), len(output_lens)))
        
        for r in config_results:
            i = input_lens.index(r['input_len'])
            j = output_lens.index(r['output_len'])
            throughput_matrix[i, j] = r['output_throughput']
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(throughput_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='Output Throughput (tokens/s)')
        
        plt.xticks(range(len(output_lens)), output_lens)
        plt.yticks(range(len(input_lens)), input_lens)
        plt.xlabel('Output Length (tokens)', fontsize=12)
        plt.ylabel('Input Length (tokens)', fontsize=12)
        plt.title(f'Throughput Heatmap: {get_config_display_name(config_name)}', fontsize=14)
        
        # Add value annotations
        for i in range(len(input_lens)):
            for j in range(len(output_lens)):
                value = throughput_matrix[i, j]
                plt.text(j, i, f'{value:.0f}', ha='center', va='center', 
                        color='white' if value > throughput_matrix.max() * 0.5 else 'black',
                        fontsize=9)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f'heatmap_{config_name}.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved: {filepath}")


def plot_overhead_comparison(results: List[Dict], output_dir: str):
    """Plot overhead/speedup compared to baseline."""
    grouped = group_results(results)
    
    if 'default' not in grouped:
        print("Warning: No baseline (default) results found for overhead comparison")
        return
    
    baseline_results = {(r['input_len'], r['output_len']): r['output_throughput'] 
                        for r in grouped['default']}
    
    # Calculate overhead for each config
    overhead_data = {}
    for config_name, config_results in grouped.items():
        if config_name == 'default':
            continue
        
        overheads = []
        for r in config_results:
            key = (r['input_len'], r['output_len'])
            if key in baseline_results:
                baseline_tp = baseline_results[key]
                current_tp = r['output_throughput']
                # Overhead as percentage slowdown (negative = faster)
                overhead = ((baseline_tp - current_tp) / baseline_tp) * 100
                overheads.append({
                    'input_len': r['input_len'],
                    'output_len': r['output_len'],
                    'overhead': overhead
                })
        overhead_data[config_name] = overheads
    
    # Bar plot of average overhead per config
    plt.figure(figsize=(12, 6))
    
    config_names = []
    avg_overheads = []
    std_overheads = []
    
    for config_name, overheads in sorted(overhead_data.items()):
        if overheads:
            config_names.append(get_config_display_name(config_name))
            overhead_vals = [o['overhead'] for o in overheads]
            avg_overheads.append(np.mean(overhead_vals))
            std_overheads.append(np.std(overhead_vals))
    
    x = np.arange(len(config_names))
    colors = ['red' if v > 0 else 'green' for v in avg_overheads]
    
    bars = plt.bar(x, avg_overheads, yerr=std_overheads, capsize=5, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(x, config_names, rotation=45, ha='right')
    plt.ylabel('Overhead (%) - Negative = Faster', fontsize=12)
    plt.title('Throughput Overhead Compared to Baseline', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'overhead_comparison.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def plot_step_size_comparison(results: List[Dict], output_dir: str):
    """Compare different min-det-step-size values."""
    # Filter for det_infer_3 configs
    det_infer_results = [r for r in results if r['config_name'].startswith('det_infer_3_step')]
    
    if not det_infer_results:
        print("No det-infer-3 results found for step size comparison")
        return
    
    # Extract step sizes
    step_sizes = sorted(set(
        int(r['config_name'].replace('det_infer_3_step', '')) 
        for r in det_infer_results
    ))
    
    # Group by input/output len combination
    combinations = sorted(set((r['input_len'], r['output_len']) for r in det_infer_results))
    
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(combinations)))
    
    for idx, (input_len, output_len) in enumerate(combinations):
        filtered = [r for r in det_infer_results 
                   if r['input_len'] == input_len and r['output_len'] == output_len]
        
        step_throughput = {}
        for r in filtered:
            step = int(r['config_name'].replace('det_infer_3_step', ''))
            step_throughput[step] = r['output_throughput']
        
        steps = sorted(step_throughput.keys())
        throughputs = [step_throughput[s] for s in steps]
        
        plt.plot(steps, throughputs, marker='o', color=colors[idx], 
                label=f'in={input_len}, out={output_len}', linewidth=2, markersize=6)
    
    plt.xlabel('min-det-step-size', fontsize=12)
    plt.ylabel('Output Throughput (tokens/s)', fontsize=12)
    plt.title('Effect of min-det-step-size on Throughput\n(enable-det-infer 3)', fontsize=14)
    plt.legend(loc='best', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xticks(step_sizes)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'step_size_comparison.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def generate_summary_table(results: List[Dict], output_dir: str):
    """Generate a summary table of all results."""
    grouped = group_results(results)
    
    lines = []
    lines.append("=" * 100)
    lines.append("BENCHMARK RESULTS SUMMARY")
    lines.append("=" * 100)
    lines.append("")
    
    for config_name in sorted(grouped.keys()):
        config_results = grouped[config_name]
        lines.append(f"\n{get_config_display_name(config_name)}")
        lines.append("-" * 80)
        lines.append(f"{'Input Len':<12} {'Output Len':<12} {'Throughput (tok/s)':<20} {'Latency (s)':<15}")
        lines.append("-" * 80)
        
        for r in sorted(config_results, key=lambda x: (x['input_len'], x['output_len'])):
            lines.append(f"{r['input_len']:<12} {r['output_len']:<12} "
                        f"{r['output_throughput']:<20.2f} {r['total_latency']:<15.2f}")
    
    lines.append("\n" + "=" * 100)
    
    summary_text = "\n".join(lines)
    print(summary_text)
    
    filepath = os.path.join(output_dir, 'summary.txt')
    with open(filepath, 'w') as f:
        f.write(summary_text)
    print(f"\nSaved summary: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Plot offline benchmark results')
    parser.add_argument('results_file', type=str, help='Path to JSONL results file')
    parser.add_argument('--output-dir', type=str, default='plots', 
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} benchmark results")
    
    if not results:
        print("No results found!")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    plot_throughput_by_input_len(results, args.output_dir)
    plot_throughput_by_output_len(results, args.output_dir)
    plot_throughput_heatmaps(results, args.output_dir)
    plot_overhead_comparison(results, args.output_dir)
    plot_step_size_comparison(results, args.output_dir)
    
    # Generate summary
    generate_summary_table(results, args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
