#!/usr/bin/env python3
"""Plot rollback stats. Usage: python plot_rollback_stats.py --input stats.json"""
import argparse, json, os
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    elapsed = [s['elapsed'] for s in stats]
    rollbacks = [s.get('sglang:num_rollbacks_total', 0) for s in stats]
    tokens = [s.get('sglang:tokens_rolled_back_total', 0) for s in stats]
    rates = [s.get('rollback_rate', 0) for s in stats]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.plot(elapsed, rollbacks, 'b-', lw=2)
    ax1.set_title('Rollback Count')
    ax1.set_xlabel('Time (s)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(elapsed, tokens, 'r-', lw=2)
    ax2.set_title('Tokens Rolled Back')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(elapsed, rates, 'g-', lw=2)
    ax3.set_title('Rollback Rate')
    ax3.set_xlabel('Time (s)')
    ax3.axhline(np.mean(rates), color='r', ls='--', label=f'Avg: {np.mean(rates):.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.axis('off')
    final = stats[-1]
    text = f"""Final Statistics:
    
Rollbacks: {final.get('sglang:num_rollbacks_total', 0):.0f}
Tokens: {final.get('sglang:tokens_rolled_back_total', 0):.0f}
Rate: {final.get('rollback_rate', 0):.4f}
Duration: {final.get('elapsed', 0):.0f}s"""
    ax4.text(0.1, 0.5, text, fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rollback_summary.png', dpi=300)
    print(f"✓ Saved to {output_dir}/rollback_summary.png")
    plt.close()

def plot_comparison(files, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for f in files:
        with open(f) as fp:
            stats = json.load(fp)
            if stats:
                final = stats[-1]
                data.append({
                    'rollbacks': final.get('sglang:num_rollbacks_total', 0),
                    'rate': final.get('rollback_rate', 0)
                })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    
    ax1.bar(x, [d['rollbacks'] for d in data], color='steelblue')
    ax1.set_title('Rollback Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, [d['rate'] for d in data], color='coral')
    ax2.set_title('Rollback Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.png', dpi=300)
    print(f"✓ Saved to {output_dir}/comparison.png")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="rollback_stats.json")
    p.add_argument("--output", default="plots")
    p.add_argument("--compare", nargs='+')
    p.add_argument("--labels", nargs='+')
    args = p.parse_args()
    
    if os.path.exists(args.input):
        with open(args.input) as f:
            stats = json.load(f)
        if stats:
            plot_stats(stats, args.output)
    
    if args.compare and args.labels:
        plot_comparison(args.compare, args.labels, args.output)
