#!/usr/bin/env python3
"""
Plot recompute cost (rollback %) bar chart for batch_size=1 across window sizes.

Reads summary.json from each model's run directory and produces a grouped
bar chart with both models side by side. The plot is also copied to
sosp-artifact/llm42-plots/figure9b.pdf (the paper's Figure 9b).

Usage:
    python plot.py --run-dir runs/8b_run --label 'Llama-3-8B'                                   --run-dir runs/70b_run --label 'Llama-3-70B'                                   --output recompute_cost_bars.pdf
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
WINDOW_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def load_rollback_pct(run_dir: Path, batch_size: int = 1) -> dict:
    summary_path = run_dir / 'summary.json'
    with open(summary_path) as f:
        data = json.load(f)

    lookup = {}
    for p in data['profiles']:
        if p['verify_batch_size'] == batch_size:
            lookup[p['window_size']] = p['rollback_pct']
    return lookup


def plot_bars(datasets, output_file):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(WINDOW_SIZES))
    n = len(datasets)
    width = 0.8 / n

    for i, (values, label) in enumerate(datasets):
        offset = (i - (n - 1) / 2) * width
        bars = [values.get(ws, 0) for ws in WINDOW_SIZES]
        ax.bar(x + offset, bars, width, label=label, color=COLORS[i % len(COLORS)], alpha=0.8)

    ax.set_xlabel('Verification window size', fontsize=24, fontweight='bold')
    ax.set_ylabel('Recompute cost (%)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels([str(ws) for ws in WINDOW_SIZES])

    if n > 1:
        ax.legend(fontsize=22, loc='best')

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=1200, bbox_inches='tight')
    print(f'Plot saved to {output_file}')
    plt.close()


def export_paper_figure(pdf_path):
    """Copy the recompute-cost bar chart to sosp-artifact/llm42-plots/figure9b.pdf (the paper's Figure 9b)."""
    if pdf_path is None or not Path(pdf_path).exists():
        return
    plots_dir = (Path(__file__).resolve().parent / ".." / "llm42-plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    dst = plots_dir / "figure9b.pdf"
    shutil.copyfile(pdf_path, dst)
    print(f"Exported paper figure: {dst}")


def main():
    parser = argparse.ArgumentParser(description='Plot recompute cost bar chart')
    parser.add_argument('--run-dir', type=Path, action='append', required=True,
                        help='Run directory containing summary.json (can specify multiple)')
    parser.add_argument('--label', type=str, action='append', default=None,
                        help='Label for each run dir (same order)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size to plot (default: 1)')
    parser.add_argument('--output', type=str, default='recompute_cost_bars.pdf',
                        help='Output PDF file')
    args = parser.parse_args()

    labels = args.label or [None] * len(args.run_dir)
    if len(labels) < len(args.run_dir):
        labels.extend([None] * (len(args.run_dir) - len(labels)))

    datasets = []
    for run_dir, label in zip(args.run_dir, labels):
        if label is None:
            label = run_dir.name
        print(f'Loading: {run_dir} (label: {label})')
        values = load_rollback_pct(run_dir, args.batch_size)
        datasets.append((values, label))

    print('Generating bar chart...')
    plot_bars(datasets, args.output)
    export_paper_figure(args.output)
    print('Done!')


if __name__ == '__main__':
    main()
