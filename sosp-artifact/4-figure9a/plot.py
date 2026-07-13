"""
Plot forward pass latency per token results.

Supports multiple CSV files for combined plots (e.g. 8B vs 70B).

The generated plot is also copied to sosp-artifact/llm42-plots/figure9a.pdf
(the paper's Figure 9a).

Usage:
    # Single model:
    python plot.py --input results.csv --output plot.pdf

    # Multiple models on one plot:
    python plot.py \
        --input 8b_results.csv --label llama-8b-tp1 \
        --input 70b_results.csv --label llama-70b-tp8 \
        --output combined.pdf
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
MARKERS = ['o', 's', '^', 'D', 'v']


def load_results(csv_file: str) -> dict:
    input_lens = []
    latency_per_token = []
    std_latency_per_token = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_lens.append(int(row['input_len']))
            latency_per_token.append(float(row['latency_per_token_ms']))
            std_latency_per_token.append(float(row['std_latency_per_token_ms']))

    return {
        'input_lens': input_lens,
        'latency_per_token': latency_per_token,
        'std_latency_per_token': std_latency_per_token,
    }


def plot_combined(datasets: list, output_file: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    all_lens = sorted(set(l for data, _ in datasets for l in data['input_lens']))
    x_indices = range(len(all_lens))
    x_labels = [str(x) for x in all_lens]

    for i, (data, label) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        xs = [all_lens.index(l) for l in data['input_lens']]

        ax.plot(
            xs,
            data['latency_per_token'],
            f'-{marker}',
            color=color,
            markersize=8,
            linewidth=2,
            label=label,
        )

    ax.set_xlabel('# Tokens in the batch', fontsize=24, fontweight='bold')
    ax.set_ylabel('Latency per token (ms)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(list(x_indices))
    ax.set_xticklabels(x_labels)

    if len(datasets) > 1:
        ax.legend(fontsize=22, loc='best')

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=1200, bbox_inches='tight')
    print(f'Plot saved to {output_file}')
    plt.close()


def export_paper_figure(pdf_path):
    """Copy the forward-pass latency plot to sosp-artifact/llm42-plots/figure9a.pdf (the paper's Figure 9a)."""
    if pdf_path is None or not Path(pdf_path).exists():
        return
    plots_dir = (Path(__file__).resolve().parent / ".." / "llm42-plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    dst = plots_dir / "figure9a.pdf"
    shutil.copyfile(pdf_path, dst)
    print(f"Exported paper figure: {dst}")


def main():
    parser = argparse.ArgumentParser(description='Plot forward pass latency results')
    parser.add_argument(
        '--input', type=str, action='append', required=True,
        help='Input CSV file(s) (can specify multiple)',
    )
    parser.add_argument(
        '--label', type=str, action='append', default=None,
        help='Label for each input CSV (same order as --input)',
    )
    parser.add_argument(
        '--output', type=str, default='forward_cost_plot.pdf',
        help='Output PDF file',
    )
    args = parser.parse_args()

    labels = args.label or [None] * len(args.input)
    if len(labels) < len(args.input):
        labels.extend([None] * (len(args.input) - len(labels)))

    datasets = []
    for csv_file, label in zip(args.input, labels):
        if label is None:
            label = csv_file.rsplit('/', 1)[-1].replace('_results.csv', '')
        print(f'Loading: {csv_file} (label: {label})')
        data = load_results(csv_file)
        datasets.append((data, label))

    print('Generating plot...')
    plot_combined(datasets, args.output)
    export_paper_figure(args.output)
    print('Done!')


if __name__ == '__main__':
    main()
