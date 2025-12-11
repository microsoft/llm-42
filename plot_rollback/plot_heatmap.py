#!/usr/bin/env python3
"""
Script to plot mismatch heatmap from summary.json with larger fonts.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def plot_mismatch_heatmap(
    mismatch_matrix: np.ndarray, 
    qps_values: List[float], 
    output_path: Path,
    figsize: tuple = (10, 8),
    title_fontsize: int = 32,
    label_fontsize: int = 24,
    tick_fontsize: int = 24,
    annotation_fontsize: int = 28,
    colorbar_fontsize: int = 24
):
    """
    Plot heatmap showing fraction of mismatches between each pair of QPS values.
    
    Args:
        mismatch_matrix: 2D array of mismatch fractions
        qps_values: List of QPS values corresponding to matrix indices
        output_path: Path to save the plot
        figsize: Figure size in inches (width, height)
        title_fontsize: Font size for the title
        label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
        annotation_fontsize: Font size for cell annotations
        colorbar_fontsize: Font size for colorbar label
    """
    plt.figure(figsize=figsize)
    im = plt.imshow(mismatch_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Colorbar with larger font
    cbar = plt.colorbar(im, label='Mismatch Fraction')
    cbar.set_label('Mismatch Fraction', fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    # Axis labels with larger font
    labels = [f"QPS {qps}" for qps in qps_values]
    plt.xticks(range(len(qps_values)), labels, rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(range(len(qps_values)), labels, fontsize=tick_fontsize)
    
    # Add text annotations with larger font
    for i in range(len(qps_values)):
        for j in range(len(qps_values)):
            text = plt.text(j, i, f'{mismatch_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=annotation_fontsize,
                          fontweight='bold')
    
    plt.title('Pairwise Output Mismatch Fractions', fontsize=title_fontsize, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"Saved heatmap to: {output_path}")
    plt.close()


def load_summary_and_plot(summary_path: Path, output_path: Path = None, **kwargs):
    """Load summary.json and plot the heatmap."""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    qps_values = data['qps_values']
    pairwise_comparisons = data['pairwise_comparisons']
    
    # Build mismatch matrix
    n = len(qps_values)
    mismatch_matrix = np.zeros((n, n))
    
    for comparison in pairwise_comparisons:
        qps_1 = comparison['qps_1']
        qps_2 = comparison['qps_2']
        mismatch_fraction = comparison['mismatch_fraction']
        
        i = qps_values.index(qps_1)
        j = qps_values.index(qps_2)
        
        mismatch_matrix[i, j] = mismatch_fraction
        mismatch_matrix[j, i] = mismatch_fraction
    
    # Determine output path
    if output_path is None:
        if 'heatmap_plot' in data:
            output_path = Path(data['heatmap_plot']).parent / "mismatch_heatmap_large_font.pdf"
        else:
            output_path = summary_path.parent / "mismatch_heatmap_large_font.pdf"
    
    plot_mismatch_heatmap(mismatch_matrix, qps_values, output_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Plot mismatch heatmap from summary.json with customizable font sizes"
    )
    parser.add_argument(
        "summary_json", 
        type=Path,
        help="Path to summary.json file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for the heatmap (default: same directory as summary.json)"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[10, 8],
        help="Figure size in inches (width height), default: 10 8"
    )
    parser.add_argument(
        "--title-fontsize",
        type=int,
        default=32,
        help="Title font size (default: 32)"
    )
    parser.add_argument(
        "--label-fontsize",
        type=int,
        default=24,
        help="Axis label font size (default: 24)"
    )
    parser.add_argument(
        "--tick-fontsize",
        type=int,
        default=24,
        help="Tick label font size (default: 24)"
    )
    parser.add_argument(
        "--annotation-fontsize",
        type=int,
        default=28,
        help="Cell annotation font size (default: 28)"
    )
    parser.add_argument(
        "--colorbar-fontsize",
        type=int,
        default=24,
        help="Colorbar label font size (default: 24)"
    )
    
    args = parser.parse_args()
    
    load_summary_and_plot(
        args.summary_json,
        args.output,
        figsize=tuple(args.figsize),
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize,
        annotation_fontsize=args.annotation_fontsize,
        colorbar_fontsize=args.colorbar_fontsize
    )


if __name__ == "__main__":
    main()
