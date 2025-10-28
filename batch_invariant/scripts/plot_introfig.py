import argparse
import re
import sys
import os
from typing import List, Tuple
from matplotlib.patches import Patch

#!/usr/bin/env python3
"""
plot_introfig.py

Read one or more log/text files given on the command line, extract lines like:
    Total token throughput (tok/s):          30298.30
and plot the extracted throughput values as a bar chart.

Usage:
    python plot_introfig.py file1.log file2.log ...
    python plot_introfig.py -o out.png file1.log file2.log
"""

import matplotlib.pyplot as plt

THROUGHPUT_RE = re.compile(r"Total token throughput\s*\(tok/s\):\s*([+-]?\d+\.\d+)")

NAME_RE = re.compile(r".*_b[0-9]+_in([0-9]+)_out([0-9]+)_(.*)\.txt", re.MULTILINE)

def extract_throughput_from_text(text: str, filename: str = None) -> float:
    m = THROUGHPUT_RE.search(text)
    key = ()
    if filename:
        m_name = NAME_RE.match(filename)
        if m_name:
            key = (int(m_name.group(1)), int(m_name.group(2)), m_name.group(3))
    if not m:
        return None
    try:
        return key, float(m.group(1))
    except ValueError:
        return None

def _extract_single_file(path: str, filename: str = None):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return None
    return extract_throughput_from_text(content, filename)


def extract_from_file(path: str):
    """
    If `path` is a file, return a float (or None).
    If `path` is a directory, return a list of (basename, float) for every regular file
    in the directory that contains a throughput value; returns None if nothing found.
    """
    if os.path.isdir(path):
        results = []
        try:
            entries = sorted(os.listdir(path))
        except OSError as e:
            print(f"Error listing directory {path}: {e}", file=sys.stderr)
            return None

        for name in entries:
            # skip hidden files
            if name.startswith("."):
                continue
            fp = os.path.join(path, name)
            if not os.path.isfile(fp):
                continue
            val = _extract_single_file(fp, name)
            print(fp, name, val)
            if val is None:
                # skip files without a throughput line
                continue
            results.append(val)

        return results if results else None

    # single file
    return _extract_single_file(path)


def parse_args():
        p = argparse.ArgumentParser(
                description="Extract 'Total token throughput (tok/s)' from files and plot a bar chart."
        )
        p.add_argument("files", nargs="+", help="Input log/text files to scan")
        p.add_argument(
                "-o",
                "--out",
                help="Optional output image file (png/svg/pdf). If omitted, show the plot interactively.",
                default=None,
        )
        p.add_argument(
                "--title",
                help="Plot title",
                default="Total token throughput (tok/s)",
        )
        return p.parse_args()


bar_names = {
    'nondet' : 'Non-deterministic',
    'det1_' : 'Default deterministic',
    'det66_' : 'Ours (Deterministic)',
}

titles = ['nondet', 'det1_', 'det66_']

def make_bar_plot(labels: List[tuple], values: List[float], title: str, outpath: str = None):
    if not labels:
        print("No throughput values found; nothing to plot.", file=sys.stderr)
        return

    # Group entries by the first element of each label tuple
    groups = {}

    for lbl, val in zip(labels, values):
        key = lbl[0] if isinstance(lbl, (list, tuple)) and len(lbl) > 0 else lbl
        groups.setdefault(key, []).append((lbl, val))

    for k in groups.keys():
        # sort each group by the order defined in `order`
        groups[k].sort(key=lambda x: titles.index(x[0][1]) if x[0][1] in titles else len(titles))

    # Sort groups by key
    sorted_keys = sorted(groups.keys())

    # Compute x positions so that bars from the same group are contiguous,
    # and add a small gap between groups.
    group_gap = 0.8
    positions = []
    bar_keys = []
    bar_values = []
    tick_labels = []
    group_centers = []
    start = 0.0
    for k in sorted_keys:
        items = groups[k]
        n = len(items)
        grp_pos = [start + i for i in range(n)]
        positions.extend(grp_pos)
        for lbl, val in items:
            bar_values.append(val)
            bar_keys.append(lbl[1])
        group_centers.append((start + (n - 1) / 2.0, str(k)))
        start = start + n + group_gap

    fig_width = max(8, 0.5 * max(1, len(positions)))
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    bars = ax.bar(positions, bar_values, color="tab:blue", edgecolor="black", alpha=0.85)

    # color bars by group and add legend
    cmap = plt.get_cmap("tab10")
    # reconstruct group key for each bar (groups were expanded in order of sorted_keys)
    color_map = {k: cmap(i % cmap.N) for i, k in enumerate(titles)}
    for rect, key in zip(bars, bar_keys):
        rect.set_facecolor(color_map[key])
    # add legend mapping colors to group keys
    handles = [Patch(facecolor=color_map[k], edgecolor="black", label=bar_names[str(k)]) for k in titles]
    ax.legend(handles=handles, title="Llama3-8B, H200", loc="upper right")
    ax.set_ylabel("Throughput (tokens/s)", fontweight='bold')
    ax.set_xlabel("(Input tokens, Output tokens)", fontweight='bold', labelpad=12)
    ax.set_xticks([], [])
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Draw separators between groups and add group labels
    for i, (center, glabel) in enumerate(group_centers):
        # separator after each group except the last
        if i < len(group_centers) - 1:
            # boundary is halfway between the end of this group and the start of next
            next_center = group_centers[i + 1][0]
            boundary = (center + next_center) / 2.0
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        # group label below x-axis
        ymin, ymax = ax.get_ylim()
        y_text = ymin - 0.01 * (ymax - ymin)
        ax.text(center, y_text, glabel, ha="center", va="top", fontsize=9)

    plt.tight_layout()
    # Make room for group labels below the ticks
    #fig.subplots_adjust(bottom=0)

    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
        print(f"Saved plot to {outpath}")
    else:
        plt.show()

def main():
        args = parse_args()
        labels: List[tuple] = []
        values: List[float] = []

        for path in args.files:
            val = extract_from_file(path)
            if isinstance(val, list):
                for key, v in val:
                    labels.append((f'({key[0]}, {key[1]})', key[2]))
                    values.append(v)
        make_bar_plot(labels, values, args.title, args.out)

if __name__ == "__main__":
        main()