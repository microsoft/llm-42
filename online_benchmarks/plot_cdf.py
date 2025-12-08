#!/usr/bin/env python3
"""
Plot CDF curves for TTFT, TPOT, and E2E latency from per-request latency files.

Usage:
    python plot_cdf.py results/latencies/ --output-dir plots/

Creates plots like: cdf_ttft_detratio_0.1.pdf
Each plot has rows for each rate (1, 4, 8, 16, 32) and columns for datasets (sharegpt, arxiv).
Each subplot has 8 config lines (baseline, global_det, det_infer_step*).
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Config display settings
CONFIG_ORDER = [
    "baseline",
    "global_det",
    "det_infer_step512",
    "det_infer_step256",
    "det_infer_step128",
    "det_infer_step64",
    "det_infer_step32",
    "det_infer_step16",
]

CONFIG_COLORS = {
    "baseline": "#1f77b4",           # blue
    "global_det": "#ff7f0e",         # orange
    "det_infer_step512": "#2ca02c",  # green
    "det_infer_step256": "#d62728",  # red
    "det_infer_step128": "#9467bd",  # purple
    "det_infer_step64": "#8c564b",   # brown
    "det_infer_step32": "#e377c2",   # pink
    "det_infer_step16": "#7f7f7f",   # gray
}

CONFIG_LABELS = {
    "baseline": "Baseline",
    "global_det": "Global Det",
    "det_infer_step512": "DetInfer-512",
    "det_infer_step256": "DetInfer-256",
    "det_infer_step128": "DetInfer-128",
    "det_infer_step64": "DetInfer-64",
    "det_infer_step32": "DetInfer-32",
    "det_infer_step16": "DetInfer-16",
}

METRICS = [
    ("ttft_ms", "TTFT", "Time to First Token (ms)"),
    ("tpot_ms", "TPOT", "Time per Output Token (ms)"),
    ("e2e_latency_ms", "E2E", "End-to-End Latency (ms)"),
]


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def parse_latency_filename(filename: str) -> dict:
    """Parse config info from filename like 'baseline_sharegpt_rate8_det0.1.jsonl'."""
    info = {"config": "", "dataset": "", "rate": "", "det": ""}
    name = filename.replace(".jsonl", "")

    for dataset in ("sharegpt", "arxiv"):
        if f"_{dataset}_" in name:
            info["config"] = name.split(f"_{dataset}_")[0]
            info["dataset"] = dataset
            remainder = name.split(f"_{dataset}_")[1]
            for part in remainder.split("_"):
                if part.startswith("rate"):
                    info["rate"] = part.replace("rate", "")
                elif part.startswith("det"):
                    info["det"] = part.replace("det", "")
            break

    return info


def load_latency_files(latency_dir: str) -> dict[str, dict]:
    """Load per-request latency files from a directory."""
    latency_path = Path(latency_dir)
    all_data = {}

    for f in latency_path.glob("*.jsonl"):
        info = parse_latency_filename(f.name)
        if not info["config"]:
            continue

        key = f"{info['config']}_{info['dataset']}_rate{info['rate']}_det{info['det']}"
        ttfts, tpots, e2es = [], [], []

        for item in load_jsonl(str(f)):
            ttfts.append(item.get("ttft_ms", 0))
            tpots.append(item.get("tpot_ms", 0))
            e2es.append(item.get("e2e_latency_ms", 0))

        all_data[key] = {
            "info": info,
            "ttft_ms": ttfts,
            "tpot_ms": tpots,
            "e2e_latency_ms": e2es,
        }

    return all_data


def plot_cdf_grid(all_data: dict, output_dir: str):
    """
    Plot CDF grids from latency data.

    Creates one PDF per metric/det_ratio combination.
    Each PDF has rows for rates and columns for datasets.
    Each subplot has lines for all 8 configs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique values
    det_ratios = sorted(
        set(d["info"]["det"] for d in all_data.values()),
        key=lambda x: float(x) if x else 0
    )

    for metric_key, metric_short, metric_label in METRICS:
        for det in det_ratios:
            # Filter data for this det_ratio
            subset = {k: v for k, v in all_data.items() if v["info"]["det"] == det}
            if not subset:
                continue

            # Get rates and datasets present
            rates = sorted(
                set(v["info"]["rate"] for v in subset.values()),
                key=lambda x: float(x) if x else 0
            )
            datasets = sorted(set(v["info"]["dataset"] for v in subset.values()))

            if not rates or not datasets:
                continue

            n_rows, n_cols = len(rates), len(datasets)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

            for row_idx, rate in enumerate(rates):
                for col_idx, dataset in enumerate(datasets):
                    ax = axes[row_idx, col_idx]

                    # Plot each config
                    for config in CONFIG_ORDER:
                        key = f"{config}_{dataset}_rate{rate}_det{det}"
                        if key not in subset:
                            continue

                        data = np.array(subset[key][metric_key])
                        if len(data) > 0:
                            sorted_data = np.sort(data)
                            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                            ax.plot(
                                sorted_data, cdf,
                                label=CONFIG_LABELS.get(config, config),
                                color=CONFIG_COLORS.get(config, "gray"),
                                linewidth=2, alpha=0.8
                            )

                    ax.set_xlabel(metric_label)
                    ax.set_ylabel("CDF")
                    ax.set_title(f"{dataset.capitalize()} @ {rate} QPS")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.02)
                    ax.set_xlim(left=0)

                    # Legend on top-right subplot only
                    if row_idx == 0 and col_idx == n_cols - 1:
                        ax.legend(fontsize=7, loc="lower right")

            fig.suptitle(f"{metric_short} CDF - Det Ratio: {det}", fontsize=14, fontweight='bold')
            plt.tight_layout()

            output_file = os.path.join(output_dir, f"cdf_{metric_short.lower()}_detratio_{det}.pdf")
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot CDF curves for latency metrics')
    parser.add_argument('latency_dir', help='Directory containing per-request latency JSONL files')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    args = parser.parse_args()

    if not os.path.isdir(args.latency_dir):
        print(f"Error: {args.latency_dir} is not a directory")
        return

    print(f"Loading latency files from: {args.latency_dir}")
    all_data = load_latency_files(args.latency_dir)

    if not all_data:
        print("No latency files found")
        return

    print(f"Loaded {len(all_data)} latency files")
    plot_cdf_grid(all_data, args.output_dir)
    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
