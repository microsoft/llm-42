#!/usr/bin/env python3
"""
Plot CDF curves for TTFT, TPOT, and E2E latency from per-request latency files.

Usage:
    python plot_cdf.py results/latencies/ --output-dir plots/

Creates directory structure:
    plots/
        sharegpt/
            qps_6/
                step_16/
                    ttft.pdf
                    tpot.pdf
                    e2e.pdf
                step_32/...
                ...
        arxiv/
            qps_6/...

Each plot shows CDF lines for:
- SGLang (Non-Deterministic): baseline with det_ratio=1.0
- SGLang (Deterministic): global_det with det_ratio=1.0
- Ours (1% Deterministic): det_infer with det_ratio=0.01
- Ours (5% Deterministic): det_infer with det_ratio=0.05
- Ours (10% Deterministic): det_infer with det_ratio=0.10
- Ours (100% Deterministic): det_infer with det_ratio=1.0
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Line display settings for each configuration
LINE_CONFIGS = [
    {"label": "SGLang (Non-Deterministic)", "config": "baseline", "det_ratio": "1.0", "color": "#1f77b4", "linestyle": "-"},
    {"label": "SGLang (Deterministic)", "config": "global_det", "det_ratio": "1.0", "color": "#ff7f0e", "linestyle": "-"},
    {"label": "Ours (1% Deterministic)", "config": None, "det_ratio": "0.01", "color": "#2ca02c", "linestyle": "-"},
    {"label": "Ours (5% Deterministic)", "config": None, "det_ratio": "0.05", "color": "#d62728", "linestyle": "--"},
    {"label": "Ours (10% Deterministic)", "config": None, "det_ratio": "0.10", "color": "#9467bd", "linestyle": "-."},
    {"label": "Ours (100% Deterministic)", "config": None, "det_ratio": "1.0", "color": "#8c564b", "linestyle": ":"},
]

METRICS = [
    ("ttft_ms", "ttft", "TTFT (ms)"),
    ("tpot_ms", "tpot", "TPOT (ms)"),
    ("e2e_latency_ms", "e2e", "E2E Latency (ms)"),
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


def plot_cdf_for_step(all_data: dict, dataset: str, rate: str, step: str, output_dir: str):
    """
    Plot CDF for a specific dataset/rate/step combination.
    Creates one plot per metric showing all configuration lines.
    """
    # Create output directory: plots/dataset/qps_rate/step_step/
    plot_dir = os.path.join(output_dir, dataset, f"qps_{rate}", f"step_{step}")
    os.makedirs(plot_dir, exist_ok=True)

    for metric_key, metric_file, metric_label in METRICS:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot each line configuration
        for line_cfg in LINE_CONFIGS:
            # For "Ours" lines, find any det_infer_step config matching this step and det_ratio
            if line_cfg["config"] is None:
                # Search for det_infer_step matching the current step size
                found = False
                for config_name in [f"det_infer_step{step}"]:
                    key = f"{config_name}_{dataset}_rate{rate}_det{line_cfg['det_ratio']}"
                    if key in all_data:
                        data = np.array(all_data[key][metric_key])
                        found = True
                        break
                
                if not found:
                    continue
            else:
                # Specific config (baseline or global_det)
                key = f"{line_cfg['config']}_{dataset}_rate{rate}_det{line_cfg['det_ratio']}"
                if key not in all_data:
                    continue
                data = np.array(all_data[key][metric_key])

            if len(data) > 0:
                sorted_data = np.sort(data)
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(
                    sorted_data, cdf,
                    label=line_cfg["label"],
                    color=line_cfg["color"],
                    linestyle=line_cfg["linestyle"],
                    linewidth=1.5,
                    alpha=0.85
                )

        ax.set_xlabel(metric_label, fontsize=16)
        ax.set_ylabel("CDF", fontsize=16)
        ax.set_title(f"{dataset.capitalize()} - {rate} QPS - Step {step}", fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.legend(fontsize=16, loc="lower right")

        output_file = os.path.join(plot_dir, f"{metric_file}.pdf")
        plt.savefig(output_file, dpi=1200, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()


def plot_all_cdfs(all_data: dict, output_dir: str):
    """
    Generate all CDF plots organized by dataset/qps/step.
    """
    # Extract unique combinations of dataset, rate, and step
    combinations = set()
    for key, value in all_data.items():
        info = value["info"]
        # Extract step size from config name if it's a det_infer config
        if "det_infer_step" in info["config"]:
            step = info["config"].replace("det_infer_step", "")
            combinations.add((info["dataset"], info["rate"], step))

    if not combinations:
        print("No det_infer configurations found in data")
        return

    # Generate plots for each combination
    for dataset, rate, step in sorted(combinations):
        print(f"\nGenerating plots for {dataset} @ {rate} QPS, step {step}")
        plot_cdf_for_step(all_data, dataset, rate, step, output_dir)


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
    plot_all_cdfs(all_data, args.output_dir)
    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
