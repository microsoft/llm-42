#!/usr/bin/env python3
"""
Plot matmul TP benchmark results from CSV data.

Reads results.csv produced by bench_matmul_tp.py and generates:
  - Slowdown heatmaps that combine all TP sizes into a single figure per op
    (rows=TP, cols=batch_size), with Triton on top and DeepGEMM on the bottom.

The llama3-70b pre-projection heatmap is also copied to
sosp-artifact/llm42-plots/figure4.pdf (the paper's Figure 4).

Can run on any machine with matplotlib+numpy — no GPU required.

Usage:
    python plot.py runs/h100/llama3-8b/results.csv
    python plot.py runs/h100/llama3-8b/results.csv --model llama3-8b
    python plot.py results1.csv results2.csv  # multiple CSVs
"""

import argparse
import csv
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import math

# Try to import model configs for GEMM shape labels; not required
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from bench_utils_tp import get_model_ops_config, get_tp_sharded_shape
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


# The paper's Figure 4 is the llama3-70b pre-projection slowdown heatmap;
# it is auto-published to sosp-artifact/llm42-plots/figure4.pdf after plotting.
FIGURE4_MODEL = "llama3-70b"
FIGURE4_OP = "preproj"


def load_csv(csv_path: str) -> list[dict]:
    """Load results CSV into list of row dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["tp_size"] = int(row["tp_size"])
            row["batch_size"] = int(row["batch_size"])
            row["M"] = int(row["M"])
            row["K"] = int(row["K"])
            row["N"] = int(row["N"])
            row["tflops"] = float(row["tflops"]) if row["tflops"] else None
            row["avg_us"] = float(row["avg_us"]) if row["avg_us"] else None
            row["slowdown_vs_cublas"] = float(row["slowdown_vs_cublas"]) if row.get("slowdown_vs_cublas") else None
            rows.append(row)
    return rows


def build_plot_data(rows: list[dict]):
    """Build nested dict: (op, tp) -> {kernel -> {batch_size -> tflops}}."""
    plot_data = {}
    for r in rows:
        key = (r["op"], r["tp_size"])
        if key not in plot_data:
            plot_data[key] = {}
        kn = r["kernel"]
        if kn not in plot_data[key]:
            plot_data[key][kn] = {}
        if r["tflops"] is not None:
            plot_data[key][kn][r["batch_size"]] = r["tflops"]
    return plot_data


def plot_slowdown_heatmap(plot_data, rows, model, out_dir):
    """Slowdown heatmaps per op: Triton on top, DeepGEMM on bottom.

    Returns a dict mapping each op name to its saved PDF path.
    """
    saved = {}
    ops = sorted(set(op for op, _ in plot_data.keys()))
    tp_sizes = sorted(set(tp for _, tp in plot_data.keys()))
    all_batches = sorted(
        bs for kd in plot_data.values() for bsd in kd.values() for bs in bsd.keys()
        if bs <= 2048
    )
    all_batches = sorted(set(all_batches))
    slowdown_kernels = ["Triton", "DeepGEMM"]

    tp_tag = "_".join(str(t) for t in tp_sizes)

    for op_name in ops:
        op_vals = []
        for tp in tp_sizes:
            data = plot_data.get((op_name, tp), {})
            for kernel_name in slowdown_kernels:
                for bs in all_batches:
                    cublas_t = data.get("cuBLAS", {}).get(bs)
                    kernel_t = data.get(kernel_name, {}).get(bs)
                    if cublas_t and kernel_t:
                        op_vals.append(cublas_t / kernel_t)
        vmax = max(2.0, math.ceil(max(op_vals, default=2.0) * 2) / 2)
        norm = TwoSlopeNorm(vmin=0.8, vcenter=1.10, vmax=vmax)

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 10),
        )
        #fig.suptitle(f"cuBLAS slowdown by kernel — {op_name}", fontsize=16, fontweight="bold")
        fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.10, hspace=0.05)
        cmap = matplotlib.colormaps["RdYlGn_r"].copy()
        cmap.set_bad(color="#d9d9d9")
        images = []

        for idx, kernel_name in enumerate(slowdown_kernels):
            ax = axes[idx]
            heatmap_data = []
            tp_labels = []

            for tp in tp_sizes:
                data = plot_data.get((op_name, tp), {})
                row = []
                for bs in all_batches:
                    cublas_t = data.get("cuBLAS", {}).get(bs)
                    kernel_t = data.get(kernel_name, {}).get(bs)
                    if cublas_t and kernel_t:
                        row.append(cublas_t / kernel_t)
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
                tp_labels.append(f"TP={tp}")

            arr = np.array(heatmap_data, dtype=float)
            masked_arr = np.ma.masked_invalid(arr)
            im = ax.imshow(
                masked_arr,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            images.append(im)

            ax.set_xticks(range(len(all_batches)))
            ax.set_yticks(range(len(tp_labels)))
            ax.set_yticklabels(tp_labels, fontsize=14, fontweight="bold")
            ax.set_ylabel(kernel_name, fontsize=18, fontweight="bold", rotation=90, labelpad=22)
            if idx == 0:
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            else:
                ax.set_xticklabels([str(b) for b in all_batches], rotation=45, fontsize=16)
                ax.set_xlabel("# Tokens in the batch", fontsize=18, fontweight="bold")

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    if np.isnan(val):
                        text = "N/A"
                        color = "black"
                    else:
                        text = f"{val:.2f}x"
                        color = "white" if val > 0.5 * (0.8 + vmax) else "black"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=color,
                        fontweight="bold",
                    )

        cbar = fig.colorbar(images[0], ax=axes, location="top", pad=0.03, fraction=0.06, aspect=40)
        cbar.set_label("Slowdown (>1 means batch-invariant is slower)", fontsize=18, fontweight="bold", labelpad=8)
        tick_candidates = [0.8, 1.0, min(1.1, vmax), min(1.5, vmax), min(2.5, vmax), vmax]
        cbar.set_ticks(sorted(set(round(t, 2) for t in tick_candidates if t <= vmax)))
        cbar.ax.tick_params(labelsize=16)

        fname = os.path.join(out_dir, f"{model}_slowdown_heatmap_{op_name}_tp{tp_tag}.pdf")
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"  Saved: {fname}")
        saved[op_name] = fname

    return saved


def export_paper_figure(saved_paths: dict, model: str):
    """Publish the pre-proj heatmap as sosp-artifact/llm42-plots/figure4.pdf.

    Figure 4 in the paper is the llama3-70b pre-projection heatmap, so only
    that model's pre-proj plot is copied; other models/ops never overwrite it.
    """
    if model != FIGURE4_MODEL:
        return
    src = saved_paths.get(FIGURE4_OP)
    if not src or not os.path.exists(src):
        return
    plots_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm42-plots")
    )
    os.makedirs(plots_dir, exist_ok=True)
    dst = os.path.join(plots_dir, "figure4.pdf")
    shutil.copyfile(src, dst)
    print(f"  Exported paper figure: {dst}")


def plot_csv(csv_path: str, model: str = None, out_dir: str = None):
    """Generate the slowdown heatmap(s) from a single results CSV."""
    rows = load_csv(csv_path)
    if not rows:
        print(f"  No data in {csv_path}")
        return

    if out_dir is None:
        out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    if model is None:
        # Infer from directory path: .../llama3-8b/results.csv
        parent = os.path.basename(os.path.dirname(csv_path))
        model = parent if parent else "unknown"

    plot_data = build_plot_data(rows)
    print(f"\nPlotting {csv_path} (model={model}, {len(rows)} rows)")

    saved = plot_slowdown_heatmap(plot_data, rows, model, out_dir)
    export_paper_figure(saved, model)

    print(f"  All plots saved to: {out_dir}")


def auto_discover_csvs(root=None):
    """Find all results.csv under runs/ and return [(csv_path, model_name), ...]."""
    if root is None:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "results.csv":
                csv_path = os.path.join(dirpath, fn)
                model = os.path.basename(dirpath)
                found.append((csv_path, model))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(
        description="Plot matmul TP benchmark results from CSV (no GPU needed)")
    parser.add_argument("csv_files", nargs="*", default=[],
                        help="Path(s) to results.csv (if none, auto-discovers under runs/)")
    parser.add_argument("--model", default=None,
                        help="Model name for plot filenames (auto-detected from path)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for plots (default: same dir as CSV)")
    parser.add_argument("--runs-dir", default=None,
                        help="Root directory for auto-discovery (default: runs/)")
    args = parser.parse_args()

    if args.csv_files:
        targets = [(p, args.model) for p in args.csv_files]
    else:
        targets = auto_discover_csvs(args.runs_dir)
        if not targets:
            print("No results.csv found. Pass CSV paths or check runs/ directory.")
            return
        print("Auto-discovered %d result(s):" % len(targets))
        for csv_path, model in targets:
            print("  %s: %s" % (model, csv_path))

    for csv_path, model in targets:
        plot_csv(csv_path, model=model, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
