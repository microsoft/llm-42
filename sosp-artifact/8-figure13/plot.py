#!/usr/bin/env python3
"""
Plot paper-ready heatmaps from rollback profiling summary.

Produces two heatmaps per model:
  - {model}_recompute_cost_heatmap.pdf         (rollback % of output tokens)
  - {model}_normalized_throughput_heatmap.pdf  (total throughput normalized to best)

Usage:
    python plot.py --run-dir runs/h100_llama-3.1-8b-instruct-tp1_fa3_20260321_153600
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.bottom"] = True

# Recompute-cost colormap (indigo -> orange)
CMAP_COLORS = ["#4f46e5", "#818cf8", "#c4b5fd", "#fcd34d", "#f97316"]
CMAP = LinearSegmentedColormap.from_list("latency_aesthetic", CMAP_COLORS, N=256)

# Throughput colormap (green -> red warm)
CMAP2_COLORS = ["#065f46", "#34d399", "#a7f3d0", "#fcd34d", "#ef4444"]
CMAP2 = LinearSegmentedColormap.from_list("latency_warm", CMAP2_COLORS, N=256)

INVALID_COLOR = "#f5f5f5"


def _pick_text_color(cmap, norm, val):
    """Return 'white' or 'black' based on the actual luminance of the cell."""
    rgba = cmap(norm(val))
    r, g, b = rgba[0] * 255, rgba[1] * 255, rgba[2] * 255
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 140 else "black"


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    with open(summary_path) as f:
        return json.load(f)


def load_throughput_from_logs(run_dir: Path) -> list:
    """Load total_throughput from client .log files in ws*_bs*/ dirs."""
    profiles = []
    for log_file in sorted(run_dir.glob("ws*_bs*/log_*.log")):
        dir_name = log_file.parent.name
        m = re.match(r"ws(\d+)_bs(\d+)", dir_name)
        if not m:
            continue
        ws, bs = int(m.group(1)), int(m.group(2))
        throughput = 0.0
        with open(log_file) as f:
            for line in f:
                if "Total token throughput" in line:
                    throughput = float(line.split(":")[-1].strip())
                    break
        profiles.append({
            "window_size": ws,
            "verify_batch_size": bs,
            "total_throughput": throughput,
        })
    return profiles


def build_matrix(profiles, ws_list, bs_list, key):
    """Build 2D matrix [batch_sizes x window_sizes] from profiles."""
    lookup = {}
    for p in profiles:
        lookup[(p["window_size"], p["verify_batch_size"])] = p[key]

    matrix = np.full((len(bs_list), len(ws_list)), np.nan)
    for i, bs in enumerate(bs_list):
        for j, ws in enumerate(ws_list):
            if (ws, bs) in lookup:
                matrix[i, j] = lookup[(ws, bs)]
    return matrix


def infer_paper_prefix(name: str) -> str:
    """Paper-figure prefix for a run/model name.

    figure13a for the 8B model, figure13b for the 70B model, else "".
    """
    s = str(name).lower()
    if "70b" in s:
        return "figure13b"
    if "8b" in s:
        return "figure13a"
    return ""


def export_paper_figure(pdf_path, figure_name: str) -> None:
    """Copy a generated heatmap PDF into sosp-artifact/llm42-plots/<figure_name>.

    The 8B run exports figure13a-*.pdf; the 70B run exports figure13b-*.pdf.
    """
    if not pdf_path or not Path(pdf_path).exists() or not figure_name:
        return
    plots_dir = (Path(__file__).resolve().parent / ".." / "llm42-plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    dst = plots_dir / figure_name
    shutil.copyfile(pdf_path, dst)
    print(f"Exported paper figure: {dst}")


def plot_heatmap(matrix, ws_list, bs_list, cbar_label, output_path,
                 fmt=".2f", suffix="", cmap_override=None):
    """Render a single heatmap to its own figure using vector cells."""
    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = (cmap_override or CMAP).copy()
    cmap.set_bad(color=INVALID_COLOR)
    masked = np.ma.masked_invalid(matrix)

    valid_vals = matrix[~np.isnan(matrix)]
    if len(valid_vals) > 0:
        vmin, vmax = valid_vals.min(), valid_vals.max()
    else:
        vmin, vmax = 0, 1
    img_norm = Normalize(vmin=vmin, vmax=vmax)

    # Vector cell patches (pcolormesh) so PDF viewers never drop the raster layer.
    x_edges = np.arange(len(ws_list) + 1) - 0.5
    y_edges = np.arange(len(bs_list) + 1) - 0.5
    im = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, norm=img_norm,
                       shading="flat")
    ax.set_xlim(-0.5, len(ws_list) - 0.5)
    ax.set_ylim(len(bs_list) - 0.5, -0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, location="top")
    cbar.ax.set_xlabel(cbar_label, fontsize=22, fontweight="bold")
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)

    ax.set_xticks(np.arange(len(ws_list)))
    ax.set_yticks(np.arange(len(bs_list)))
    ax.set_xticklabels(ws_list, fontsize=20, fontweight="medium")
    ax.set_yticklabels(bs_list, fontsize=20, fontweight="medium")
    ax.set_xlabel("Per-request window size", fontsize=22, fontweight="bold")
    ax.set_ylabel("Number of requests", fontsize=22, fontweight="bold")

    for i in range(len(bs_list)):
        for j in range(len(ws_list)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = _pick_text_color(cmap, img_norm, val)
            ax.text(j, i, f"{val:{fmt}}{suffix}", ha="center", va="center",
                    color=text_color, fontsize=13, fontweight="semibold")

    ax.set_xticks(np.arange(len(ws_list) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(bs_list) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#e5e5e5", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot rollback profiling heatmaps")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to run directory containing summary.json")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory for PDFs (default: the run directory)")
    parser.add_argument("--paper-prefix", type=str, default=None,
                        help="Prefix for the exported paper figures in ../llm42-plots/ "
                             "(default: inferred; 8B->figure13a, 70B->figure13b). "
                             "Exports <prefix>-recompute.pdf and <prefix>-throughput.pdf.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.run_dir)
    profiles = summary["profiles"]

    # Extract model name from run directory name (e.g. "llama-3.1-8b-instruct"
    # from "h100_llama-3.1-8b-instruct-tp1_fa3_20260321_153600")
    dir_name = args.run_dir.name
    m = re.search(r"_(.+?)[-_]tp\d+", dir_name)
    model_name = m.group(1) if m else dir_name

    # Paper-figure prefix (8B->figure13a, 70B->figure13b); "" disables export.
    paper_prefix = args.paper_prefix or infer_paper_prefix(dir_name)

    # Determine axes from data
    ws_list = sorted(set(p["window_size"] for p in profiles))
    bs_list = sorted(set(p["verify_batch_size"] for p in profiles), reverse=True)

    # --- Heatmap 1: recompute cost (rollback %) ---
    rb_matrix = build_matrix(profiles, ws_list, bs_list, "rollback_pct")
    rb_path = output_dir / f"{model_name}_recompute_cost_heatmap.pdf"
    plot_heatmap(
        rb_matrix, ws_list, bs_list,
        cbar_label="Recompute Cost",
        output_path=rb_path,
        fmt=".2f", suffix="%", cmap_override=CMAP,
    )
    if paper_prefix:
        export_paper_figure(rb_path, f"{paper_prefix}-recompute.pdf")

    # --- Heatmap 2: normalized total throughput ---
    # Prefer throughput parsed from client logs; fall back to summary.json.
    tp_profiles = load_throughput_from_logs(args.run_dir)
    if not tp_profiles or all(p["total_throughput"] == 0 for p in tp_profiles):
        tp_profiles = profiles
    tp_matrix = build_matrix(tp_profiles, ws_list, bs_list, "total_throughput")
    valid_tp = tp_matrix[~np.isnan(tp_matrix)]
    if len(valid_tp) > 0 and valid_tp.max() > 0:
        norm_tp_matrix = tp_matrix / valid_tp.max()
    else:
        norm_tp_matrix = tp_matrix

    tp_path = output_dir / f"{model_name}_normalized_throughput_heatmap.pdf"
    plot_heatmap(
        norm_tp_matrix, ws_list, bs_list,
        cbar_label="Normalized Throughput",
        output_path=tp_path,
        fmt=".2f", suffix="x", cmap_override=CMAP2,
    )
    if paper_prefix:
        export_paper_figure(tp_path, f"{paper_prefix}-throughput.pdf")


if __name__ == "__main__":
    main()
