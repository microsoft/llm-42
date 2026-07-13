#!/usr/bin/env python3
"""
Generate PDF throughput-comparison plots from multi-dataset benchmark results.

Produces one plot per LLM42 config (e.g. ws_32_bs_16, ws_64_bs_32).
Each plot contains:
  - X-axis: dataset configurations
  - Grouped bars: SGLang non-det, SGLang global-det, LLM42@2%, @5%, …, @100%
  - Y-axis: total throughput (tokens/s)
  - Speedup annotations relative to non-deterministic baseline

The visual style matches the paper figure (paper-plots/plot.py): compact
SOSP two-column format, serif fonts, solid fills with white edges, a framed
legend, and a K-formatted throughput axis.

Also saves a CSV with all extracted data.

Usage:
    # Process all model runs under runs/:
    python plot.py

    # Process specific results directories:
    python plot.py \\
        --results-dirs runs/h100_model_tp4_fa3_n2048/results/ \\
        --output-dir plots/
"""

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_logs import load_logs_from_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Publication style — SOSP two-column format (matches paper-plots/plot.py)
# ---------------------------------------------------------------------------
COLUMN_WIDTH = 3.33  # inches (single column)
TEXT_WIDTH = 7.1     # inches (two-column span)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 6,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.6,
    "patch.linewidth": 0.6,
})

# Baseline bar styles: key -> (color, alpha, hatch, label).  Solid fills with
# white edges, matching the paper palette (sglang = red/orange family).
BASELINE_STYLES = {
    "non_det":           ("#d62728", 1.0,  "",   "SGLang-nondet"),
    "global_det":        ("#8b0000", 0.7,  "//", "SGLang-det-deepgemm"),
    "global_det_triton": ("#ff7f0e", 0.85, "",   "SGLang-det-triton"),
}

# LLM-42 ratio bars — green/teal spectrum, styled by ratio order of appearance.
LLM42_COLOR_CYCLE = ["#2ca02c", "#006d2c", "#74c476", "#31a354", "#a1d99b", "#00441b"]
LLM42_ALPHA_CYCLE = [1.0, 0.9, 1.0, 0.75, 0.9, 0.6]
LLM42_HATCH_CYCLE = ["", "//", "\\\\", "", "xx", "//"]


def ratio_style(idx: int, ratio: float):
    """(color, alpha, hatch, label) for an LLM-42 ratio bar by appearance order."""
    color = LLM42_COLOR_CYCLE[idx % len(LLM42_COLOR_CYCLE)]
    alpha = LLM42_ALPHA_CYCLE[idx % len(LLM42_ALPHA_CYCLE)]
    hatch = LLM42_HATCH_CYCLE[idx % len(LLM42_HATCH_CYCLE)]
    return color, alpha, hatch, f"LLM-42 @ {ratio * 100:.0f}%"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(filepath: Path) -> list:
    results = []
    if not filepath.exists():
        return results
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def load_summary_csv(filepath: Path) -> list:
    """Load results from a summary.csv produced by summarize_results_csv.py.

    Returns a list of dicts with keys matching what the rest of the script
    expects: config_name, deterministic_ratio, throughput.
    """
    results = []
    if not filepath.exists():
        return results
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row["config"].strip()
            tp = float(row["tokens-per-second"])
            # Extract ratio from config names like llm42_ws_32_bs_16_ratio_0.05
            ratio_match = re.search(r"_ratio_(\d+(?:\.\d+)?)", config)
            det_ratio = float(ratio_match.group(1)) if ratio_match else 1.0
            # Normalise config_name to what downstream code expects
            if config == "sglang_non_deterministic":
                config_name = config
            elif config == "sglang_global_deterministic":
                config_name = config
            elif config == "sglang_global_deterministic_triton":
                config_name = config
            else:
                config_name = config  # e.g. llm42_ws_32_bs_16_ratio_0.02
            results.append({
                "config_name": config_name,
                "deterministic_ratio": det_ratio,
                "throughput": tp,
            })
    return results


def _fmt_size(n_str: str) -> str:
    """Format token count: 1024 -> 1K, 4096 -> 4K, else as-is."""
    n = int(n_str)
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return n_str


def dataset_label(dir_name: str) -> str:
    """Human-readable label from the result directory name."""
    if "arxiv" in dir_name:
        return "ArXiv"
    if "sharegpt" in dir_name:
        return "ShareGPT"
    m = re.search(r"random_in(\d+)_out(\d+)", dir_name)
    if m:
        return f"({_fmt_size(m.group(1))},{_fmt_size(m.group(2))})"
    return dir_name


def dataset_sort_key(label: str):
    """Sort order: ShareGPT, ArXiv, then random (in, out) ascending."""
    if label == "ShareGPT":
        return (0, 0, 0)
    if label == "ArXiv":
        return (0, 1, 0)
    m = re.search(r"\((\d+)(K?),\s*(\d+)(K?)\)", label)
    if m:
        a = int(m.group(1)) * (1024 if m.group(2) else 1)
        b = int(m.group(3)) * (1024 if m.group(4) else 1)
        return (1, a, b)
    return (2, 0, 0)


def llm42_base_config(config_name: str) -> str:
    """
    Extract the LLM42 window/batch-size portion of a config name.
    e.g. 'llm42_ws_32_bs_16' -> 'ws_32_bs_16'
    """
    m = re.search(r"(ws_\d+_bs_\d+)", config_name)
    return m.group(1) if m else config_name


def infer_model_label(name: str) -> str:
    """Model + TP label from a run-dir name.

    e.g. 'h100_pcie_llama-3.1-8b-instruct-tp1_fa3_n256' -> 'Llama-3-8B (TP-1)'.
    """
    s = name.lower()
    if "70b" in s:
        model = "Llama-3-70B"
    elif "8b" in s:
        model = "Llama-3-8B"
    else:
        return ""
    m = re.search(r"tp(\d+)", s)
    return f"{model} (TP-{m.group(1)})" if m else model


def infer_paper_figure(name: str) -> str:
    """Paper-figure filename for a run/model name.

    figure10a.pdf for the 8B model, figure10b.pdf for the 70B model, else "".
    """
    s = str(name).lower()
    if "70b" in s:
        return "figure10b.pdf"
    if "8b" in s:
        return "figure10a.pdf"
    return ""


def export_paper_figure(pdf_path, figure_name: str) -> None:
    """Copy a generated throughput PDF into sosp-artifact/llm42-plots/<figure_name>.

    The 8B run exports figure10a.pdf; the 70B run exports figure10b.pdf.
    """
    if not pdf_path or not Path(pdf_path).exists() or not figure_name:
        return
    plots_dir = (Path(__file__).resolve().parent / ".." / "llm42-plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    dst = plots_dir / figure_name
    shutil.copyfile(pdf_path, dst)
    print(f"Exported paper figure: {dst}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    data: dict,
    llm42_cfg: str,
    ratios: list,
    output_path: Path,
    model_label: str = "",
):
    """
    Draw a paper-styled grouped-bar chart for one LLM42 configuration.

    Parameters
    ----------
    data : {dataset_label: {"non_det": tp, "global_det": tp, "global_det_triton": tp, ratio: tp, …}}
    llm42_cfg : e.g. "ws_32_bs_16"
    ratios : sorted list of det_ratio floats present for this config
    output_path : where to save the PDF
    model_label : e.g. "Llama-3-8B (TP-1)" shown in a box on the plot
    """
    datasets = sorted(data.keys(), key=dataset_sort_key)
    n_datasets = len(datasets)

    # Bar spec in draw order: sglang baselines, then LLM-42 ratios.
    # Only include a baseline if it was actually collected for >=1 dataset, so a
    # baseline that never ran doesn't leave an empty slot + legend entry.
    present_keys = set()
    for d in data.values():
        present_keys.update(d.keys())

    bar_specs = []  # (key, color, alpha, hatch, label)
    for key in ("non_det", "global_det", "global_det_triton"):
        if key not in present_keys:
            continue
        color, alpha, hatch, label = BASELINE_STYLES[key]
        bar_specs.append((key, color, alpha, hatch, label))
    for idx, r in enumerate(ratios):
        color, alpha, hatch, label = ratio_style(idx, r)
        bar_specs.append((r, color, alpha, hatch, label))

    n_bars = len(bar_specs)
    bar_w = max(0.10, min(0.22, 1.5 / max(n_bars, 1)))
    group_gap = bar_w * 1.2
    group_w = bar_w * n_bars
    x_pos = np.arange(n_datasets) * (group_w + group_gap)

    # Width scales with dataset count, capped at the two-column text width and
    # floored so the multi-entry legend still fits.
    fig_w = min(TEXT_WIDTH, max(4.7, n_datasets * 1.15 + 2.3))
    fig_h = 1.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    non_det_tp = [data[d].get("non_det", 0) for d in datasets]

    for i, (key, color, alpha, hatch, label) in enumerate(bar_specs):
        vals = [data[d].get(key, 0) for d in datasets]
        x = x_pos + i * bar_w
        bars = ax.bar(
            x, vals, bar_w,
            facecolor=color, edgecolor="white",
            linewidth=0.4, hatch=hatch, alpha=alpha,
        )
        # Speedup annotation (skip non-det itself)
        if key != "non_det":
            for bar, val, base in zip(bars, vals, non_det_tp):
                if val > 0 and base > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 50,
                        f"{val / base:.2f}x",
                        ha="center", va="bottom", fontsize=5, rotation=90,
                    )

    # Axes
    ax.set_ylabel("Throughput (tokens/s)", fontweight="bold", fontsize=6)
    group_centers = x_pos + (n_bars - 1) * bar_w / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(datasets, fontweight="bold", rotation=25, ha="right")
    ax.tick_params(axis="x", pad=2)

    # Legend (framed, compact) above the axes
    legend_handles = [
        Patch(facecolor=color, edgecolor="white", hatch=hatch, linewidth=0.4,
              alpha=alpha, label=label)
        for _, color, alpha, hatch, label in bar_specs
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.99),
        ncol=min(n_bars, 5), fontsize=6, handlelength=1.5, handleheight=1.0,
        columnspacing=0.8, handletextpad=0.4,
        frameon=True, fancybox=False, edgecolor="0.4", framealpha=0.95,
    )

    # Model + TP label box (top-left inside the axes)
    if model_label:
        ax.text(
            0.02, 0.95, model_label, transform=ax.transAxes,
            fontsize=7, fontweight="bold", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.7),
        )

    ax.yaxis.grid(True, linestyle="--", alpha=0.6, linewidth=0.6)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5000))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x // 1000)}K" if x >= 1000 else f"{int(x)}"
    ))
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.30)
    ax.set_xlim(left=-bar_w, right=x_pos[-1] + n_bars * bar_w + 0.1)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_results_dir(results_dir: Path, output_dir: Path, num_prompts: int = 0,
                        paper_figure: str = None) -> int:
    """
    Process a single results directory (or parent containing dataset subdirs)
    and generate throughput comparison plots + CSV.

    Returns 0 on success, 1 if no LLM42 data found.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model + TP label inferred from the run-dir name (parent of results/).
    model_label = (infer_model_label(Path(results_dir).parent.name)
                   or infer_model_label(Path(output_dir).name))

    # Paper-figure filename: 8B -> figure10a.pdf, 70B -> figure10b.pdf.
    fig_name = (paper_figure
                or infer_paper_figure(Path(results_dir).parent.name)
                or infer_paper_figure(Path(output_dir).name)
                or infer_paper_figure(model_label))

    all_data: dict = defaultdict(lambda: defaultdict(dict))
    llm42_cfgs_ratios: dict = defaultdict(set)

    # Expand: if results_dir has no data files, use its subdirectories.
    expanded_dirs = []
    rp = Path(results_dir)
    has_data = any(rp.glob("log_*.log")) or (rp / "benchmark_results.jsonl").exists() or (rp / "summary.csv").exists()
    if has_data:
        expanded_dirs.append(rp)
    else:
        subdirs = sorted(p for p in rp.iterdir() if p.is_dir())
        if subdirs:
            expanded_dirs.extend(subdirs)
        else:
            print(f"Warning: no data files in {rp}")

    for dp in expanded_dirs:
        ds_label = dataset_label(dp.name)

        # Primary: parse log files directly
        results = load_logs_from_dir(dp)

        # Fallback: summary.csv, then benchmark_results.jsonl
        if not results:
            summary_file = dp / "summary.csv"
            jsonl_file = dp / "benchmark_results.jsonl"
            if summary_file.exists():
                results = load_summary_csv(summary_file)
            else:
                results = load_results(jsonl_file)

        if not results:
            print(f"Warning: no results in {dp}")
            continue

        print(f"Loading: {ds_label}  ({len(results)} records)")

        for r in results:
            config_name = r.get("config_name", "unknown")
            det_ratio = r.get("deterministic_ratio", 1.0)

            if "throughput" in r:
                tp = r["throughput"]
            else:
                total_input = r.get("total_input_tokens", 0)
                total_output = r.get("total_output_tokens", 0)
                duration = r.get("duration", 1)
                tp = (total_input + total_output) / duration if duration > 0 else 0

            if config_name == "sglang_non_deterministic":
                for cfg in all_data:
                    all_data[cfg][ds_label]["non_det"] = tp
                all_data["__baseline__"][ds_label]["non_det"] = tp

            elif config_name == "sglang_global_deterministic":
                for cfg in all_data:
                    all_data[cfg][ds_label]["global_det"] = tp
                all_data["__baseline__"][ds_label]["global_det"] = tp

            elif config_name == "sglang_global_deterministic_triton":
                for cfg in all_data:
                    all_data[cfg][ds_label]["global_det_triton"] = tp
                all_data["__baseline__"][ds_label]["global_det_triton"] = tp

            elif "llm42" in config_name:
                lcfg = llm42_base_config(config_name)
                llm42_cfgs_ratios[lcfg].add(det_ratio)
                all_data[lcfg][ds_label][det_ratio] = tp

    # Propagate baselines into every discovered LLM42 config
    baselines = all_data.pop("__baseline__", {})
    for lcfg in llm42_cfgs_ratios:
        for ds_label, bl in baselines.items():
            for key in ("non_det", "global_det", "global_det_triton"):
                if key in bl:
                    all_data[lcfg][ds_label].setdefault(key, bl[key])

    if not llm42_cfgs_ratios:
        print(f"Warning: no LLM42 data found in {results_dir}")
        return 1

    # Determine num_prompts for filenames
    if not num_prompts:
        for rd in expanded_dirs:
            m = re.search(r"_n(\d+)", rd.name)
            if m:
                num_prompts = int(m.group(1))
                break
    n_tag = f"_n{num_prompts}" if num_prompts else ""

    # ---- Generate one PDF per LLM42 config ----
    generated = []
    for lcfg in sorted(llm42_cfgs_ratios):
        data = all_data[lcfg]
        if not data:
            continue
        ratios = sorted(llm42_cfgs_ratios[lcfg])
        out_pdf = output_dir / f"throughput_{lcfg}{n_tag}.pdf"
        plot_comparison(data, lcfg, ratios, out_pdf, model_label=model_label)
        generated.append((lcfg, out_pdf))

    # ---- Export the paper figure into llm42-plots/ (8B -> figure10a, 70B -> figure10b) ----
    if fig_name and generated:
        primary_cfg, primary_pdf = generated[0]
        if len(generated) > 1:
            print(f"NOTE: {len(generated)} LLM42 configs found; exporting '{primary_cfg}' "
                  f"as {fig_name} (override with --paper-figure).")
        export_paper_figure(primary_pdf, fig_name)

    # ---- Save CSV with all extracted numbers ----
    csv_path = output_dir / "throughput_data.csv"
    with open(csv_path, "w") as f:
        f.write("llm42_config,dataset,bar_key,throughput_tokens_per_sec\n")
        for lcfg in sorted(llm42_cfgs_ratios):
            for ds in all_data[lcfg]:
                ds_clean = ds.replace("\n", " ")
                for key, tp in all_data[lcfg][ds].items():
                    f.write(f"{lcfg},{ds_clean},{key},{tp:.2f}\n")
    print(f"Saved CSV: {csv_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Plot throughput comparison across datasets (one PDF per LLM42 config)",
    )
    parser.add_argument(
        "--results-dirs", nargs="+", default=None,
        help="Results directories to process (one per dataset, or a parent dir). "
             "If omitted, auto-discovers all runs/*/results/ under the script directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output PDFs and CSV (default: each model's run dir)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=0,
        help="Number of prompts (added to PDF filenames). Auto-detected from dir names if omitted.",
    )
    parser.add_argument(
        "--paper-figure", default=None,
        help="Filename to export into llm42-plots/ (default: inferred; 8B->figure10a.pdf, "
             "70B->figure10b.pdf).",
    )
    args = parser.parse_args()

    # Auto-discover all model runs if no --results-dirs given
    if args.results_dirs is None:
        script_dir = Path(__file__).resolve().parent
        runs_dir = script_dir / "runs"
        if not runs_dir.is_dir():
            print(f"Error: no runs/ directory found at {runs_dir}")
            return 1
        run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "results").is_dir())
        if not run_dirs:
            print(f"Error: no model run directories with results/ found under {runs_dir}")
            return 1

        print(f"Auto-discovered {len(run_dirs)} model run(s):\n")
        rc = 0
        # An explicit --paper-figure only makes sense for a single run; with
        # multiple runs, rely on per-run inference (8B->10a, 70B->10b).
        forced_fig = args.paper_figure if len(run_dirs) == 1 else None
        for run_dir in run_dirs:
            results_path = run_dir / "results"
            out = args.output_dir if args.output_dir else run_dir
            print(f"--- {run_dir.name} ---")
            process_results_dir(results_path, out, args.num_prompts, forced_fig)
            print()
        return rc

    # Explicit --results-dirs: process them as a single batch
    if args.output_dir is None:
        first = Path(args.results_dirs[0])
        if first.name == "results" or not (
            any(first.glob("log_*.log"))
            or (first / "summary.csv").exists()
            or (first / "benchmark_results.jsonl").exists()
        ):
            args.output_dir = first.parent
        else:
            args.output_dir = first.parent.parent

    # When multiple dirs are given, merge them into one run
    merged_parent = Path(args.results_dirs[0])
    return process_results_dir(merged_parent, args.output_dir, args.num_prompts,
                               args.paper_figure)


if __name__ == "__main__":
    raise SystemExit(main())
