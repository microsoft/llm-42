#!/usr/bin/env python3
"""
Generate comparison plots and tables for online (QPS-driven) benchmarks.

Produces per QPS value:
  1. CDF of E2E latency (all requests)
  2. CDF of E2E latency (non-deterministic requests only)
  3. TTFT table CSV (P50, P90, P99 for each config)
  4. Combined 2x2 CDF figure for paper-ready QPS selections

Usage:
    python plot.py --results-dirs runs/.../results/
    python plot.py --results-dirs runs/.../results/ --output-dir plots/
"""

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

# Publication style — match ../7-offline for two-column paper figures
COLUMN_WIDTH = 3.33  # inches
TEXT_WIDTH = 7.1     # inches (two-column span)
PAPER_STYLE = {
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 6,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "patch.linewidth": 0.6,
}

# ---------------------------------------------------------------------------
# Style palette
# ---------------------------------------------------------------------------
CONFIG_STYLES = {
    "sglang_non_deterministic": {"color": "#d62728", "linestyle": "-", "marker": "o",
                                   "label": "SGLang-nondet"},
    "sglang_deterministic":     {"color": "#8b0000", "linestyle": "--", "marker": "s",
                                   "label": "SGLang-det-deepgemm"},
    "sglang_global_deterministic": {"color": "#ff7f0e", "linestyle": "-.", "marker": "s",
                                      "label": "SGLang deterministic"},
}

LLM42_RATIO_STYLES = {
    0.02: {"color": "#2ca02c", "linestyle": "-", "marker": "^", "label": "Axiom @ 2%"},
    0.05: {"color": "#006d2c", "linestyle": "--", "marker": "D", "label": "Axiom @ 5%"},
    0.1:  {"color": "#74c476", "linestyle": "-.", "marker": "v", "label": "Axiom @ 10%"},
    0.2:  {"color": "#31a354", "linestyle": ":", "marker": "<", "label": "Axiom @ 20%"},
    0.5:  {"color": "#a1d99b", "linestyle": (0, (5, 1.5)), "marker": ">", "label": "Axiom @ 50%"},
    1.0:  {"color": "#00441b", "linestyle": (0, (3, 1, 1, 1)), "marker": "p", "label": "Axiom @ 100%"},
}


def get_style(config_name: str, det_ratio: float) -> dict:
    if config_name in CONFIG_STYLES:
        return CONFIG_STYLES[config_name]
    if "llm42" in config_name:
        return LLM42_RATIO_STYLES.get(det_ratio, {
            "color": "tab:gray", "linestyle": "--", "marker": "x",
            "label": f"LLM-42 @{int(det_ratio*100)}%"
        })
    return {"color": "tab:gray", "linestyle": "--", "marker": "x", "label": config_name}


def get_label(config_name: str, det_ratio: float) -> str:
    return get_style(config_name, det_ratio)["label"]


def config_sort_key(config_name: str, det_ratio: float):
    if config_name == "sglang_non_deterministic":
        return (0, config_name, 0)
    if config_name in ("sglang_deterministic", "sglang_global_deterministic"):
        return (1, config_name, 0)
    if "llm42" in config_name:
        return (2, config_name, det_ratio)
    return (3, config_name, det_ratio)


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


def compute_cdf(data: list):
    if not data:
        return np.array([]), np.array([])
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def build_output_prefix(experiment_name: str) -> str:
    lname = experiment_name.lower()
    model_match = re.search(r"(llama)-(\d+)(?:\.\d+)?-(\d+b)", lname)
    n_match = re.search(r"_n(\d+)", lname)

    if model_match and n_match:
        model_name = f"{model_match.group(1)}-{model_match.group(2)}-{model_match.group(3)}"
        return f"{model_name}_{n_match.group(1) and f'n{n_match.group(1)}'}_"

    sanitized = re.sub(r"[^a-z0-9]+", "_", lname).strip("_")
    return f"{sanitized}_"


def infer_paper_figure(name: str) -> str:
    """Paper-figure filename for a run/model name.

    figure11a.pdf for the 8B model, figure11b.pdf for the 70B model, else "".
    """
    s = str(name).lower()
    if "70b" in s:
        return "figure11b.pdf"
    if "8b" in s:
        return "figure11a.pdf"
    return ""


def infer_ttft_paper_figure(name: str) -> str:
    """Paper-figure filename for the P99 TTFT-ratio plot of a run/model name.

    figure12a.pdf for the 8B model, figure12b.pdf for the 70B model, else "".
    """
    s = str(name).lower()
    if "70b" in s:
        return "figure12b.pdf"
    if "8b" in s:
        return "figure12a.pdf"
    return ""


def export_paper_figure(pdf_path, figure_name: str) -> None:
    """Copy a generated PDF into sosp-artifact/llm42-plots/<figure_name>.

    E2E-latency combined CDF: 8B exports figure11a.pdf, 70B figure11b.pdf.
    P99 TTFT-ratio plot:      8B exports figure12a.pdf, 70B figure12b.pdf.
    """
    if not pdf_path or not Path(pdf_path).exists() or not figure_name:
        return
    plots_dir = (Path(__file__).resolve().parent / ".." / "llm42-plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    dst = plots_dir / figure_name
    shutil.copyfile(pdf_path, dst)
    print(f"Exported paper figure: {dst}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(results_dir: Path) -> dict:
    data = defaultdict(dict)
    rp = Path(results_dir)
    expanded_dirs = []
    if (rp / "benchmark_results.jsonl").exists():
        expanded_dirs.append(rp)
    else:
        expanded_dirs.extend(sorted(p for p in rp.iterdir() if p.is_dir()))

    for dp in expanded_dirs:
        ds_label = dp.name
        results = load_results(dp / "benchmark_results.jsonl")
        if not results:
            continue
        for r in results:
            config_name = r.get("config_name", "unknown")
            det_ratio = r.get("deterministic_ratio", 1.0)
            qps = r.get("qps", 0)
            data[ds_label][(config_name, det_ratio, qps)] = r
    return data


# ---------------------------------------------------------------------------
# Plot 1: CDF of E2E latency — all requests
# ---------------------------------------------------------------------------

def plot_cdf_all_per_qps(data: dict, dataset_label: str, output_dir: Path, output_prefix: str):
    qps_groups = defaultdict(dict)
    for (config_name, det_ratio, qps), r in data.items():
        qps_groups[qps][(config_name, det_ratio)] = r

    for qps in sorted(qps_groups.keys()):
        group = qps_groups[qps]
        fig, ax = plt.subplots(figsize=(10, 7))

        for (config_name, det_ratio) in sorted(group.keys(), key=lambda x: config_sort_key(*x)):
            r = group[(config_name, det_ratio)]
            raw = r.get("latencies", [])
            if not raw:
                continue
            raw_ms = [x * 1000 for x in raw if x is not None]
            x_vals, y_vals = compute_cdf(raw_ms)
            style = get_style(config_name, det_ratio)
            ax.plot(x_vals, y_vals, color=style["color"],
                    linestyle=style["linestyle"], linewidth=2,
                    label=style["label"])

        ax.set_xlabel("E2E Latency — all requests (ms)", fontsize=20, fontweight="bold")
        ax.set_ylabel("CDF", fontsize=20, fontweight="bold")
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=16, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        fname = f"{output_prefix}cdf_e2e_all_{dataset_label}_qps{qps}.pdf"
        plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def infer_combined_qps(experiment_name: str, available_qps: list[float]) -> list[float]:
    lname = experiment_name.lower()
    if "8b" in lname:
        preferred = [4, 4.5, 5, 5.5]
    elif "70b" in lname:
        preferred = [2.5, 3, 3.5, 4]
    elif any(q > 4 for q in available_qps):
        preferred = [4, 4.5, 5, 5.5]
    else:
        preferred = [2.5, 3, 3.5, 4]

    selected = [q for q in preferred if q in set(available_qps)]
    if len(selected) < 4:
        for q in sorted(available_qps):
            if q not in selected:
                selected.append(q)
            if len(selected) == 4:
                break
    return selected[:4]


def plot_cdf_all_combined(data: dict, dataset_label: str, experiment_name: str, output_dir: Path, output_prefix: str):
    qps_groups = defaultdict(dict)
    for (config_name, det_ratio, qps), r in data.items():
        qps_groups[qps][(config_name, det_ratio)] = r

    selected_qps = infer_combined_qps(experiment_name, sorted(qps_groups.keys()))
    if not selected_qps:
        print("  (no QPS values available for combined CDF plot)")
        return

    is_8b_model = "8b" in experiment_name.lower()
    show_legend = True

    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 1.55), sharey=True)
        legend_handles = OrderedDict()

        for idx, ax in enumerate(np.atleast_1d(axes)):
            if idx >= len(selected_qps):
                ax.set_visible(False)
                continue

            qps = selected_qps[idx]
            group = qps_groups.get(qps, {})
            for (config_name, det_ratio) in sorted(group.keys(), key=lambda x: config_sort_key(*x)):
                r = group[(config_name, det_ratio)]
                raw = r.get("latencies", [])
                if not raw:
                    continue
                raw_sec = [x for x in raw if x is not None]
                x_vals, y_vals = compute_cdf(raw_sec)
                if len(x_vals) == 0:
                    continue

                style = get_style(config_name, det_ratio)
                (line,) = ax.plot(
                    x_vals, y_vals,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.0,
                    label=style["label"],
                )
                legend_handles.setdefault(style["label"], line)

            ax.set_title(f"QPS = {qps:g}", fontweight="bold", fontsize=6, pad=2)
            ax.set_ylim(0, 1.02)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", pad=1.5)
            if is_8b_model:
                ax.xaxis.set_major_locator(MultipleLocator(40))
            else:
                ax.xaxis.set_major_locator(MultipleLocator(120))

        model_label = "Llama-3-8B (TP-1)" if is_8b_model else "Llama-3-70B (TP-8)"
        for ax in axes:
            ax.text(0.65, 0.1, model_label, transform=ax.transAxes,
                    fontsize=6, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
        # Place model name in bottom-right of the last visible subplot
        #last_ax = np.atleast_1d(axes)[min(len(selected_qps), len(np.atleast_1d(axes))) - 1]
        #last_ax.text(0.95, 0.05, model_label, transform=last_ax.transAxes,
        #             fontsize=6, fontweight="bold", ha="right", va="bottom",
        #             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", #edgecolor="none", alpha=0.7))

        fig.supxlabel("E2E latency (seconds)", fontsize=6, fontweight="bold", y=0.03)
        fig.supylabel("CDF", fontsize=8, fontweight="bold", x=0.01)
        if show_legend:
            fig.legend(
                list(legend_handles.values()),
                list(legend_handles.keys()),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=min(4, max(1, len(legend_handles))),
                frameon=True,
                handlelength=2.2,
                columnspacing=1.2,
            )
        fig.subplots_adjust(
            left=0.07,
            right=0.995,
            bottom=0.25,
            top=0.70,
            wspace=0.18,
        )

        fname = f"{output_prefix}cdf_e2e_all_combined_{dataset_label}.pdf"
        out_path = output_dir / fname
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")
        return out_path


# ---------------------------------------------------------------------------
# Table: TTFT P50/P90/P99 per QPS (one CSV per QPS)
# ---------------------------------------------------------------------------

def generate_ttft_tables(data: dict, dataset_label: str, output_dir: Path, output_prefix: str):
    qps_groups = defaultdict(dict)
    for (config_name, det_ratio, qps), r in data.items():
        qps_groups[qps][(config_name, det_ratio)] = r

    for qps in sorted(qps_groups.keys()):
        group = qps_groups[qps]
        rows = []

        for (config_name, det_ratio) in sorted(group.keys(), key=lambda x: config_sort_key(*x)):
            r = group[(config_name, det_ratio)]
            label = get_label(config_name, det_ratio)

            # Compute TTFT percentiles from raw data
            ttfts = r.get("ttfts", [])
            if ttfts:
                ttfts_ms = [x * 1000 for x in ttfts if x is not None]
                if ttfts_ms:
                    p50 = round(np.percentile(ttfts_ms, 50), 2)
                    p90 = round(np.percentile(ttfts_ms, 90), 2)
                    p99 = round(np.percentile(ttfts_ms, 99), 2)
                else:
                    p50 = p90 = p99 = ""
            else:
                # Fall back to pre-computed values
                p50 = r.get("p50_ttft_ms", "")
                p90 = r.get("p90_ttft_ms", "")
                p99 = r.get("p99_ttft_ms", "")

            rows.append({
                "config": label,
                "TTFT_P50_ms": p50,
                "TTFT_P90_ms": p90,
                "TTFT_P99_ms": p99,
            })

        fname = f"{output_prefix}ttft_{dataset_label}_qps{qps}.csv"
        csv_path = output_dir / fname
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["config", "TTFT_P50_ms", "TTFT_P90_ms", "TTFT_P99_ms"])
            writer.writeheader()
            writer.writerows(rows)

        # Print to console
        print(f"  Saved: {fname}")
        print(f"    {'Config':<30s} {'P50':>10s} {'P90':>10s} {'P99':>10s}")
        for row in rows:
            print(f"    {row['config']:<30s} {str(row['TTFT_P50_ms']):>10s} {str(row['TTFT_P90_ms']):>10s} {str(row['TTFT_P99_ms']):>10s}")


# ---------------------------------------------------------------------------
# Plot 3: TTFT ratio vs baseline across all QPS values
# ---------------------------------------------------------------------------

def plot_ttft_ratio(data: dict, dataset_label: str, output_dir: Path, output_prefix: str, experiment_name: str):
    """Generate TTFT ratio plots (config / non-det baseline) with all QPS on one chart.

    One PDF per percentile (P50, P90, P99).  Y-axis = ratio relative to
    sglang_non_deterministic at that QPS.  A horizontal line at y=1 marks parity.
    """
    # Group by (config, ratio) -> {qps: result}
    config_groups = defaultdict(dict)
    for (config_name, det_ratio, qps), r in data.items():
        config_groups[(config_name, det_ratio)][qps] = r

    # Baseline TTFT per QPS
    baseline_key = None
    for key in config_groups:
        if key[0] == "sglang_non_deterministic":
            baseline_key = key
            break
    if baseline_key is None:
        print("  (no non-det baseline for TTFT ratio plots)")
        return

    baseline_by_qps = config_groups[baseline_key]
    show_legend = True
    fig_height = 1.56
    p99_pdf_path = None

    for pctl_label, pctl_idx in [("P50", 50), ("P90", 90), ("P99", 99)]:
        with plt.rc_context(PAPER_STYLE):
            fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, fig_height))

            for (config_name, det_ratio) in sorted(config_groups.keys(), key=lambda x: config_sort_key(*x)):
                if config_name == "sglang_non_deterministic":
                    continue  # skip baseline itself (always 1.0)
                qps_data = config_groups[(config_name, det_ratio)]
                qps_vals = sorted(qps_data.keys())
                ratios = []
                valid_qps = []
                for q in qps_vals:
                    r = qps_data[q]
                    bl = baseline_by_qps.get(q)
                    if bl is None:
                        continue
                    # Compute from raw TTFT arrays
                    ttfts = r.get("ttfts", [])
                    bl_ttfts = bl.get("ttfts", [])
                    if ttfts and bl_ttfts:
                        ttfts_clean = [x for x in ttfts if x is not None]
                        bl_clean = [x for x in bl_ttfts if x is not None]
                        if ttfts_clean and bl_clean:
                            val = np.percentile(ttfts_clean, pctl_idx)
                            bl_val = np.percentile(bl_clean, pctl_idx)
                            if bl_val > 0:
                                ratios.append(val / bl_val)
                                valid_qps.append(q)
                                continue
                    # Fallback to pre-computed
                    key_map = {50: "p50_ttft_ms", 90: "p90_ttft_ms", 99: "p99_ttft_ms"}
                    val = r.get(key_map[pctl_idx])
                    bl_val = bl.get(key_map[pctl_idx])
                    if val and bl_val and bl_val > 0:
                        ratios.append(val / bl_val)
                        valid_qps.append(q)

                if not ratios:
                    continue
                style = get_style(config_name, det_ratio)
                ax.plot(valid_qps, ratios, color=style["color"],
                        linestyle=style["linestyle"], marker=style["marker"],
                        linewidth=1.0, markersize=3.2, label=style["label"])

            ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("QPS (requests/sec)", fontsize=7, fontweight="bold")
            ax.set_ylabel(f"{pctl_label} TTFT ratio\nvs SGLang-nondet", fontsize=7, fontweight="bold")
            ax.tick_params(axis="both", labelsize=6)
            if "8b" in experiment_name.lower():
                ax.text(0.57, 0.95, f"Llama-3-8B (TP-1)", transform=ax.transAxes,
                        fontsize=6, fontweight="bold", ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
            if "70b" in experiment_name.lower():
                ax.text(0.05, 0.95, f"Llama-3-70B (TP-8)", transform=ax.transAxes,
                        fontsize=6, fontweight="bold", ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
            if show_legend:
                fig.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.59, 1.12),
                    ncol=3,
                    fontsize=5,
                    frameon=True,
                    columnspacing=0.9,
                    handlelength=1.6,
                )
            ax.grid(True, linestyle="--", alpha=0.7)
            fig.subplots_adjust(left=0.25, right=0.99, bottom=0.22, top=0.80)
            fname = f"{output_prefix}ttft_ratio_{pctl_label.lower()}_{dataset_label}.pdf"
            out_path = output_dir / fname
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fname}")
            if pctl_idx == 99:
                p99_pdf_path = out_path

    return p99_pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def filter_ratios(all_data: dict, ratios: list[float]) -> dict:
    """Keep only baseline configs + LLM-42 entries whose det_ratio is in *ratios*."""
    filtered = {}
    for ds_label, data in all_data.items():
        fd = {}
        for (config_name, det_ratio, qps), r in data.items():
            if "llm42" in config_name and det_ratio not in ratios:
                continue
            fd[(config_name, det_ratio, qps)] = r
        if fd:
            filtered[ds_label] = fd
    return filtered


def process_results_dir(results_dir: Path, output_dir: Path, ratios: list[float] | None = None,
                        paper_figure: str | None = None, ttft_paper_figure: str | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir

    all_data = load_all_data(results_dir)
    if not all_data:
        print(f"Warning: no data found in {results_dir}")
        return 1

    if ratios is not None:
        all_data = filter_ratios(all_data, ratios)
        if not all_data:
            print(f"Warning: no data left after filtering to ratios {ratios}")
            return 1
        print(f"Filtered to LLM-42 ratios: {ratios}")

    experiment_name = results_dir.parent.name if results_dir.name == "results" else results_dir.name
    output_prefix = build_output_prefix(experiment_name)

    # Paper-figure filename: 8B -> figure11a.pdf, 70B -> figure11b.pdf.
    fig_name = (paper_figure
                or infer_paper_figure(experiment_name)
                or infer_paper_figure(output_dir.name))
    primary_combined_pdf = None

    # P99 TTFT-ratio paper figure: 8B -> figure12a.pdf, 70B -> figure12b.pdf.
    ttft_fig_name = (ttft_paper_figure
                     or infer_ttft_paper_figure(experiment_name)
                     or infer_ttft_paper_figure(output_dir.name))
    primary_ttft_pdf = None

    for ds_label, data in sorted(all_data.items()):
        print(f"\n--- {ds_label} ({len(data)} records) ---")
        plot_cdf_all_per_qps(data, ds_label, plot_dir, output_prefix)
        combined_pdf = plot_cdf_all_combined(data, ds_label, experiment_name, plot_dir, output_prefix)
        if primary_combined_pdf is None and combined_pdf is not None:
            primary_combined_pdf = combined_pdf
        generate_ttft_tables(data, ds_label, plot_dir, output_prefix)
        ttft_p99_pdf = plot_ttft_ratio(data, ds_label, plot_dir, output_prefix, experiment_name)
        if primary_ttft_pdf is None and ttft_p99_pdf is not None:
            primary_ttft_pdf = ttft_p99_pdf

    # Also save a combined CSV with all data
    csv_path = output_dir / f"{output_prefix}online_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "config", "det_ratio", "qps",
                          "throughput_tps", "p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
                          "p50_e2e_ms", "p90_e2e_ms", "p99_e2e_ms"])
        for ds_label, data in sorted(all_data.items()):
            for (config_name, det_ratio, qps), r in sorted(data.items()):
                ti = r.get("total_input_tokens", 0)
                to = r.get("total_output_tokens", 0)
                dur = r.get("duration", 1)
                tp = (ti + to) / dur if dur > 0 else 0
                writer.writerow([
                    ds_label, config_name, det_ratio, qps,
                    f"{tp:.2f}",
                    r.get("p50_ttft_ms", ""), r.get("p90_ttft_ms", ""), r.get("p99_ttft_ms", ""),
                    r.get("p50_e2e_latency_ms", ""), r.get("p90_e2e_latency_ms", ""), r.get("p99_e2e_latency_ms", ""),
                ])
    print(f"\nSaved CSV: {csv_path}")

    # ---- Export the paper figure into llm42-plots/ (8B -> figure11a, 70B -> figure11b) ----
    if fig_name and primary_combined_pdf is not None:
        export_paper_figure(primary_combined_pdf, fig_name)

    # ---- Export the P99 TTFT-ratio figure into llm42-plots/ (8B -> figure12a, 70B -> figure12b) ----
    if ttft_fig_name and primary_ttft_pdf is not None:
        export_paper_figure(primary_ttft_pdf, ttft_fig_name)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Plot online benchmark comparison (CDF + TTFT tables)",
    )
    parser.add_argument(
        "--results-dirs", nargs="+", default=None,
        help="Results directories to process. If omitted, auto-discovers under runs/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output (default: parent of first results dir)",
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=None,
        help="LLM-42 ratios to include (e.g. --ratios 0.05 0.1). Baselines always included.",
    )
    parser.add_argument(
        "--paper-figure", default=None,
        help="Filename to export the combined CDF into llm42-plots/ "
             "(default: inferred; 8B->figure11a.pdf, 70B->figure11b.pdf).",
    )
    parser.add_argument(
        "--ttft-paper-figure", default=None,
        help="Filename to export the P99 TTFT-ratio plot into llm42-plots/ "
             "(default: inferred; 8B->figure12a.pdf, 70B->figure12b.pdf).",
    )
    args = parser.parse_args()

    ratios = args.ratios

    if args.results_dirs is None:
        script_dir = Path(__file__).resolve().parent
        runs_dir = script_dir.parent / "runs"
        if not runs_dir.is_dir():
            print(f"Error: no runs/ directory found at {runs_dir}")
            return 1
        run_dirs = sorted(p for p in runs_dir.iterdir()
                          if p.is_dir() and p.name.endswith("_online")
                          and (p / "results" / "benchmark_results.jsonl").exists())
        if not run_dirs:
            print(f"Error: no online run directories found under {runs_dir}")
            return 1

        print(f"Auto-discovered {len(run_dirs)} online run(s):\n")
        for run_dir in run_dirs:
            out = args.output_dir if args.output_dir else script_dir
            print(f"=== {run_dir.name} ===")
            process_results_dir(run_dir / "results", out, ratios=ratios,
                                 paper_figure=args.paper_figure,
                                 ttft_paper_figure=args.ttft_paper_figure)
            print()
        return 0

    first = Path(args.results_dirs[0])
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent

    return process_results_dir(first, args.output_dir, ratios=ratios,
                               paper_figure=args.paper_figure,
                               ttft_paper_figure=args.ttft_paper_figure)


if __name__ == "__main__":
    raise SystemExit(main())
