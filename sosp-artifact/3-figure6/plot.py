#!/usr/bin/env python3
"""
Plot the CDF of the first and second consistent spans from a comparison run.

Reads mismatch_per_request.jsonl (produced by compare_sharegpt_runs.py) and writes
mismatch_cdf.pdf. Each plot is also copied into sosp-artifact/llm42-plots/ as the paper's
Figure 6: the 8B run becomes figure6a.pdf and the 70B run becomes figure6b.pdf
(inferred from the model in the run path, or forced with --paper-figure).

With no arguments the script auto-discovers run directories under runs/ (each
containing a mismatch_per_request.jsonl) and plots each, exporting every run to
its inferred paper figure. Pass --mismatch-file to plot a specific run.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np


def load_records(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_spans(records: List[dict]) -> Tuple[List[int], List[int]]:
    """Return (first_spans, second_spans).

    Prefer the spans stored by the benchmark; fall back to recomputing from the mismatch
    indices for older data files.
    """
    first_spans, second_spans = [], []
    for r in records:
        if "first_consistent_span" in r and "second_consistent_span" in r:
            first_spans.append(r["first_consistent_span"])
            second_spans.append(r["second_consistent_span"])
        else:
            output_length = r["output_length"]
            first_mismatch = r["first_mismatch_index"]
            second_mismatch = r["second_mismatch_index"]
            first_spans.append(output_length if first_mismatch == output_length else first_mismatch + 1)
            second_spans.append(second_mismatch - first_mismatch)
    return first_spans, second_spans


def plot_cdf(first_spans: List[int], second_spans: List[int], path: Path, model_name: str = None):
    """Plot empirical CDFs of the first and second consistent spans on a symlog x-axis.

    The x-axis is symmetric-log (linear within [0, 1], log base 2 beyond), which reveals
    fine-grained structure in the low-token range while keeping zero-length spans distinct
    from 1-token spans and still showing the long tail.

    If ``model_name`` is given it is shown in a boxed label in the bottom-right corner.
    """
    plt.figure(figsize=(10, 5))

    series = [
        (r"1$^{\mathrm{st}}$ consistent span", first_spans, "tab:blue", "-"),
        (r"2$^{\mathrm{nd}}$ consistent span", second_spans, "red", "--"),
    ]
    for label, values, color, linestyle in series:
        arr = np.sort(np.asarray(values, dtype=float))
        if arr.size == 0:
            continue
        cdf = np.arange(1, arr.size + 1) / arr.size
        # Anchor the step at the true origin (x == 0). A symlog axis can render 0,
        # so zero-length spans stay distinct from 1-token spans.
        x = np.concatenate(([0.0], arr))
        y = np.concatenate(([0.0], cdf))
        plt.step(x, y, where="post", label=label, color=color, linestyle=linestyle, linewidth=2.5)

    ax = plt.gca()
    # Symmetric-log so x == 0 has its own position (linear within [0, 1], log2 beyond),
    # keeping zero-length spans distinct from 1-token spans.
    ax.set_xscale("symlog", base=2, linthresh=1, linscale=0.5)
    # Label 0 and every power of two (1, 2, 4, ...) so the low-token region is finely marked.
    max_val = max(
        float(np.max(first_spans)) if len(first_spans) else 1.0,
        float(np.max(second_spans)) if len(second_spans) else 1.0,
        1.0,
    )
    n = int(np.ceil(np.log2(max_val)))
    ax.set_xticks([0] + [2 ** i for i in range(0, n + 1)])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    plt.xlabel("# Tokens in consistent span", fontweight="bold", fontsize=22)
    plt.ylabel("CDF", fontweight="bold", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1.02)
    plt.xlim(left=-0.5)
    plt.grid(True, which="major", alpha=0.3)
    plt.grid(True, which="minor", alpha=0.15)
    # Model name (if any) sits in a boxed label in the bottom-right corner,
    # matching the model labels used by the other paper figures.
    plt.legend(fontsize=20, loc="upper center", bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False)
    if model_name:
        ax.text(0.97, 0.05, model_name, transform=ax.transAxes,
                fontsize=20, fontweight="bold", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.7))
    plt.tight_layout()
    plt.savefig(path, dpi=1200, bbox_inches="tight")
    plt.close()


def infer_model_name(path):
    """Infer a display model name from a run path (no TP): Llama-3-8B / Llama-3-70B."""
    s = str(path).lower()
    if "70b" in s:
        return "Llama-3-70B"
    if "8b" in s:
        return "Llama-3-8B"
    return None


def infer_paper_figure(path) -> str:
    """Infer the paper figure filename from the model in a run path.

    figure6a.pdf for the 8B model, figure6b.pdf for the 70B model, else figure6.pdf.
    """
    s = str(path).lower()
    if "70b" in s:
        return "figure6b.pdf"
    if "8b" in s:
        return "figure6a.pdf"
    return "figure6.pdf"


def export_paper_figure(pdf_path, figure_name="figure6.pdf"):
    """Copy the consistent-span CDF to sosp-artifact/llm42-plots/<figure_name> (the paper's Figure 6)."""
    if pdf_path is None or not Path(pdf_path).exists() or not figure_name:
        return
    plots_dir = (Path(__file__).resolve().parent / ".." / "llm42-plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    dst = plots_dir / figure_name
    shutil.copyfile(pdf_path, dst)
    print(f"Exported paper figure: {dst}")


def discover_mismatch_files(runs_dir: Path) -> List[Path]:
    """Return sorted mismatch_per_request.jsonl paths under runs/<run>/.

    Sorted ascending so the most recent run (timestamped dir name) is last.
    """
    return sorted(runs_dir.glob("*/mismatch_per_request.jsonl"))


def main():
    parser = argparse.ArgumentParser(description="Plot the CDF of first/second consistent spans")
    parser.add_argument(
        "--mismatch-file",
        type=Path,
        default=None,
        help="Path to mismatch_per_request.jsonl (default: auto-discover under runs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the CDF plot (default: mismatch_cdf.pdf next to each mismatch file)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Root directory for auto-discovery (default: runs/ next to this script)",
    )
    parser.add_argument(
        "--paper-figure",
        type=str,
        default=None,
        help="Filename to export into llm42-plots/ (default: inferred from the model in the "
             "run path -- figure6a.pdf for 8B, figure6b.pdf for 70B)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name shown in the bottom-right corner (default: inferred from the run path, "
             "e.g. Llama-3-8B or Llama-3-70B)",
    )
    args = parser.parse_args()

    if args.mismatch_file is not None:
        mismatch_files = [args.mismatch_file]
    else:
        runs_dir = args.runs_dir or (Path(__file__).resolve().parent / "runs")
        if not runs_dir.is_dir():
            raise SystemExit(f"No --mismatch-file given and runs/ directory not found at {runs_dir}")
        mismatch_files = discover_mismatch_files(runs_dir)
        if not mismatch_files:
            raise SystemExit(f"No mismatch_per_request.jsonl found under {runs_dir}/")
        print(f"Auto-discovered {len(mismatch_files)} run(s):")
        for mf in mismatch_files:
            print(f"  {mf}")

    generated = None
    for mismatch_file in mismatch_files:
        output = args.output if args.output is not None else mismatch_file.parent / "mismatch_cdf.pdf"
        records = load_records(mismatch_file)
        if not records:
            print(f"No records found in {mismatch_file}, skipping.")
            continue
        first_spans, second_spans = extract_spans(records)
        model_name = args.model_name or infer_model_name(mismatch_file)
        plot_cdf(first_spans, second_spans, output, model_name=model_name)
        print(f"Saved plot to {output}")
        # export each run to its paper figure (8B -> figure6a.pdf, 70B -> figure6b.pdf)
        figure_name = args.paper_figure or infer_paper_figure(mismatch_file)
        export_paper_figure(output, figure_name)
        generated = output

    if generated is None:
        raise SystemExit("No plots generated (no valid records found).")


if __name__ == "__main__":
    main()
