#!/usr/bin/env python3
"""
Compare mismatches across ALL batches of a given experiment.

Within each batch, only QPS values in that batch are compared pairwise.
This script loads the raw qps_*.jsonl files from every batch directory,
then performs pairwise comparison across ALL QPS values (including those
from different batches).

Usage:
    python compare_across_batches.py /path/to/reqs_60000
    python compare_across_batches.py /path/to/reqs_60000 --output-dir /path/to/cross_batch_results
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def first_mismatch(a: Sequence[int], b: Sequence[int]) -> int:
    """Return first mismatch index; treat length difference as a mismatch.

    If sequences are identical and equal length, return the length.
    """
    len_a, len_b = len(a), len(b)
    limit = min(len_a, len_b)

    for i in range(limit):
        if a[i] != b[i]:
            return i

    if len_a != len_b:
        return limit

    return len_a


def load_qps_file(path: Path) -> dict:
    """Load a qps_*.jsonl benchmark output file.

    Returns dict with keys: output_ids, generated_texts, input_lens, meta_info, etc.
    """
    with path.open() as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {path}")

    # bench_serving writes one JSON object per line (typically just one line)
    data = json.loads(lines[0])
    return data


def write_jsonl(path: Path, rows: List[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def plot_mismatch_heatmap(
    mismatch_matrix: np.ndarray, qps_values: List[float], output_path: Path
):
    """Plot heatmap showing fraction of mismatches between each pair of QPS values."""
    n = len(qps_values)
    fig_size = max(8, n * 0.6)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    im = plt.imshow(mismatch_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Mismatch Fraction")

    labels = [f"QPS {qps}" for qps in qps_values]
    plt.xticks(range(n), labels, rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    plt.yticks(range(n), labels, fontsize=max(6, 10 - n // 5))

    # Add text annotations
    fontsize = max(5, 9 - n // 5)
    for i in range(n):
        for j in range(n):
            plt.text(
                j, i, f"{mismatch_matrix[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=fontsize,
            )

    plt.title("Cross-Batch Pairwise Output Mismatch Fractions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def discover_qps_files(experiment_dir: Path) -> Dict[float, Path]:
    """Find all qps_*.jsonl files across batch subdirectories.

    Returns {qps_value: path} sorted by QPS.
    """
    qps_files: Dict[float, Path] = {}
    pattern = re.compile(r"^qps_([\d.]+)\.jsonl$")

    # Look in batch_* subdirectories
    batch_dirs = sorted(experiment_dir.glob("batch_*"))
    if not batch_dirs:
        # Maybe the user pointed directly at a directory with qps files
        batch_dirs = [experiment_dir]

    for batch_dir in batch_dirs:
        if not batch_dir.is_dir():
            continue
        for f in sorted(batch_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                qps_val = float(m.group(1))
                if qps_val in qps_files:
                    print(
                        f"WARNING: Duplicate QPS={qps_val} found in {batch_dir} "
                        f"(already loaded from {qps_files[qps_val].parent}). Skipping."
                    )
                    continue
                qps_files[qps_val] = f

    return dict(sorted(qps_files.items()))


def main():
    parser = argparse.ArgumentParser(
        description="Compare mismatches across all batches of a given experiment"
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing batch_* subdirectories (e.g. reqs_60000/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for cross-batch results (default: <experiment_dir>/cross_batch_comparison)",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        print(f"Error: {experiment_dir} is not a directory")
        return 1

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else experiment_dir / "cross_batch_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all qps_*.jsonl files
    print(f"Scanning {experiment_dir} for QPS output files...")
    qps_files = discover_qps_files(experiment_dir)

    if len(qps_files) < 2:
        print(f"Error: Need at least 2 QPS files, found {len(qps_files)}")
        return 1

    qps_values = list(qps_files.keys())
    print(f"Found {len(qps_values)} QPS values: {qps_values}")
    print()

    # Load all QPS data
    qps_data: Dict[float, dict] = {}
    for qps_val, path in qps_files.items():
        print(f"  Loading QPS={qps_val} from {path.relative_to(experiment_dir)} ...", end=" ")
        data = load_qps_file(path)

        tokens = data.get("output_ids", [])
        texts = data.get("generated_texts", [])
        input_lens = data.get("input_lens", [None] * len(texts))
        meta_info = data.get("meta_info", [])

        if not tokens or not any(tokens):
            print(f"WARNING: No output_ids for QPS={qps_val}, skipping")
            continue

        # Extract rollback stats
        det_num_rollbacks = []
        det_tokens_rolled_back = []
        for meta in meta_info:
            if meta:
                det_num_rollbacks.append(meta.get("llm42_num_rollbacks", 0))
                det_tokens_rolled_back.append(meta.get("llm42_tokens_rolled_back", 0))

        qps_data[qps_val] = {
            "tokens": tokens,
            "texts": texts,
            "input_lens": input_lens,
            "det_num_rollbacks": det_num_rollbacks,
            "det_tokens_rolled_back": det_tokens_rolled_back,
            "source_file": str(path),
        }
        print(f"{len(tokens)} responses")

    if len(qps_data) < 2:
        print(f"Error: Only {len(qps_data)} QPS values loaded successfully, need at least 2")
        return 1

    # Update qps_values to only include successfully loaded ones
    qps_values = sorted(qps_data.keys())
    num_runs = len(qps_values)

    # Validate all QPS runs have the same number of prompts
    num_prompts_set = {len(qps_data[q]["tokens"]) for q in qps_values}
    if len(num_prompts_set) > 1:
        print(f"WARNING: Different number of prompts across QPS values: {num_prompts_set}")
        print("Comparisons will use the minimum common length.")
    num_prompts = min(num_prompts_set)

    print()
    print(f"Comparing {num_runs} QPS values pairwise ({num_runs * (num_runs - 1) // 2} pairs)...")
    print()

    # Pairwise comparison
    mismatch_matrix = np.zeros((num_runs, num_runs))
    pairwise_details = []

    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            qps_i = qps_values[i]
            qps_j = qps_values[j]
            d_i = qps_data[qps_i]
            d_j = qps_data[qps_j]

            tokens_i = d_i["tokens"][:num_prompts]
            tokens_j = d_j["tokens"][:num_prompts]
            texts_i = d_i["texts"][:num_prompts]
            texts_j = d_j["texts"][:num_prompts]
            prompt_lens_i = d_i["input_lens"][:num_prompts]
            prompt_lens_j = d_j["input_lens"][:num_prompts]
            det_rollbacks_i = d_i["det_num_rollbacks"]
            det_tokens_rb_i = d_i["det_tokens_rolled_back"]
            det_rollbacks_j = d_j["det_num_rollbacks"]
            det_tokens_rb_j = d_j["det_tokens_rolled_back"]

            mismatches = []
            for req_id in range(num_prompts):
                tok_i = tokens_i[req_id]
                tok_j = tokens_j[req_id]
                first_mm = first_mismatch(tok_i, tok_j)
                delta = max(len(tok_i), len(tok_j)) - first_mm

                mismatches.append({
                    "request_id": req_id,
                    "prompt_len": prompt_lens_i[req_id],
                    f"output_len_qps_{qps_i}": len(tok_i),
                    f"output_len_qps_{qps_j}": len(tok_j),
                    "first_mismatch_index": first_mm,
                    "output_length": len(tok_i),
                    "delta": delta,
                })

            # Calculate statistics
            deltas = np.array([m["delta"] for m in mismatches])
            num_mismatches = int(np.sum(deltas > 0))
            # Count mismatches where both outputs are non-empty
            both_nonempty = np.array([
                m[f"output_len_qps_{qps_i}"] > 0 and m[f"output_len_qps_{qps_j}"] > 0
                for m in mismatches
            ])
            num_nonzero_mismatches = int(np.sum((deltas > 0) & both_nonempty))
            mismatch_frac = float(np.mean(deltas > 0))
            num_delta_gt_64 = int(np.sum(deltas > 64))
            num_delta_gt_128 = int(np.sum(deltas > 128))
            mismatch_matrix[i, j] = mismatch_frac
            mismatch_matrix[j, i] = mismatch_frac

            # Save pairwise summary (compact — no full token dumps for cross-batch)
            summary_mismatches = []
            for m in mismatches:
                entry = {
                    "request_id": m["request_id"],
                    "prompt_len": m["prompt_len"],
                    f"output_len_qps_{qps_i}": m[f"output_len_qps_{qps_i}"],
                    f"output_len_qps_{qps_j}": m[f"output_len_qps_{qps_j}"],
                    "first_mismatch_index": m["first_mismatch_index"],
                    "output_length": m["output_length"],
                    "delta": m["delta"],
                }
                if det_rollbacks_i and m["request_id"] < len(det_rollbacks_i):
                    entry[f"det_num_rollbacks_qps_{qps_i}"] = det_rollbacks_i[m["request_id"]]
                    entry[f"det_tokens_rolled_back_qps_{qps_i}"] = det_tokens_rb_i[m["request_id"]]
                if det_rollbacks_j and m["request_id"] < len(det_rollbacks_j):
                    entry[f"det_num_rollbacks_qps_{qps_j}"] = det_rollbacks_j[m["request_id"]]
                    entry[f"det_tokens_rolled_back_qps_{qps_j}"] = det_tokens_rb_j[m["request_id"]]
                summary_mismatches.append(entry)

            pair_file = output_dir / f"compare_qps_{qps_i}_vs_{qps_j}_summary.jsonl"
            write_jsonl(pair_file, summary_mismatches)

            pairwise_details.append({
                "qps_1": qps_i,
                "qps_2": qps_j,
                "num_mismatches": num_mismatches,
                "num_nonzero_mismatches": num_nonzero_mismatches,
                "mismatch_fraction": mismatch_frac,
                "zero_mismatch_fraction": float(np.mean(deltas == 0)),
                "num_delta_gt_64": num_delta_gt_64,
                "num_delta_gt_128": num_delta_gt_128,
                "comparison_file": str(pair_file),
            })

            print(
                f"  QPS {qps_i} vs QPS {qps_j}: "
                f"{num_mismatches} mismatches (non-zero mismatches {num_nonzero_mismatches}), "
                f"{num_delta_gt_64} deltas > 64, "
                f"{num_delta_gt_128} deltas > 128"
            )

    # Plot heatmap
    heatmap_path = output_dir / "cross_batch_mismatch_heatmap.pdf"
    plot_mismatch_heatmap(mismatch_matrix, qps_values, heatmap_path)

    # Per-QPS output & rollback stats
    qps_output_stats = []
    qps_rollback_stats = []
    for qps_val in qps_values:
        d = qps_data[qps_val]
        total_output_len = sum(len(t) for t in d["tokens"][:num_prompts])
        total_tokens_rolled_back = sum(d["det_tokens_rolled_back"][:num_prompts]) if d["det_tokens_rolled_back"] else 0
        rollback_pct = (total_tokens_rolled_back / total_output_len * 100) if total_output_len > 0 else 0.0
        qps_output_stats.append({
            "qps": qps_val,
            "total_output_len": total_output_len,
            "total_tokens_rolled_back": total_tokens_rolled_back,
            "rollback_pct": rollback_pct,
        })

        if d["det_num_rollbacks"]:
            rb = d["det_num_rollbacks"][:num_prompts]
            trb = d["det_tokens_rolled_back"][:num_prompts]
            qps_rollback_stats.append({
                "qps": qps_val,
                "total_rollbacks": sum(rb),
                "total_tokens_rolled_back": sum(trb),
                "avg_rollbacks_per_request": float(np.mean(rb)),
                "max_rollbacks_per_request": max(rb),
                "avg_tokens_rolled_back_per_request": float(np.mean(trb)),
                "max_tokens_rolled_back_per_request": max(trb),
                "requests_with_rollbacks": sum(1 for x in rb if x > 0),
            })

    # Save summary JSON
    summary = {
        "experiment_dir": str(experiment_dir),
        "qps_values": qps_values,
        "num_prompts": num_prompts,
        "num_qps_values": num_runs,
        "num_pairwise_comparisons": len(pairwise_details),
        "qps_source_files": {str(q): qps_data[q]["source_file"] for q in qps_values},
        "qps_output_stats": qps_output_stats,
        "qps_rollback_stats": qps_rollback_stats,
        "pairwise_comparisons": pairwise_details,
        "heatmap_plot": str(heatmap_path),
    }

    summary_file = output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)

    # Write human-readable summary
    txt_path = output_dir / "summary.txt"
    with txt_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("Cross-Batch Pairwise Comparison Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Experiment Dir: {experiment_dir}\n")
        f.write(f"Num Prompts: {num_prompts}\n")
        f.write(f"QPS Values ({num_runs}): {qps_values}\n\n")

        f.write("-" * 70 + "\n")
        f.write("Per-QPS Output & Rollback Stats:\n")
        f.write("-" * 70 + "\n")
        for stat in qps_output_stats:
            f.write(f"QPS {stat['qps']}:\n")
            f.write(f"  Total Output Tokens: {stat['total_output_len']}\n")
            f.write(f"  Total Tokens Rolled Back: {stat['total_tokens_rolled_back']}\n")
            f.write(f"  Rollback %: {stat['rollback_pct']:.4f}%\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("Pairwise Comparisons:\n")
        f.write("-" * 70 + "\n")
        for pd in pairwise_details:
            f.write(
                f"QPS {pd['qps_1']} vs QPS {pd['qps_2']}: "
                f"{pd['num_mismatches']} mismatches (non-zero mismatches {pd['num_nonzero_mismatches']}), "
                f"{pd['num_delta_gt_64']} deltas > 64, "
                f"{pd['num_delta_gt_128']} deltas > 128\n"
            )
        f.write("\n")
        f.write(f"Heatmap: {heatmap_path}\n")

    # Print final summary
    print()
    print("=" * 60)
    print("Cross-Batch Comparison Summary")
    print("=" * 60)

    print(f"\n{num_runs} QPS values, {num_prompts} prompts, {len(pairwise_details)} pairwise comparisons")

    print("\nPer-QPS Output & Rollback Stats:")
    for stat in qps_output_stats:
        print(
            f"  QPS {stat['qps']}: {stat['total_output_len']} output tokens, "
            f"{stat['total_tokens_rolled_back']} rolled back ({stat['rollback_pct']:.4f}%)"
        )

    print("\nPairwise Comparisons:")
    for pd in pairwise_details:
        print(
            f"  QPS {pd['qps_1']} vs QPS {pd['qps_2']}: "
            f"{pd['num_mismatches']} mismatches (non-zero mismatches {pd['num_nonzero_mismatches']}), "
            f"{pd['num_delta_gt_64']} deltas > 64, "
            f"{pd['num_delta_gt_128']} deltas > 128"
        )

    print(f"\nResults saved to: {output_dir}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Summary: {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
