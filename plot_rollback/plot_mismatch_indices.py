#!/usr/bin/env python3
"""
Plot per-request mismatch indices and output lengths.
X-axis: request_id
Lines: first_mismatch_index, second_mismatch_index, output_length
Shaded area under each curve for quick visual comparison.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def plot_indices(records: List[Dict], out_path: Path):
    num_records = 80
    xs = np.array([r["request_id"] for r in records])
    first = np.array([r["first_mismatch_index"] for r in records])
    second = np.array([r["second_mismatch_index"] for r in records])
    outlen = np.array([r["output_length"] for r in records])

    first_consistent_span = []
    second_consistent_span = []
    output_span = []
    for i in range(num_records):
        if first[i] == outlen[i]:
            first_consistent_span.append(outlen[i])
        else:
            first_consistent_span.append(first[i] + 1)
        second_consistent_span.append(second[i] - first[i])
        output_span.append(outlen[i])

    plt.figure(figsize=(10, 5))

    plt.plot(xs[:num_records], first_consistent_span[:num_records], label="first_consistent_span", color="tab:red")
    plt.fill_between(xs[:num_records], 0, first_consistent_span[:num_records], color="tab:red", alpha=0.06)

    
    plt.plot(xs[:num_records],  second_consistent_span[:num_records], label="second_consistent_span", color="tab:green")
    plt.fill_between(xs[:num_records], 0, second_consistent_span[:num_records], color="tab:green", alpha=0.04)

    plt.plot(xs[:num_records], output_span[:num_records], label="output_span", color="tab:blue")
    plt.fill_between(xs[:num_records], 0, output_span[:num_records], color="tab:blue", alpha=0.02)

    # plt.yscale("log", base=2)
    plt.xlabel("Request Id", fontweight="bold", fontsize=24)
    plt.ylabel("# Tokens", fontweight="bold", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title("Mismatch indices vs output length")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14.5, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=1200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot mismatch indices per request")
    parser.add_argument(
        "--mismatch-file",
        type=Path,
        default=Path("sharegpt_compare_out/mismatch_per_request.jsonl"),
        help="Path to mismatch_per_request.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sharegpt_compare_out/mismatch_indices.pdf"),
        help="Path to save the plot (PDF recommended)",
    )
    args = parser.parse_args()

    records = load_records(args.mismatch_file)
    if not records:
        raise SystemExit(f"No records found in {args.mismatch_file}")

    plot_indices(records, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
