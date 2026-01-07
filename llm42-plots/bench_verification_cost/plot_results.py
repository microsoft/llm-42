"""
Plot forward pass latency per token results.

Usage:
    python plot_results.py --input forward_cost_results.csv --output forward_cost_plot.pdf
"""

import argparse
import csv

import matplotlib.pyplot as plt


def load_results(csv_file: str) -> dict:
    """Load benchmark results from CSV file."""
    input_lens = []
    latency_per_token = []
    std_latency_per_token = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_lens.append(int(row["input_len"]))
            latency_per_token.append(float(row["latency_per_token_ms"]))
            std_latency_per_token.append(float(row["std_latency_per_token_ms"]))

    return {
        "input_lens": input_lens,
        "latency_per_token": latency_per_token,
        "std_latency_per_token": std_latency_per_token,
    }


def plot_results(data: dict, output_file: str):
    """Plot latency per token with error bars."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use indices for equal spacing
    x_indices = range(len(data["input_lens"]))
    x_labels = [str(x) for x in data["input_lens"]]

    # Plot with error bars
    ax.errorbar(
        x_indices,
        data["latency_per_token"],
        yerr=data["std_latency_per_token"],
        fmt="-o",
        color="tab:blue",
        capsize=5,
        capthick=2,
        markersize=8,
        linewidth=2,
        elinewidth=2,
    )

    # Fill area under curve
    ax.fill_between(
        x_indices,
        data["latency_per_token"],
        alpha=0.15,
        color="tab:green",
    )

    # Axis labels (font size 24)
    ax.set_xlabel("Number of Tokens", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latency per Token (ms)", fontsize=24, fontweight="bold")

    # Tick font size (20)
    ax.tick_params(axis="both", labelsize=20)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set x-axis to show all data points with equal spacing
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)

    # Tight layout
    plt.tight_layout()

    # Save as PDF with DPI 1200
    plt.savefig(output_file, format="pdf", dpi=1200, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot forward pass latency results")
    parser.add_argument(
        "--input",
        type=str,
        default="forward_cost_results.csv",
        help="Input CSV file with benchmark results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="forward_cost_plot.pdf",
        help="Output PDF file for the plot",
    )

    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    data = load_results(args.input)

    print("Generating plot...")
    plot_results(data, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
