import sys
import os
import re
from collections import defaultdict
# Plot results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, batch_sizes, hidden_sizes, all_batch_sizes, output_dir="."):
    """Plot performance comparison: raw execution times and speedup vs SGLang-Native side by side"""

    # Get indices of filtered batch_sizes in all_batch_sizes
    batch_indices = [all_batch_sizes.index(bs) for bs in batch_sizes]

    # Create figure with subplots for each hidden size (2 rows x 2 columns)
    # Each row: raw time (left) and speedup (right)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RMSNorm Performance Comparison', fontsize=16, fontweight='bold')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8']
    baseline_name = "SGLang-Native"

    for idx, hidden_size in enumerate(hidden_sizes):
        print(f"Plotting results for hidden size: {hidden_size}")
        ax_time = axes[idx, 0]
        ax_speedup = axes[idx, 1]

        # Get baseline times for speedup calculation
        baseline_times = None
        if baseline_name in results and hidden_size in results[baseline_name]['times']:
            all_baseline_times = np.array(results[baseline_name]['times'][hidden_size])
            baseline_times = all_baseline_times[batch_indices]
            print(baseline_times)

        # Create equally spaced x positions for batch sizes
        x_positions = np.arange(len(batch_sizes))

        # Plot raw execution times and speedups
        for (name, _), color, marker in zip(results.items(), colors, markers):
            if hidden_size in results[name]['times']:
                all_times = np.array(results[name]['times'][hidden_size])
                times = all_times[batch_indices]

                # Plot raw times at equally spaced positions
                ax_time.plot(x_positions, times, marker=marker, color=color,
                           label=name, linewidth=2, markersize=8)

                # Plot speedup (if not the baseline itself)
                if baseline_times is not None and name != baseline_name:
                    speedups = baseline_times / times
                    ax_speedup.plot(x_positions, speedups, marker=marker, color=color,
                                  label=name, linewidth=2, markersize=8)

        # Configure raw time subplot
        ax_time.set_xlabel('Batch Size', fontsize=12)
        ax_time.set_ylabel('Execution Time (ms)', fontsize=12)
        ax_time.set_title(f'Raw Execution Time - Hidden Size {hidden_size}', fontsize=13, fontweight='bold')
        ax_time.set_xticks(x_positions)
        ax_time.set_xticklabels([str(bs) for bs in batch_sizes])
        ax_time.grid(True, alpha=0.3, linestyle='--')
        ax_time.legend(fontsize=9, loc='best')

        # Configure speedup subplot
        ax_speedup.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                          label=f'Baseline ({baseline_name})', alpha=0.7)
        ax_speedup.set_xlabel('Batch Size', fontsize=12)
        ax_speedup.set_ylabel('Speedup vs SGLang-Native', fontsize=12)
        ax_speedup.set_title(f'Speedup vs Native - Hidden Size {hidden_size}', fontsize=13, fontweight='bold')
        ax_speedup.set_xticks(x_positions)
        ax_speedup.set_xticklabels([str(bs) for bs in batch_sizes])
        ax_speedup.grid(True, alpha=0.3, linestyle='--')
        ax_speedup.legend(fontsize=9, loc='best')

    plt.tight_layout()

    # Save plot as PDF
    output_path = os.path.join(output_dir, 'rmsnorm_benchmark_all_configs.png')
    plt.savefig(output_path, format='png', bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    plt.close('all')


def plot_one(results, batch_sizes, hidden_size, lines, all_batch_sizes, titles=None, output_dir=".", file_name="rmsnorm_benchmark_one.png"):
    """Plot performance comparison: raw execution times and speedup vs SGLang-Native side by side"""

    # Get indices of filtered batch_sizes in all_batch_sizes
    batch_indices = [all_batch_sizes.index(bs) for bs in batch_sizes]

    # Create figure with subplots for each hidden size (2 rows x 2 columns)
    # Each row: raw time (left) and speedup (right)
    fig, axes = plt.subplots(1, 1, figsize=(9, 6))
    #fig.suptitle('RMSNorm Performance Comparison', fontsize=16, fontweight='bold')

    colors = ['tab:pink', 'tab:blue', 'tab:gray', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:red', 'tab:green', 'tab:olive', 'tab:cyan']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8']

    print(f"Plotting results for hidden size: {hidden_size}")
    ax_time = axes

    # Create equally spaced x positions for batch sizes
    x_positions = np.arange(len(batch_sizes))

    # Plot raw execution times and speedups
    for (name, _), color, marker in zip(results.items(), colors, markers):
        if hidden_size in results[name]['times'] and name in lines:
            all_times = np.array(results[name]['times'][hidden_size])
            times = all_times[batch_indices]

            if titles:
                name = titles[lines.index(name)]

            # Plot raw times at equally spaced positions
            ax_time.plot(x_positions, times, marker=marker, color=color,
                        label=name, markersize=6, linewidth=1.6)

    # Configure raw time subplot
    ax_time.set_xlabel('Tokens', fontsize=24, fontweight='bold')
    ax_time.set_ylabel('Execution Time (ms)', fontsize=24, fontweight='bold')
    #ax_time.set_title(f'Raw Execution Time - Hidden Size {hidden_size}', fontsize=13, fontweight='bold')
    ax_time.set_xticks(x_positions)
    ax_time.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=20)
    ax_time.tick_params(labelsize=20)
    ax_time.grid(True, alpha=0.5, linestyle='--', which='both')
    ax_time.legend(fontsize=20, loc='best', frameon=False)

    plt.tight_layout()

    # Save plot as PDF
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path, bbox_inches='tight', dpi=1200)
    print(f"\n✓ Plot saved to: {output_path}")

    plt.close('all')

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print("Usage: python plot_llama_matmul.py <results_file>")
    sys.exit(1)

if not os.path.exists(fname):
    raise FileNotFoundError(f"Input file not found: {fname}")

#print(f"\nHidden Size: {hidden_size}")
#print(f"{batch_size:<12} | {name:<25} | {time_ms:>10.4f} | {bandwidth_gbs:>10.2f}")

op_name_re = re.compile(r"Hidden Size:\s+([0-9]+)")

results = defaultdict(dict)   # op_name -> seq_len -> {name: time}
batch_sizes = set()
hidden_sizes = set()
hidden_size = "unknown"

with open(fname, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue

        if op_name_re.match(line):
            m_op = op_name_re.match(line)
            hidden_size = int(m_op.group(1))
            print(f"Processing hidden size: {hidden_size}")
            hidden_sizes.add(hidden_size)
            continue

        # Extend lines: look for seq_len=...
        m = re.search(r"([0-9]+)\s+\| ([\S]+)\s+\|\s+([0-9]+\.[0-9]+) \|\s+([0-9]+\.[0-9]+)", line)
        if not m:
            continue

        batch_size = int(m.group(1).strip())
        if batch_size not in batch_sizes:
            batch_sizes.add(batch_size)
        name = m.group(2).strip()
        time_ms = float(m.group(3).strip())
        bandwidth_gbs = float(m.group(4).strip())

        if name not in results:
            results[name] = {'times': {}, 'bandwidths': {}}

        # Store for plotting
        if hidden_size not in results[name]['times']:
            results[name]['times'][hidden_size] = []
            results[name]['bandwidths'][hidden_size] = []
        results[name]['times'][hidden_size].append(time_ms)
        results[name]['bandwidths'][hidden_size].append(bandwidth_gbs)

# Filter batch_sizes to only include specific values
allowed_batch_sizes = [1, 8, 32, 128, 256, 512, 1024, 2048, 4096]
all_batch_sizes = sorted(batch_sizes)
filtered_batch_sizes = sorted([bs for bs in batch_sizes if bs in allowed_batch_sizes])

# plot_results(results, filtered_batch_sizes, sorted(hidden_sizes), all_batch_sizes, output_dir="figures/")
# plot_one(results, filtered_batch_sizes, 8192,
#          ['vLLM-Dynamic', 'vLLM-BS=128', 'vLLM-BS=256', 'vLLM-BS=512', 'vLLM-BS=1024'],
#          all_batch_sizes,
#          output_dir="figures/", file_name="figure5.png")

# 'SGLang (Batch-invariant)',
#'SGLang-Native',
print("Plotting RMSNorm implementations comparison...")
print(f"{results=}")
plot_one(results, filtered_batch_sizes, 8192,
         ['SGLang-Native', 'SGLang-Default', 'Triton-BatchInv'],
         all_batch_sizes,
         titles=['Batch-invariant (Python)', 'Non-batch-invariant (CUDA)', 'Batch-invariant (Triton)'],
         output_dir="../llm42-plots/microbenchmarks/rms_norm/", file_name="rmsnorm_impl.pdf")