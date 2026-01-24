import sys
import os
import re
from collections import defaultdict
# Plot results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print("Usage: python plot.py <results_file>")
    sys.exit(1)

if not os.path.exists(fname):
    raise FileNotFoundError(f"Input file not found: {fname}")

os.makedirs("./plots", exist_ok=True)

# Helpers
time_entry_re = re.compile(r"([0-9a-zA-Z\-]+):\s+([0-9]+\.[0-9]+) TFLOPS")
#
op_name_re = re.compile(r"([a-zA-Z\_]+)\s+\(.*Projection")
full_results = defaultdict(dict)   # op_name -> seq_len -> {name: time}

op_name = "unknown"

with open(fname, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue

        if op_name_re.match(line):
            m_op = op_name_re.match(line)
            op_name = m_op.group(1)
            print(f"Processing operation: {op_name}")
            continue

        # Extend lines: look for seq_len=...
        if "Batch Size" in line:
            m = re.search(r"Batch Size:\s+([0-9]+) |", line)
            if not m:
                continue
            seq = int(m.group(1))
            if seq not in full_results:
                full_results[op_name][seq] = {}
            for match in time_entry_re.finditer(line):
                name = match.group(1).strip()
                t = float(match.group(2))
                full_results[op_name][seq][name] = t

# Prepare data (preserve the BATCHES order)
for op_name in full_results.keys():
    tflops_results = full_results[op_name]
    x = list(tflops_results.keys())

    plt.figure(figsize=(8, 6))

    torch_vals = [tflops_results[b].get('PyTorch', 0.0) for b in x]
    bi_vals = [tflops_results[b].get('ThinkingMachines', 0.0) for b in x]
    bi_fused_vals = [tflops_results[b].get('Ours', 0.0) for b in x]
    plt.plot(x, torch_vals, marker='o', label='Non-batch-invariant (cuBLAS)', color='tab:blue')
    plt.plot(x, bi_vals, marker='s', label='Batch-invariant (Triton)', color='tab:red')

    print(x, flush=True)
    plt.xscale('log', base=2)
    plt.xticks(x, [str(b) for b in x], rotation=45, fontsize=20)
    plt.yscale('log', base=10)
    plt.yticks(fontsize=20)
    plt.xlabel('# Tokens', fontsize=24, fontweight='bold')
    plt.ylabel('TFLOPS', fontsize=24, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.3)
    plt.legend(fontsize=20, loc='best', frameon=True)
    plt.tight_layout()

    outfile = f'./plots/tflops_vs_batch_{op_name}.pdf'
    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=1200)
    print(f"  Saved plot to {outfile}")