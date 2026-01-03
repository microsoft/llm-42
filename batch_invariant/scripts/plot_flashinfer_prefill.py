import re
import sys
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read filename from argv or default to "results.txt"
fname = sys.argv[1] if len(sys.argv) > 1 else "results.txt"
if not os.path.exists(fname):
    raise FileNotFoundError(f"Input file not found: {fname}")

# Helpers
time_entry_re = re.compile(r"([A-Za-z0-9\-\s\)\()]+?)\s*time\s*=\s*([\d.]+)s")
pref_series = defaultdict(dict)   # seq_len -> {name: time}

with open(fname, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue

        # Extend lines: look for seq_len=...
        if "Extend" in line and "seq_len=" in line:
            m = re.search(r"seq_len=(\d+)", line)
            if not m:
                continue
            seq = int(m.group(1))
            for match in time_entry_re.finditer(line):
                name = match.group(1).strip()
                t = float(match.group(2))
                pref_series[seq][name] = t

print(pref_series)

# Preference (extend) plot: x = seq_len
x_pref = sorted(pref_series.keys())
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
for idx, name in enumerate(pref_series[x_pref[0]].keys()):
    y_vals = [pref_series[seq].get(name, None) for seq in x_pref]
    # Replace "deterministic" with "split size" in display label
    plt.plot(x_pref, y_vals, marker=markers[idx % len(markers)],
             color=colors[idx % len(colors)],
             label=name)
plt.xscale("log", base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.ticklabel_format(axis='x', style='plain')
plt.xlabel("Sequence length", fontsize=24, fontweight='bold')
plt.ylabel("Time (s)", fontsize=24, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title("Prefill performance")
plt.legend(fontsize=18, loc="best", frameon=False)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
# plt.savefig("figures/fi_pref_compare.png")
outfile = f'../llm42-plots/microbenchmarks/attention/fi_pref_compare.pdf'
import os
os.makedirs(os.path.dirname(outfile), exist_ok=True)
plt.savefig(outfile, dpi=1200)
#plt.show()

'''
# Decode plot: x = batch_size
x_dec = sorted(dec_non_det.keys())
y_dec_non = [dec_non_det[k] for k in x_dec]
y_dec_det = [dec_det[k] for k in x_dec]

plt.figure(figsize=(8, 5))
plt.plot(x_dec, y_dec_non, marker="o", label="Non-determinstic")
plt.plot(x_dec, y_dec_det, marker="o", label="Deterministic")
plt.xscale("log", base=2)
plt.xlabel("Batch size")
plt.ylabel("time (s)")
plt.title("Decode performance")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/fi_dec_compare.png")
plt.show()
'''