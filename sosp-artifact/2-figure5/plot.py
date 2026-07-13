#!/usr/bin/env python3
"""
Plot output throughput vs batch size for global-det vs llm42 (LLM-42).
Creates a bar graph showing how throughput scales with batch size.

Data source: log_*.log files produced by the benchmark client.

The generated plot is also copied to sosp-artifact/llm42-plots/figure5.pdf
(the paper's Figure 5).
"""

import argparse
import re
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Regex for output throughput line in benchmark logs
RE_OUTPUT_THROUGHPUT = re.compile(
    r"Output token throughput \(tok/s\):\s+([\d.]+)"
)
# Filename pattern: log_{config}_bs{N}.log
RE_LOG_FILENAME = re.compile(r"^log_(.+)_bs(\d+)\.log$")


def parse_log(filepath: Path) -> dict | None:
    """Extract config_name, batch_size, and output_throughput from a log file."""
    m = RE_LOG_FILENAME.match(filepath.name)
    if not m:
        return None
    config_name = m.group(1)
    batch_size = int(m.group(2))

    text = filepath.read_text(errors="replace")
    tm = RE_OUTPUT_THROUGHPUT.search(text)
    if not tm:
        print(f"Warning: no throughput found in {filepath}")
        return None

    return {
        "config_name": config_name,
        "batch_size": batch_size,
        "output_throughput": float(tm.group(1)),
    }


def load_results_from_logs(run_dir: Path) -> list:
    """Load benchmark results by parsing all log_*.log files in a run dir."""
    results = []
    for logfile in sorted(run_dir.glob("log_*.log")):
        rec = parse_log(logfile)
        if rec is not None:
            results.append(rec)
    return results


def plot_throughput_vs_batchsize(results: list, output_path: Path):
    """
    Plot bar graph of output throughput vs batch size.
    """
    # Organize data by config
    data = {}  # config_name -> {batch_size: throughput}
    
    for r in results:
        config = r.get('config_name', 'unknown')
        batch_size = r.get('batch_size', r.get('num_prompts', 0))
        throughput = r.get('output_throughput', 0)
        
        if config not in data:
            data[config] = {}
        data[config][batch_size] = throughput
    
    if not data:
        print("No data to plot!")
        return None
    
    # Setup plot
    plt.figure(figsize=(15, 6))
    
    # Config styles - plot order determines z-order (first = back)
    # non_det should be widest (back), then global_det, then llm42 (front)
    styles = {
        'non_det': {
            'label': 'SGLang-nondet',
            #'color': '#E74C3C',
            'color': "#88B0CB",
            'hatch': '',
            'width': 1.0,
            'zorder': 1,
        },
        'global_det': {
            'label': 'SGLang-det',
            #'color': '#F39C12',
            'color': "#DB978F",
            'hatch': '',
            'width': 1.0,
            'zorder': 3,
        },
        'llm42': {
            'label': 'LLM-42',
            #'color': '#2980B9',
            #'color': '#E74C3C',
            'color': "#DCE1A7",
            'hatch': '/',
            'width': 1.0,
            'zorder': 2,
        },
    }
    
    # Get all batch sizes
    all_batch_sizes = sorted(set(
        bs for batch_data in data.values() for bs in batch_data.keys()
    ))
    
    # Bar plot settings
    x = np.arange(len(all_batch_sizes))
    
    # Plot bars for each config (overlapped - same position, decreasing widths)
    # Plot in order: non_det (widest, back), global_det, llm42 (narrowest, front)
    plot_order = ['non_det', 'llm42', 'global_det']
    for config_name in plot_order:
        if config_name not in data:
            continue
        batch_data = data[config_name]
        throughputs = [batch_data.get(bs, 0) for bs in all_batch_sizes]
        
        style = styles.get(config_name, {
            'label': config_name,
            'color': 'tab:gray',
            'hatch': '',
            'width': 0.8,
            'zorder': 1,
        })
        
        width = style['width']
        bars = plt.bar(x, throughputs,
                       width=width,
                       label=style['label'],
                       color=style['color'],
                       hatch=style['hatch'],
                       edgecolor='black',
                       linewidth=1.5,
                       alpha=1.0,
                       zorder=style['zorder'])
    
    # Formatting
    plt.xlabel('Batch size (# requests)', fontsize=28, fontweight='bold')
    plt.ylabel('Output tokens/second', fontsize=26, fontweight='bold')
    
    # Set x-ticks at center of each bar group
    plt.xticks(x, [str(bs) for bs in all_batch_sizes], fontsize=26)
    plt.yticks(fontsize=24)
    
    # Set axis limits
    plt.xlim(-0.5, len(all_batch_sizes) - 0.5)
    
    # Set y-axis limit based on max throughput
    max_throughput = max(
        t for batch_data in data.values() for t in batch_data.values()
    )
    plt.ylim(bottom=0, top=max_throughput * 1.3)
    
    # plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=26, loc='upper left', ncol=2, frameon=False)
    
    # Add some padding
    plt.tight_layout()
    
    # Save PDF only
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save data as CSV
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write('config,batch_size,output_throughput\n')
        for config_name, batch_data in sorted(data.items()):
            for batch_size in sorted(batch_data.keys()):
                f.write(f'{config_name},{batch_size},{batch_data[batch_size]:.2f}\n')
    print(f"CSV saved to: {csv_path}")

    return output_path


def export_paper_figure(pdf_path):
    """Copy the throughput plot to sosp-artifact/llm42-plots/figure5.pdf (the paper's Figure 5)."""
    if pdf_path is None or not Path(pdf_path).exists():
        return
    plots_dir = (Path(__file__).resolve().parent / ".." / "llm42-plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    dst = plots_dir / "figure5.pdf"
    shutil.copyfile(pdf_path, dst)
    print(f"Exported paper figure: {dst}")


def main():
    parser = argparse.ArgumentParser(description='Plot throughput vs batch size')
    parser.add_argument('--input', '-i', type=Path, default=None,
                        help='Input run directory containing log_*.log files')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output plot file (PDF)')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    runs_dir = script_dir / 'runs'

    if args.input is not None:
        # Single directory mode
        run_dirs = [args.input]
    elif runs_dir.is_dir():
        # Auto-discover run directories that contain log files
        run_dirs = sorted(
            d for d in runs_dir.iterdir()
            if d.is_dir() and list(d.glob("log_*.log"))
        )
        if not run_dirs:
            print(f"No run directories with log_*.log found under {runs_dir}/")
            return
    else:
        print("No --input provided and runs/ directory not found.")
        return

    generated = None
    for run_dir in run_dirs:
        output_path = args.output if args.output else run_dir / 'throughput_vs_batchsize.pdf'
        results = load_results_from_logs(run_dir)
        if not results:
            print(f"No results in {run_dir}, skipping.")
            continue
        print(f"[{run_dir.name}] Loaded {len(results)} results from log files")
        if plot_throughput_vs_batchsize(results, output_path) is not None:
            generated = output_path

    export_paper_figure(generated)

    print(f"\nDone. Processed {len(run_dirs)} run(s).")


if __name__ == '__main__':
    main()
