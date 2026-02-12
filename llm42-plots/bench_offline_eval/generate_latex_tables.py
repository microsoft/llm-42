#!/usr/bin/env python3
"""
Generate LaTeX tables for rollback metrics.
Creates two tables (one per LLM42 config: ws_32_bs_16, ws_64_bs_8).

Each table has:
- Rows: Dataset configurations (ArXiv, ShareGPT, in=X out=Y)
- Columns: LLM42@2pct, @5pct, @10pct, @20pct, @50pct, @100pct
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_results(filepath: Path) -> list:
    """Load benchmark results from JSONL file."""
    results = []
    if not filepath.exists():
        return results
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def parse_dataset_config_from_dir(dir_name: str) -> str:
    """Extract dataset config name from directory name."""
    if 'arxiv' in dir_name:
        return 'ArXiv'
    if 'sharegpt' in dir_name:
        return 'ShareGPT'
    elif 'random_in512_out256' in dir_name:
        return 'in=512, out=256'
    elif 'random_in1024_out256' in dir_name:
        return 'in=1024, out=256'
    elif 'random_in1024_out512' in dir_name:
        return 'in=1024, out=512'
    elif 'random_in1024_out1024' in dir_name:
        return 'in=1024, out=1024'
    elif 'random_in2048_out256' in dir_name:
        return 'in=2048, out=256'
    elif 'random_in2048_out512' in dir_name:
        return 'in=2048, out=512'
    elif 'random_in4096_out256' in dir_name:
        return 'in=4096, out=256'
    elif 'random_in4096_out512' in dir_name:
        return 'in=4096, out=512'
    else:
        return dir_name


# Config order for columns (LLM42 only for rollback tables)
LLM42_CONFIGS = [
    ('llm42_0.02', '2pct'),
    ('llm42_0.05', '5pct'),
    ('llm42_0.1', '10pct'),
    ('llm42_0.2', '20pct'),
    ('llm42_0.5', '50pct'),
    ('llm42_1.0', '100pct'),
]

# Dataset order for rows
DATASET_ORDER = [
    'ArXiv',
    'ShareGPT',
    'in=512, out=256',
    'in=1024, out=256',
    'in=1024, out=512',
    'in=1024, out=1024',
    'in=2048, out=256',
    'in=2048, out=512',
    'in=4096, out=256',
    'in=4096, out=512',
]


def generate_rollback_table(data: dict, llm42_config: str, output_path: Path):
    """Generate LaTeX table for rollback metrics."""
    
    config_label = "ws=32, bs=16" if "32" in llm42_config else "ws=64, bs=8"
    
    lines = []
    lines.append("% LaTeX Table: Rollback Metrics")
    lines.append(f"% LLM42 Config: {config_label}")
    lines.append("% Format: rollbacks/recomputed_tokens")
    lines.append("% Required packages: booktabs, graphicx")
    lines.append("")
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Rollback Metrics ({config_label}). Format: rollbacks/recomputed\\_tokens}}")
    lines.append("\\label{tab:rollback_" + llm42_config + "}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    
    # Header
    lines.append("\\begin{tabular}{l|cccccc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{6}{c}{\\textbf{LLM-42 (LLM42) Deterministic Ratio}} \\\\")
    lines.append("\\cmidrule(lr){2-7}")
    
    header_row = "\\textbf{Dataset Config}"
    for _, label in LLM42_CONFIGS:
        header_row += f" & {label}"
    header_row += " \\\\"
    lines.append(header_row)
    lines.append("\\midrule")
    
    # Data rows
    for dataset in DATASET_ORDER:
        if dataset not in data:
            continue
        
        row = dataset
        for config_key, _ in LLM42_CONFIGS:
            if config_key in data[dataset]:
                rollbacks = data[dataset][config_key].get('rollbacks', 0)
                tokens = data[dataset][config_key].get('recomputed_tokens', 0)
                row += f" & {rollbacks}/{tokens}"
            else:
                row += " & --"
        row += " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved rollback table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for throughput and rollback metrics")
    parser.add_argument("--results-dirs", nargs='+', required=True,
                       help="List of result directories to process")
    parser.add_argument("--output-dir", type=Path, default=Path("tables"),
                       help="Output directory for LaTeX tables")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data structure: {llm42_config: {dataset: {config_key: {throughput, rollbacks, recomputed_tokens}}}}
    all_data = {
        'ws_32_bs_16': defaultdict(lambda: defaultdict(dict)),
        'ws_64_bs_8': defaultdict(lambda: defaultdict(dict)),
    }
    
    for results_dir in args.results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: Results directory not found: {results_path}")
            continue
        
        dataset_config = parse_dataset_config_from_dir(results_path.name)
        
        # Load results from JSONL file
        results_file = results_path / "benchmark_results.jsonl"
        results = load_results(results_file)
        
        if not results:
            continue
        
        print(f"Processing: {dataset_config}")
        
        for r in results:
            config_name = r.get('config_name', 'unknown')
            det_ratio = r.get('deterministic_ratio', 0)
            
            # Calculate throughput
            total_input = r.get('total_input_tokens', 0)
            total_output = r.get('total_output_tokens', 0)
            duration = r.get('duration', 1)
            total_tp = (total_input + total_output) / duration if duration > 0 else 0
            
            # Get rollback metrics (can be at top level or nested in rollback_stats)
            rollback_stats = r.get('rollback_stats', {})
            rollbacks = r.get('total_rollbacks', rollback_stats.get('total_rollbacks', 0))
            recomputed_tokens = r.get('total_recomputed_tokens', rollback_stats.get('total_tokens_rolled_back', 0))
            
            # Determine config key
            if config_name == 'sglang_non_deterministic':
                config_key = 'non_det'
                # Add to both llm42 configs
                for di_cfg in all_data.keys():
                    all_data[di_cfg][dataset_config][config_key] = {
                        'throughput': total_tp,
                        'rollbacks': rollbacks,
                        'recomputed_tokens': recomputed_tokens,
                    }
            elif config_name == 'sglang_global_deterministic':
                config_key = 'global_det'
                for di_cfg in all_data.keys():
                    all_data[di_cfg][dataset_config][config_key] = {
                        'throughput': total_tp,
                        'rollbacks': rollbacks,
                        'recomputed_tokens': recomputed_tokens,
                    }
            elif 'llm42' in config_name:
                # Format ratio consistently
                if det_ratio == int(det_ratio):
                    config_key = f'llm42_{int(det_ratio)}.0'
                else:
                    config_key = f'llm42_{det_ratio}'
                
                # Determine which llm42 config
                if 'ws_32_bs_16' in config_name or 'ws32' in config_name:
                    all_data['ws_32_bs_16'][dataset_config][config_key] = {
                        'throughput': total_tp,
                        'rollbacks': rollbacks,
                        'recomputed_tokens': recomputed_tokens,
                    }
                elif 'ws_64_bs_8' in config_name or 'ws64' in config_name:
                    all_data['ws_64_bs_8'][dataset_config][config_key] = {
                        'throughput': total_tp,
                        'rollbacks': rollbacks,
                        'recomputed_tokens': recomputed_tokens,
                    }
    
    if not any(all_data[k] for k in all_data):
        print("Error: No data found")
        return 1
    
    # Generate tables for each llm42 config
    for di_cfg, data in all_data.items():
        if not data:
            continue
        
        # Convert defaultdict to regular dict
        data = {k: dict(v) for k, v in data.items()}
        
        # Generate rollback table
        rollback_path = args.output_dir / f"rollback_{di_cfg}.tex"
        generate_rollback_table(data, di_cfg, rollback_path)
    
    print(f"\nAll tables saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
