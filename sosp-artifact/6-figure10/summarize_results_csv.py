#!/usr/bin/env python3
"""
Generate a summary CSV from benchmark_results.jsonl.

Rows   = config (e.g. sglang_non_deterministic, llm42_ws_32_bs_16_ratio_0.1)
Columns = workload details + aggregate metrics

Usage:
    python summarize_results_csv.py --input benchmark_results.jsonl --output summary.csv
    python summarize_results_csv.py --input-dirs results/sharegpt_n4096 results/random_in1024_out256_n4096 --output summary.csv
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(filepath: Path) -> List[dict]:
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def workload_label(r: dict) -> str:
    """Human-readable workload label derived from one result record."""
    ds = r.get("dataset_name", "unknown")
    if ds == "random":
        return f"random_in{r.get('input_len', '?')}_out{r.get('output_len', '?')}"
    return ds


def config_label(r: dict) -> str:
    """Row label that folds LLM42 ratio into the config name."""
    name = r.get("config_name", "unknown")
    ratio = r.get("deterministic_ratio", 1.0)
    if "llm42" in name:
        return f"{name}_ratio_{ratio}"
    return name


def build_row(r: dict) -> Dict[str, Any]:
    """Extract throughput metrics from a single JSONL record."""
    total_input = r.get("total_input_tokens", 0)
    total_output = r.get("total_output_tokens", 0)
    duration = r.get("duration", 0)
    total_throughput = (total_input + total_output) / duration if duration > 0 else 0

    # Compute rollback metrics over deterministic requests only so that
    # non-deterministic requests don't dilute the numbers.
    avg_rb = ""
    rb_pct = ""
    per_rb = r.get("per_request_rollbacks", [])
    per_rb_tok = r.get("per_request_tokens_rolled_back", [])
    is_det = r.get("is_deterministic", [])
    output_lens = r.get("output_lens", [])

    if per_rb and is_det and len(per_rb) == len(is_det):
        det_rb = [per_rb[i] for i in range(len(per_rb)) if is_det[i]]
        det_rb_tok = [per_rb_tok[i] for i in range(len(per_rb_tok)) if is_det[i]] if len(per_rb_tok) == len(is_det) else []
        det_out = [output_lens[i] for i in range(len(output_lens)) if is_det[i]] if len(output_lens) == len(is_det) else []

        if det_rb:
            avg_rb = round(sum(det_rb) / len(det_rb), 4)
        if det_rb_tok:
            det_total_out = sum(det_out) if det_out else 0
            if det_total_out:
                rb_pct = round(sum(det_rb_tok) / det_total_out * 100, 4)
    else:
        # Fallback to aggregate rollback_stats when per-request data is unavailable
        rs = r.get("rollback_stats", {})
        if "avg_rollbacks_per_request" in rs:
            avg_rb = round(rs["avg_rollbacks_per_request"], 4)
        total_rb_tokens = rs.get("total_tokens_rolled_back", 0)
        total_out = rs.get("total_output_tokens", total_output)
        if total_out:
            rb_pct = round(total_rb_tokens / total_out * 100, 4)

    return OrderedDict([
        ("tokens-per-second", round(total_throughput, 2)),
        ("normalized-tokens-per-second", ""),
        ("avg_rollbacks_per_req", avg_rb),
        ("rollback_token_pct", rb_pct),
    ])


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results into a CSV")
    parser.add_argument("--input", "-i", type=Path, help="Single benchmark_results.jsonl file")
    parser.add_argument("--input-dirs", nargs="+", type=Path,
                        help="One or more result directories (each must contain benchmark_results.jsonl)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    # Collect all JSONL records, tagging each with its source directory name
    tagged: List[tuple] = []  # (workload, config, metrics_dict, record)
    sources: List[Path] = []

    if args.input:
        sources.append(args.input)
    if args.input_dirs:
        for d in args.input_dirs:
            if d.is_dir():
                # Try .jsonl first, then .json (vLLM uses .json with JSONL content)
                f = d / "benchmark_results.jsonl"
                if not f.exists():
                    f = d / "benchmark_results.json"
            else:
                f = d
            if f.exists():
                sources.append(f)
            else:
                print(f"Warning: {d} has no benchmark_results.jsonl or .json, skipping")

    if not sources:
        print("Error: provide --input or --input-dirs")
        return 1

    for src in sources:
        for r in load_jsonl(src):
            wl = workload_label(r)
            cfg = config_label(r)
            metrics = build_row(r)
            tagged.append((wl, cfg, metrics))

    if not tagged:
        print("No results found")
        return 1

    # Discover all unique workloads (column groups) and configs (rows)
    workloads = list(OrderedDict.fromkeys(wl for wl, _, _ in tagged))
    configs = list(OrderedDict.fromkeys(cfg for _, cfg, _ in tagged))

    # Sort configs: baselines first, then llm42 grouped by ws/bs, then by ratio
    def config_sort_key(c: str):
        if c in ("sglang_non_deterministic", "vllm_non_deterministic"):
            return (0, c, 0)
        if c == "sglang_global_deterministic":
            return (1, c, 0)
        if c in ("sglang_global_deterministic_triton", "vllm_deterministic",
                 "vllm_deterministic_optimized"):
            return (1, c, 1)
        if "llm42" in c:
            # Extract ratio for sub-sorting
            parts = c.rsplit("_ratio_", 1)
            base = parts[0] if len(parts) == 2 else c
            ratio = float(parts[1]) if len(parts) == 2 else 0
            return (2, base, ratio)
        return (3, c, 0)

    configs.sort(key=config_sort_key)

    # Build lookup: (workload, config) -> metrics
    lookup: Dict[tuple, dict] = {}
    for wl, cfg, metrics in tagged:
        lookup[(wl, cfg)] = metrics

    # Normalize tokens-per-second to non-deterministic baseline
    # Support both sglang and vLLM baseline config names
    baseline_names = ["sglang_non_deterministic", "vllm_non_deterministic"]
    for wl in OrderedDict.fromkeys(wl for wl, _, _ in tagged):
        base_tps = 0
        for bname in baseline_names:
            baseline = lookup.get((wl, bname), {})
            base_tps = baseline.get("tokens-per-second", 0)
            if base_tps:
                break
        if base_tps:
            for cfg in OrderedDict.fromkeys(cfg for _, cfg, _ in tagged):
                m = lookup.get((wl, cfg))
                if m and m["tokens-per-second"] != "":
                    m["normalized-tokens-per-second"] = round(m["tokens-per-second"] / base_tps, 4)

    # Determine metric keys from first record
    metric_keys = list(tagged[0][2].keys())

    # Write CSV
    # Header row 1: config | workload1 ... | workload2 ... |
    # Header row 2: config | metric1 metric2 ... | metric1 metric2 ... |
    # If only one workload, simplify to just metric columns
    single_workload = len(workloads) == 1
    single_metric = len(metric_keys) == 1

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)

        if single_metric:
            # One value per (config, workload) — workloads become columns directly
            writer.writerow(["config"] + workloads)
            for cfg in configs:
                row = [cfg]
                for wl in workloads:
                    m = lookup.get((wl, cfg), {})
                    row.append(m.get(metric_keys[0], ""))
                writer.writerow(row)
        elif single_workload:
            # Simple header
            writer.writerow(["config"] + metric_keys)
            for cfg in configs:
                m = lookup.get((workloads[0], cfg), {})
                writer.writerow([cfg] + [m.get(k, "") for k in metric_keys])
        else:
            # Multi-workload: two header rows
            header1 = ["config"]
            header2 = [""]
            for wl in workloads:
                header1 += [wl] + [""] * (len(metric_keys) - 1)
                header2 += metric_keys
            writer.writerow(header1)
            writer.writerow(header2)

            for cfg in configs:
                row = [cfg]
                for wl in workloads:
                    m = lookup.get((wl, cfg), {})
                    row += [m.get(k, "") for k in metric_keys]
                writer.writerow(row)

    n_cells = len(configs) * len(workloads)
    missing = sum(1 for cfg in configs for wl in workloads if (wl, cfg) not in lookup)
    print(f"Wrote {args.output}  ({len(configs)} configs × {len(workloads)} workloads, "
          f"{n_cells - missing}/{n_cells} cells populated)")

    # Console preview
    print(f"\nConfigs ({len(configs)}):")
    for cfg in configs:
        print(f"  {cfg}")
    print(f"\nWorkloads ({len(workloads)}):")
    for wl in workloads:
        print(f"  {wl}")

    return 0


if __name__ == "__main__":
    exit(main())
