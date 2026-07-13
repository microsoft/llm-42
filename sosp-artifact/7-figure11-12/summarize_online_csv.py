#!/usr/bin/env python3
"""
Generate a summary CSV from online benchmark results (benchmark_results.jsonl).

Rows   = (config, qps)
Columns = latency percentiles (TTFT, E2E), throughput, rollback stats

Usage:
    python summarize_online_csv.py --input benchmark_results.jsonl --output summary.csv
    python summarize_online_csv.py --input-dirs results/sharegpt_n1024 --output summary.csv
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


def config_label(r: dict) -> str:
    """Row label that folds LLM42 ratio into the config name."""
    name = r.get("config_name", "unknown")
    ratio = r.get("deterministic_ratio", 1.0)
    if "llm42" in name:
        return f"{name}_ratio_{ratio}"
    return name


def build_row(r: dict) -> Dict[str, Any]:
    """Extract online benchmark metrics from a single JSONL record."""
    total_input = r.get("total_input_tokens", 0)
    total_output = r.get("total_output_tokens", 0)
    duration = r.get("duration", 0)
    total_throughput = (total_input + total_output) / duration if duration > 0 else 0

    # Rollback metrics (deterministic requests only)
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
        rs = r.get("rollback_stats", {})
        if "avg_rollbacks_per_request" in rs:
            avg_rb = round(rs["avg_rollbacks_per_request"], 4)
        total_rb_tokens = rs.get("total_tokens_rolled_back", 0)
        total_out = rs.get("total_output_tokens", total_output)
        if total_out:
            rb_pct = round(total_rb_tokens / total_out * 100, 4)

    return OrderedDict([
        ("qps", r.get("qps", "")),
        ("tokens-per-second", round(total_throughput, 2)),
        ("p50_ttft_ms", r.get("p50_ttft_ms", "")),
        ("p90_ttft_ms", r.get("p90_ttft_ms", "")),
        ("p99_ttft_ms", r.get("p99_ttft_ms", "")),
        ("p50_e2e_latency_ms", r.get("p50_e2e_latency_ms", "")),
        ("p90_e2e_latency_ms", r.get("p90_e2e_latency_ms", "")),
        ("p99_e2e_latency_ms", r.get("p99_e2e_latency_ms", "")),
        ("avg_rollbacks_per_req", avg_rb),
        ("rollback_token_pct", rb_pct),
    ])


def main():
    parser = argparse.ArgumentParser(description="Summarize online benchmark results into a CSV")
    parser.add_argument("--input", "-i", type=Path, help="Single benchmark_results.jsonl file")
    parser.add_argument("--input-dirs", nargs="+", type=Path,
                        help="One or more result directories (each must contain benchmark_results.jsonl)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    # Collect all JSONL records
    records: List[dict] = []
    sources: List[Path] = []

    if args.input:
        sources.append(args.input)
    if args.input_dirs:
        for d in args.input_dirs:
            if d.is_dir():
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
            records.append(r)

    if not records:
        print("No results found")
        return 1

    # Sort by config, then QPS
    def sort_key(r):
        cfg = config_label(r)
        qps = r.get("qps", 0)
        if cfg == "sglang_non_deterministic":
            return (0, cfg, 0, qps)
        if cfg == "sglang_deterministic":
            return (1, cfg, 0, qps)
        if "llm42" in cfg:
            parts = cfg.rsplit("_ratio_", 1)
            base = parts[0] if len(parts) == 2 else cfg
            ratio = float(parts[1]) if len(parts) == 2 else 0
            return (2, base, ratio, qps)
        return (3, cfg, 0, qps)

    records.sort(key=sort_key)

    # Write CSV
    metric_keys = list(build_row(records[0]).keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config"] + metric_keys)
        for r in records:
            cfg = config_label(r)
            metrics = build_row(r)
            writer.writerow([cfg] + [metrics.get(k, "") for k in metric_keys])

    print(f"Wrote {args.output}  ({len(records)} records)")

    # Console preview
    configs = list(OrderedDict.fromkeys(config_label(r) for r in records))
    qps_vals = sorted(set(r.get("qps", 0) for r in records))
    print(f"\nConfigs ({len(configs)}):")
    for cfg in configs:
        print(f"  {cfg}")
    print(f"\nQPS values: {qps_vals}")

    return 0


if __name__ == "__main__":
    exit(main())
