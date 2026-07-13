#!/usr/bin/env python3
"""
Summarize rollback profiling results across all (window_size, verify_batch_size) configs.

Reads per-config rollback_stats.json files from a run directory and produces:
  - summary.json  (machine-readable aggregate)
  - summary.txt   (human-readable table)

Usage:
    python summarize_profiles.py --run-dir runs/h100_llama-3.1-8b-instruct_20260318_120000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_profiles(run_dir: Path) -> List[dict]:
    """Load all rollback_stats.json files from subdirectories."""
    profiles = []
    for stats_file in sorted(run_dir.glob("ws*_bs*/rollback_stats.json")):
        with open(stats_file) as f:
            profile = json.load(f)
        # Read output_throughput from benchmark.jsonl if available
        bench_file = stats_file.parent / "benchmark.jsonl"
        if bench_file.exists():
            with open(bench_file) as f:
                bench = json.loads(f.readline())
                profile["total_throughput"] = bench.get("total_throughput", 0.0)
        profiles.append(profile)
    return profiles


def build_summary(profiles: List[dict]) -> dict:
    """Build a summary dict from all profiles."""
    rows = []
    for p in profiles:
        cfg = p["config"]
        rs = p.get("rollback_stats", {})
        ls = p.get("latency_stats", {})
        rows.append({
            "window_size": cfg["window_size"],
            "verify_batch_size": cfg["verify_batch_size"],
            "name": cfg["name"],
            "total_requests": rs.get("total_requests", 0),
            "total_output_tokens": rs.get("total_output_tokens", 0),
            "total_rollbacks": rs.get("total_rollbacks", 0),
            "total_tokens_rolled_back": rs.get("total_tokens_rolled_back", 0),
            "rollback_pct": rs.get("rollback_pct", 0.0),
            "avg_rollbacks_per_request": rs.get("avg_rollbacks_per_request", 0.0),
            "median_rollbacks_per_request": rs.get("median_rollbacks_per_request", 0.0),
            "max_rollbacks_per_request": rs.get("max_rollbacks_per_request", 0),
            "p90_rollbacks": rs.get("p90_rollbacks", 0.0),
            "p99_rollbacks": rs.get("p99_rollbacks", 0.0),
            "requests_with_rollbacks": rs.get("requests_with_rollbacks", 0),
            "pct_requests_with_rollbacks": rs.get("pct_requests_with_rollbacks", 0.0),
            "avg_tokens_rolled_back_per_request": rs.get("avg_tokens_rolled_back_per_request", 0.0),
            "p90_tokens_rolled_back": rs.get("p90_tokens_rolled_back", 0.0),
            "p99_tokens_rolled_back": rs.get("p99_tokens_rolled_back", 0.0),
            "avg_e2e_latency": ls.get("avg_e2e_latency", 0.0),
            "median_e2e_latency": ls.get("median_e2e_latency", 0.0),
            "p90_e2e_latency": ls.get("p90_e2e_latency", 0.0),
            "p99_e2e_latency": ls.get("p99_e2e_latency", 0.0),
            "avg_ttft": ls.get("avg_ttft", 0.0),
            "total_throughput": p.get("total_throughput", 0.0),
        })

    rows.sort(key=lambda r: (r["window_size"], r["verify_batch_size"]))

    workload = profiles[0].get("workload", {}) if profiles else {}
    return {"workload": workload, "profiles": rows}


def write_summary_txt(summary: dict, out_path: Path):
    """Write a human-readable summary table."""
    rows = summary["profiles"]
    workload = summary.get("workload", {})

    with open(out_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("Rollback Profile Summary\n")
        f.write("=" * 100 + "\n\n")

        if workload:
            f.write(f"Workload: qps={workload.get('qps')}, "
                    f"num_prompts={workload.get('num_prompts')}, "
                    f"select_seed={workload.get('select_seed')}, "
                    f"det_ratio={workload.get('deterministic_ratio')}\n\n")

        # Table header
        hdr = (
            f"{'Config':>14s} | {'Reqs':>5s} | {'Out Toks':>9s} | "
            f"{'Rollbacks':>9s} | {'Toks RB':>8s} | {'RB%':>7s} | "
            f"{'Avg RB':>6s} | {'P90 RB':>6s} | {'P99 RB':>6s} | {'Max RB':>6s} | "
            f"{'% Reqs w/RB':>11s} | {'Avg E2E':>8s}"
        )
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")

        for r in rows:
            line = (
                f"{r['name']:>14s} | "
                f"{r['total_requests']:5d} | "
                f"{r['total_output_tokens']:9d} | "
                f"{r['total_rollbacks']:9d} | "
                f"{r['total_tokens_rolled_back']:8d} | "
                f"{r['rollback_pct']:7.4f} | "
                f"{r['avg_rollbacks_per_request']:6.2f} | "
                f"{r['p90_rollbacks']:6.1f} | "
                f"{r['p99_rollbacks']:6.1f} | "
                f"{r['max_rollbacks_per_request']:6d} | "
                f"{r['pct_requests_with_rollbacks']:10.2f}% | "
                f"{r['avg_e2e_latency']:7.3f}s"
            )
            f.write(line + "\n")

        f.write("\n" + "=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize rollback profiling results"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing ws*_bs*/ subdirs",
    )

    args = parser.parse_args()

    profiles = load_profiles(args.run_dir)
    if not profiles:
        print(f"No rollback_stats.json files found in {args.run_dir}/ws*_bs*/")
        return 1

    print(f"Loaded {len(profiles)} profile(s) from {args.run_dir}")

    summary = build_summary(profiles)

    # Write outputs
    json_path = args.run_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")

    txt_path = args.run_dir / "summary.txt"
    write_summary_txt(summary, txt_path)
    print(f"Wrote {txt_path}")

    # Print to console
    with open(txt_path) as f:
        print()
        print(f.read())

    return 0


if __name__ == "__main__":
    exit(main())
