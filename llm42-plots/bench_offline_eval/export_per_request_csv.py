#!/usr/bin/env python3
"""
Export per-request benchmark data to CSV.

Extracts the following fields per request:
- prompt_hash: Hash of the prompt for matching across configs
- is_deterministic: Whether the request was marked as deterministic
- input_len: Input/prompt length in tokens
- output_len: Output length in tokens
- rollbacks: Number of rollback events (llm_42_num_rollbacks)
- tokens_rolled_back: Total tokens rolled back (llm_42_tokens_rolled_back)
- ttft: Time to first token (seconds)
- latency: End-to-end latency (seconds)

Usage:
    python export_per_request_csv.py --input results.jsonl --output per_request.csv
    python export_per_request_csv.py --input-dir results_dir/ --output per_request.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_per_request_data(result: Dict[str, Any], config_name: str = "", det_ratio: float = 0.0) -> List[Dict[str, Any]]:
    """Extract per-request data from a single benchmark result."""
    rows = []
    
    # Get lists from result
    prompt_hashes = result.get("prompt_hashes", [])
    is_deterministic_list = result.get("is_deterministic", [])
    input_lens = result.get("input_lens", [])
    output_lens = result.get("output_lens", [])
    meta_infos = result.get("meta_info", [])
    ttfts = result.get("ttfts", [])
    latencies = result.get("latencies", [])
    errors = result.get("errors", [])
    
    # Determine number of requests
    num_requests = len(meta_infos) if meta_infos else len(output_lens)
    if num_requests == 0:
        return rows
    
    # Pad lists if needed
    if len(prompt_hashes) < num_requests:
        prompt_hashes = prompt_hashes + [None] * (num_requests - len(prompt_hashes))
    if len(is_deterministic_list) < num_requests:
        is_deterministic_list = is_deterministic_list + [None] * (num_requests - len(is_deterministic_list))
    if len(input_lens) < num_requests:
        input_lens = input_lens + [None] * (num_requests - len(input_lens))
    if len(output_lens) < num_requests:
        output_lens = output_lens + [None] * (num_requests - len(output_lens))
    if len(ttfts) < num_requests:
        ttfts = ttfts + [None] * (num_requests - len(ttfts))
    if len(latencies) < num_requests:
        latencies = latencies + [None] * (num_requests - len(latencies))
    if len(errors) < num_requests:
        errors = errors + [""] * (num_requests - len(errors))
    
    for i in range(num_requests):
        meta = meta_infos[i] if i < len(meta_infos) and meta_infos[i] else {}
        
        row = {
            "config_name": config_name or result.get("config_name", ""),
            "dataset_name": result.get("dataset_name", ""),
            "det_ratio": det_ratio or result.get("deterministic_ratio", 0.0),
            "request_idx": i,
            "prompt_hash": prompt_hashes[i] if prompt_hashes[i] else "",
            "is_deterministic": is_deterministic_list[i] if is_deterministic_list[i] is not None else "",
            "input_len": input_lens[i] if input_lens[i] is not None else "",
            "output_len": output_lens[i] if output_lens[i] is not None else "",
            "rollbacks": meta.get("llm_42_num_rollbacks", 0),
            "tokens_rolled_back": meta.get("llm_42_tokens_rolled_back", 0),
            "ttft_s": ttfts[i] if ttfts[i] is not None else "",
            "latency_s": latencies[i] if latencies[i] is not None else "",
            "error": errors[i] if errors[i] else "",
        }
        rows.append(row)
    
    return rows


def process_jsonl_file(filepath: Path, config_name: str = "", det_ratio: float = 0.0) -> List[Dict[str, Any]]:
    """Process a JSONL file and extract per-request data."""
    all_rows = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                rows = extract_per_request_data(result, config_name, det_ratio)
                all_rows.extend(rows)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}")
    
    return all_rows


def process_results_dir(dirpath: Path) -> List[Dict[str, Any]]:
    """Process all JSONL files in a results directory."""
    all_rows = []
    
    # Look for benchmark_results.jsonl
    jsonl_file = dirpath / "benchmark_results.jsonl"
    if jsonl_file.exists():
        rows = process_jsonl_file(jsonl_file)
        all_rows.extend(rows)
    else:
        # Look for any .jsonl files
        for jsonl_file in dirpath.glob("*.jsonl"):
            rows = process_jsonl_file(jsonl_file)
            all_rows.extend(rows)
    
    return all_rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path):
    """Write rows to CSV file."""
    if not rows:
        print("No data to write")
        return
    
    fieldnames = [
        "config_name",
        "dataset_name", 
        "det_ratio",
        "request_idx",
        "prompt_hash",
        "is_deterministic",
        "input_len",
        "output_len",
        "rollbacks",
        "tokens_rolled_back",
        "ttft_s",
        "latency_s",
        "error",
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export per-request benchmark data to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--input-dir", "-d",
        type=Path,
        help="Input directory containing JSONL files"
    )
    parser.add_argument(
        "--input-dirs",
        type=str,
        help="Comma-separated list of input directories"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output CSV file"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="",
        help="Config name to use (if not in the JSONL)"
    )
    parser.add_argument(
        "--det-ratio",
        type=float,
        default=0.0,
        help="Deterministic ratio to use (if not in the JSONL)"
    )
    
    args = parser.parse_args()
    
    all_rows = []
    
    if args.input:
        rows = process_jsonl_file(args.input, args.config_name, args.det_ratio)
        all_rows.extend(rows)
    
    if args.input_dir:
        rows = process_results_dir(args.input_dir)
        all_rows.extend(rows)
    
    if args.input_dirs:
        for dir_path in args.input_dirs.split(","):
            dir_path = Path(dir_path.strip())
            if dir_path.exists():
                rows = process_results_dir(dir_path)
                all_rows.extend(rows)
            else:
                print(f"Warning: Directory not found: {dir_path}")
    
    if not all_rows:
        print("No data found. Please provide --input or --input-dir")
        return
    
    write_csv(all_rows, args.output)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total requests: {len(all_rows)}")
    
    # Count by config
    configs = {}
    for row in all_rows:
        key = (row["config_name"], row["det_ratio"])
        configs[key] = configs.get(key, 0) + 1
    
    print(f"  Configs:")
    for (config, ratio), count in sorted(configs.items()):
        print(f"    {config} (det_ratio={ratio}): {count} requests")
    
    # Count deterministic requests
    det_count = sum(1 for row in all_rows if row["is_deterministic"] == True)
    if det_count > 0:
        print(f"  Deterministic requests: {det_count}")
    
    # Count requests with rollbacks
    rollback_count = sum(1 for row in all_rows if row["rollbacks"] and row["rollbacks"] > 0)
    if rollback_count > 0:
        print(f"  Requests with rollbacks: {rollback_count}")


if __name__ == "__main__":
    main()
