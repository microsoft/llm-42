#!/usr/bin/env python3
"""
Wrapper script to run etalon benchmark with mixed is_deterministic distribution.

This script modifies the etalon benchmark to randomly assign is_deterministic values
to requests while maintaining Poisson arrival order. Temperature is set to 0.0 for all.

For example, --temp0-pct 10 with --max-requests 256 means:
  - 25 random requests (10%) will have is_deterministic=True
  - 231 requests (90%) will have is_deterministic=False
  - All requests use temperature=0.0
  - All requests follow the same Poisson arrival pattern
"""

import sys
import os
import json
import random
import subprocess
import argparse
from pathlib import Path

def create_is_deterministic_assignment_file(output_dir: Path, det_pct: int, max_requests: int, 
                                           assignment_mode: str = 'random', seed: int = 42):
    """Create a file that maps request IDs to is_deterministic values
    
    Args:
        output_dir: Directory to save the is_deterministic map
        det_pct: Percentage of requests that should have is_deterministic=True
        max_requests: Total number of requests
        assignment_mode: 'random' for random selection with seed, 'fixed' for first N requests
        seed: Random seed (only used when assignment_mode='random')
    """
    
    # Calculate how many requests should be deterministic
    num_det = int(max_requests * det_pct / 100)
    
    # Create a list of request IDs
    request_ids = list(range(max_requests))
    
    # Select which requests get is_deterministic=True based on mode
    if assignment_mode == 'fixed':
        # Every Nth request gets is_deterministic=True (0, N, 2N, 3N, ...)
        # Calculate the step size to get approximately det_pct% of requests
        step = int(100 / det_pct) if det_pct > 0 else max_requests + 1
        det_requests = set(range(0, max_requests, step))
        # If we have too many or too few, adjust
        det_list = sorted(det_requests)
        if len(det_list) > num_det:
            det_requests = set(det_list[:num_det])
        elif len(det_list) < num_det:
            # Add more evenly spaced requests
            remaining = num_det - len(det_list)
            available = [i for i in request_ids if i not in det_requests]
            step = len(available) // remaining if remaining > 0 else 1
            det_requests.update(available[::step][:remaining])
        print(f"  Using FIXED order: every ~{step}th request gets is_deterministic=True")
        print(f"  Request IDs with is_deterministic=True: {sorted(list(det_requests))[:10]}{'...' if len(det_requests) > 10 else ''}")
    else:  # random
        # Randomly select which requests get is_deterministic=True (with fixed seed)
        random.seed(seed)
        det_requests = set(random.sample(request_ids, num_det))
        print(f"  Using RANDOM selection with seed={seed}")
    
    # Create mapping
    is_det_map = {}
    for req_id in request_ids:
        is_det_map[req_id] = True if req_id in det_requests else False
    
    # Save to file
    is_det_file = output_dir / 'is_deterministic_map.json'
    with open(is_det_file, 'w') as f:
        json.dump(is_det_map, f)
    
    print(f"  Created is_deterministic map: {is_det_file}")
    print(f"    - {num_det} requests with is_deterministic=True")
    print(f"    - {max_requests - num_det} requests with is_deterministic=False")
    
    return str(is_det_file)

def main():
    parser = argparse.ArgumentParser(
        description='Run etalon benchmark with mixed is_deterministic distribution'
    )
    parser.add_argument('--temp0-pct', type=int, required=True, 
                        help='Percentage of requests with is_deterministic=True (0-100)')
    parser.add_argument('--assignment-mode', type=str, default='random',
                        choices=['random', 'fixed'],
                        help='How to assign is_deterministic: random (with seed) or fixed (evenly distributed)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for is_deterministic assignment (only used with --assignment-mode=random)')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max-requests', type=int, required=True)
    parser.add_argument('--timeout', type=int, required=True)
    parser.add_argument('--num-clients', type=int, required=True)
    parser.add_argument('--concurrent', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--qps', type=float, required=True)
    parser.add_argument('--trace-file', type=str, required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    
    args = parser.parse_args()
    
    det_pct = args.temp0_pct  # Keeping same arg name for compatibility
    non_det_pct = 100 - det_pct
    
    # Calculate number of requests for each mode
    num_det_requests = int(args.max_requests * det_pct / 100)
    num_non_det_requests = args.max_requests - num_det_requests
    
    print("=" * 60)
    print(f"Mixed is_deterministic Benchmark (temperature=0.0 for all)")
    print("=" * 60)
    print(f"Assignment mode: {args.assignment_mode.upper()}")
    if args.assignment_mode == 'random':
        print(f"  Random seed: {args.seed}")
    print(f"Total requests: {args.max_requests}")
    print(f"  - {num_det_requests} requests with is_deterministic=True ({det_pct}%)")
    print(f"  - {num_non_det_requests} requests with is_deterministic=False ({non_det_pct}%)")
    print(f"  - ALL requests use temperature=0.0")
    if args.assignment_mode == 'fixed':
        print(f"  - Every ~{int(100/det_pct) if det_pct > 0 else 'N'}th request gets is_deterministic=True (evenly distributed)")
    else:
        print(f"  - Random {num_det_requests} requests get is_deterministic=True (reproducible with seed={args.seed})")
    print(f"  - Poisson arrival pattern maintained")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create is_deterministic assignment file
    is_det_map_file = create_is_deterministic_assignment_file(
        output_dir, det_pct, args.max_requests, 
        args.assignment_mode, args.seed
    )
    
    # Set environment variable for etalon to use
    env = os.environ.copy()
    env['IS_DETERMINISTIC_MAP_FILE'] = is_det_map_file
    
    print(f"\nRunning etalon benchmark with mixed is_deterministic...")
    print(f"  is_deterministic map: {is_det_map_file}")
    
    # Set temperature=0.0 for all requests in additional_params
    additional_params = json.dumps({"temperature": 0.0})
    
    cmd = [
        'python3', '-m', 'etalon.run_benchmark',
        '--client_config_model', args.model,
        '--max_completed_requests', str(args.max_requests),
        '--timeout', str(args.timeout),
        '--client_config_num_clients', str(args.num_clients),
        '--client_config_num_concurrent_requests_per_client', str(args.concurrent),
        '--metrics_config_output_dir', str(output_dir),
        '--metrics_config_should_write_metrics',
        '--request_interval_generator_config_type', 'poisson',
        '--poisson_request_interval_generator_config_qps', str(args.qps),
        '--request_length_generator_config_type', 'trace',
        '--trace_request_length_generator_config_trace_file', args.trace_file,
        '--trace_request_length_generator_config_max_tokens', str(args.max_tokens),
        '--deadline_config_ttft_deadline', '0.3',
        '--deadline_config_tbt_deadline', '0.03',
        '--client_config_additional_sampling_params', additional_params,
    ]
    
    result = subprocess.run(cmd, env=env)
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
