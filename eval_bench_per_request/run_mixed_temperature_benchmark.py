#!/usr/bin/env python3
"""
Wrapper script to run etalon benchmark with mixed temperature distribution.

This script modifies the etalon benchmark to randomly assign temperature values
to requests while maintaining Poisson arrival order.

For example, --temp0-pct 10 with --max-requests 256 means:
  - 25 random requests (10%) will have temperature=0
  - 231 requests (90%) will have temperature=1
  - All requests follow the same Poisson arrival pattern
"""

import sys
import os
import json
import random
import subprocess
import argparse
from pathlib import Path

def create_temperature_assignment_file(output_dir: Path, temp0_pct: int, max_requests: int, 
                                      assignment_mode: str = 'random', seed: int = 42):
    """Create a file that maps request IDs to temperature values
    
    Args:
        output_dir: Directory to save the temperature map
        temp0_pct: Percentage of requests that should have temperature=0
        max_requests: Total number of requests
        assignment_mode: 'random' for random selection with seed, 'fixed' for first N requests
        seed: Random seed (only used when assignment_mode='random')
    """
    
    # Calculate how many requests should have each temperature
    num_temp0 = int(max_requests * temp0_pct / 100)
    
    # Create a list of request IDs
    request_ids = list(range(max_requests))
    
    # Select which requests get temperature=0 based on mode
    if assignment_mode == 'fixed':
        # Every 5th request gets temperature=0 (0, 5, 10, 15, ...)
        # Calculate the step size to get approximately temp0_pct% of requests
        step = int(100 / temp0_pct) if temp0_pct > 0 else max_requests + 1
        temp0_requests = set(range(0, max_requests, step))
        # If we have too many or too few, adjust
        temp0_list = sorted(temp0_requests)
        if len(temp0_list) > num_temp0:
            temp0_requests = set(temp0_list[:num_temp0])
        elif len(temp0_list) < num_temp0:
            # Add more evenly spaced requests
            remaining = num_temp0 - len(temp0_list)
            available = [i for i in request_ids if i not in temp0_requests]
            step = len(available) // remaining if remaining > 0 else 1
            temp0_requests.update(available[::step][:remaining])
        print(f"  Using FIXED order: every ~{step}th request gets temperature=0")
        print(f"  Request IDs with temp=0: {sorted(list(temp0_requests))[:10]}{'...' if len(temp0_requests) > 10 else ''}")
    else:  # random
        # Randomly select which requests get temperature=0 (with fixed seed)
        random.seed(seed)
        temp0_requests = set(random.sample(request_ids, num_temp0))
        print(f"  Using RANDOM selection with seed={seed}")
    
    # Create mapping
    temp_map = {}
    for req_id in request_ids:
        temp_map[req_id] = 0.0 if req_id in temp0_requests else 1.0
    
    # Save to file
    temp_file = output_dir / 'temperature_map.json'
    with open(temp_file, 'w') as f:
        json.dump(temp_map, f)
    
    print(f"  Created temperature map: {temp_file}")
    print(f"    - {num_temp0} requests with temperature=0")
    print(f"    - {max_requests - num_temp0} requests with temperature=1")
    
    return str(temp_file)

def main():
    parser = argparse.ArgumentParser(
        description='Run etalon benchmark with mixed temperature distribution'
    )
    parser.add_argument('--temp0-pct', type=int, required=True, 
                        help='Percentage of requests with temperature=0 (0-100)')
    parser.add_argument('--assignment-mode', type=str, default='random',
                        choices=['random', 'fixed'],
                        help='How to assign temperatures: random (with seed) or fixed (evenly distributed)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for temperature assignment (only used with --assignment-mode=random)')
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
    
    temp0_pct = args.temp0_pct
    temp1_pct = 100 - temp0_pct
    
    # Calculate number of requests for each temperature
    num_temp0_requests = int(args.max_requests * temp0_pct / 100)
    num_temp1_requests = args.max_requests - num_temp0_requests
    
    print("=" * 60)
    print(f"Mixed Temperature Benchmark")
    print("=" * 60)
    print(f"Assignment mode: {args.assignment_mode.upper()}")
    if args.assignment_mode == 'random':
        print(f"  Random seed: {args.seed}")
    print(f"Total requests: {args.max_requests}")
    print(f"  - {num_temp0_requests} requests with temperature=0 ({temp0_pct}%)")
    print(f"  - {num_temp1_requests} requests with temperature=1 ({temp1_pct}%)")
    if args.assignment_mode == 'fixed':
        print(f"  - Every ~{int(100/temp0_pct) if temp0_pct > 0 else 'N'}th request gets temp=0 (evenly distributed)")
    else:
        print(f"  - Random {num_temp0_requests} requests get temp=0 (reproducible with seed={args.seed})")
    print(f"  - Poisson arrival pattern maintained")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temperature assignment file
    temp_map_file = create_temperature_assignment_file(
        output_dir, temp0_pct, args.max_requests, 
        args.assignment_mode, args.seed
    )
    
    # Set environment variable for etalon to use
    env = os.environ.copy()
    env['TEMPERATURE_MAP_FILE'] = temp_map_file
    
    print(f"\nRunning etalon benchmark with mixed temperatures...")
    print(f"  Temperature map: {temp_map_file}")
    
    # Don't set temperature in additional_params - let per-request assignment handle it
    additional_params = json.dumps({})
    
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
