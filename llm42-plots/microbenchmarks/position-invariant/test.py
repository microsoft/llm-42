"""
Test to confirm non-determinism of default NCCL all-reduce with batch size invariance.

This test uses the default torch.distributed.all_reduce (NCCL) which can be
NON-DETERMINISTIC. NCCL auto-selects algorithms (ring, tree, etc.) based on
message size and topology. Ring is typically used for large messages and is
deterministic, but tree algorithms (used for smaller messages) don't guarantee
fixed accumulation order for bfloat16/float16.

This test compares:
1. Default all-reduce (same batch size) - should be DETERMINISTIC
2. Default all-reduce (different batch size) - typically NON-DETERMINISTIC for bfloat16
3. Default all-reduce (position invariance) - fixed batch size, permuted positions
4. NCCL config sweep with position invariance and performance measurement using CUDA events
5. Performance comparison: Default vs Deterministic NCCL settings

Usage:
    python test_ar.py --test 1      # Run only test 1
    python test_ar.py --test 2      # Run only test 2
    python test_ar.py --test 3      # Run only test 3
    python test_ar.py --test 4      # Run only test 4 (NCCL sweep with perf)
    python test_ar.py --test 5      # Run only test 5 (Default vs Deterministic perf)
    python test_ar.py --test 1 2 3  # Run tests 1, 2, and 3
    python test_ar.py               # Run all tests
"""

import argparse
import multiprocessing as mp
import socket

import torch
import torch.distributed as dist


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test1_same_batch_size(rank, device, num_trials, BS, hidden_dim):
    """TEST 1: Default all-reduce (same batch size) - should be DETERMINISTIC"""
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 1: Default NCCL all_reduce (same batch size)")
        print(f"{'='*70}")
    dist.barrier()

    for batch_size in range(1, BS + 1):
        if rank == 0:
            print(f"  Batch size: {batch_size}")
        dist.barrier()

        base_input = torch.empty((batch_size, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)

        dist.barrier()

        results_allreduce_only = []
        for trial in range(num_trials):
            # Clone the same input
            inp = base_input.clone()

            # Use default NCCL all-reduce
            dist.all_reduce(inp)
            torch.cuda.synchronize()

            # Only compare output corresponding to first request
            out_first_req = inp[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_allreduce_only.append((checksum, first_vals))

        # Check determinism
        if rank == 0:
            ref_sum, ref_vals = results_allreduce_only[0]
            all_match = True
            for i, (s, vals) in enumerate(results_allreduce_only[1:], 1):
                if not torch.equal(ref_vals, vals) or ref_sum != s:
                    all_match = False
                    print(f"    Trial {i+1} DIFFERS! ref_sum={ref_sum:.6f}, got={s:.6f}")

            if all_match:
                print(f"    ✓ DEFAULT ALL_REDUCE (fixed BS={batch_size}): DETERMINISTIC (as expected)")
            else:
                print(f"    ✗ DEFAULT ALL_REDUCE (fixed BS={batch_size}): NON-DETERMINISTIC (unexpected!)")

    dist.barrier()


def test2_different_batch_size(rank, device, num_trials, BS, hidden_dim):
    """TEST 2: Default all-reduce (different batch size) - typically NON-DETERMINISTIC"""
    base_input = torch.empty((1, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    base_input_rand = torch.empty((1, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)

    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 2: Default NCCL all_reduce (different batch size)")
        print("Batches: [a], [a,x], [a,x,x], ...")
        print(f"{'='*70}")
    dist.barrier()

    results_allreduce_only = {trial: [] for trial in range(num_trials)}
    for trial in range(num_trials):
        for bs in range(1, BS + 1):
            # Construct batch: (batch_size, hidden_dim)
            # First element is base_input, rest are base_input_rand
            batch = torch.stack([base_input] + [base_input_rand] * (bs - 1), dim=0)
            # Shape: (bs, hidden_dim)

            # Flatten for all-reduce: (bs * hidden_dim,)
            batch_flat = batch.view(-1)

            # Use default NCCL all-reduce
            dist.all_reduce(batch_flat)
            torch.cuda.synchronize()

            # Reshape back to (bs, hidden_dim)
            batch_out = batch_flat.view(bs, hidden_dim)

            # Only compare output corresponding to first request
            out_first_req = batch_out[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_allreduce_only[trial].append((bs, checksum, first_vals))

    # Check determinism
    if rank == 0:
        for trial in range(num_trials):
            results = results_allreduce_only[trial]

            _, ref_sum, ref_vals = results[0]
            all_match = True
            for _, s, vals in results[1:]:
                if abs(ref_sum - s) > 1e-3 or not torch.allclose(
                    ref_vals, vals, rtol=1e-3
                ):
                    all_match = False

        if all_match:
            print("  ✓ DEFAULT ALL_REDUCE (variant BS): DETERMINISTIC")
        else:
            print("  ✗ DEFAULT ALL_REDUCE (variant BS): NON-DETERMINISTIC")

    dist.barrier()


def test3_position_invariance(rank, device, num_trials, hidden_dim, fixed_batch_sizes=None):
    """TEST 3: Default all-reduce (position invariance) - fixed batch size, permuted positions"""
    if fixed_batch_sizes is None:
        fixed_batch_sizes = [8, 128, 256, 512]

    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 3: Default NCCL all_reduce (position invariance)")
        print("Fixed batch size, element 'a' at different positions")
        print("Batches: [a,x,x,...], [x,a,x,...], [x,x,a,...], ...")
        print(f"{'='*70}")
    dist.barrier()
    for fixed_bs in fixed_batch_sizes:
        results_position_invariance = []  # Reset for each batch size
        if rank == 0:
            print(f"\n  Testing batch size: {fixed_bs}")
        dist.barrier()

        # Create base inputs - different per rank
        torch.manual_seed(142 + rank)
        target_input = torch.empty((1, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        filler_input = torch.empty((fixed_bs, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        dist.barrier()
        for pos in range(fixed_bs):
            # Construct batch: target element at position 'pos', rest are filler
            batch = filler_input.clone()
            batch[pos] = target_input.clone()
            # Shape: (fixed_bs, hidden_dim)
            torch.cuda.synchronize()

            # Use default NCCL all-reduce
            dist.all_reduce(batch)
            torch.cuda.synchronize()

            # Extract output corresponding to target element (at position 'pos')
            out_target = batch[pos].clone()
            checksum = out_target.sum().item()
            results_position_invariance.append((pos, checksum, out_target.clone()))

        # Check position invariance
        if rank == 0:
            all_trials_match = True
            _, ref_sum, ref_vals = results_position_invariance[0]
            for pos, s, vals in results_position_invariance[1:]:
                if ref_sum != s or not torch.equal(ref_vals, vals):
                    all_trials_match = False
                    print(f"    Position {pos} DIFFERS! ref_sum={ref_sum:.6f}, got={s:.6f}")

            if all_trials_match:
                print(f"    ✓ DEFAULT ALL_REDUCE (position invariance, BS={fixed_bs}): POSITION-INVARIANT", flush=True)
            else:
                print(f"    ✗ DEFAULT ALL_REDUCE (position invariance, BS={fixed_bs}): NOT POSITION-INVARIANT", flush=True)

        dist.barrier()


def test4_position_invariance_with_nccl_sweep(rank, device, hidden_dim, fixed_batch_sizes=None, world_size=4):
    """TEST 4: Position invariance with NCCL config sweep and performance measurement
    
    NOTE: This function is called from test4_worker which sets NCCL env vars BEFORE init_process_group.
    """
    import os

    if fixed_batch_sizes is None:
        fixed_batch_sizes = [8, 128, 256, 512, 1024, 2048, 4096]

    num_warmup = 10
    num_perf_trials = 30
    num_positions_to_test = 16  # Test a subset of positions for speed

    config_name = os.environ.get("TEST4_CONFIG_NAME", "unknown")

    if rank == 0:
        print(f"  Testing batch sizes: {fixed_batch_sizes}")
    dist.barrier()

    # Results storage: {batch_size: (is_invariant, avg_time_ms, min_time_ms, max_time_ms)}
    results = {}

    for fixed_bs in fixed_batch_sizes:
        # Synchronize seeds across tests but different per rank
        torch.manual_seed(142 + rank)

        target_input = torch.empty((1, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        filler_input = torch.empty((fixed_bs, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)

        # Select positions to test (evenly distributed)
        positions_to_test = list(range(0, fixed_bs, max(1, fixed_bs // num_positions_to_test)))
        if len(positions_to_test) > num_positions_to_test:
            positions_to_test = positions_to_test[:num_positions_to_test]
        # Always include first and last position
        if 0 not in positions_to_test:
            positions_to_test.insert(0, 0)
        if fixed_bs - 1 not in positions_to_test:
            positions_to_test.append(fixed_bs - 1)

        dist.barrier()

        # Warmup
        for _ in range(num_warmup):
            batch = filler_input.clone()
            batch[0] = target_input.clone()
            dist.all_reduce(batch)
        torch.cuda.synchronize()
        dist.barrier()

        # Test position invariance
        position_results = []
        for pos in positions_to_test:
            batch = filler_input.clone()
            batch[pos] = target_input.clone()
            dist.all_reduce(batch)
            torch.cuda.synchronize()

            out_target = batch[pos].clone()
            checksum = out_target.sum().item()
            position_results.append((pos, checksum, out_target.clone()))

        # Check invariance
        is_invariant = True
        _, ref_sum, ref_vals = position_results[0]
        for pos, s, vals in position_results[1:]:
            if ref_sum != s or not torch.equal(ref_vals, vals):
                is_invariant = False
                break

        dist.barrier()

        # Performance measurement using CUDA events
        times_ms = []
        
        for i in range(num_perf_trials):
            batch = filler_input.clone()
            batch[0] = target_input.clone()

            # Ensure previous work is done before starting timing
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            dist.all_reduce(batch)
            end_event.record()
            
            # Wait for this operation to complete before measuring
            torch.cuda.synchronize()
            
            times_ms.append(start_event.elapsed_time(end_event))

        dist.barrier()

        # Calculate timing
        avg_time_ms = sum(times_ms) / len(times_ms)
        min_time_ms = min(times_ms)
        max_time_ms = max(times_ms)

        results[fixed_bs] = (is_invariant, avg_time_ms, min_time_ms, max_time_ms)

        if rank == 0:
            status = "✓" if is_invariant else "✗"
            print(f"    BS={fixed_bs:>6}: {status} invariant={is_invariant}, "
                  f"avg={avg_time_ms:.3f}ms, min={min_time_ms:.3f}ms, max={max_time_ms:.3f}ms")

        dist.barrier()

    return results


def test4_worker(world_size, rank, port, config, fixed_batch_sizes, hidden_dim, result_queue):
    """Worker for test4 - sets NCCL env vars BEFORE init_process_group"""
    import os
    import sys
    
    try:
        # Set NCCL config BEFORE init_process_group - this is critical!
        os.environ["NCCL_ALGO"] = config["NCCL_ALGO"]
        os.environ["NCCL_PROTO"] = config["NCCL_PROTO"]
        os.environ["NCCL_NTHREADS"] = config["NCCL_NTHREADS"]
        os.environ["NCCL_SOCKET_NTHREADS"] = config["NCCL_SOCKET_NTHREADS"]
        os.environ["NCCL_LAUNCH_MODE"] = config["NCCL_LAUNCH_MODE"]
        os.environ["NCCL_MIN_NCHANNELS"] = config["NCCL_MIN_NCHANNELS"]
        os.environ["NCCL_MAX_NCHANNELS"] = config["NCCL_MAX_NCHANNELS"]
        os.environ["TEST4_CONFIG_NAME"] = config["name"]
        
        # Additional NCCL settings
        os.environ["NCCL_COLLNET_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout,
        )

        results = test4_position_invariance_with_nccl_sweep(
            rank, device, hidden_dim, fixed_batch_sizes, world_size
        )
        
        # Only rank 0 reports results
        if rank == 0 and result_queue is not None:
            result_queue.put((config["name"], results))

        dist.destroy_process_group()
        
    except Exception as e:
        if rank == 0:
            print(f"    ⚠ Worker error: {e}", flush=True)
        sys.exit(1)


def run_test4_sweep(world_size, fixed_batch_sizes, hidden_dim):
    """Run test4 by spawning separate process groups for each NCCL config"""
    from itertools import product
    import multiprocessing as mp
    
    # NCCL parameter options to sweep
    algo_options = ["allreduce:ring", "allreduce:tree"]
    proto_options = ["Simple", "LL", "LL128"]
    nthreads_options = ["1", "64", "128", "256", "512"]
    socket_nthreads_options = ["1", "2", "4", "8", "16"]
    launch_mode_options = ["GROUP", "PARALLEL"]
    nchannels_options = ["1", "2", "4", "8", "16", "32"]

    # Generate all configurations
    nccl_configs = []
    for algo, proto, nthreads, sock_nthreads, launch_mode, nchannels in product(
        algo_options, proto_options, nthreads_options, socket_nthreads_options, launch_mode_options, nchannels_options
    ):
        algo_short = "ring" if "ring" in algo else "tree"
        config_name = f"{algo_short}_{proto}_nt{nthreads}_snt{sock_nthreads}_{launch_mode}_ch{nchannels}"
        nccl_configs.append({
            "name": config_name,
            "NCCL_ALGO": algo,
            "NCCL_PROTO": proto,
            "NCCL_NTHREADS": nthreads,
            "NCCL_SOCKET_NTHREADS": sock_nthreads,
            "NCCL_LAUNCH_MODE": launch_mode,
            "NCCL_MIN_NCHANNELS": nchannels,
            "NCCL_MAX_NCHANNELS": nchannels,
        })

    print(f"\n{'='*70}")
    print("TEST 4: NCCL Config Sweep with Position Invariance & Performance")
    print(f"{'='*70}")
    print(f"Batch sizes: {fixed_batch_sizes}")
    print(f"Total NCCL configs: {len(nccl_configs)}")
    print(f"  Algorithms: {algo_options}")
    print(f"  Protocols: {proto_options}")
    print(f"  NCCL_NTHREADS: {nthreads_options}")
    print(f"  NCCL_SOCKET_NTHREADS: {socket_nthreads_options}")
    print(f"  NCCL_LAUNCH_MODE: {launch_mode_options}")
    print(f"  NCCL_MIN/MAX_NCHANNELS: {nchannels_options}")
    print(f"{'='*70}\n", flush=True)

    # Collect all results: {config_name: {batch_size: (is_invariant, avg, min, max)}}
    all_results = {}
    failed_configs = []
    
    TIMEOUT_SECONDS = 120  # 2 minutes timeout per config
    
    for config_idx, config in enumerate(nccl_configs):
        config_name = config["name"]
        print(f"\n--- [{config_idx+1}/{len(nccl_configs)}] Testing config: {config_name} ---", flush=True)
        
        port = get_open_port()
        result_queue = mp.Queue()
        
        procs = []
        try:
            for rank in range(world_size):
                p = mp.Process(
                    target=test4_worker,
                    args=(world_size, rank, port, config, fixed_batch_sizes, hidden_dim, result_queue if rank == 0 else None)
                )
                p.start()
                procs.append(p)

            # Wait for processes with timeout
            all_finished = True
            for p in procs:
                p.join(timeout=TIMEOUT_SECONDS)
                if p.is_alive():
                    all_finished = False
            
            # Check if any process failed or timed out
            any_failed = False
            for p in procs:
                if p.is_alive():
                    print(f"    ⚠ TIMEOUT: Process still running after {TIMEOUT_SECONDS}s, terminating...", flush=True)
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                    any_failed = True
                elif p.exitcode != 0:
                    print(f"    ⚠ FAILED: Process exited with code {p.exitcode}", flush=True)
                    any_failed = True
            
            if any_failed:
                failed_configs.append((config_name, "timeout or crash"))
                # Clean up any remaining processes
                for p in procs:
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=1)
                continue
            
            # Get results from rank 0
            if not result_queue.empty():
                name, results = result_queue.get()
                all_results[name] = results
            else:
                print(f"    ⚠ No results returned from config", flush=True)
                failed_configs.append((config_name, "no results"))
                
        except Exception as e:
            print(f"    ⚠ EXCEPTION: {e}", flush=True)
            failed_configs.append((config_name, str(e)))
            # Clean up processes
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()

    # Print failed configs summary
    if failed_configs:
        print(f"\n{'='*100}")
        print(f"FAILED CONFIGS: {len(failed_configs)} out of {len(nccl_configs)}")
        print(f"{'='*100}")
        for config_name, reason in failed_configs:
            print(f"  ✗ {config_name}: {reason}")
        print("-" * 100)

    # Print summary tables
    print(f"\n{'='*100}")
    print("SUMMARY TABLE: Best Position-Invariant Config per Batch Size")
    print(f"{'='*100}")
    print(f"{'Batch Size':>12} | {'Best Config':>50} | {'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10}")
    print("-" * 100)

    for bs in fixed_batch_sizes:
        # Filter to only position-invariant configs
        invariant_configs = []
        for config_name, bs_results in all_results.items():
            if bs in bs_results:
                is_inv, avg, mn, mx = bs_results[bs]
                if is_inv:
                    invariant_configs.append((config_name, avg, mn, mx))

        if invariant_configs:
            # Sort by average time
            invariant_configs.sort(key=lambda x: x[1])
            best_name, best_avg, best_min, best_max = invariant_configs[0]
            print(f"{bs:>12} | {best_name:>50} | {best_avg:>10.3f} | {best_min:>10.3f} | {best_max:>10.3f}")
        else:
            print(f"{bs:>12} | {'NO INVARIANT CONFIG':>50} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")

    print("-" * 100)

    # Print all configs performance
    print(f"\n{'='*120}")
    print("ALL CONFIGS PERFORMANCE (sorted by avg time per batch size)")
    print(f"{'='*120}")
    
    for bs in fixed_batch_sizes:
        print(f"\n--- Batch Size: {bs} ---")
        print(f"{'Config':>55} | {'Invariant':>10} | {'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10}")
        print("-" * 100)
        
        configs_for_bs = []
        for config_name, bs_results in all_results.items():
            if bs in bs_results:
                is_inv, avg, mn, mx = bs_results[bs]
                configs_for_bs.append((config_name, is_inv, avg, mn, mx))
        
        # Sort by avg time
        configs_for_bs.sort(key=lambda x: x[2])
        
        for name, is_inv, avg, mn, mx in configs_for_bs[:20]:  # Top 20
            inv_str = "✓" if is_inv else "✗"
            print(f"{name:>55} | {inv_str:>10} | {avg:>10.3f} | {mn:>10.3f} | {mx:>10.3f}")
    
    print("-" * 100)


def test5_perf_worker(world_size, rank, port, config, fixed_batch_sizes, hidden_dim, result_queue):
    """Worker for test5 - performance comparison between default and deterministic settings"""
    import os
    import sys
    
    try:
        config_name = config["name"]
        
        # Set NCCL config BEFORE init_process_group
        if config_name == "deterministic":
            os.environ["NCCL_COLLNET_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            os.environ["NCCL_P2P_NET_DISABLE"] = "1"
            os.environ["NCCL_MIN_NCHANNELS"] = "1"
            os.environ["NCCL_MAX_NCHANNELS"] = "1"
            os.environ["NCCL_ALGO"] = "allreduce:tree"
        # For "default", don't set any NCCL env vars - let NCCL auto-select
        
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout,
        )

        num_warmup = 10
        num_perf_trials = 100
        
        results = {}
        
        for fixed_bs in fixed_batch_sizes:
            torch.manual_seed(142 + rank)
            
            data = torch.empty((fixed_bs, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
            
            dist.barrier()
            
            # Warmup
            for _ in range(num_warmup):
                batch = data.clone()
                dist.all_reduce(batch)
            torch.cuda.synchronize()
            dist.barrier()
            
            # Performance measurement using CUDA events
            times_ms = []
            
            for i in range(num_perf_trials):
                batch = data.clone()
                
                # Ensure previous work is done before starting timing
                torch.cuda.synchronize()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                dist.all_reduce(batch)
                end_event.record()
                
                # Wait for this operation to complete before measuring
                torch.cuda.synchronize()
                
                times_ms.append(start_event.elapsed_time(end_event))

            dist.barrier()
            avg_time_ms = sum(times_ms) / len(times_ms)
            min_time_ms = min(times_ms)
            max_time_ms = max(times_ms)
            std_time_ms = (sum((t - avg_time_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

            results[fixed_bs] = (avg_time_ms, min_time_ms, max_time_ms, std_time_ms)

            if rank == 0:
                print(f"    BS={fixed_bs:>6}: avg={avg_time_ms:.3f}ms, min={min_time_ms:.3f}ms, "
                      f"max={max_time_ms:.3f}ms, std={std_time_ms:.3f}ms")

            dist.barrier()
        
        # Only rank 0 reports results
        if rank == 0 and result_queue is not None:
            result_queue.put((config_name, results))

        dist.destroy_process_group()
        
    except Exception as e:
        if rank == 0:
            print(f"    ⚠ Worker error: {e}", flush=True)
        sys.exit(1)


def run_test5_perf_comparison(world_size, fixed_batch_sizes, hidden_dim):
    """Run test5 - compare performance between default and deterministic NCCL settings"""
    import multiprocessing as mp
    
    configs = [
        {"name": "default"},
        {"name": "deterministic"},
    ]

    print(f"\n{'='*70}")
    print("TEST 5: Performance Comparison - Default vs Deterministic NCCL")
    print(f"{'='*70}")
    print(f"Batch sizes: {fixed_batch_sizes}")
    print(f"Configs: {[c['name'] for c in configs]}")
    print(f"Deterministic settings:")
    print(f"  NCCL_COLLNET_ENABLE=0")
    print(f"  NCCL_NVLS_ENABLE=0")
    print(f"  NCCL_P2P_NET_DISABLE=1")
    print(f"  NCCL_MIN_NCHANNELS=1")
    print(f"  NCCL_MAX_NCHANNELS=1")
    print(f"  NCCL_ALGO=allreduce:tree")
    print(f"{'='*70}\n", flush=True)

    all_results = {}
    failed_configs = []
    
    TIMEOUT_SECONDS = 300  # 5 minutes timeout per config
    
    for config in configs:
        config_name = config["name"]
        print(f"\n--- Testing config: {config_name} ---", flush=True)
        
        port = get_open_port()
        result_queue = mp.Queue()
        
        procs = []
        try:
            for rank in range(world_size):
                p = mp.Process(
                    target=test5_perf_worker,
                    args=(world_size, rank, port, config, fixed_batch_sizes, hidden_dim, result_queue if rank == 0 else None)
                )
                p.start()
                procs.append(p)

            # Wait for processes with timeout
            for p in procs:
                p.join(timeout=TIMEOUT_SECONDS)
            
            # Check if any process failed or timed out
            any_failed = False
            for p in procs:
                if p.is_alive():
                    print(f"    ⚠ TIMEOUT: Process still running after {TIMEOUT_SECONDS}s, terminating...", flush=True)
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                    any_failed = True
                elif p.exitcode != 0:
                    print(f"    ⚠ FAILED: Process exited with code {p.exitcode}", flush=True)
                    any_failed = True
            
            if any_failed:
                failed_configs.append((config_name, "timeout or crash"))
                for p in procs:
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=1)
                continue
            
            # Get results from rank 0
            if not result_queue.empty():
                name, results = result_queue.get()
                all_results[name] = results
            else:
                print(f"    ⚠ No results returned from config", flush=True)
                failed_configs.append((config_name, "no results"))
                
        except Exception as e:
            print(f"    ⚠ EXCEPTION: {e}", flush=True)
            failed_configs.append((config_name, str(e)))
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()

    # Print summary table
    print(f"\n{'='*100}")
    print("PERFORMANCE COMPARISON: Default vs Deterministic")
    print(f"{'='*100}")
    print(f"{'Batch Size':>12} | {'Default Avg':>12} | {'Determ Avg':>12} | {'Overhead':>12} | {'Default Min':>12} | {'Determ Min':>12}")
    print("-" * 100)

    for bs in fixed_batch_sizes:
        default_results = all_results.get("default", {}).get(bs)
        determ_results = all_results.get("deterministic", {}).get(bs)
        
        if default_results and determ_results:
            def_avg, def_min, def_max, def_std = default_results
            det_avg, det_min, det_max, det_std = determ_results
            overhead_pct = ((det_avg - def_avg) / def_avg) * 100
            print(f"{bs:>12} | {def_avg:>10.3f}ms | {det_avg:>10.3f}ms | {overhead_pct:>+10.1f}% | {def_min:>10.3f}ms | {det_min:>10.3f}ms")
        else:
            print(f"{bs:>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}")

    print("-" * 100)
    
    # Detailed results
    print(f"\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    
    for config_name in ["default", "deterministic"]:
        if config_name in all_results:
            print(f"\n--- {config_name.upper()} ---")
            print(f"{'Batch Size':>12} | {'Avg (ms)':>12} | {'Min (ms)':>12} | {'Max (ms)':>12} | {'Std (ms)':>12}")
            print("-" * 70)
            for bs in fixed_batch_sizes:
                if bs in all_results[config_name]:
                    avg, mn, mx, std = all_results[config_name][bs]
                    print(f"{bs:>12} | {avg:>12.3f} | {mn:>12.3f} | {mx:>12.3f} | {std:>12.3f}")
    
    print("-" * 100)
    
    if failed_configs:
        print(f"\n⚠ FAILED CONFIGS: {failed_configs}")


def worker(world_size, rank, port, tests_to_run):
    import os
    os.environ["NCCL_ALGO"] = "allreduce:tree"  # Force ring algorithm for deterministic reduction order
    # # NCCL determinism settings
    # os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    # os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    # os.environ["NCCL_NTHREADS"] = "512"
    # os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    num_trials = 200

    # Matrix sizes similar to real model layers
    # Format: (batch_size, hidden_dim) - typical tensor shape for all-reduce
    BS = 32768  # max batch_size (1..BS)
    hidden_dim = 4096  # hidden dimension / intermediate dimension

    # Different seed per rank - each GPU has DIFFERENT input
    torch.manual_seed(42 + rank)

    # Run selected tests
    if 1 in tests_to_run:
        test1_same_batch_size(rank, device, num_trials, BS, hidden_dim)

    if 2 in tests_to_run:
        test2_different_batch_size(rank, device, num_trials, BS, hidden_dim)

    if 3 in tests_to_run:
        test3_position_invariance(rank, device, num_trials, hidden_dim, fixed_batch_sizes=[8, 128, 256, 512, 617, 1999, 2772, 4096, 8192, 9199, 17689, 24576, 32768, 49786])

    # Note: test4 is handled separately in main() - it spawns its own process groups

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="NCCL All-Reduce Determinism Test")
    parser.add_argument(
        "--test", "-t",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Which tests to run (1, 2, 3, 4, 5, or combination). Default: all tests"
    )
    args = parser.parse_args()

    tests_to_run = set(args.test)

    world_size = 4
    available_gpus = torch.cuda.device_count()

    print("=" * 70)
    print("Default NCCL All-Reduce Determinism Test")
    print("=" * 70)
    print(f"Available GPUs: {available_gpus}")
    print(f"Using world_size: {world_size}")
    print(f"Running tests: {sorted(tests_to_run)}", flush=True)

    if available_gpus < world_size:
        print(
            f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead"
        )
        world_size = available_gpus

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this test")
        return

    mp.set_start_method("spawn", force=True)
    port = get_open_port()

    # Run tests 1, 2, 3 in shared process group
    tests_123 = tests_to_run - {4, 5}
    if tests_123:
        procs = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(world_size, rank, port, tests_123))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    # Run test 4 separately - it spawns its own process groups per config
    if 4 in tests_to_run:
        hidden_dim = 4096
        fixed_batch_sizes = [8, 256]
        run_test4_sweep(world_size, fixed_batch_sizes, hidden_dim)

    # Run test 5 - performance comparison default vs deterministic
    if 5 in tests_to_run:
        hidden_dim = 4096
        fixed_batch_sizes = [8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        run_test5_perf_comparison(world_size, fixed_batch_sizes, hidden_dim)


if __name__ == "__main__":
    main()