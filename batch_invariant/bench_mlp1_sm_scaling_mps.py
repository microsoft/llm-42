"""
Microbenchmark to measure impact of reducing SM count on MLP1 (gate_up_proj) of Llama3.3-70B.
Configured for 8-GPU tensor parallelism (TP=8).

Uses CUDA MPS (Multi-Process Service) to limit SM percentage per process.
The script automatically starts and stops the MPS daemon.

Usage:
    python bench_mlp1_sm_scaling_mps.py                    # Run with default SM percentages
    python bench_mlp1_sm_scaling_mps.py --sm-percentages 50,75,90
    python bench_mlp1_sm_scaling_mps.py --batch-sizes 32,128,512,2048
    python bench_mlp1_sm_scaling_mps.py --no-auto-mps      # Don't auto-manage MPS daemon
"""

import torch
import torch.nn.functional as F
import argparse
import statistics
import subprocess
import os
import sys
import time
import atexit
import signal

# Global flag to track if we started MPS
_mps_started_by_script = False

torch.set_default_device('cuda')

# Llama3.3-70B dimensions (TP=8, per-GPU)
HIDDEN_SIZE = 8192
INTERMEDIATE_SIZE = 28672  # 3.5 * hidden_size
TP_SIZE = 8

# MLP1 = gate_up_proj: [B, HIDDEN_SIZE] @ [HIDDEN_SIZE, INTERMEDIATE_SIZE*2/TP]
# The gate and up projections are fused into a single matmul, sharded across TP GPUs
MLP1_K = HIDDEN_SIZE
MLP1_N = (INTERMEDIATE_SIZE * 2) // TP_SIZE  # 7168 per GPU (gate + up fused, sharded)


def get_gpu_sm_count():
    """Get the number of SMs on the current GPU."""
    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count


def flush_cache():
    """Flush GPU L2 cache by allocating and touching a large tensor."""
    cache_flush = torch.empty(10 * 1024 * 1024, dtype=torch.float32, device='cuda')
    cache_flush.zero_()
    del cache_flush
    torch.cuda.synchronize()


def bench_mlp1(batch_size, weight, warmup=20, iterations=200):
    """Benchmark MLP1 using F.linear."""
    x = torch.randn(batch_size, MLP1_K, device='cuda', dtype=torch.bfloat16)
    
    # Warm-up
    for _ in range(warmup):
        _ = F.linear(x, weight)
    torch.cuda.synchronize()
    
    # Measure each iteration individually for median calculation
    times_ms = []
    for _ in range(iterations):
        flush_cache()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = F.linear(x, weight)
        end_event.record()
        
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    
    median_time_ms = statistics.median(times_ms)
    tflops = 2 * batch_size * MLP1_K * MLP1_N / (median_time_ms / 1000 * 1e12)
    return median_time_ms, tflops


def run_benchmark_subprocess(batch_sizes, sm_percentage, warmup, iterations):
    """Run benchmark in a subprocess with specific MPS thread percentage."""
    import json
    
    env = os.environ.copy()
    env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(sm_percentage)
    
    # Create a simple inline script to run the benchmark
    # IMPORTANT: Set env var BEFORE importing torch to ensure CUDA sees it
    script = f'''
import os
# Verify the environment variable is set (for debugging)
mps_pct = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'NOT SET')

import torch
import torch.nn.functional as F
import statistics
import json
import sys

# Now set default device after env var is in place
torch.set_default_device('cuda')

MLP1_K = {MLP1_K}
MLP1_N = {MLP1_N}

def flush_cache():
    cache_flush = torch.empty(10 * 1024 * 1024, dtype=torch.float32, device='cuda')
    cache_flush.zero_()
    del cache_flush
    torch.cuda.synchronize()

def bench_mlp1(batch_size, weight, warmup, iterations):
    x = torch.randn(batch_size, MLP1_K, device='cuda', dtype=torch.bfloat16)
    
    for _ in range(warmup):
        _ = F.linear(x, weight)
    torch.cuda.synchronize()
    
    times_ms = []
    for _ in range(iterations):
        flush_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = F.linear(x, weight)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    
    median_time_ms = statistics.median(times_ms)
    tflops = 2 * batch_size * MLP1_K * MLP1_N / (median_time_ms / 1000 * 1e12)
    return median_time_ms, tflops

# Print MPS percentage to stderr for debugging
print(f"MPS thread percentage: {{mps_pct}}", file=sys.stderr)

weight = torch.randn(MLP1_N, MLP1_K, device='cuda', dtype=torch.bfloat16)
batch_sizes = {batch_sizes}
warmup = {warmup}
iterations = {iterations}

results = {{}}
for batch in batch_sizes:
    time_ms, tflops = bench_mlp1(batch, weight, warmup, iterations)
    results[batch] = (time_ms, tflops)

print(json.dumps(results))
'''
    
    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running subprocess with {sm_percentage}% threads:")
        print(result.stderr)
        return None
    
    # Print the MPS percentage confirmation from stderr
    if result.stderr:
        print(f"    [{result.stderr.strip()}]")
    
    try:
        # Parse the JSON output
        results = json.loads(result.stdout.strip())
        # Convert string keys back to int
        return {int(k): tuple(v) for k, v in results.items()}
    except json.JSONDecodeError:
        print(f"Failed to parse output: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return None


def check_mps_running():
    """Check if MPS daemon is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'nvidia-cuda-mps-control'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def start_mps_daemon():
    """Start the MPS daemon."""
    global _mps_started_by_script
    
    if check_mps_running():
        print("MPS daemon is already running.")
        return True
    
    print("Starting MPS daemon...")
    
    # Set CUDA_VISIBLE_DEVICES if not already set
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        # Start MPS control daemon
        result = subprocess.run(
            ['nvidia-cuda-mps-control', '-d'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Failed to start MPS daemon: {result.stderr}")
            return False
        
        # Wait a moment for daemon to initialize
        time.sleep(1)
        
        if check_mps_running():
            print("MPS daemon started successfully.")
            _mps_started_by_script = True
            return True
        else:
            print("MPS daemon failed to start.")
            return False
            
    except FileNotFoundError:
        print("Error: nvidia-cuda-mps-control not found. Is CUDA installed?")
        return False
    except PermissionError:
        print("Error: Permission denied. You may need to run with sudo or ensure proper permissions.")
        return False


def stop_mps_daemon():
    """Stop the MPS daemon if we started it."""
    global _mps_started_by_script
    
    if not _mps_started_by_script:
        return
    
    if not check_mps_running():
        return
    
    print("\nStopping MPS daemon...")
    
    try:
        # Send quit command to MPS control
        result = subprocess.run(
            ['nvidia-cuda-mps-control'],
            input='quit\n',
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Wait a moment for daemon to stop
        time.sleep(1)
        
        if not check_mps_running():
            print("MPS daemon stopped successfully.")
        else:
            # Force kill if still running
            subprocess.run(['pkill', '-f', 'nvidia-cuda-mps'], capture_output=True)
            print("MPS daemon force stopped.")
            
        _mps_started_by_script = False
        
    except subprocess.TimeoutExpired:
        print("Timeout stopping MPS daemon, force killing...")
        subprocess.run(['pkill', '-f', 'nvidia-cuda-mps'], capture_output=True)
    except Exception as e:
        print(f"Error stopping MPS daemon: {e}")


def run_mps_benchmark(batch_sizes, sm_percentages, warmup=20, iterations=100, auto_mps=True):
    """Run benchmarks with different SM percentages using MPS."""
    total_sms = get_gpu_sm_count()
    
    print("\n" + "="*80)
    print("MLP1 (gate_up_proj) SM Scaling Benchmark - Llama3.3-70B (TP=8, per-GPU)")
    print("Using CUDA MPS for SM limiting")
    print("="*80)
    print(f"\nMLP1 Shape: [B, {MLP1_K}] @ [{MLP1_K}, {MLP1_N}]")
    print(f"GPU Total SMs: {total_sms}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    
    results = {}
    
    # Start MPS first if needed
    mps_available = False
    if auto_mps:
        mps_available = start_mps_daemon()
        if mps_available:
            # Register cleanup handlers
            atexit.register(stop_mps_daemon)
            signal.signal(signal.SIGINT, lambda s, f: (stop_mps_daemon(), sys.exit(1)))
            signal.signal(signal.SIGTERM, lambda s, f: (stop_mps_daemon(), sys.exit(1)))
    else:
        mps_available = check_mps_running()
        if not mps_available:
            print("\n" + "!"*80)
            print("WARNING: MPS daemon is not running and --no-auto-mps was specified!")
            print("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE may not work without MPS.")
            print("!"*80)
    
    # Run baseline (100% threads) in subprocess for consistency
    print(f"\n--- Baseline (100% = {total_sms} SMs) ---")
    baseline_results = run_benchmark_subprocess(batch_sizes, 100, warmup, iterations)
    if baseline_results:
        results[100] = baseline_results
        for batch in batch_sizes:
            if batch in baseline_results:
                time_ms, tflops = baseline_results[batch]
                print(f"  Batch {batch:5d}: {time_ms:8.3f} ms, {tflops:6.2f} TFLOPS")
    else:
        print("Failed to run baseline benchmark!")
        return {}, total_sms
    
    # Run with different SM percentages
    for pct in sm_percentages:
        if pct >= 100:
            continue
            
        effective_sms = int(total_sms * pct / 100)
        print(f"\n--- MPS {pct}% (~{effective_sms} SMs) ---")
        
        # Run in subprocess to set MPS environment variable
        pct_results = run_benchmark_subprocess(batch_sizes, pct, warmup, iterations)
        
        if pct_results:
            results[pct] = pct_results
            for batch in batch_sizes:
                if batch in pct_results:
                    time_ms, tflops = pct_results[batch]
                    print(f"  Batch {batch:5d}: {time_ms:8.3f} ms, {tflops:6.2f} TFLOPS")
        else:
            print(f"  Failed to run benchmark at {pct}%")
    
    return results, total_sms


def print_comparison_table(results, total_sms):
    """Print a comparison table of results across SM percentages."""
    print("\n" + "="*80)
    print("Summary: TFLOPS vs SM Percentage")
    print("="*80)
    
    sm_pcts = sorted(results.keys(), reverse=True)
    batch_sizes = sorted(results[sm_pcts[0]].keys())
    
    # Header
    header = f"{'Batch':>8}"
    for pct in sm_pcts:
        effective_sms = int(total_sms * pct / 100)
        if pct == 100:
            header += f" {pct}% ({effective_sms}SM)".rjust(18)
        else:
            header += f" {pct}% (~{effective_sms}SM)".rjust(18)
    print(header)
    print("-" * (8 + 18 * len(sm_pcts)))
    
    # Data rows - TFLOPS
    for batch in batch_sizes:
        row = f"{batch:>8}"
        baseline_tflops = results[100].get(batch, (0, 0))[1] if 100 in results else None
        for pct in sm_pcts:
            if pct in results and batch in results[pct]:
                tflops = results[pct][batch][1]
                if baseline_tflops and pct != 100:
                    ratio = tflops / baseline_tflops * 100
                    row += f" {tflops:>6.1f} ({ratio:>4.0f}%)".rjust(18)
                else:
                    row += f" {tflops:>6.1f} TFLOPS".rjust(18)
            else:
                row += " N/A".rjust(18)
        print(row)
    
    print("\n" + "="*80)
    print("Summary: Latency (ms) vs SM Percentage")
    print("="*80)
    
    # Header
    header = f"{'Batch':>8}"
    for pct in sm_pcts:
        effective_sms = int(total_sms * pct / 100)
        if pct == 100:
            header += f" {pct}% ({effective_sms}SM)".rjust(18)
        else:
            header += f" {pct}% (~{effective_sms}SM)".rjust(18)
    print(header)
    print("-" * (8 + 18 * len(sm_pcts)))
    
    # Data rows - Latency
    for batch in batch_sizes:
        row = f"{batch:>8}"
        baseline_time = results[100].get(batch, (0, 0))[0] if 100 in results else None
        for pct in sm_pcts:
            if pct in results and batch in results[pct]:
                time_ms = results[pct][batch][0]
                if baseline_time and pct != 100:
                    slowdown = time_ms / baseline_time
                    row += f" {time_ms:>6.2f} ({slowdown:>4.2f}x)".rjust(18)
                else:
                    row += f" {time_ms:>6.2f} ms".rjust(18)
            else:
                row += " N/A".rjust(18)
        print(row)


def save_results_to_markdown(results, total_sms, output_path):
    """Save results to a markdown file."""
    sm_pcts = sorted(results.keys(), reverse=True)
    batch_sizes = sorted(results[sm_pcts[0]].keys())
    
    lines = []
    lines.append("# MLP1 (gate_up_proj) SM Scaling Benchmark - Llama3.3-70B (TP=8, per-GPU)\n")
    lines.append(f"- **MLP1 Shape**: `[B, {MLP1_K}] @ [{MLP1_N}, {MLP1_K}].T`")
    lines.append(f"- **Total GPU SMs**: {total_sms}")
    lines.append(f"- **Method**: CUDA MPS (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)")
    lines.append(f"- **Timing**: Median of iterations with cache flush\n")
    lines.append("")
    
    # TFLOPS Table
    lines.append("## TFLOPS vs SM Percentage\n")
    header = "| Batch |"
    separator = "|------:|"
    for pct in sm_pcts:
        effective_sms = int(total_sms * pct / 100)
        header += f" {pct}% (~{effective_sms} SMs) |"
        separator += "------:|"
    lines.append(header)
    lines.append(separator)
    
    for batch in batch_sizes:
        row = f"| {batch} |"
        baseline_tflops = results[100].get(batch, (0, 0))[1] if 100 in results else None
        for pct in sm_pcts:
            if pct in results and batch in results[pct]:
                tflops = results[pct][batch][1]
                if baseline_tflops and pct != 100:
                    ratio = tflops / baseline_tflops * 100
                    row += f" {tflops:.1f} ({ratio:.0f}%) |"
                else:
                    row += f" {tflops:.1f} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark MLP1 (gate_up_proj) with SM scaling using CUDA MPS'
    )
    parser.add_argument('--batch-sizes', type=str, default='32,64,128,256,512,1024,2048,4096,8192',
                        help='Comma-separated batch sizes')
    parser.add_argument('--sm-percentages', type=str, default='97,94,88,76,64,52',
                        help='Comma-separated SM percentages (100 is always included as baseline)')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--output', type=str, default='mlp1_sm_scaling_mps_results.md',
                        help='Output markdown file path')
    parser.add_argument('--no-auto-mps', action='store_true',
                        help='Disable automatic MPS daemon start/stop (use if MPS is already running)')
    
    args = parser.parse_args()
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    sm_percentages = [int(x) for x in args.sm_percentages.split(',')]
    
    total_sms = get_gpu_sm_count()
    print(f"Detected GPU with {total_sms} SMs")
    print(f"SM percentages to test: {sm_percentages}")
    print(f"Effective SMs: {[int(total_sms * p / 100) for p in sm_percentages]}")
    print(f"Auto MPS management: {'disabled' if args.no_auto_mps else 'enabled'}")
    
    try:
        results, total_sms = run_mps_benchmark(
            batch_sizes, sm_percentages, args.warmup, args.iterations,
            auto_mps=not args.no_auto_mps
        )
        print_comparison_table(results, total_sms)
        save_results_to_markdown(results, total_sms, args.output)
    finally:
        # Ensure MPS is stopped even if there's an error
        stop_mps_daemon()
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
