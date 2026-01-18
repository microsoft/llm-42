"""
Microbenchmark to measure impact of reducing SM count on MLP1 (gate_up_proj) of Llama3.3-70B.
Configured for 8-GPU tensor parallelism (TP=8).

Uses CUDA Green Contexts (via flashinfer) to limit SM count per kernel.
Uses CUDA Graphs for accurate timing (10 iterations per graph, 20 graph replays).

Usage:
    python bench_mlp1_sm_scaling.py                    # Run with default SM splits
    python bench_mlp1_sm_scaling.py --sm-counts 32,48,64,80,96
    python bench_mlp1_sm_scaling.py --batch-sizes 1,128,1024,4096
"""

import torch
import torch.nn.functional as F
import argparse
import statistics
import time
import subprocess
import flashinfer.green_ctx as green_ctx

torch.set_default_device('cuda')

# CUDA Graph configuration
GRAPH_ITERATIONS = 10  # Number of iterations captured in each CUDA graph
NUM_GRAPH_REPLAYS = 20  # Number of times to replay the graph for measurement
CUDA_SLEEP_NS = 1_000_000  # 1ms sleep before launching CUDA graph


def set_gpu_clock_to_tdp():
    """Set GPU clock speed to TDP for consistent benchmarking."""
    print("Setting GPU clock to TDP...")
    subprocess.run("nvidia-smi -pm ENABLED", shell=True, check=False)
    subprocess.run("nvidia-smi -lgc tdp", shell=True, check=False)
    print("GPU clock locked to TDP")


def reset_gpu_clock():
    """Reset GPU clock speed to default (unlocked) values."""
    print("Resetting GPU clock to default...")
    subprocess.run("nvidia-smi -pm ENABLED", shell=True, check=False)
    subprocess.run("nvidia-smi -rgc", shell=True, check=False)
    print("GPU clock reset to default")

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
    # Allocate ~40MB to flush L2 cache (typical L2 is 40-50MB on H100)
    cache_flush = torch.empty(10 * 1024 * 1024, dtype=torch.float32, device='cuda')
    cache_flush.zero_()
    del cache_flush
    torch.cuda.synchronize()


def bench_mlp1_with_green_ctx(batch_size, weight, gstream, warmup=20, iterations=200):
    """Benchmark MLP1 using F.linear within a green context stream with CUDA graphs."""
    x = torch.randn(batch_size, MLP1_K, device='cuda', dtype=torch.bfloat16)
    # Pre-allocate output for graph capture
    out = torch.empty(batch_size, MLP1_N, device='cuda', dtype=torch.bfloat16)
    
    with torch.cuda.stream(gstream):
        # Warm-up (outside graph)
        for _ in range(warmup):
            out = F.linear(x, weight)
        torch.cuda.synchronize()
        
        # Capture CUDA graph with GRAPH_ITERATIONS iterations
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gstream):
            for _ in range(GRAPH_ITERATIONS):
                out = F.linear(x, weight)
        
        # Measure graph replays
        times_ms = []
        for _ in range(NUM_GRAPH_REPLAYS):
            flush_cache()
            
            # CUDA sleep before launching graph
            torch.cuda._sleep(CUDA_SLEEP_NS)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record(gstream)
            graph.replay()
            end_event.record(gstream)
            
            torch.cuda.synchronize()
            # Time for all GRAPH_ITERATIONS, divide to get per-iteration time
            times_ms.append(start_event.elapsed_time(end_event) / GRAPH_ITERATIONS)
    
    median_time_ms = statistics.median(times_ms)
    tflops = 2 * batch_size * MLP1_K * MLP1_N / (median_time_ms / 1000 * 1e12)
    return median_time_ms, tflops


def bench_mlp1_baseline(batch_size, weight, warmup=20, iterations=200):
    """Benchmark MLP1 using standard F.linear (all SMs) with CUDA graphs."""
    x = torch.randn(batch_size, MLP1_K, device='cuda', dtype=torch.bfloat16)
    # Pre-allocate output for graph capture
    out = torch.empty(batch_size, MLP1_N, device='cuda', dtype=torch.bfloat16)
    
    # Warm-up (outside graph)
    for _ in range(warmup):
        out = F.linear(x, weight)
    torch.cuda.synchronize()
    
    # Capture CUDA graph with GRAPH_ITERATIONS iterations
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_ITERATIONS):
            out = F.linear(x, weight)
    
    # Measure graph replays
    times_ms = []
    for _ in range(NUM_GRAPH_REPLAYS):
        flush_cache()
        
        # CUDA sleep before launching graph
        torch.cuda._sleep(CUDA_SLEEP_NS)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        graph.replay()
        end_event.record()
        
        torch.cuda.synchronize()
        # Time for all GRAPH_ITERATIONS, divide to get per-iteration time
        times_ms.append(start_event.elapsed_time(end_event) / GRAPH_ITERATIONS)
    
    median_time_ms = statistics.median(times_ms)
    tflops = 2 * batch_size * MLP1_K * MLP1_N / (median_time_ms / 1000 * 1e12)
    return median_time_ms, tflops


def run_green_ctx_benchmark(batch_sizes, sm_counts, warmup=10, iterations=100):
    """Run benchmarks with different SM counts using green contexts."""
    total_sms = get_gpu_sm_count()
    
    # Set GPU clock to TDP for consistent results
    set_gpu_clock_to_tdp()
    
    print("\n" + "="*80)
    print("MLP1 (gate_up_proj) SM Scaling Benchmark - Llama3.3-70B (TP=8, per-GPU)")
    print("Using CUDA Green Contexts for SM limiting")
    print("Using CUDA Graphs for timing")
    print("="*80)
    print(f"\nMLP1 Shape: [B, {MLP1_K}] @ [{MLP1_K}, {MLP1_N}]")
    print(f"GPU Total SMs: {total_sms}")
    print(f"CUDA Graph: {GRAPH_ITERATIONS} iterations/graph, {NUM_GRAPH_REPLAYS} graph replays")
    print(f"Warmup: {warmup}, Iterations (ignored, using graphs): {iterations}")
    
    # Weight shape matches actual model: [out_features, in_features]
    # F.linear computes x @ weight.T internally
    weight = torch.randn(MLP1_N, MLP1_K, device='cuda', dtype=torch.bfloat16)
    
    results = {}
    
    # First, run baseline (all SMs)
    print(f"\n--- Baseline (All {total_sms} SMs) ---")
    results[total_sms] = {}
    for batch in batch_sizes:
        time_ms, tflops = bench_mlp1_baseline(batch, weight, warmup, iterations)
        results[total_sms][batch] = (time_ms, tflops)
        print(f"  Batch {batch:5d}: {time_ms:8.3f} ms, {tflops:6.2f} TFLOPS")
        time.sleep(1)  # Sleep 1 second between batches
    
    # Now run with different SM counts using green contexts
    for sm in sm_counts:
        if sm >= total_sms:
            continue  # Skip if SM count >= total (already covered by baseline)
            
        print(f"\n--- Green Context: {sm} SMs ---")
        results[sm] = {}
        
        # Create green context with specified SM count
        # split_device_green_ctx_by_sm_count returns streams for [sm, total_sms - sm]
        gstreams, _ = green_ctx.split_device_green_ctx_by_sm_count(
            torch.device('cuda:0'), [sm]
        )
        assert len(gstreams) == 2, f"Expected 2 streams, got {len(gstreams)}"
        
        # Use the first stream which has 'sm' SMs
        for batch in batch_sizes:
            time_ms, tflops = bench_mlp1_with_green_ctx(
                batch, weight, gstreams[0], warmup, iterations
            )
            results[sm][batch] = (time_ms, tflops)
            print(f"  Batch {batch:5d}: {time_ms:8.3f} ms, {tflops:6.2f} TFLOPS")
            time.sleep(1)  # Sleep 1 second between batches
    
    # Reset GPU clock to default after benchmarking
    reset_gpu_clock()
    
    return results


def print_comparison_table(results, total_sms):
    """Print a comparison table of results across SM counts."""
    print("\n" + "="*80)
    print("Summary: TFLOPS vs SM Count")
    print("="*80)
    
    sm_counts = sorted(results.keys(), reverse=True)
    batch_sizes = sorted(results[sm_counts[0]].keys())
    
    # Header
    header = f"{'Batch':>8}"
    for sm in sm_counts:
        if sm == total_sms:
            header += f" {sm} SMs (base)".rjust(16)
        else:
            header += f" {sm} SMs".rjust(16)
    print(header)
    print("-" * (8 + 16 * len(sm_counts)))
    
    # Data rows - TFLOPS
    for batch in batch_sizes:
        row = f"{batch:>8}"
        baseline_tflops = results[total_sms].get(batch, (0, 0))[1] if total_sms in results else None
        for sm in sm_counts:
            if batch in results[sm]:
                tflops = results[sm][batch][1]
                if baseline_tflops and sm != total_sms:
                    ratio = tflops / baseline_tflops * 100
                    row += f" {tflops:>6.1f} ({ratio:>4.0f}%)".rjust(16)
                else:
                    row += f" {tflops:>6.1f} TFLOPS".rjust(16)
            else:
                row += " N/A".rjust(16)
        print(row)
    
    print("\n" + "="*80)
    print("Summary: Latency (ms) vs SM Count")
    print("="*80)
    
    # Header
    header = f"{'Batch':>8}"
    for sm in sm_counts:
        if sm == total_sms:
            header += f" {sm} SMs (base)".rjust(16)
        else:
            header += f" {sm} SMs".rjust(16)
    print(header)
    print("-" * (8 + 16 * len(sm_counts)))
    
    # Data rows - Latency
    for batch in batch_sizes:
        row = f"{batch:>8}"
        baseline_time = results[total_sms].get(batch, (0, 0))[0] if total_sms in results else None
        for sm in sm_counts:
            if batch in results[sm]:
                time_ms = results[sm][batch][0]
                if baseline_time and sm != total_sms:
                    slowdown = time_ms / baseline_time
                    row += f" {time_ms:>6.2f} ({slowdown:>4.2f}x)".rjust(16)
                else:
                    row += f" {time_ms:>6.2f} ms".rjust(16)
            else:
                row += " N/A".rjust(16)
        print(row)
    
    # Print SM efficiency analysis
    print("\n" + "="*80)
    print("SM Efficiency Analysis (TFLOPS per SM)")
    print("="*80)
    
    header = f"{'Batch':>8}"
    for sm in sm_counts:
        header += f" {sm} SMs".rjust(12)
    print(header)
    print("-" * (8 + 12 * len(sm_counts)))
    
    for batch in batch_sizes:
        row = f"{batch:>8}"
        for sm in sm_counts:
            if batch in results[sm]:
                tflops = results[sm][batch][1]
                tflops_per_sm = tflops / sm
                row += f" {tflops_per_sm:>6.3f}".rjust(12)
            else:
                row += " N/A".rjust(12)
        print(row)


def save_results_to_markdown(results, total_sms, output_path):
    """Save results to a markdown file."""
    sm_counts = sorted(results.keys(), reverse=True)
    batch_sizes = sorted(results[sm_counts[0]].keys())
    
    lines = []
    lines.append("# MLP1 (gate_up_proj) SM Scaling Benchmark - Llama3.3-70B (TP=8, per-GPU)\n")
    lines.append(f"- **MLP1 Shape**: `[B, {MLP1_K}] @ [{MLP1_N}, {MLP1_K}].T`")
    lines.append(f"- **Total GPU SMs**: {total_sms}")
    lines.append(f"- **Timing**: Median of iterations with cache flush\n")
    lines.append("")
    
    # TFLOPS Table
    lines.append("## TFLOPS vs SM Count\n")
    header = "| Batch |"
    separator = "|------:|"
    for sm in sm_counts:
        if sm == total_sms:
            header += f" {sm} SMs (base) |"
        else:
            header += f" {sm} SMs (-{total_sms - sm}) |"
        separator += "------:|"
    lines.append(header)
    lines.append(separator)
    
    for batch in batch_sizes:
        row = f"| {batch} |"
        baseline_tflops = results[total_sms].get(batch, (0, 0))[1] if total_sms in results else None
        for sm in sm_counts:
            if batch in results[sm]:
                tflops = results[sm][batch][1]
                if baseline_tflops and sm != total_sms:
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
        description='Benchmark MLP1 (gate_up_proj) with SM scaling using Green Contexts'
    )
    parser.add_argument('--batch-sizes', type=str, default='32,64,128,256,512,1024,2048,4096,8192,16384,32768',
                        help='Comma-separated batch sizes')
    parser.add_argument('--sm-reductions', type=str, default='4,8,16,32',
                        help='Comma-separated SM reductions from total (e.g., "4,8" means total-4, total-8). '
                             'Note: Green contexts have granularity constraints, very small reductions may fail.')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=4, help='Benchmark iterations')
    parser.add_argument('--output', type=str, default='mlp1_sm_scaling_results.md',
                        help='Output markdown file path')
    
    args = parser.parse_args()
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    sm_reductions = [int(x) for x in args.sm_reductions.split(',')]
    
    total_sms = get_gpu_sm_count()
    print(f"Detected GPU with {total_sms} SMs")
    
    # Convert SM reductions to actual SM counts
    sm_counts = [total_sms - r for r in sm_reductions if total_sms - r > 0]
    if not sm_counts:
        print(f"Warning: All SM reductions result in <= 0 SMs, using defaults")
        sm_counts = [total_sms // 4, total_sms // 2, total_sms * 3 // 4]
    
    print(f"SM counts to test: {sm_counts} (from reductions: {sm_reductions})")
    
    results = run_green_ctx_benchmark(batch_sizes, sm_counts, args.warmup, args.iterations)
    print_comparison_table(results, total_sms)
    save_results_to_markdown(results, total_sms, args.output)
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
