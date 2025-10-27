#!/usr/bin/env python3
"""
Benchmark script for comparing different RMSNorm implementations:
1. sglang-fused-rmsnorm-deterministic (default, CUDA kernel)
2. sglang-native-deterministic (SGLANG_ENABLE_DETERMINISTIC_INFERENCE without bit 64)
3. vllm-fused-rmsnorm-dynamic (via SGLANG_USE_VLLM_RMSNORM=dynamic)
4. vllm-fused-rmsnorm-256 (via SGLANG_USE_VLLM_RMSNORM=256)
5. vllm-fused-rmsnorm-1024 (via SGLANG_USE_VLLM_RMSNORM=1024)

This benchmark measures both performance and batch invariance properties.
It also tests correctness by comparing all implementations against the naive PyTorch version.
All tests use residual connections (residual != None).
"""

import argparse
import itertools
import os
import sys
import time
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add parent paths to import sglang
sys.path.insert(0, "/mnt/ddn/t-rajagond/batch_inv/sglang-deterministic/python")

# Import sglang RMSNorm layer
from sglang.srt.layers.layernorm import RMSNorm


# ============================================================================
# Helper RMSNorm implementations
# ============================================================================

class HuggingFaceRMSNorm(nn.Module):
    """Reference PyTorch implementation"""
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    """Naive PyTorch implementation for reference"""
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output



# ============================================================================
# Wrapper functions for different implementations
# ============================================================================

def create_rmsnorm_layer(hidden_size: int, eps: float = 1e-6, vllm_mode: Optional[str] = None, use_native: bool = False):
    """Create RMSNorm layer with optional vLLM mode or native mode configuration"""
    # Clear any existing environment variables
    os.environ.pop("SGLANG_USE_VLLM_RMSNORM", None)
    os.environ.pop("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", None)
    
    if use_native:
        # Set deterministic inference without bit 64 to force native mode
        os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
    elif vllm_mode:
        os.environ["SGLANG_USE_VLLM_RMSNORM"] = vllm_mode
    
    # Force reimport to pick up environment variable changes
    import importlib
    import sglang.srt.layers.layernorm
    importlib.reload(sglang.srt.layers.layernorm)
    from sglang.srt.layers.layernorm import RMSNorm
    
    return RMSNorm(hidden_size, eps).cuda()


def rmsnorm_sglang_default(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
):
    """SGLang default deterministic implementation (CUDA kernel)"""
    norm_layer = create_rmsnorm_layer(x.shape[-1], eps, vllm_mode=None, use_native=False)
    norm_layer.weight.data = weight
    return norm_layer(x, residual)


def rmsnorm_sglang_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
):
    """SGLang native deterministic implementation (PyTorch native)"""
    norm_layer = create_rmsnorm_layer(x.shape[-1], eps, vllm_mode=None, use_native=True)
    norm_layer.weight.data = weight
    return norm_layer(x, residual)


def rmsnorm_vllm_dynamic(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
):
    """vLLM-style implementation with dynamic BLOCK_SIZE"""
    norm_layer = create_rmsnorm_layer(x.shape[-1], eps, vllm_mode="dynamic", use_native=False)
    norm_layer.weight.data = weight
    return norm_layer(x, residual)


def rmsnorm_vllm_256(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
):
    """vLLM-style implementation with BLOCK_SIZE=256"""
    norm_layer = create_rmsnorm_layer(x.shape[-1], eps, vllm_mode="256", use_native=False)
    norm_layer.weight.data = weight
    return norm_layer(x, residual)


def rmsnorm_vllm_1024(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
):
    """vLLM-style implementation with BLOCK_SIZE=1024"""
    norm_layer = create_rmsnorm_layer(x.shape[-1], eps, vllm_mode="1024", use_native=False)
    norm_layer.weight.data = weight
    return norm_layer(x, residual)



# ============================================================================
# Correctness testing
# ============================================================================

def test_correctness(batch_size=4, hidden_size=4096):
    """
    Test correctness by comparing all implementations against the naive PyTorch version.
    Always tests with residual connections.
    """
    dtype = torch.bfloat16
    eps = 1e-6
    
    # Generate test data
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    
    # Get reference output from naive implementation
    output_naive = rmsnorm_naive(x.clone(), weight, residual.clone(), eps)
    
    implementations = [
        ("SGLang-Default", rmsnorm_sglang_default),
        ("SGLang-Native", rmsnorm_sglang_native),
        ("vLLM-Dynamic", rmsnorm_vllm_dynamic),
        ("vLLM-BS=256", rmsnorm_vllm_256),
        ("vLLM-BS=1024", rmsnorm_vllm_1024),
    ]
    
    print(f"\n{'='*80}")
    print(f"Correctness Test (batch={batch_size}, hidden={hidden_size}, with residual)")
    print(f"{'='*80}")
    print(f"{'Implementation':<25} | {'Status':<10}")
    print(f"{'-'*80}")
    
    all_correct = True
    for name, func in implementations:
        try:
            output = func(x.clone(), weight, residual.clone(), eps)
            
            output_test = output[0]
            output_ref = output_naive[0]
            
            # Use torch.testing.assert_close for correctness check
            torch.testing.assert_close(
                output_test,
                output_ref,
                rtol=1e-2,
                atol=1e-2,
                msg=f"Mismatch between {name} and reference",
            )
            
            status = "✓ PASS"
            print(f"{name:<25} | {status:<10}")
        except AssertionError as e:
            print(f"{name:<25} | {'✗ FAIL':<10}")
            print(f"  {str(e)[:100]}")
            all_correct = False
        except Exception as e:
            print(f"{name:<25} | {'✗ ERROR':<10}")
            print(f"  {str(e)[:100]}")
            all_correct = False
    
    print(f"{'-'*80}")
    print(f"Overall: {'✓ All tests passed' if all_correct else '✗ Some tests failed'}")
    return all_correct


# ============================================================================
# Batch invariance testing
# ============================================================================

def test_batch_invariance(rmsnorm_func, batch_size=2048, hidden_size=4096):
    """
    Test if an RMSNorm implementation is batch-invariant.
    Always tests with residual connections.
    Returns True if results are identical across different batch sizes.
    """
    dtype = torch.bfloat16
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)

    # Method 1: Single row
    out1 = rmsnorm_func(x[:1].clone(), weight, residual[:1].clone())[0]

    # Method 2: Batch size 10, take first row
    out2 = rmsnorm_func(x[:10].clone(), weight, residual[:10].clone())[0][:1]

    # Method 3: Batch size 128, take first row
    out3 = rmsnorm_func(x[:128].clone(), weight, residual[:128].clone())[0][:1]

    # Method 4: Full batch, take first row
    out_full = rmsnorm_func(x.clone(), weight, residual.clone())[0][:1]

    # Check if results are identical
    diff12 = (out1 - out2).abs().max().item()
    diff13 = (out1 - out3).abs().max().item()
    diff1_full = (out1 - out_full).abs().max().item()
    max_diff = max(diff12, diff13, diff1_full)

    is_invariant = max_diff < 1e-6
    return is_invariant, max_diff


def run_batch_invariance_test(rmsnorm_func, name, iterations=5):
    """Run batch invariance test multiple times"""
    print(f"\n{name}:")
    difflist = []
    is_deterministic = True
    
    for i in range(iterations):
        try:
            isd, df = test_batch_invariance(rmsnorm_func)
            is_deterministic = is_deterministic and isd
            difflist.append(df)
        except Exception as e:
            print(f"  Error: {e}")
            return
    
    print(f"  Batch Invariant: {is_deterministic}")
    print(f"  Max diff: {max(difflist):.2e}, Min diff: {min(difflist):.2e}")
    print(f"  Run-to-run variation: {max(difflist) - min(difflist):.2e}")


# ============================================================================
# Performance benchmarking
# ============================================================================

def benchmark_rmsnorm(
    norm_layer,
    batch_size: int,
    hidden_size: int,
    warmup_iters: int = 50,
    bench_iters: int = 1000,
):
    """Benchmark a single RMSNorm layer. Always uses residual."""
    dtype = torch.bfloat16
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    norm_layer.weight.data = weight
    residual = torch.randn_like(x)

    # Use inference mode for better performance
    with torch.inference_mode():
        # Warmup
        for _ in range(warmup_iters):
            try:
                input_tensor = x.clone()
                residual_tensor = residual.clone()
                _ = norm_layer(input_tensor, residual_tensor)
            except Exception as e:
                return None, str(e)

        torch.cuda.synchronize()
        
        # Benchmark using CUDA events for accurate timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
        
        for i in range(bench_iters):
            # Clone BEFORE recording the event
            input_tensor = x.clone()
            residual_tensor = residual.clone()
            # Now measure only the RMSNorm operation
            start_events[i].record()
            _ = norm_layer(input_tensor, residual_tensor)
            end_events[i].record()
        
        # Wait for events to complete
        torch.cuda.synchronize()
        
        # Calculate elapsed time in milliseconds
        elapsed_time_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(bench_iters)]
        avg_time_ms = sum(elapsed_time_ms) / bench_iters

    # Calculate bandwidth (GB/s)
    # Read: batch_size * hidden_size (input) + hidden_size (weight) + batch_size * hidden_size (residual)
    # Write: batch_size * hidden_size (output) + batch_size * hidden_size (residual)
    bytes_per_element = 2  # bfloat16
    total_bytes = batch_size * hidden_size * 4 * bytes_per_element + hidden_size * bytes_per_element
    
    bandwidth_gbs = (total_bytes / (avg_time_ms / 1000)) / 1e9
    
    return avg_time_ms, bandwidth_gbs


def run_benchmark_suite():
    """Run comprehensive benchmark suite and return results for plotting"""
    # Test configurations
    batch_sizes = [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    hidden_sizes = [4096, 8192]
    
    # Store results for plotting (initialize once for all hidden sizes)
    layer_names = ["SGLang-Default", "SGLang-Native", "vLLM-Dynamic", "vLLM-BS=256", "vLLM-BS=1024"]
    results = {name: {'batch_sizes': batch_sizes, 'times': {}, 'bandwidths': {}} for name in layer_names}
    
    print(f"\n{'='*80}")
    print(f"Performance Benchmark (with residual)")
    print(f"{'='*80}")

    for hidden_size in hidden_sizes:
        # Create all RMSNorm layers ONCE per hidden size to avoid overhead
        layers = {
            "SGLang-Default": create_rmsnorm_layer(hidden_size, vllm_mode=None, use_native=False),
            "SGLang-Native": create_rmsnorm_layer(hidden_size, vllm_mode=None, use_native=True),
            "vLLM-Dynamic": create_rmsnorm_layer(hidden_size, vllm_mode="dynamic", use_native=False),
            "vLLM-BS=256": create_rmsnorm_layer(hidden_size, vllm_mode="256", use_native=False),
            "vLLM-BS=1024": create_rmsnorm_layer(hidden_size, vllm_mode="1024", use_native=False),
        }
        
        print(f"\nHidden Size: {hidden_size}")
        print(f"{'-'*80}")
        print(f"{'Batch Size':<12} | {'Implementation':<25} | {'Time (ms)':<12} | {'BW (GB/s)':<12}")
        print(f"{'-'*80}")

        for batch_size in batch_sizes:
            for name, layer in layers.items():
                result = benchmark_rmsnorm(layer, batch_size, hidden_size)
                if result[0] is not None:
                    time_ms, bandwidth_gbs = result
                    print(f"{batch_size:<12} | {name:<25} | {time_ms:>10.4f} | {bandwidth_gbs:>10.2f}")
                    
                    # Store for plotting
                    if hidden_size not in results[name]['times']:
                        results[name]['times'][hidden_size] = []
                        results[name]['bandwidths'][hidden_size] = []
                    results[name]['times'][hidden_size].append(time_ms)
                    results[name]['bandwidths'][hidden_size].append(bandwidth_gbs)
                else:
                    print(f"{batch_size:<12} | {name:<25} | {'ERROR':<12} | {result[1]}")
    
    return results, batch_sizes, hidden_sizes


# ============================================================================
# Plotting
# ============================================================================

def plot_results(results, batch_sizes, hidden_sizes, output_dir="."):
    """Plot performance comparison: raw execution times and speedup vs SGLang-Native side by side"""
    
    # Create figure with subplots for each hidden size (2 rows x 2 columns)
    # Each row: raw time (left) and speedup (right)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RMSNorm Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    baseline_name = "SGLang-Native"
    
    for idx, hidden_size in enumerate(hidden_sizes):
        ax_time = axes[idx, 0]
        ax_speedup = axes[idx, 1]
        
        # Get baseline times for speedup calculation
        baseline_times = None
        if baseline_name in results and hidden_size in results[baseline_name]['times']:
            baseline_times = np.array(results[baseline_name]['times'][hidden_size])
        
        # Create equally spaced x positions for batch sizes
        x_positions = np.arange(len(batch_sizes))
        
        # Plot raw execution times and speedups
        for (name, _), color, marker in zip(results.items(), colors, markers):
            if hidden_size in results[name]['times']:
                times = np.array(results[name]['times'][hidden_size])
                
                # Plot raw times at equally spaced positions
                ax_time.plot(x_positions, times, marker=marker, color=color, 
                           label=name, linewidth=2, markersize=8)
                
                # Plot speedup (if not the baseline itself)
                if baseline_times is not None and name != baseline_name:
                    speedups = baseline_times / times
                    ax_speedup.plot(x_positions, speedups, marker=marker, color=color,
                                  label=name, linewidth=2, markersize=8)
        
        # Configure raw time subplot
        ax_time.set_xlabel('Batch Size', fontsize=12)
        ax_time.set_ylabel('Execution Time (ms)', fontsize=12)
        ax_time.set_title(f'Raw Execution Time - Hidden Size {hidden_size}', fontsize=13, fontweight='bold')
        ax_time.set_xticks(x_positions)
        ax_time.set_xticklabels([str(bs) for bs in batch_sizes])
        ax_time.grid(True, alpha=0.3, linestyle='--')
        ax_time.legend(fontsize=9, loc='best')
        
        # Configure speedup subplot
        ax_speedup.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
                          label=f'Baseline ({baseline_name})', alpha=0.7)
        ax_speedup.set_xlabel('Batch Size', fontsize=12)
        ax_speedup.set_ylabel('Speedup vs SGLang-Native', fontsize=12)
        ax_speedup.set_title(f'Speedup vs Native - Hidden Size {hidden_size}', fontsize=13, fontweight='bold')
        ax_speedup.set_xticks(x_positions)
        ax_speedup.set_xticklabels([str(bs) for bs in batch_sizes])
        ax_speedup.grid(True, alpha=0.3, linestyle='--')
        ax_speedup.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    
    # Save plot as PDF
    output_path = os.path.join(output_dir, 'rmsnorm_benchmark_all_configs.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    plt.close('all')



# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm implementations")
    parser.add_argument("--test-correctness", action="store_true",
                       help="Test correctness against naive PyTorch implementation")
    parser.add_argument("--test-invariance", action="store_true", 
                       help="Test batch invariance properties")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--plot", action="store_true",
                       help="Generate performance plots")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Directory to save plots (default: current directory)")
    args = parser.parse_args()

    if not any([args.test_correctness, args.test_invariance, args.benchmark, args.plot, args.all]):
        args.all = True

    print("="*80)
    print("RMSNorm Implementation Benchmark (5 Configurations)")
    print("="*80)
    print("Testing SGLang RMSNorm with different configurations:")
    print("  1. SGLang-Default (deterministic CUDA kernel)")
    print("  2. SGLang-Native (SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1, native PyTorch)")
    print("  3. vLLM-Dynamic (SGLANG_USE_VLLM_RMSNORM=dynamic)")
    print("  4. vLLM-BS=256 (SGLANG_USE_VLLM_RMSNORM=256)")
    print("  5. vLLM-BS=1024 (SGLANG_USE_VLLM_RMSNORM=1024)")
    print("\nAll tests use residual connections (residual != None)")
    print("="*80)

    if args.test_correctness or args.all:
        print("\n" + "="*80)
        print("Correctness Tests")
        print("="*80)
        
        test_correctness(batch_size=4, hidden_size=4096)
        
        # Test with different sizes
        print("\n--- Large Hidden Size (8192) ---")
        test_correctness(batch_size=8, hidden_size=8192)

    if args.test_invariance or args.all:
        print("\n" + "="*80)
        print("Batch Invariance Tests")
        print("="*80)
        
        run_batch_invariance_test(rmsnorm_naive, "Naive (PyTorch)")
        run_batch_invariance_test(rmsnorm_sglang_default, "SGLang-Default")
        run_batch_invariance_test(rmsnorm_sglang_native, "SGLang-Native")
        run_batch_invariance_test(rmsnorm_vllm_dynamic, "vLLM-Dynamic")
        run_batch_invariance_test(rmsnorm_vllm_256, "vLLM-BS=256")
        run_batch_invariance_test(rmsnorm_vllm_1024, "vLLM-BS=1024")

    results = None
    if args.benchmark or args.all or args.plot:
        results, batch_sizes, hidden_sizes = run_benchmark_suite()
    
    if args.plot or args.all:
        if results is not None:
            print("\n" + "="*80)
            print("Generating Performance Plots")
            print("="*80)
            plot_results(results, batch_sizes, hidden_sizes, args.output_dir)
        else:
            print("\n⚠ No benchmark results available for plotting. Run with --benchmark first.")

    print("\n" + "="*80)
    print("Benchmark Complete")
    print("="*80)


if __name__ == "__main__":
    main()
