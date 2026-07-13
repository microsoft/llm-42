#!/usr/bin/env python3
"""
TP-aware matmul micro-benchmark: cuBLAS vs deterministic kernels.

Compares kernel performance at the actual GEMM shapes seen by each GPU
under tensor parallelism (TP=1,2,4,8). With TP, weight matrices are
sharded so each rank does a smaller GEMM — this is where cuBLAS can
have a bigger advantage over deterministic (batch-invariant) kernels.

Usage:
    python bench_matmul_tp.py                         # defaults
    python bench_matmul_tp.py --model llama3-70b      # 70B shapes
    python bench_matmul_tp.py --tp-sizes 1,4          # only TP=1,4
    python bench_matmul_tp.py --batch-sizes 1,64,256  # custom batches

Outputs: console table + CSV + PDF plots in runs/<gpu>/matmul_tp/
"""

import argparse
import csv
import os
import sys
import time

import torch

# Ensure the local benchmark utilities are importable
sys.path.insert(0, os.path.dirname(__file__))

from bench_utils_tp import (
    get_model_ops_config,
    get_short_gpu_name,
    get_tp_sharded_shape,
    Tee,
)

torch.set_default_device("cuda")


# ── Kernel wrappers ──────────────────────────────────────────────────────────

def cublas_mm(a, b):
    """Standard cuBLAS matmul (non-deterministic default)."""
    return torch.mm(a, b)


def _make_col_major(b):
    """Ensure b is column-major (b.T is contiguous), required by DeepGEMM."""
    if b.transpose(0, 1).is_contiguous():
        return b
    return b.T.contiguous().T


def _get_deepgemm_mm():
    """Lazy import of DeepGEMM."""
    try:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            _matmul_persistent_deepgemm,
        )
        def wrapper(a, b):
            return _matmul_persistent_deepgemm(a, _make_col_major(b))
        return wrapper
    except Exception:
        return None


def _get_triton_persistent_mm():
    """Lazy import of Triton persistent matmul."""
    try:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            _matmul_persistent_triton,
        )
        return _matmul_persistent_triton
    except Exception:
        return None


# ── Benchmarking ─────────────────────────────────────────────────────────────

def bench_kernel(matmul_func, M, K, N, warmup=10, iters=50):
    """Benchmark a matmul kernel. Returns (avg_time_s, tflops) or (None, None)."""
    # Simulate F.linear layout: weight is (N,K) row-major, mm gets weight.t() = (K,N) col-major
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).T

    try:
        for _ in range(warmup):
            matmul_func(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            matmul_func(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_s = elapsed / iters
        tflops = 2 * M * K * N / (avg_s * 1e12)
        return avg_s, tflops
    except RuntimeError:
        return None, None


def bench_kernel_cuda_events(matmul_func, M, K, N, warmup=10, iters=100):
    """Benchmark with CUDA events for more accurate GPU timing."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).T

    try:
        for _ in range(warmup):
            matmul_func(a, b)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iters):
            matmul_func(a, b)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        avg_ms = elapsed_ms / iters
        avg_s = avg_ms / 1000
        tflops = 2 * M * K * N / (avg_s * 1e12)
        return avg_s, tflops
    except RuntimeError:
        return None, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TP-aware matmul microbenchmark")
    parser.add_argument("--model", default="llama3-8b", choices=list(MODEL_NAMES))
    parser.add_argument("--tp-sizes", default="1,2,4,8",
                        help="Comma-separated TP sizes to simulate")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192",
                        help="Comma-separated M (token batch) sizes")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--use-cuda-events", action="store_true", default=True,
                        help="Use CUDA events for timing (default: True)")
    args = parser.parse_args()

    tp_sizes = [int(x) for x in args.tp_sizes.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    ops_config = get_model_ops_config(args.model)

    # Set up output dir
    gpu_name = get_short_gpu_name()
    run_dir = os.path.join(os.path.dirname(__file__), "runs", gpu_name, args.model)
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(run_dir, "results.txt"))

    bench_fn = bench_kernel_cuda_events if args.use_cuda_events else bench_kernel

    # Discover available kernels
    kernel_funcs = {"cuBLAS": cublas_mm}
    _deepgemm = _get_deepgemm_mm()
    _triton = _get_triton_persistent_mm()

    if _deepgemm:
        kernel_funcs["DeepGEMM"] = _deepgemm
    if _triton:
        kernel_funcs["Triton"] = _triton

    kernel_names = list(kernel_funcs.keys())

    print("=" * 100)
    print(f"TP-Aware Matmul Micro-Benchmark")
    print(f"  GPU:       {gpu_name} ({torch.cuda.get_device_name(0)})")
    print(f"  Model:     {args.model}")
    print(f"  TP sizes:  {tp_sizes}")
    print(f"  Batches:   {batch_sizes}")
    print(f"  Kernels:   {kernel_names}")
    print(f"  Iters:     {args.iters} (warmup: {args.warmup})")
    print(f"  Timing:    {'CUDA events' if args.use_cuda_events else 'wall clock'}")
    print(f"  Output:    {run_dir}")
    print("=" * 100)

    # CSV output
    csv_path = os.path.join(run_dir, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "op", "tp_shard", "tp_size", "batch_size",
        "M", "K", "N", "kernel", "avg_us", "tflops",
        "slowdown_vs_cublas",
    ])

    # Collect all data for plotting
    plot_data = {}  # (op_name, tp_size) -> {kernel_name -> {batch_size -> tflops}}

    for op_name, op_cfg in ops_config.items():
        for tp in tp_sizes:
            K, N = get_tp_sharded_shape(op_cfg, tp)

            print(f"\n{'─'*90}")
            print(f"  {op_name} ({op_cfg['desc']})  TP={tp}  shard={op_cfg['tp_shard']}")
            print(f"  GEMM shape per rank: [M, {K}] @ [{K}, {N}]")
            print(f"{'─'*90}")

            # Table header
            hdr = f"  {'M':>6}"
            for kn in kernel_names:
                hdr += f" | {kn:>12} TFLOPS {kn:>10} us"
            hdr += " | slowdown"
            print(hdr)
            print(f"  {'-' * (len(hdr) - 2)}")

            plot_data[(op_name, tp)] = {kn: {} for kn in kernel_names}

            for bs in batch_sizes:
                M = bs
                row = f"  {M:>6}"

                cublas_tflops = None
                results_this_row = {}

                for kn in kernel_names:
                    avg_s, tflops = bench_fn(kernel_funcs[kn], M, K, N, args.warmup, args.iters)
                    results_this_row[kn] = (avg_s, tflops)

                    if tflops is not None:
                        avg_us = avg_s * 1e6
                        row += f" | {tflops:>12.2f}       {avg_us:>10.1f}   "
                        plot_data[(op_name, tp)][kn][bs] = tflops
                        if kn == "cuBLAS":
                            cublas_tflops = tflops
                    else:
                        row += f" | {'N/A':>12}       {'N/A':>10}   "

                # Compute slowdown for best deterministic kernel vs cuBLAS
                if cublas_tflops and cublas_tflops > 0:
                    det_tflops_list = [
                        results_this_row[kn][1]
                        for kn in kernel_names if kn != "cuBLAS"
                        and results_this_row[kn][1] is not None
                    ]
                    if det_tflops_list:
                        best_det = max(det_tflops_list)
                        slowdown = cublas_tflops / best_det
                        row += f" | {slowdown:>6.2f}x"
                    else:
                        row += f" | {'N/A':>6}"
                        slowdown = None
                else:
                    row += f" | {'N/A':>6}"
                    slowdown = None

                print(row)

                # Write CSV rows
                for kn in kernel_names:
                    avg_s, tflops = results_this_row[kn]
                    sd = ""
                    if kn != "cuBLAS" and cublas_tflops and tflops:
                        sd = f"{cublas_tflops / tflops:.4f}"
                    csv_writer.writerow([
                        op_name, op_cfg["tp_shard"], tp, bs,
                        M, K, N, kn,
                        f"{avg_s*1e6:.2f}" if avg_s else "",
                        f"{tflops:.2f}" if tflops else "",
                        sd,
                    ])

    csv_file.close()
    print(f"\n  CSV saved to: {csv_path}")

    # ── Summary table: slowdown at key batch sizes across TP ─────────────
    print("\n" + "=" * 100)
    print("Summary: Best-deterministic / cuBLAS slowdown by TP size")
    print("=" * 100)

    summary_batches = [1, 8, 64, 256, 1024, 4096]
    summary_batches = [b for b in summary_batches if b in batch_sizes]

    for op_name in ops_config:
        print(f"\n  {op_name}:")
        hdr = f"    {'TP':>4} {'shape':>20}"
        for bs in summary_batches:
            hdr += f" | M={bs:>5}"
        print(hdr)
        print(f"    {'-' * (len(hdr) - 4)}")

        for tp in tp_sizes:
            K, N = get_tp_sharded_shape(ops_config[op_name], tp)
            data = plot_data.get((op_name, tp), {})
            row = f"    {tp:>4} [{K:>5}x{N:>5}]"

            for bs in summary_batches:
                cublas_t = data.get("cuBLAS", {}).get(bs)
                det_ts = [
                    data[kn].get(bs) for kn in kernel_names
                    if kn != "cuBLAS" and data.get(kn, {}).get(bs) is not None
                ]
                if cublas_t and det_ts:
                    best_det = max(det_ts)
                    sd = cublas_t / best_det
                    row += f" | {sd:>6.2f}x"
                else:
                    row += f" | {'N/A':>6}"
            print(row)


    # ── Generate plots from CSV ──────────────────────────────────────────
    try:
        from plot import plot_csv
        plot_csv(csv_path, model=args.model, out_dir=run_dir)
    except ImportError:
        print("  (plot.py not found or matplotlib missing, skipping plots)")
    except Exception as e:
        print(f"  WARNING: plotting failed: {e}")

    print(f"\nAll results in: {run_dir}")


MODEL_NAMES = ["llama3-8b", "llama3-70b"]

if __name__ == "__main__":
    main()
