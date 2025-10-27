import torch
from sglang.srt.batch_invariant_ops import matmul_persistent
from sgl_kernel import bf16_batch_invariant_mm, bf16_batch_invariant_fused_mm

torch.set_default_device('cuda')

def bi_kernel_wrapper(a, b):
    return bf16_batch_invariant_mm(a, b, a.dtype)

def split_kernel_wrapper(a, b, split_frac=0.0):
    return bf16_batch_invariant_fused_mm(a, b, a.dtype, split_frac=split_frac)

def test_equality():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D).to(torch.float16).cuda()
    b = torch.linspace(-100, 100, D*D).reshape(D, D).to(torch.float16).cuda()

    out1 = torch.mm(a, b)
    out2 = matmul_persistent(a, b)
    out3 = bi_kernel_wrapper(a, b)
    out4 = split_kernel_wrapper(a, b, split_frac=0.25)
    out5 = split_kernel_wrapper(a, b, split_frac=0.5)

    diff2 = (out1 - out2).abs().max()
    diff3 = (out1 - out3).abs().max()
    diff4 = (out1 - out4).abs().max()
    diff5 = (out1 - out5).abs().max()
    print(f"Diff (matmul_persistent vs torch.mm): {diff2.item()}")
    print(f"Diff (bi_kernel vs torch.mm): {diff3.item()}")
    print(f"Diff (split 25% vs torch.mm): {diff4.item()}")
    print(f"Diff (split 50% vs torch.mm): {diff5.item()}")

def test_batch_invariance(matmul_func):
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D).to(torch.float16)
    b = torch.linspace(-100, 100, D*D).reshape(D, D).to(torch.float16)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = matmul_func(a[:1], b)

    # Batch size 10
    out2 = matmul_func(a[:10], b)[:1]

    # Batch size 128
    out3 = matmul_func(a[:128], b)[:1]

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out_full = matmul_func(a, b)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max() + (out1 - out3).abs().max() + (out1 - out_full).abs().max()
    return diff.item() == 0, diff.item()

def run_iters(func, iters=10):
    difflist = []
    is_deterministic = True
    for i in range (iters):
        isd, df = test_batch_invariance(func)
        is_deterministic = is_deterministic and isd
        difflist.append(df)
    print( f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {iters} iterations")

def bench_perf(matmul_func, B, K=16384, D=4096, iterations=10):
    M = B
    N = D
    # K = D * 4

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # Warm-up
    for _ in range(5):
        _ = matmul_func(a, b)

    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        _ = matmul_func(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    tflops = 2 * M * K * N / (avg_time * 1e12)
    return tflops

test_equality()

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
run_iters(torch.mm)

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
run_iters(matmul_persistent)

# Test with batch-invariant operations
print("\nBatch-Invariant-Fused-kernel Mode:")
run_iters(bi_kernel_wrapper)

print("\nBatch-Invariant-Fused-kernel 25% Mode:")
run_iters(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.25))

print("\nBatch-Invariant-Fused-kernel 50% Mode:")
run_iters(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.5))
print()

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
CONFIGS = [
    "torch",
    "bi",
    "bi_fused",
    # "split_12.5",
    # "split_25",
    # "split_50",
    #"split_75",
    #"split_87.5"
]

tflops_results = {}
for config in CONFIGS:
    tflops_results[config] = {}

# Benchmark performance
for batch_size in BATCHES:
    tflops_results["torch"][batch_size] = bench_perf(torch.mm, B=batch_size)
    tflops_results["bi"][batch_size] = bench_perf(matmul_persistent, B=batch_size)
    tflops_results["bi_fused"][batch_size] = bench_perf(bi_kernel_wrapper, B=batch_size)
    # tflops_results["split_12.5"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.125), B=batch_size)
    # tflops_results["split_25"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.25), B=batch_size)
    # tflops_results["split_50"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.5), B=batch_size)
    #tflops_results["split_75"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.75), B=batch_size)
    #tflops_results["split_87.5"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.875), B=batch_size)

    #print(tflops_results)
    #print([f'{k}: {tflops_results[k][batch_size]:.2f}' for k in CONFIGS if k.startswith('split_')])
    #BI TFLOPS: {tflops_results["bi"][batch_size]:.2f} |
    print(f"Batch Size: {batch_size} | PyTorch TFLOPS: {tflops_results["torch"][batch_size]:.2f} | "
          f" BI-Fused TFLOPS: {tflops_results["bi_fused"][batch_size]:.2f} | "
          f"Split TFLOPS: {', '.join([f'{k}: {tflops_results[k][batch_size]:.2f}' for k in CONFIGS if k.startswith('split_')])}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Prepare data (preserve the BATCHES order)
x = BATCHES
torch_vals = [tflops_results['torch'].get(b, 0.0) for b in x]
bi_vals = [tflops_results['bi'].get(b, 0.0) for b in x]
bi_fused_vals = [tflops_results['bi_fused'].get(b, 0.0) for b in x]

plt.figure(figsize=(10, 6))
plt.plot(x, torch_vals, marker='o', label='PyTorch (torch.mm)')
plt.plot(x, bi_vals, marker='s', label='Batch-Invariant (matmul_persistent)')
plt.plot(x, bi_fused_vals, marker='^', label='BI-Fused (bi_kernel_wrapper)')
for config in CONFIGS:
    if config.startswith('split_'):
        split_vals = [tflops_results[config].get(b, 0.0) for b in x]
        plt.plot(x, split_vals, marker='x', label=f'Split {config.split("_")[1]}')

plt.xscale('log', base=2)
plt.xticks(x, x, rotation=45)
plt.yscale('log', base=10)
plt.xlabel('Batch size')
plt.ylabel('TFLOPS')
plt.title('TFLOPS vs Batch size, H100, bf16')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

outfile = 'tflops_vs_batch.png'
plt.savefig(outfile)
print(f"Saved plot to {outfile}")