import torch
from sglang.srt.batch_invariant_ops import matmul_persistent
from sgl_kernel import bf16_batch_invariant_mm, bf16_batch_invariant_fused_mm

torch.set_default_device('cuda')

def bi_kernel_wrapper(a, b, bias):
    return bf16_batch_invariant_mm(a, b, a.dtype, bias=bias)

def test_equality():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D).to(torch.bfloat16).cuda()
    b = torch.linspace(-100, 100, D*D).reshape(D, D).to(torch.bfloat16).cuda()
    bias = torch.linspace(-10, 10, D).reshape(1, D).to(torch.bfloat16).cuda()

    out1 = torch.addmm(bias, a, b)
    out2 = matmul_persistent(a, b, bias=bias)
    out3 = bi_kernel_wrapper(a, b, bias=bias)

    diff2 = (out1 - out2).abs().max()
    diff3 = (out1 - out3).abs().max()
    print(f"Diff (matmul_persistent vs torch.addmm): {diff2.item()}")
    print(f"Diff (bi_kernel vs torch.addmm): {diff3.item()}")

def test_batch_invariance(matmul_func):
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D).to(torch.bfloat16)
    b = torch.linspace(-100, 100, D*D).reshape(D, D).to(torch.bfloat16)
    bias = torch.linspace(-10, 10, D).reshape(1, D).to(torch.bfloat16)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = matmul_func(a[:1], b, bias=bias)

    # Batch size 10
    out2 = matmul_func(a[:10], b, bias=bias)[:1]

    # Batch size 128
    out3 = matmul_func(a[:128], b, bias=bias)[:1]

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out_full = matmul_func(a, b, bias=bias)[:1]

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

def bench_perf(matmul_func, a, b, bias, warmup=5, iterations=10):
    # Warm-up
    for _ in range(warmup):
        _ = matmul_func(a, b, bias=bias)

    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        _ = matmul_func(a, b, bias=bias)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    tflops = 2 * M * K * N / (avg_time * 1e12)
    return tflops

'''
test_equality()

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
run_iters(torch.addmm)

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
run_iters(matmul_persistent)

# Test with batch-invariant operations
print("\nBatch-Invariant-Fused-kernel Mode:")
run_iters(bi_kernel_wrapper)
print()
'''

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
CONFIGS = [
    "torch",
    "bi",
    "bi_fused",
    #"split_12.5",
    #"split_25",
    #"split_50",
    #"split_75",
    #"split_87.5"
]

tflops_results = {}
for config in CONFIGS:
    tflops_results[config] = {}

# Benchmark performance
for batch_size in BATCHES:
    D = 4096
    M = batch_size
    N = D
    K = D * 4
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(1, N, device='cuda', dtype=torch.bfloat16)

    tflops_results["torch"][batch_size] = bench_perf(lambda a, b, bias: torch.addmm(bias, a, b), a, b, bias)
    #tflops_results["bi"][batch_size] = bench_perf(matmul_persistent, a, b)
    tflops_results["bi_fused"][batch_size] = bench_perf(bi_kernel_wrapper, a, b, bias)
    #tflops_results["split_12.5"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.125), a, b)
    #tflops_results["split_25"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.25), a, b)
    #tflops_results["split_50"][batch_size] = bench_perf(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.5), a, b)

    print(f"Batch Size: {batch_size} | PyTorch TFLOPS: {tflops_results["torch"][batch_size]:.2f} | "
          f" BI-Fused TFLOPS: {tflops_results["bi_fused"][batch_size]:.2f} | "
          #f"Split TFLOPS: {', '.join([f'{k}: {tflops_results[k][batch_size]:.2f}' for k in CONFIGS if k.startswith('split_')])}"
          )

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Prepare data (preserve the BATCHES order)
x = BATCHES
torch_vals = [tflops_results['torch'].get(b, 0.0) for b in x]
#bi_vals = [tflops_results['bi'].get(b, 0.0) for b in x]
bi_fused_vals = [tflops_results['bi_fused'].get(b, 0.0) for b in x]

plt.figure(figsize=(10, 6))
plt.plot(x, torch_vals, marker='o', label='PyTorch (torch.addmm)')
#plt.plot(x, bi_vals, marker='s', label='Batch-Invariant (matmul_persistent)')
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
plt.title('TFLOPS vs Batch size')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

outfile = 'tflops_vs_batch_bias.png'
plt.savefig(outfile)
print(f"Saved plot to {outfile}")