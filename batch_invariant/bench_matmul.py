import torch
from sglang.srt.batch_invariant_ops import matmul_persistent
from sgl_kernel import bf16_batch_invariant_mm

torch.set_default_device('cuda')

def bi_kernel_wrapper(a, b):
    return bf16_batch_invariant_mm(a, b, a.dtype)

def test_equality():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D).to(torch.float16).cuda()
    b = torch.linspace(-100, 100, D*D).reshape(D, D).to(torch.float16).cuda()

    out1 = matmul_persistent(a, b)
    out2 = bi_kernel_wrapper(a, b)
    out3 = torch.mm(a, b)

    diff1 = (out1 - out2).abs().max()
    diff2 = (out1 - out3).abs().max()
    diff3 = (out2 - out3).abs().max()
    print(f"Diff (matmul_persistent vs bi_kernel): {diff1.item()}")
    print(f"Diff (matmul_persistent vs torch.mm): {diff2.item()}")
    print(f"Diff (bi_kernel vs torch.mm): {diff3.item()}")

def test_batch_invariance(matmul_func):
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D).to(torch.float16)
    b = torch.linspace(-100, 100, D*D).reshape(D, D).to(torch.float16)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = matmul_func(a[:1], b)

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = matmul_func(a, b)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

def bench_perf(matmul_func, B=2048, D=4096, iterations=50):
    a = torch.randn(B, D, device='cuda', dtype=torch.float16)
    b = torch.randn(D, D, device='cuda', dtype=torch.float16)

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
    tflops = 2 * B * D * D / (avg_time * 1e12)
    return tflops
    print(f"Avg Time: {avg_time*1000:.2f} ms, TFLOPS: {gflops:.2f}")

test_equality()

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
is_deterministic = test_batch_invariance(torch.mm)
print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
is_deterministic = test_batch_invariance(matmul_persistent)
print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant-Fused-kernel Mode:")
is_deterministic = test_batch_invariance(bi_kernel_wrapper)
print(f"Deterministic: {is_deterministic}")


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384]

torch_tflops = {}
bi_tflops = {}
bi_fuse_tflops = {}

# Benchmark performance
for batch_size in BATCHES:
    torch_tflops[batch_size] = bench_perf(torch.mm, B=batch_size)
    bi_tflops[batch_size] = bench_perf(matmul_persistent, B=batch_size)
    bi_fuse_tflops[batch_size] = 0#bench_perf(bi_kernel_wrapper, B=batch_size)
    print(f"Batch Size: {batch_size} | PyTorch TFLOPS: {torch_tflops[batch_size]:.2f} | BI TFLOPS: {bi_tflops[batch_size]:.2f} | BI-Fused TFLOPS: {bi_fuse_tflops[batch_size]:.2f}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Prepare data (preserve the BATCHES order)
x = BATCHES
torch_vals = [torch_tflops.get(b, 0.0) for b in x]
bi_vals = [bi_tflops.get(b, 0.0) for b in x]
bi_fused_vals = [bi_fuse_tflops.get(b, 0.0) for b in x]

plt.figure(figsize=(10, 6))
plt.plot(x, torch_vals, marker='o', label='PyTorch (torch.mm)')
plt.plot(x, bi_vals, marker='s', label='Batch-Invariant (matmul_persistent)')
plt.plot(x, bi_fused_vals, marker='^', label='BI-Fused (bi_kernel_wrapper)')

plt.xscale('log', base=2)
plt.xticks(x, x, rotation=45)
plt.yscale('log', base=10)
plt.xlabel('Batch size')
plt.ylabel('TFLOPS')
plt.title('TFLOPS vs Batch size')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

outfile = 'tflops_vs_batch.png'
plt.savefig(outfile)
print(f"Saved plot to {outfile}")
