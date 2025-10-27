import torch
from sglang.srt.batch_invariant_ops import matmul_persistent
from sgl_kernel import bf16_batch_invariant_mm, bf16_batch_invariant_fused_mm

print("Setting default device to CUDA...", flush=True)

torch.set_default_device('cuda')

def bi_kernel_wrapper(a, b, out=None):
    return bf16_batch_invariant_mm(a, b, a.dtype, out=out)

def split_kernel_wrapper(a, b, split_frac=0, out=None):
    return bf16_batch_invariant_fused_mm(a, b, a.dtype, split_frac=split_frac, out=out)

def bench_perf(matmul_func, a, b, out=None):
    WARMUP = 0
    ITERS = 1
    # Warm-up
    for _ in range(WARMUP):
        _ = matmul_func(a, b, out=out)
    import time
    #torch.cuda.synchronize()
    start = time.time()
    for _ in range(ITERS):
        _ = matmul_func(a, b, out=out)
    #torch.cuda.synchronize()
    end = time.time()
    print(f"{matmul_func.__name__}: {end - start:.6f} seconds", flush=True)

print("Preparing data...", flush=True)

'''
# Random for graph capture
a = torch.randn(32, 4096 * 4, device='cuda', dtype=torch.float16)
b = torch.randn(4096 * 4, 4096, device='cuda', dtype=torch.float16)
out = torch.empty((32, 4096), device='cuda', dtype=torch.float16)

print("Warming up...", flush=True)

# Warmup
bench_perf(torch.mm, a, b, out)
#bench_perf(matmul_persistent, a, b)
bench_perf(bi_kernel_wrapper, a, b, out)
bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.125, out=out), a, b, out)
bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.25, out=out), a, b, out)
bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.5, out=out), a, b, out)


print("Starting CUDA graph capture...", flush=True)

# Capture into CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    bench_perf(torch.mm, a, b, out)
    #bench_perf(matmul_persistent, a, b)
    bench_perf(bi_kernel_wrapper, a, b, out)
    bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.125, out=out), a, b, out)
    bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.25, out=out), a, b, out)
    bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.5, out=out), a, b, out)

'''

BATCHES = [32, 256, 8192]
# Benchmark performance
for batch_size in BATCHES:
    B = batch_size
    D = 4096

    M = B
    N = D
    K = D * 4

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    out = torch.empty((M, N), device='cuda', dtype=torch.float16)

    bench_perf(torch.mm, a, b)
    #bench_perf(matmul_persistent, a, b)
    bench_perf(bi_kernel_wrapper, a, b)
    bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.125, out=out), a, b, out)
    bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.25, out=out), a, b, out)
    bench_perf(lambda a, b, out: split_kernel_wrapper(a, b, split_frac=0.5, out=out), a, b, out)

#    torch.cuda.synchronize()
