import torch
from sglang.srt.batch_invariant_ops import matmul_persistent
from sgl_kernel import bf16_batch_invariant_mm, bf16_batch_invariant_fused_mm

torch.set_default_device('cuda')

def bi_kernel_wrapper(a, b):
    return bf16_batch_invariant_mm(a, b, a.dtype)

def split_kernel_wrapper(a, b, split_frac=0.0):
    return bf16_batch_invariant_fused_mm(a, b, a.dtype, split_frac=split_frac)

# Llama3-8B dimensions
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Operation dimensions:
# preproj (q_proj, k_proj, v_proj): [B, HIDDEN_SIZE] @ [HIDDEN_SIZE, HIDDEN_SIZE]
# postproj (o_proj): [B, HIDDEN_SIZE] @ [HIDDEN_SIZE, HIDDEN_SIZE]
# mlp_up (gate_up_proj): [B, HIDDEN_SIZE] @ [HIDDEN_SIZE, INTERMEDIATE_SIZE*2]
# mlp_down (down_proj): [B, INTERMEDIATE_SIZE] @ [INTERMEDIATE_SIZE, HIDDEN_SIZE]

OPS_CONFIG = {
    "preproj": {
        "M": lambda B: B,
        "K": HIDDEN_SIZE,
        "N": HIDDEN_SIZE,
        "desc": "Attention Q/K/V Projection"
    },
    "postproj": {
        "M": lambda B: B,
        "K": HIDDEN_SIZE,
        "N": HIDDEN_SIZE,
        "desc": "Attention O Projection"
    },
    "mlp_up": {
        "M": lambda B: B,
        "K": HIDDEN_SIZE,
        "N": INTERMEDIATE_SIZE * 2,  # gate_up_proj is fused
        "desc": "MLP Gate+Up Projection"
    },
    "mlp_down": {
        "M": lambda B: B,
        "K": INTERMEDIATE_SIZE,
        "N": HIDDEN_SIZE,
        "desc": "MLP Down Projection"
    }
}

def test_equality(op_name):
    config = OPS_CONFIG[op_name]
    B = 2048
    M = config["M"](B)
    K = config["K"]
    N = config["N"]
    
    a = torch.linspace(-100, 100, M*K).reshape(M, K).to(torch.float16).cuda()
    b = torch.linspace(-100, 100, K*N).reshape(K, N).to(torch.float16).cuda()

    out1 = torch.mm(a, b)
    out2 = matmul_persistent(a, b)
    out3 = bi_kernel_wrapper(a, b)
    out4 = split_kernel_wrapper(a, b, split_frac=0.25)
    out5 = split_kernel_wrapper(a, b, split_frac=0.5)

    diff2 = (out1 - out2).abs().max()
    diff3 = (out1 - out3).abs().max()
    diff4 = (out1 - out4).abs().max()
    diff5 = (out1 - out5).abs().max()
    print(f"\n{op_name} ({config['desc']}):")
    print(f"  Shape: [{M}, {K}] @ [{K}, {N}]")
    print(f"  Diff (matmul_persistent vs torch.mm): {diff2.item()}")
    print(f"  Diff (bi_kernel vs torch.mm): {diff3.item()}")
    print(f"  Diff (split 25% vs torch.mm): {diff4.item()}")
    print(f"  Diff (split 50% vs torch.mm): {diff5.item()}")

def test_batch_invariance(matmul_func, op_name):
    config = OPS_CONFIG[op_name]
    B = 2048
    M = config["M"](B)
    K = config["K"]
    N = config["N"]
    
    a = torch.linspace(-100, 100, M*K).reshape(M, K).to(torch.float16)
    b = torch.linspace(-100, 100, K*N).reshape(K, N).to(torch.float16)

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

def run_iters(func, op_name, iters=10):
    difflist = []
    is_deterministic = True
    for i in range(iters):
        isd, df = test_batch_invariance(func, op_name)
        is_deterministic = is_deterministic and isd
        difflist.append(df)
    print(f"  Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {iters} iterations")

def bench_perf(matmul_func, op_name, B, iterations=10):
    config = OPS_CONFIG[op_name]
    M = config["M"](B)
    K = config["K"]
    N = config["N"]

    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

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

print("="*80)
print("Llama3-8B Operation Benchmarks")
print("="*80)

# Test equality for all operations
print("\n--- Testing Correctness ---")
for op_name in OPS_CONFIG.keys():
    test_equality(op_name)

# Test batch invariance
print("\n\n--- Testing Batch Invariance ---")
for op_name in OPS_CONFIG.keys():
    config = OPS_CONFIG[op_name]
    print(f"\n{op_name} ({config['desc']}):")
    
    print("  Standard PyTorch:")
    run_iters(torch.mm, op_name)
    
    print("  Batch-Invariant Mode:")
    run_iters(matmul_persistent, op_name)
    
    print("  Batch-Invariant-Fused-kernel Mode:")
    run_iters(bi_kernel_wrapper, op_name)
    
    print("  Batch-Invariant-Fused-kernel 25% Mode:")
    run_iters(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.25), op_name)
    
    print("  Batch-Invariant-Fused-kernel 50% Mode:")
    run_iters(lambda a, b: split_kernel_wrapper(a, b, split_frac=0.5), op_name)

# Performance benchmarks
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
CONFIGS = [
    "torch",
    "bi",
    "bi_fused",
]

print("\n\n--- Performance Benchmarks ---")
for op_name in OPS_CONFIG.keys():
    config = OPS_CONFIG[op_name]
    print(f"\n{op_name} ({config['desc']}):")
    
    tflops_results = {}
    for impl in CONFIGS:
        tflops_results[impl] = {}

    # Benchmark performance
    for batch_size in BATCHES:
        tflops_results["torch"][batch_size] = bench_perf(torch.mm, op_name, B=batch_size)
        tflops_results["bi"][batch_size] = bench_perf(matmul_persistent, op_name, B=batch_size)
        tflops_results["bi_fused"][batch_size] = bench_perf(bi_kernel_wrapper, op_name, B=batch_size)

        print(f"  Batch Size: {batch_size:5d} | PyTorch: {tflops_results['torch'][batch_size]:6.2f} TFLOPS | "
              f"BI: {tflops_results['bi'][batch_size]:6.2f} TFLOPS | "
              f"BI-Fused: {tflops_results['bi_fused'][batch_size]:6.2f} TFLOPS")

    # Plot results
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

    plt.xscale('log', base=2)
    plt.xticks(x, [str(b) for b in x], rotation=45)
    plt.yscale('log', base=10)
    plt.xlabel('Batch size')
    plt.ylabel('TFLOPS')
    plt.title(f'TFLOPS vs Batch size - {op_name} ({config["desc"]}), H100-PCIe, bf16')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    outfile = f'tflops_vs_batch_{op_name}.pdf'
    plt.savefig(outfile, dpi=1200)
    print(f"  Saved plot to {outfile}")

print("\n" + "="*80)
print("Benchmarking Complete!")
print("="*80)
