"""
Direct test to verify if Fused MoE with different BLOCK_SIZE configs produces different results.

This test directly invokes the kernel with different configs to isolate the effect
of BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N on floating-point accumulation.
"""

import torch
from typing import Tuple
from itertools import product

def test_moe_all_block_sizes():
    """Test if same input with different BLOCK_SIZE combinations gives same output."""
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    num_iters = 20  # Run each batch size 20 times with different data
    
    # Different batch sizes (num tokens) to test
    batch_sizes = [
        3072
    ]
    
    # Import MoE components
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
    from sglang.srt.layers.moe.fused_moe_triton import override_config
    
    # MoE dimensions - must be divisible by largest block sizes
    # Test multiple hidden dimensions
    hidden_dims = [4096]
    num_experts = 128
    top_k = 8
    
    # Parameter ranges to test
    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [32, 64, 128, 256]
    block_k_range = [64, 128, 256]
    num_warps_range = [4, 8]
    group_m_range = [1, 16, 32, 64]
    num_stage_range = [2, 3, 4, 5]
    
    param_ranges = {
        "BLOCK_SIZE_M": block_m_range,
        "BLOCK_SIZE_N": block_n_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
    }
    
    total_configs = (len(block_m_range) * len(block_n_range) * len(block_k_range) * 
                     len(group_m_range) * len(num_warps_range) * len(num_stage_range))
    
    print("=" * 80)
    print("Testing MoE BLOCK_SIZE Combinations")
    print("=" * 80)
    print(f"Hidden dims (K): {hidden_dims}")
    print(f"Batch sizes (num tokens): {batch_sizes}")
    print(f"Iterations per batch size: {num_iters}")
    print(f"Parameter ranges: {param_ranges}")
    print(f"Configs per run: {total_configs}")
    print(flush=True)
    
    # Track failures across all runs
    all_failures = []  # List of (hidden_dim, batch_size, iter_idx, m, k, n, max_diff)
    
    for hidden_dim in hidden_dims:
        # Intermediate size is typically hidden_dim // 8 for MoE, must be >= max(BLOCK_SIZE_N)
        intermediate_size = max(hidden_dim // 8, 256)
        
        print(f"\n{'='*80}")
        print(f"Hidden dim: {hidden_dim}, Intermediate size: {intermediate_size}")
        print(f"{'='*80}", flush=True)
        
        for batch_size in batch_sizes:
            print(f"\n--- Batch size: {batch_size} ---", flush=True)
            
            for iter_idx in range(num_iters):
                # Create fresh random data for each iteration (uniform between -1 and 1)
                torch.manual_seed(hidden_dim * 10000 + batch_size * 1000 + iter_idx)
                w1 = (2 * torch.rand(num_experts, intermediate_size * 2, hidden_dim,
                                     dtype=dtype, device=device) - 1).contiguous()
                w2 = (2 * torch.rand(num_experts, hidden_dim, intermediate_size,
                                     dtype=dtype, device=device) - 1).contiguous()
                
                input_token = (2 * torch.rand(batch_size, hidden_dim, dtype=dtype, device=device) - 1).contiguous()
                
                router_logits = 2 * torch.rand(batch_size, num_experts, dtype=torch.float32, device=device) - 1
                routing_weights = torch.softmax(router_logits, dim=-1)
                topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)
                topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)
                topk_ids = topk_ids.to(torch.int32)
                
                # Compute reference output with first config
                reference_config = {
                    "BLOCK_SIZE_M": block_m_range[0],
                    "BLOCK_SIZE_N": block_n_range[0],
                    "BLOCK_SIZE_K": block_k_range[0],
                    "GROUP_SIZE_M": group_m_range[0],
                    "num_warps": num_warps_range[0],
                    "num_stages": num_stage_range[0],
                }
                
                with override_config(reference_config):
                    reference_output = outplace_fused_experts(
                        input_token, w1, w2, topk_weights, topk_ids,
                        activation="silu"
                    )
                    torch.cuda.synchronize()
                
                # Test all combinations for this iteration
                iter_failures = []
            
                for bm, bn, bk, gm, nw, ns in product(block_m_range, block_n_range, block_k_range,
                                                       group_m_range, num_warps_range, num_stage_range):
                    config = {
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_N": bn,
                        "BLOCK_SIZE_K": bk,
                        "GROUP_SIZE_M": gm,
                        "num_warps": nw,
                        "num_stages": ns,
                    }
                    
                    try:
                        with override_config(config):
                            output = outplace_fused_experts(
                                input_token, w1, w2, topk_weights, topk_ids,
                                activation="silu"
                            )
                            torch.cuda.synchronize()
                    except Exception as e:
                        # Skip configs that fail (e.g., out of shared memory)
                        if hidden_dim == hidden_dims[0] and batch_size == batch_sizes[0] and iter_idx == 0:
                            print(f"  Skipping M={bm:3d}, N={bn:3d}, K={bk:3d}, G={gm:2d}, W={nw}, S={ns}: {type(e).__name__}", flush=True)
                        continue
                    
                    if not torch.equal(reference_output, output):
                        diff = (reference_output - output).abs().max().item()
                        iter_failures.append((bm, bn, bk, gm, nw, ns, diff))
                        all_failures.append((hidden_dim, batch_size, iter_idx, bm, bn, bk, gm, nw, ns, diff))
                
                # Print progress
                if iter_failures:
                    print(f"  Iter {iter_idx + 1:3d}/{num_iters}: {len(iter_failures)} configs differ ✗", flush=True)
                    for bm, bn, bk, gm, nw, ns, diff in iter_failures:
                        print(f"    M={bm:3d}, N={bn:3d}, K={bk:3d}, G={gm:2d}, W={nw}, S={ns}: max_diff={diff:.6e}")
                else:
                    print(f"  Iter {iter_idx + 1:3d}/{num_iters}: ALL {total_configs} configs IDENTICAL ✓", flush=True)
    
    # Summary
    print(flush=True)
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_runs = len(hidden_dims) * len(batch_sizes) * num_iters
    print(f"Hidden dims tested: {hidden_dims}")
    print(f"Batch sizes tested: {batch_sizes}")
    print(f"Iterations per batch size: {num_iters}")
    print(f"Total runs: {total_runs}")
    print(f"Configs per run: {total_configs}")
    print(f"Total failures: {len(all_failures)}")
    
    if all_failures:
        # Group failures by config
        from collections import defaultdict
        config_failure_counts = defaultdict(list)
        for hidden_dim, batch_size, iter_idx, bm, bn, bk, gm, nw, ns, diff in all_failures:
            config_failure_counts[(hidden_dim, bm, bn, bk, gm, nw, ns)].append((batch_size, iter_idx, diff))
        
        print(f"\nConfigurations that failed (across any run):")
        for (hidden_dim, bm, bn, bk, gm, nw, ns), fails in sorted(config_failure_counts.items()):
            max_diff = max(d for _, _, d in fails)
            batch_sizes_failed = set(bs for bs, _, _ in fails)
            print(f"  K={hidden_dim:4d}, M={bm:3d}, N={bn:3d}, BK={bk:3d}, G={gm:2d}, W={nw}, S={ns}: "
                  f"failed {len(fails)}/{total_runs} runs, batch_sizes={sorted(batch_sizes_failed)}, max_diff={max_diff:.6e}")
    else:
        print(f"\n✓ ALL configurations produce IDENTICAL results across all {total_runs} runs!")
        print(f"  Fused MoE is fully deterministic and invariant to all config parameters")


if __name__ == "__main__":
    test_moe_all_block_sizes()
