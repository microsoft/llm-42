"""
Microbenchmark to test determinism of flash_attn_with_kvcache for overlapping prefill chunks.

This benchmark:
1. Creates a prior KV cache (prefix tokens)
2. Computes attention for prefill chunks of the same size but with sliding boundaries
3. Checks if a token's attention output is consistent across all chunks it appears in

The goal is to identify if non-determinism in attention is causing output variance.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import argparse

try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
except ImportError:
    # Fallback to flash_attn package
    from flash_attn import flash_attn_with_kvcache

def create_kv_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create KV cache tensors."""
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, dtype=dtype, device=device)
    v_cache = torch.randn(shape, dtype=dtype, device=device)
    return k_cache, v_cache

def RMSNORM(
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor = None,
        variance_epsilon: float = 1e-6,
    ):
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)
        x_var = x
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = (x * weight).to(orig_dtype)
        return x, residual 

def run_chunk_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    residual: torch.Tensor,
    weight: torch.Tensor,
    num_splits: int = 0,
) -> torch.Tensor:
    """Run flash attention with KV cache for a chunk."""
    # Apply fused_add_rmsnorm to q tensor before attention
    # Note: fused_add_rmsnorm modifies input and residual in-place
    # fused_add_rmsnorm expects 2D tensor: (num_tokens, hidden_dim)
    # q shape is (chunk_size, num_q_heads, head_dim), reshape to (chunk_size, num_q_heads * head_dim)
    eps = 1e-6
    original_shape = q.shape  # (chunk_size, num_q_heads, head_dim)
    hidden_dim = original_shape[1] * original_shape[2]  # num_q_heads * head_dim
    q_2d = q.clone().view(original_shape[0], hidden_dim)  # (chunk_size, hidden_dim)
    residual_2d = residual.clone().view(original_shape[0], hidden_dim)
    q_normed_2d, _ = RMSNORM(
        x=q_2d,
        weight=weight,
        residual=residual_2d,
        variance_epsilon=eps,
    )
    # Reshape back to original q shape
    q = q_normed_2d.view(original_shape)
    
    result = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        num_splits=num_splits,
    )
    return result


def benchmark_chunk_determinism(
    prefix_len: int = 1024,
    chunk_size: int = 128,
    num_chunks: int = 10,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 1,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    num_splits: int = 0,
    num_iterations: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Benchmark to test determinism of flash attention with overlapping chunks.
    
    Args:
        prefix_len: Number of prefix tokens already in KV cache
        chunk_size: Size of each prefill chunk
        num_chunks: Number of sliding chunks to test (each shifted by 1 token)
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        page_size: Page size for KV cache
        dtype: Data type
        device: Device to run on
        num_splits: Number of splits for flash attention (0 = auto, 1 = deterministic)
        num_iterations: Number of times to repeat each chunk for variance measurement
        seed: Random seed
    
    Returns:
        Dictionary with statistics about determinism
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Calculate total sequence length needed
    # We need prefix + enough tokens for all chunks
    # If chunks slide by 1 token each time, we need prefix_len + chunk_size + (num_chunks - 1) tokens
    total_seq_len = prefix_len + chunk_size + (num_chunks - 1)
    num_pages = (total_seq_len + page_size - 1) // page_size
    
    print(f"Configuration:")
    print(f"  Prefix length: {prefix_len}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Num chunks: {num_chunks}")
    print(f"  Total sequence length: {total_seq_len}")
    print(f"  Num Q heads: {num_q_heads}, Num KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Page size: {page_size}")
    print(f"  Num splits: {num_splits} ({'auto' if num_splits == 0 else 'deterministic' if num_splits == 1 else 'custom'})")
    print(f"  Num iterations per chunk: {num_iterations}")
    print()
    
    # Create KV cache with all tokens pre-filled
    k_cache, v_cache = create_kv_cache(
        num_pages, page_size, num_kv_heads, head_dim, dtype, device
    )
    
    # Create all Q vectors for the entire sequence (we'll slice into chunks)
    all_q = torch.randn(
        total_seq_len, num_q_heads, head_dim, dtype=dtype, device=device
    )
    
    # Page table: simple sequential mapping for single-batch
    # Shape: (1, num_pages) - single sequence
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Create residual tensor for the entire sequence (per-global-token)
    # This ensures the same global token gets the same residual regardless of which chunk it's in
    # Shape: (total_seq_len, num_q_heads, head_dim)
    all_residual = torch.randn(
        total_seq_len, num_q_heads, head_dim, dtype=dtype, device=device
    )
    # Create weight tensor for RMSNorm (size = hidden_dim = num_q_heads * head_dim)
    rmsnorm_weight = torch.ones(num_q_heads * head_dim, dtype=dtype, device=device)
    
    # Dictionary to store outputs for each token position across chunks
    # Key: global token position, Value: list of (chunk_id, local_position, outputs_across_iterations)
    token_outputs: Dict[int, List[Tuple[int, int, List[torch.Tensor]]]] = {}
    
    print("Running attention for overlapping chunks...")
    
    for chunk_id in range(num_chunks):
        # Chunk starts at position (prefix_len + chunk_id) and has chunk_size tokens
        chunk_start = prefix_len + chunk_id
        chunk_end = chunk_start + chunk_size
        
        # The KV cache length for this chunk's attention
        # Each token in the chunk attends to all tokens up to and including itself
        cache_seqlen = chunk_end  # Total KV length up to end of chunk
        
        # Extract Q for this chunk
        q_chunk = all_q[chunk_start:chunk_end].unsqueeze(0).contiguous()  # (1, chunk_size, num_q_heads, head_dim)
        q_chunk = q_chunk.view(chunk_size, num_q_heads, head_dim)  # (chunk_size, num_q_heads, head_dim)
        
        # Prepare metadata for varlen attention
        cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, cache_seqlen], dtype=torch.int32, device=device)
        
        # Run multiple iterations to check for non-determinism
        chunk_outputs = []
        # Get the residual slice for this chunk's global positions
        chunk_residual = all_residual[chunk_start:chunk_end]
        for iter_id in range(num_iterations):
            output = run_chunk_attention(
                q=q_chunk.clone(),  # Clone since fused_add_rmsnorm modifies in-place
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=chunk_size,
                softmax_scale=softmax_scale,
                residual=chunk_residual,
                weight=rmsnorm_weight,
                num_splits=num_splits,
            )
            chunk_outputs.append(output.clone())
        
        # Store outputs for each token position
        for local_pos in range(chunk_size):
            global_pos = chunk_start + local_pos
            if global_pos not in token_outputs:
                token_outputs[global_pos] = []
            # Extract output for this token across all iterations
            token_iter_outputs = [out[local_pos].clone() for out in chunk_outputs]
            token_outputs[global_pos].append((chunk_id, local_pos, token_iter_outputs))
    
    # Analyze determinism
    print("\n" + "="*80)
    print("Determinism Analysis")
    print("="*80)
    
    # 1. Intra-chunk determinism: Same chunk, multiple iterations
    intra_chunk_max_diffs = []
    intra_chunk_mean_diffs = []
    
    for global_pos, appearances in token_outputs.items():
        for chunk_id, local_pos, iter_outputs in appearances:
            if len(iter_outputs) > 1:
                # Compare all pairs of iterations
                for i in range(len(iter_outputs)):
                    for j in range(i + 1, len(iter_outputs)):
                        diff = (iter_outputs[i] - iter_outputs[j]).abs()
                        intra_chunk_max_diffs.append(diff.max().item())
                        intra_chunk_mean_diffs.append(diff.mean().item())
    
    print("\n1. INTRA-CHUNK DETERMINISM (same chunk, multiple iterations)")
    print("-" * 60)
    if intra_chunk_max_diffs:
        print(f"   Max absolute difference: {max(intra_chunk_max_diffs):.2e}")
        print(f"   Mean absolute difference: {np.mean(intra_chunk_mean_diffs):.2e}")
        print(f"   Std of max differences: {np.std(intra_chunk_max_diffs):.2e}")
        if max(intra_chunk_max_diffs) > 0:
            print("   ⚠️  NON-DETERMINISTIC: Same input produces different outputs!")
        else:
            print("   ✓  DETERMINISTIC: Same input produces identical outputs")
    else:
        print("   No data (need num_iterations > 1)")
    
    # 2. Inter-chunk determinism: Same token appearing in different chunks
    inter_chunk_max_diffs = []
    inter_chunk_mean_diffs = []
    inter_chunk_details = []
    
    for global_pos, appearances in token_outputs.items():
        if len(appearances) > 1:
            # This token appears in multiple chunks
            # Compare outputs across chunks (using first iteration of each)
            for i in range(len(appearances)):
                for j in range(i + 1, len(appearances)):
                    chunk_i, local_i, outputs_i = appearances[i]
                    chunk_j, local_j, outputs_j = appearances[j]
                    
                    # Compare first iteration of each chunk
                    diff = (outputs_i[0] - outputs_j[0]).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    
                    inter_chunk_max_diffs.append(max_diff)
                    inter_chunk_mean_diffs.append(mean_diff)
                    
                    if max_diff > 1e-5:  # Significant difference
                        inter_chunk_details.append({
                            'global_pos': global_pos,
                            'chunk_i': chunk_i,
                            'chunk_j': chunk_j,
                            'local_i': local_i,
                            'local_j': local_j,
                            'max_diff': max_diff,
                            'mean_diff': mean_diff,
                        })
    
    print("\n2. INTER-CHUNK DETERMINISM (same token in different chunks)")
    print("-" * 60)
    if inter_chunk_max_diffs:
        print(f"   Max absolute difference: {max(inter_chunk_max_diffs):.2e}")
        print(f"   Mean absolute difference: {np.mean(inter_chunk_mean_diffs):.2e}")
        print(f"   Std of max differences: {np.std(inter_chunk_max_diffs):.2e}")
        
        if max(inter_chunk_max_diffs) > 1e-5:
            print("   ⚠️  NON-DETERMINISTIC: Same token has different outputs in different chunks!")
            print("\n   Top 10 largest differences:")
            sorted_details = sorted(inter_chunk_details, key=lambda x: x['max_diff'], reverse=True)[:10]
            for detail in sorted_details:
                print(f"      Token {detail['global_pos']}: "
                      f"chunk {detail['chunk_i']} (local={detail['local_i']}) vs "
                      f"chunk {detail['chunk_j']} (local={detail['local_j']}) -> "
                      f"max_diff={detail['max_diff']:.2e}")
        else:
            print("   ✓  DETERMINISTIC: Same token produces consistent outputs across chunks")
    else:
        print("   No overlapping tokens found (increase num_chunks or decrease chunk_size)")
    
    # Summary statistics
    results = {
        'intra_chunk': {
            'max_diff': max(intra_chunk_max_diffs) if intra_chunk_max_diffs else 0,
            'mean_diff': np.mean(intra_chunk_mean_diffs) if intra_chunk_mean_diffs else 0,
            'is_deterministic': max(intra_chunk_max_diffs) == 0 if intra_chunk_max_diffs else True,
        },
        'inter_chunk': {
            'max_diff': max(inter_chunk_max_diffs) if inter_chunk_max_diffs else 0,
            'mean_diff': np.mean(inter_chunk_mean_diffs) if inter_chunk_mean_diffs else 0,
            'is_deterministic': max(inter_chunk_max_diffs) < 1e-5 if inter_chunk_max_diffs else True,
            'num_overlapping_tokens': len([p for p, a in token_outputs.items() if len(a) > 1]),
        },
        'details': inter_chunk_details,
    }
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Overlapping token positions analyzed: {results['inter_chunk']['num_overlapping_tokens']}")
    print(f"Overall intra-chunk determinism: {'✓ PASS' if results['intra_chunk']['is_deterministic'] else '✗ FAIL'}")
    print(f"Overall inter-chunk determinism: {'✓ PASS' if results['inter_chunk']['is_deterministic'] else '✗ FAIL'}")
    
    return results


def run_comparison_splits(
    prefix_len: int = 1024,
    chunk_size: int = 128,
    num_chunks: int = 10,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_iterations: int = 5,
):
    """Compare determinism with different num_splits settings."""
    print("\n" + "#"*80)
    print("# Comparing num_splits=0 (auto) vs num_splits=1 (deterministic)")
    print("#"*80)
    
    print("\n>>> Testing with num_splits=0 (auto heuristic) <<<")
    results_auto = benchmark_chunk_determinism(
        prefix_len=prefix_len,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_splits=0,
        num_iterations=num_iterations,
    )
    
    print("\n\n>>> Testing with num_splits=1 (deterministic) <<<")
    results_det = benchmark_chunk_determinism(
        prefix_len=prefix_len,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_splits=1,
        num_iterations=num_iterations,
    )
    
    print("\n" + "#"*80)
    print("# COMPARISON SUMMARY")
    print("#"*80)
    print(f"\nnum_splits=0 (auto):")
    print(f"  Intra-chunk max diff: {results_auto['intra_chunk']['max_diff']:.2e}")
    print(f"  Inter-chunk max diff: {results_auto['inter_chunk']['max_diff']:.2e}")
    
    print(f"\nnum_splits=1 (deterministic):")
    print(f"  Intra-chunk max diff: {results_det['intra_chunk']['max_diff']:.2e}")
    print(f"  Inter-chunk max diff: {results_det['inter_chunk']['max_diff']:.2e}")
    
    if results_auto['intra_chunk']['max_diff'] > results_det['intra_chunk']['max_diff']:
        print("\n✓ num_splits=1 improves intra-chunk determinism")
    if results_auto['inter_chunk']['max_diff'] > results_det['inter_chunk']['max_diff']:
        print("✓ num_splits=1 improves inter-chunk determinism")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark flash attention determinism with overlapping chunks")
    parser.add_argument("--prefix-len", type=int, default=1024, help="Length of prefix in KV cache")
    parser.add_argument("--chunk-size", type=int, default=128, help="Size of each prefill chunk")
    parser.add_argument("--num-chunks", type=int, default=100, help="Number of overlapping chunks to test")
    parser.add_argument("--num-q-heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--page-size", type=int, default=16, help="Page size for KV cache")
    parser.add_argument("--num-splits", type=int, default=1, help="Number of splits (0=auto, 1=deterministic)")
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of iterations per chunk")
    parser.add_argument("--compare", action="store_true", help="Compare num_splits=0 vs num_splits=1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison_splits(
            prefix_len=args.prefix_len,
            chunk_size=args.chunk_size,
            num_chunks=args.num_chunks,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            num_iterations=args.num_iterations,
        )
    else:
        benchmark_chunk_determinism(
            prefix_len=args.prefix_len,
            chunk_size=args.chunk_size,
            num_chunks=args.num_chunks,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            num_splits=args.num_splits,
            num_iterations=args.num_iterations,
            seed=args.seed,
        )
