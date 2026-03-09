"""
Comprehensive test suite for batch-invariance and position-invariance of LLM operators.

Tests the following operator categories:
1. Matmul: CuBLAS GEMM, Fused MoE (Triton)
2. Attention: FA3, FlashInfer (FA3 backend), FlashInfer (FA2 backend)
3. Communication: Ring AllReduce, Tree AllReduce
4. Normalization: RMSNorm, Fused RMSNorm+Residual
5. Embeddings: Rotary Embedding (RoPE)
6. Activations: SiLU, GELU, ReLU

Invariance Definitions:
- Batch-Invariant: output[i] is the same regardless of batch size (BS=1 vs BS=N)
- Position-Invariant: output[i] is the same regardless of position in batch

Usage:
    python test_all_operators.py                    # Run all tests
    python test_all_operators.py --category matmul  # Run only matmul tests
    python test_all_operators.py --category attention normalization  # Multiple categories
    python test_all_operators.py --list             # List all available tests
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

@dataclass
class TestResult:
    """Result of an invariance test."""
    operator_name: str
    category: str
    batch_invariant: Optional[bool]
    position_invariant: Optional[bool]
    batch_inv_details: str = ""
    pos_inv_details: str = ""
    error: Optional[str] = None


class OperatorTester:
    """Base class for testing operator invariance properties."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.num_trials = 10
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.hidden_dim = 4096
        
    def check_equal(self, a: torch.Tensor, b: torch.Tensor, rtol: float = 0, atol: float = 0) -> bool:
        """Check if two tensors are exactly equal (bitwise)."""
        return torch.equal(a, b)
    
    def check_close(self, a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
        """Check if two tensors are approximately equal."""
        return torch.allclose(a, b, rtol=rtol, atol=atol)


# =============================================================================
# MATMUL OPERATORS
# =============================================================================

class GEMMTester(OperatorTester):
    """Test CuBLAS GEMM (torch.matmul) for invariance.
    
    NOTE: CuBLAS can be non-deterministic because it selects different
    algorithms/tile sizes based on problem dimensions (M, N, K). Different batch
    sizes change M, leading to different accumulation orders within each dot product.
    """
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Matmul"
        self.name = "CuBLAS GEMM"
        # GEMM dimensions: [M, K] x [K, N] -> [M, N]
        self.M = 2048  # Full batch size
        self.K = 8192
        self.N = 7178
        self.num_iters = 10
        
    def test_batch_invariance(self) -> Tuple[bool, str]:
        """
        Test if GEMM output for first row is the same when computed as:
        - torch.mm(a[:1], b)  # batch size 1
        - torch.mm(a, b)[:1]  # full batch, then slice
        
        cuBLAS may use different algorithms for different M dimensions.
        """
        is_deterministic = True
        diffs = []
        
        for _ in range(self.num_iters):
            # Use linspace for reproducible, non-random data
            a = torch.linspace(-100, 100, self.M * self.K, dtype=self.dtype, device=self.device).reshape(self.M, self.K)
            b = torch.linspace(-100, 100, self.K * self.N, dtype=self.dtype, device=self.device).reshape(self.K, self.N)
            
            # Method 1: Matrix-vector multiplication (batch size 1)
            out1 = torch.mm(a[:1], b)
            torch.cuda.synchronize()
            
            # Method 2: Matrix-matrix multiplication, then slice (full batch)
            out2 = torch.mm(a, b)[:1]
            torch.cuda.synchronize()
            
            # Check if results are identical (bitwise)
            diff = (out1 - out2).abs().max().item()
            diffs.append(diff)
            
            if diff != 0:
                is_deterministic = False
        
        details = f"M={self.M}, K={self.K}, N={self.N}, iters={self.num_iters}"
        if not is_deterministic:
            details += f", max_diff={max(diffs):.6e}, min_diff={min(diffs):.6e}"
        return is_deterministic, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        """
        Test if GEMM output for a row is the same regardless of its position.
        """
        is_deterministic = True
        diffs = []
        fixed_bs = 512
        
        for _ in range(self.num_iters):
            # Fixed weight matrix
            b = torch.linspace(-100, 100, self.K * self.N, dtype=self.dtype, device=self.device).reshape(self.K, self.N)
            
            # Target row
            target_row = torch.linspace(-50, 50, self.K, dtype=self.dtype, device=self.device).reshape(1, self.K)
            
            # Compute reference with target at position 0
            a_ref = torch.linspace(-100, 100, fixed_bs * self.K, dtype=self.dtype, device=self.device).reshape(fixed_bs, self.K)
            a_ref[0] = target_row[0]
            out_ref = torch.mm(a_ref, b)[0]
            torch.cuda.synchronize()
            
            # Test at different positions
            for pos in range(0, fixed_bs, 1):  # Test every 16th position
                a_test = torch.linspace(-100, 100, fixed_bs * self.K, dtype=self.dtype, device=self.device).reshape(fixed_bs, self.K)
                a_test[pos] = target_row[0]
                out_test = torch.mm(a_test, b)[pos]
                torch.cuda.synchronize()
                
                diff = (out_ref - out_test).abs().max().item()
                diffs.append(diff)
                
                if diff != 0:
                    is_deterministic = False
        
        details = f"Tested positions in BS={fixed_bs}, iters={self.num_iters}"
        if not is_deterministic:
            details += f", max_diff={max(diffs):.6e}"
        return is_deterministic, details


class FusedMoETester(OperatorTester):
    """Test Fused MoE (Triton) for invariance using SGLang's implementation.
    
    NOTE: Similar to CuBLAS, Fused MoE kernels can be non-deterministic because
    they may select different tile sizes or accumulation orders based on the
    number of tokens routed to each expert, which changes with batch size.
    
    The config (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) is selected based on M.
    Different BLOCK_SIZE_K can lead to different floating-point accumulation orders.
    """
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Matmul"
        self.name = "Fused MoE (Triton)"
        # Use E=128, N=384 which has tuned config for H100 PCIe with varying BLOCK_SIZE_K
        # Config file: E=128,N=384,device_name=NVIDIA_H100_PCIe.json
        # BS=32 and BS=96 use BLOCK_SIZE_K=128, others use BLOCK_SIZE_K=64
        self.num_experts = 128
        self.intermediate_size = 384  # N value
        self.hidden_dim = 4096  # K dimension - must be divisible by BLOCK_SIZE_K
        self.top_k = 2
        self.num_iters = 10
        self.available = False
        self._check_availability()
        
    def _check_availability(self):
        """Check if fused MoE kernel is available."""
        try:
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
                try_get_optimal_moe_config, get_moe_configs
            )
            self.outplace_fused_experts = outplace_fused_experts
            self.try_get_optimal_moe_config = try_get_optimal_moe_config
            self.get_moe_configs = get_moe_configs
            self.available = True
        except ImportError:
            self.available = False
    
    def _get_config_for_m(self, M: int) -> dict:
        """Get the MoE config for a given M value."""
        # w1 shape: (E, 2*N, K) -> w1_shape = (E, 2*N, hidden_dim)
        # w2 shape: (E, K, N) -> w2_shape = (E, hidden_dim, N)
        w1_shape = (self.num_experts, self.intermediate_size * 2, self.hidden_dim)
        w2_shape = (self.num_experts, self.hidden_dim, self.intermediate_size)
        config = self.try_get_optimal_moe_config(
            w1_shape, w2_shape, self.top_k, dtype=None, M=M
        )
        return config
    
    def _compute_topk(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k routing weights and indices."""
        # Softmax over experts
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        # Select top-k
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        # Renormalize
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(self.dtype), topk_ids.to(torch.int32)
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        """
        Test if MoE output for first token is the same when computed as:
        - MoE(input[:1])  # batch size 1
        - MoE(input)[:1]  # full batch, then slice
        
        Similar to CuBLAS, MoE kernels may use different algorithms for different
        batch sizes due to varying expert loads.
        
        Key source of non-determinism: Different BLOCK_SIZE_K leads to different
        floating-point accumulation orders in the K-dimension reduction.
        """
        if not self.available:
            return None, "Fused MoE not available (sglang not installed)"
        
        is_deterministic = True
        diffs = []
        config_info = {}
        
        # Log configs for different batch sizes
        # Focus on batch sizes where BLOCK_SIZE_K differs:
        # BS=32 and BS=96 use BLOCK_SIZE_K=128, others use BLOCK_SIZE_K=64
        print(f"\n    MoE Config check (E={self.num_experts}, N={self.intermediate_size}, K={self.hidden_dim}):")
        test_batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1534, 2048, 4097]
        for bs in test_batch_sizes:
            config = self._get_config_for_m(bs)
            config_info[bs] = config
            print(f"      BS={bs:4d}: BLOCK_SIZE_M={config['BLOCK_SIZE_M']:3d}, "
                  f"BLOCK_SIZE_N={config['BLOCK_SIZE_N']:3d}, "
                  f"BLOCK_SIZE_K={config['BLOCK_SIZE_K']:3d}")
        
        # Check if configs differ - this is the source of non-determinism
        unique_k_sizes = set(c['BLOCK_SIZE_K'] for c in config_info.values())
        if len(unique_k_sizes) > 1:
            print(f"    WARNING: Different BLOCK_SIZE_K values detected: {unique_k_sizes}")
            print(f"             This WILL cause non-batch-invariance due to different FP accumulation orders!")
        else:
            print(f"    INFO: Same BLOCK_SIZE_K={list(unique_k_sizes)[0]} for all batch sizes (batch-invariant)")
        
        # Use RANDOM data to better expose floating-point non-determinism
        # (linspace creates very regular patterns that might mask differences)
        torch.manual_seed(42)
        w1 = torch.randn(self.num_experts, self.intermediate_size * 2, self.hidden_dim,
                         dtype=self.dtype, device=self.device).contiguous()
        w2 = torch.randn(self.num_experts, self.hidden_dim, self.intermediate_size,
                         dtype=self.dtype, device=self.device).contiguous()
        
        # Target input (single token) - also random
        target_input = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device).contiguous()
        
        # Fixed router logits for target (deterministic routing)
        target_router = torch.randn(1, self.num_experts, dtype=torch.float32, device=self.device)
        target_topk_weights, target_topk_ids = self._compute_topk(target_router)
        
        for _ in range(self.num_iters):
            # Method 1: Compute with BS=1
            out_bs1 = self.outplace_fused_experts(
                target_input, w1, w2, target_topk_weights, target_topk_ids,
                activation="silu"
            )
            torch.cuda.synchronize()
            
            # Method 2: Compute with full batch, then slice
            for bs in [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048]:
                # Create filler with random data
                torch.manual_seed(42 + bs)  # Different seed per batch size for variety
                filler_input = torch.randn(bs - 1, self.hidden_dim,
                                            dtype=self.dtype, device=self.device)
                filler_router = torch.randn(bs - 1, self.num_experts,
                                             dtype=torch.float32, device=self.device)
                filler_topk_weights, filler_topk_ids = self._compute_topk(filler_router)
                
                batch_input = torch.cat([target_input, filler_input], dim=0).contiguous()
                batch_topk_weights = torch.cat([target_topk_weights, filler_topk_weights], dim=0).contiguous()
                batch_topk_ids = torch.cat([target_topk_ids, filler_topk_ids], dim=0).contiguous()
                
                out_batch = self.outplace_fused_experts(
                    batch_input, w1, w2, batch_topk_weights, batch_topk_ids,
                    activation="silu"
                )
                torch.cuda.synchronize()
                
                # Check if results are identical (bitwise)
                diff = (out_bs1[0] - out_batch[0]).abs().max().item()
                diffs.append((bs, diff))
                
                # Debug: print actual difference for batch sizes that should differ (BLOCK_SIZE_K=128)
                if bs in [32, 96] and _ == 0:  # First iter, these use different BLOCK_SIZE_K
                    print(f"    DEBUG BS={bs}: max_diff={diff:.6e}")
                
                if diff != 0:
                    is_deterministic = False
        
        details = f"E={self.num_experts}, N={self.intermediate_size}, K={self.hidden_dim}, iters={self.num_iters}"
        if not is_deterministic:
            max_diff = max(d[1] for d in diffs)
            failed_bs = list(set(d[0] for d in diffs if d[1] > 0))
            details += f", max_diff={max_diff:.6e}, failed_BS={failed_bs}"
        return is_deterministic, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        """
        Test if MoE output for a token is the same regardless of its position in batch.
        """
        if not self.available:
            return None, "Fused MoE not available"
        
        is_deterministic = True
        diffs = []
        fixed_bs = 512
        
        # Use linspace for reproducible data
        w1 = torch.linspace(-1, 1, self.num_experts * self.intermediate_size * 2 * self.hidden_dim,
                            dtype=self.dtype, device=self.device).reshape(
                                self.num_experts, self.intermediate_size * 2, self.hidden_dim).contiguous()
        w2 = torch.linspace(-1, 1, self.num_experts * self.hidden_dim * self.intermediate_size,
                            dtype=self.dtype, device=self.device).reshape(
                                self.num_experts, self.hidden_dim, self.intermediate_size).contiguous()
        
        # Target token
        target_input = torch.linspace(-50, 50, self.hidden_dim,
                                       dtype=self.dtype, device=self.device).reshape(1, self.hidden_dim)
        target_router = torch.linspace(-5, 5, self.num_experts,
                                        dtype=torch.float32, device=self.device).reshape(1, self.num_experts)
        target_topk_weights, target_topk_ids = self._compute_topk(target_router)
        
        for _ in range(self.num_iters):
            # Compute reference with target at position 0
            filler_input = torch.linspace(-100, 100, fixed_bs * self.hidden_dim,
                                           dtype=self.dtype, device=self.device).reshape(fixed_bs, self.hidden_dim)
            filler_router = torch.linspace(-10, 10, fixed_bs * self.num_experts,
                                            dtype=torch.float32, device=self.device).reshape(fixed_bs, self.num_experts)
            filler_topk_weights, filler_topk_ids = self._compute_topk(filler_router)
            
            batch_input_ref = filler_input.clone()
            batch_topk_weights_ref = filler_topk_weights.clone()
            batch_topk_ids_ref = filler_topk_ids.clone()
            batch_input_ref[0] = target_input[0]
            batch_topk_weights_ref[0] = target_topk_weights[0]
            batch_topk_ids_ref[0] = target_topk_ids[0]
            
            out_ref = self.outplace_fused_experts(
                batch_input_ref.contiguous(), w1, w2,
                batch_topk_weights_ref.contiguous(), batch_topk_ids_ref.contiguous(),
                activation="silu"
            )
            torch.cuda.synchronize()
            ref_output = out_ref[0].clone()
            
            # Test at different positions
            for pos in range(1, fixed_bs, 1):  # Test every position
                batch_input = filler_input.clone()
                batch_topk_weights = filler_topk_weights.clone()
                batch_topk_ids = filler_topk_ids.clone()
                batch_input[pos] = target_input[0]
                batch_topk_weights[pos] = target_topk_weights[0]
                batch_topk_ids[pos] = target_topk_ids[0]
                
                out = self.outplace_fused_experts(
                    batch_input.contiguous(), w1, w2,
                    batch_topk_weights.contiguous(), batch_topk_ids.contiguous(),
                    activation="silu"
                )
                torch.cuda.synchronize()
                
                diff = (ref_output - out[pos]).abs().max().item()
                diffs.append((pos, diff))
                
                if diff != 0:
                    is_deterministic = False
        
        details = f"Tested positions in BS={fixed_bs}, iters={self.num_iters}"
        if not is_deterministic:
            max_diff = max(d[1] for d in diffs)
            failed_positions = [d[0] for d in diffs if d[1] > 0][:10]  # First 10
            details += f", max_diff={max_diff:.6e}, failed_pos={failed_positions}..."
        return is_deterministic, details


# =============================================================================
# ATTENTION OPERATORS
# =============================================================================

class FlashAttention3Tester(OperatorTester):
    """Test FlashAttention3 for invariance."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 num_splits: int = None):
        super().__init__(device, dtype)
        self.category = "Attention"
        self.num_splits = num_splits if num_splits is not None else 1
        if num_splits is None:
            self.name = "FA3 (default)"
        else:
            self.name = f"FA3 (num_splits={num_splits})"
        self.num_heads = 32
        self.num_kv_heads = 8  # GQA
        self.head_dim = 128
        self.available = False
        self._check_availability()
        
    def _check_availability(self):
        try:
            # Use sgl_kernel's flash_attn_varlen_func (FA3)
            from sgl_kernel.flash_attn import flash_attn_varlen_func
            self.flash_attn_varlen_func = flash_attn_varlen_func
            self.available = True
        except ImportError:
            try:
                # Fallback to flash_attn package
                from flash_attn import flash_attn_varlen_func
                self.flash_attn_varlen_func = flash_attn_varlen_func
                self.available = True
            except ImportError:
                self.available = False
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        if not self.available:
            return None, "FlashAttention not available"
        
        torch.manual_seed(42)
        seq_len = 128
        
        # For varlen API: q, k, v are (total_tokens, num_heads, head_dim)
        # cu_seqlens are cumulative sequence lengths
        
        # Target sequence (BS=1)
        target_q = torch.randn(seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_k = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_v = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        
        # BS=1 cu_seqlens
        cu_seqlens_q_bs1 = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        cu_seqlens_k_bs1 = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        
        out_bs1 = self.flash_attn_varlen_func(
            target_q, target_k, target_v,
            cu_seqlens_q_bs1, cu_seqlens_k_bs1,
            max_seqlen_q=seq_len, max_seqlen_k=seq_len,
            causal=True, num_splits=self.num_splits
        )
        torch.cuda.synchronize()
        ref_output = out_bs1.clone()
        
        results = []
        for bs in [2, 4, 8]:
            # Create batched data
            total_tokens = seq_len * bs
            batch_q = torch.randn(total_tokens, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            batch_k = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
            batch_v = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
            
            # Put target at first sequence
            batch_q[:seq_len] = target_q
            batch_k[:seq_len] = target_k
            batch_v[:seq_len] = target_v
            
            # cu_seqlens for batch
            cu_seqlens_q = torch.tensor([i * seq_len for i in range(bs + 1)], dtype=torch.int32, device=self.device)
            cu_seqlens_k = torch.tensor([i * seq_len for i in range(bs + 1)], dtype=torch.int32, device=self.device)
            
            out_batch = self.flash_attn_varlen_func(
                batch_q, batch_k, batch_v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                causal=True, num_splits=self.num_splits
            )
            torch.cuda.synchronize()
            
            # Compare first sequence
            out_first = out_batch[:seq_len]
            match = self.check_equal(ref_output, out_first)
            results.append((bs, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested BS: {[r[0] for r in results]}, seq_len={seq_len}"
        if not all_match:
            failed = [r[0] for r in results if not r[1]]
            details += f", Failed at BS: {failed}"
        return all_match, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        if not self.available:
            return None, "FlashAttention not available"
        
        torch.manual_seed(42)
        fixed_bs = 8
        seq_len = 128
        
        target_q = torch.randn(seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_k = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_v = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        
        total_tokens = seq_len * fixed_bs
        filler_q = torch.randn(total_tokens, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        filler_k = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        filler_v = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        
        cu_seqlens_q = torch.tensor([i * seq_len for i in range(fixed_bs + 1)], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([i * seq_len for i in range(fixed_bs + 1)], dtype=torch.int32, device=self.device)
        
        results = []
        ref_output = None
        
        for pos in range(fixed_bs):
            batch_q = filler_q.clone()
            batch_k = filler_k.clone()
            batch_v = filler_v.clone()
            
            # Put target at position `pos`
            start_idx = pos * seq_len
            end_idx = (pos + 1) * seq_len
            batch_q[start_idx:end_idx] = target_q
            batch_k[start_idx:end_idx] = target_k
            batch_v[start_idx:end_idx] = target_v
            
            out = self.flash_attn_varlen_func(
                batch_q, batch_k, batch_v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                causal=True, num_splits=self.num_splits
            )
            torch.cuda.synchronize()
            
            target_out = out[start_idx:end_idx].clone()
            
            if ref_output is None:
                ref_output = target_out
            else:
                match = self.check_equal(ref_output, target_out)
                results.append((pos, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested positions 0-{fixed_bs-1}"
        if not all_match:
            failed = [r[0] for r in results if not r[1]]
            details += f", Failed at positions: {failed}"
        return all_match, details


class FlashInferTester(OperatorTester):
    """Test FlashInfer with ragged batch for invariance."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 backend: str = "FA3"):
        super().__init__(device, dtype)
        self.category = "Attention"
        self.name = f"FlashInfer ({backend} backend)"
        self.backend = backend
        self.num_heads = 32
        self.num_kv_heads = 8  # GQA
        self.head_dim = 128
        self.available = False
        self._check_availability()
        
    def _check_availability(self):
        try:
            import flashinfer
            self.flashinfer = flashinfer
            self.available = True
        except ImportError:
            self.available = False
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        if not self.available:
            return None, "FlashInfer not available"
        
        torch.manual_seed(42)
        seq_len = 128
        
        # Single sequence test using single_prefill_with_kv_cache
        target_q = torch.randn(seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_k = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_v = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        
        # BS=1 output using single_prefill
        out_bs1 = self.flashinfer.single_prefill_with_kv_cache(
            target_q, target_k, target_v, causal=True
        )
        torch.cuda.synchronize()
        
        results = []
        for bs in [2, 4, 8]:
            # Use ragged batch API
            workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            prefill_wrapper = self.flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            
            # Create batched data with target at position 0
            total_tokens = seq_len * bs
            batch_q = torch.randn(total_tokens, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            batch_k = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
            batch_v = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
            
            # Put target at first sequence position
            batch_q[:seq_len] = target_q
            batch_k[:seq_len] = target_k
            batch_v[:seq_len] = target_v
            
            qo_indptr = torch.tensor([i * seq_len for i in range(bs + 1)], dtype=torch.int32, device=self.device)
            kv_indptr = torch.tensor([i * seq_len for i in range(bs + 1)], dtype=torch.int32, device=self.device)
            
            prefill_wrapper.begin_forward(
                qo_indptr, kv_indptr,
                self.num_heads, self.num_kv_heads, self.head_dim,
                q_data_type=self.dtype
            )
            
            out_batch = prefill_wrapper.forward(batch_q, batch_k, batch_v, causal=True)
            torch.cuda.synchronize()
            
            # Extract first sequence output
            out_first = out_batch[:seq_len]
            match = self.check_equal(out_bs1, out_first)
            results.append((bs, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested BS: {[r[0] for r in results]}, seq_len={seq_len}"
        if not all_match:
            failed = [r[0] for r in results if not r[1]]
            details += f", Failed at BS: {failed}"
        return all_match, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        if not self.available:
            return None, "FlashInfer not available"
        
        torch.manual_seed(42)
        fixed_bs = 8
        seq_len = 128
        
        target_q = torch.randn(seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_k = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_v = torch.randn(seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        
        # Create filler data
        total_tokens = seq_len * fixed_bs
        filler_q = torch.randn(total_tokens, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        filler_k = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        filler_v = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        
        workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        qo_indptr = torch.tensor([i * seq_len for i in range(fixed_bs + 1)], dtype=torch.int32, device=self.device)
        kv_indptr = torch.tensor([i * seq_len for i in range(fixed_bs + 1)], dtype=torch.int32, device=self.device)
        
        results = []
        ref_output = None
        
        for pos in range(fixed_bs):
            batch_q = filler_q.clone()
            batch_k = filler_k.clone()
            batch_v = filler_v.clone()
            
            # Put target at position `pos`
            start_idx = pos * seq_len
            end_idx = (pos + 1) * seq_len
            batch_q[start_idx:end_idx] = target_q
            batch_k[start_idx:end_idx] = target_k
            batch_v[start_idx:end_idx] = target_v
            
            prefill_wrapper = self.flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            prefill_wrapper.begin_forward(
                qo_indptr, kv_indptr,
                self.num_heads, self.num_kv_heads, self.head_dim,
                q_data_type=self.dtype
            )
            
            out = prefill_wrapper.forward(batch_q, batch_k, batch_v, causal=True)
            torch.cuda.synchronize()
            
            # Extract output at position `pos`
            target_out = out[start_idx:end_idx].clone()
            
            if ref_output is None:
                ref_output = target_out
            else:
                match = self.check_equal(ref_output, target_out)
                results.append((pos, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested positions 0-{fixed_bs-1}"
        if not all_match:
            failed = [r[0] for r in results if not r[1]]
            details += f", Failed at positions: {failed}"
        return all_match, details


# =============================================================================
# NORMALIZATION OPERATORS
# =============================================================================

class RMSNormTester(OperatorTester):
    """Test RMSNorm for invariance using actual vLLM kernel from vllm._custom_ops."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Normalization"
        self.name = "vLLM RMSNorm (actual)"
        self.eps = 1e-6
        self.available = False
        self._check_availability()
        # Test multiple hidden dimensions
        self.hidden_dims = [256, 1024, 2048, 4096, 8192]
        self.num_iters = 2000
        self.test_batch_sizes = [1, 10, 55, 128, 255, 256, 257, 512, 528, 2048]
        
    def _check_availability(self):
        try:
            from vllm._custom_ops import rms_norm
            self.vllm_rms_norm = rms_norm
            self.available = True
        except (ImportError, AttributeError):
            self.available = False
        
    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm using actual vLLM kernel from vllm._custom_ops."""
        if self.available:
            out = torch.empty_like(x)
            self.vllm_rms_norm(out, x, weight, self.eps)
            return out
        else:
            # Fallback: reference implementation
            orig_dtype = x.dtype
            x_float = x.float()
            variance = x_float.pow(2).mean(-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.eps)
            return (x_norm * weight.float()).to(orig_dtype)
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        """
        Test if RMSNorm output for first row is the same across different batch sizes.
        Tests multiple hidden dimensions with many iterations each.
        """
        full_batch_size = max(self.test_batch_sizes)
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            failed_bs_set = set()  # Track which batch sizes fail
            failed_iters = []  # Track (iter_idx, bs, diff) for failures
            
            for iter_idx in range(self.num_iters):
                # Generate fresh random test data each iteration
                x = torch.randn(full_batch_size, hidden_dim, dtype=self.dtype, device=self.device)
                weight = torch.ones(hidden_dim, dtype=self.dtype, device=self.device)
                
                # Reference: Single row (BS=1)
                out_ref = self.rms_norm(x[:1].clone(), weight)
                torch.cuda.synchronize()
                
                # Test at different batch sizes
                for bs in self.test_batch_sizes[1:]:  # Skip BS=1 (reference)
                    out_bs = self.rms_norm(x[:bs].clone(), weight)
                    torch.cuda.synchronize()
                    
                    diff = (out_ref[0] - out_bs[0]).abs().max().item()
                    all_diffs.append(diff)
                    
                    if diff > dim_max_diff:
                        dim_max_diff = diff
                    
                    if diff != 0:
                        dim_invariant = False
                        all_invariant = False
                        failed_bs_set.add(bs)
                        failed_iters.append((iter_idx, bs, diff))
            
            status = "✓" if dim_invariant else "✗"
            if dim_invariant:
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            else:
                # Show first 5 failures with iteration numbers
                fail_summary = [(it, bs, f"{d:.2e}") for it, bs, d in failed_iters[:5]]
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}, failed_BS={sorted(failed_bs_set)}, "
                      f"failures({len(failed_iters)} total)={fail_summary}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff, sorted(failed_bs_set), len(failed_iters)))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={self.test_batch_sizes}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        """
        Test if output for a row is the same regardless of position in batch.
        Tests multiple hidden dimensions with many iterations each.
        """
        fixed_bs = 512
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            
            for iter_idx in range(self.num_iters):
                # Generate fresh random test data each iteration
                weight = torch.randn(hidden_dim, dtype=self.dtype, device=self.device)
                target_x = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device)
                filler_x = torch.randn(fixed_bs, hidden_dim, dtype=self.dtype, device=self.device)
                
                ref_output = None
                
                # Test at every position
                for pos in range(fixed_bs):
                    batch_x = filler_x.clone()
                    batch_x[pos] = target_x[0]
                    
                    out = self.rms_norm(batch_x, weight)
                    torch.cuda.synchronize()
                    
                    target_out = out[pos].clone()
                    
                    if ref_output is None:
                        ref_output = target_out
                    else:
                        diff = (ref_output - target_out).abs().max().item()
                        all_diffs.append(diff)
                        
                        if diff > dim_max_diff:
                            dim_max_diff = diff
                        
                        if diff != 0:
                            dim_invariant = False
                            all_invariant = False
            
            status = "✓" if dim_invariant else "✗"
            print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={fixed_bs}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details


class FusedRMSNormResidualTester(OperatorTester):
    """Test Fused RMSNorm + Residual for invariance using actual vLLM kernel from vllm._custom_ops."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Normalization"
        self.name = "vLLM Fused RMSNorm+Residual (actual)"
        self.eps = 1e-6
        self.available = False
        self._check_availability()
        # Test multiple hidden dimensions
        self.hidden_dims = [256, 1024, 2048, 4096, 8192]
        self.num_iters = 2000
        self.test_batch_sizes = [1, 10, 55, 128, 255, 256, 257, 512, 528, 2048]
        
    def _check_availability(self):
        try:
            from vllm._custom_ops import fused_add_rms_norm
            self.vllm_fused_add_rms_norm = fused_add_rms_norm
            self.available = True
        except (ImportError, AttributeError):
            self.available = False
    
    def fused_rms_norm_residual(self, x: torch.Tensor, residual: torch.Tensor, 
                                 weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused RMSNorm + Residual using actual vLLM kernel: out = RMSNorm(x + residual), also returns updated residual."""
        if self.available:
            x = x.clone()
            residual = residual.clone()
            self.vllm_fused_add_rms_norm(x, residual, weight, self.eps)
            return x, residual
        else:
            # Fallback implementation
            print(f"Warning: Using fallback implementation for Fused RMSNorm+Residual")
            orig_dtype = x.dtype
            x_float = x.float() + residual.float()
            new_residual = x_float.to(orig_dtype)
            variance = x_float.pow(2).mean(-1, keepdim=True)
            out = (x_float * torch.rsqrt(variance + self.eps) * weight.float()).to(orig_dtype)
            return out, new_residual
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        """
        Test if Fused RMSNorm+Residual output for first row is the same across different batch sizes.
        Tests multiple hidden dimensions (256, 1024, 2048, 4096, 8192) with 100 iterations each.
        """
        full_batch_size = max(self.test_batch_sizes)
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            failed_bs_set = set()  # Track which batch sizes fail
            failed_iters = []  # Track (iter_idx, bs, diff) for failures
            
            for iter_idx in range(self.num_iters):
                # Generate fresh random test data each iteration
                x = torch.randn(full_batch_size, hidden_dim, dtype=self.dtype, device=self.device)
                weight = torch.ones(hidden_dim, dtype=self.dtype, device=self.device)
                residual = torch.randn(full_batch_size, hidden_dim, dtype=self.dtype, device=self.device)
                
                # Reference: Single row (BS=1)
                out_ref, _ = self.fused_rms_norm_residual(x[:1].clone(), residual[:1].clone(), weight)
                torch.cuda.synchronize()
                
                # Test at different batch sizes
                for bs in self.test_batch_sizes[1:]:  # Skip BS=1 (reference)
                    out_bs, _ = self.fused_rms_norm_residual(x[:bs].clone(), residual[:bs].clone(), weight)
                    torch.cuda.synchronize()
                    
                    diff = (out_ref[0] - out_bs[0]).abs().max().item()
                    all_diffs.append(diff)
                    
                    if diff > dim_max_diff:
                        dim_max_diff = diff
                    
                    if diff != 0:
                        dim_invariant = False
                        all_invariant = False
                        failed_bs_set.add(bs)
                        failed_iters.append((iter_idx, bs, diff))
            
            status = "✓" if dim_invariant else "✗"
            if dim_invariant:
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            else:
                # Show first 5 failures with iteration numbers
                fail_summary = [(it, bs, f"{d:.2e}") for it, bs, d in failed_iters[:5]]
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}, failed_BS={sorted(failed_bs_set)}, "
                      f"failures({len(failed_iters)} total)={fail_summary}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff, sorted(failed_bs_set), len(failed_iters)))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={self.test_batch_sizes}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        """
        Test if output for a row is the same regardless of position in batch.
        Tests multiple hidden dimensions with 100 iterations each.
        """
        fixed_bs = 512
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            
            for iter_idx in range(self.num_iters):
                # Generate fresh random test data each iteration
                weight = torch.randn(hidden_dim, dtype=self.dtype, device=self.device)
                target_x = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device)
                target_res = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device)
                filler_x = torch.randn(fixed_bs, hidden_dim, dtype=self.dtype, device=self.device)
                filler_res = torch.randn(fixed_bs, hidden_dim, dtype=self.dtype, device=self.device)
                
                ref_output = None
                
                # Test at every position
                for pos in range(fixed_bs):
                    batch_x = filler_x.clone()
                    batch_res = filler_res.clone()
                    batch_x[pos] = target_x[0]
                    batch_res[pos] = target_res[0]
                    
                    out, _ = self.fused_rms_norm_residual(batch_x, batch_res, weight)
                    torch.cuda.synchronize()
                    
                    target_out = out[pos].clone()
                    
                    if ref_output is None:
                        ref_output = target_out
                    else:
                        diff = (ref_output - target_out).abs().max().item()
                        all_diffs.append(diff)
                        
                        if diff > dim_max_diff:
                            dim_max_diff = diff
                        
                        if diff != 0:
                            dim_invariant = False
                            all_invariant = False
            
            status = "✓" if dim_invariant else "✗"
            # print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={fixed_bs}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details


class SGLangRMSNormTester(OperatorTester):
    """Test RMSNorm for invariance using SGLang default kernel from sgl_kernel (with PDL)."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Normalization"
        self.name = "SGLang RMSNorm (PDL)"
        self.eps = 1e-6
        self.available = False
        self._check_availability()
        # Test multiple hidden dimensions
        self.hidden_dims = [256, 1024, 2048, 4096, 8192]
        self.num_iters = 2000
        self.test_batch_sizes = [1, 10, 55, 128, 255, 256, 257, 512, 528, 2048]
        
    def _check_availability(self):
        try:
            import sgl_kernel
            self.sgl_rmsnorm = sgl_kernel.rmsnorm
            self.available = True
        except (ImportError, AttributeError):
            self.available = False
        
    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm using SGLang kernel from sgl_kernel."""
        if self.available:
            return self.sgl_rmsnorm(x, weight, eps=self.eps)
        else:
            # Fallback: reference implementation
            orig_dtype = x.dtype
            x_float = x.float()
            variance = x_float.pow(2).mean(-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.eps)
            return (x_norm * weight.float()).to(orig_dtype)
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        """
        Test if RMSNorm output for first row is the same across different batch sizes.
        Tests multiple hidden dimensions with many iterations each.
        """
        full_batch_size = max(self.test_batch_sizes)
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            failed_bs_set = set()
            failed_iters = []
            
            for iter_idx in range(self.num_iters):
                x = torch.randn(full_batch_size, hidden_dim, dtype=self.dtype, device=self.device)
                weight = torch.ones(hidden_dim, dtype=self.dtype, device=self.device)
                
                out_ref = self.rms_norm(x[:1].clone(), weight)
                torch.cuda.synchronize()
                
                for bs in self.test_batch_sizes[1:]:
                    out_bs = self.rms_norm(x[:bs].clone(), weight)
                    torch.cuda.synchronize()
                    
                    diff = (out_ref[0] - out_bs[0]).abs().max().item()
                    all_diffs.append(diff)
                    
                    if diff > dim_max_diff:
                        dim_max_diff = diff
                    
                    if diff != 0:
                        dim_invariant = False
                        all_invariant = False
                        failed_bs_set.add(bs)
                        failed_iters.append((iter_idx, bs, diff))
            
            status = "✓" if dim_invariant else "✗"
            if dim_invariant:
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            else:
                fail_summary = [(it, bs, f"{d:.2e}") for it, bs, d in failed_iters[:5]]
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}, failed_BS={sorted(failed_bs_set)}, "
                      f"failures({len(failed_iters)} total)={fail_summary}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff, sorted(failed_bs_set), len(failed_iters)))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={self.test_batch_sizes}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        """Test if output for a row is the same regardless of position in batch."""
        fixed_bs = 512
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            
            for iter_idx in range(self.num_iters):
                weight = torch.randn(hidden_dim, dtype=self.dtype, device=self.device)
                target_x = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device)
                filler_x = torch.randn(fixed_bs, hidden_dim, dtype=self.dtype, device=self.device)
                
                ref_output = None
                
                for pos in range(fixed_bs):
                    batch_x = filler_x.clone()
                    batch_x[pos] = target_x[0]
                    
                    out = self.rms_norm(batch_x, weight)
                    torch.cuda.synchronize()
                    
                    target_out = out[pos].clone()
                    
                    if ref_output is None:
                        ref_output = target_out
                    else:
                        diff = (ref_output - target_out).abs().max().item()
                        all_diffs.append(diff)
                        
                        if diff > dim_max_diff:
                            dim_max_diff = diff
                        
                        if diff != 0:
                            dim_invariant = False
                            all_invariant = False
            
            status = "✓" if dim_invariant else "✗"
            print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={fixed_bs}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details


class SGLangFusedRMSNormResidualTester(OperatorTester):
    """Test Fused RMSNorm + Residual for invariance using SGLang default kernel from sgl_kernel (with PDL)."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Normalization"
        self.name = "SGLang Fused RMSNorm+Residual (PDL)"
        self.eps = 1e-6
        self.available = False
        self._check_availability()
        # Test multiple hidden dimensions
        self.hidden_dims = [256, 1024, 2048, 4096, 8192]
        self.num_iters = 2000
        self.test_batch_sizes = [1, 10, 55, 128, 255, 256, 257, 512, 528, 2048]
        
    def _check_availability(self):
        try:
            import sgl_kernel
            self.sgl_fused_add_rmsnorm = sgl_kernel.fused_add_rmsnorm
            self.available = True
        except (ImportError, AttributeError):
            self.available = False
    
    def fused_rms_norm_residual(self, x: torch.Tensor, residual: torch.Tensor, 
                                 weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused RMSNorm + Residual using SGLang kernel."""
        if self.available:
            x = x.clone()
            residual = residual.clone()
            self.sgl_fused_add_rmsnorm(x, residual, weight, self.eps)
            return x, residual
        else:
            orig_dtype = x.dtype
            x_float = x.float() + residual.float()
            new_residual = x_float.to(orig_dtype)
            variance = x_float.pow(2).mean(-1, keepdim=True)
            out = (x_float * torch.rsqrt(variance + self.eps) * weight.float()).to(orig_dtype)
            return out, new_residual
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        """Test if Fused RMSNorm+Residual output for first row is the same across different batch sizes."""
        full_batch_size = max(self.test_batch_sizes)
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            failed_bs_set = set()
            failed_iters = []
            
            for iter_idx in range(self.num_iters):
                x = torch.randn(full_batch_size, hidden_dim, dtype=self.dtype, device=self.device)
                weight = torch.ones(hidden_dim, dtype=self.dtype, device=self.device)
                residual = torch.randn(full_batch_size, hidden_dim, dtype=self.dtype, device=self.device)
                
                out_ref, _ = self.fused_rms_norm_residual(x[:1].clone(), residual[:1].clone(), weight)
                torch.cuda.synchronize()
                
                for bs in self.test_batch_sizes[1:]:
                    out_bs, _ = self.fused_rms_norm_residual(x[:bs].clone(), residual[:bs].clone(), weight)
                    torch.cuda.synchronize()
                    
                    diff = (out_ref[0] - out_bs[0]).abs().max().item()
                    all_diffs.append(diff)
                    
                    if diff > dim_max_diff:
                        dim_max_diff = diff
                    
                    if diff != 0:
                        dim_invariant = False
                        all_invariant = False
                        failed_bs_set.add(bs)
                        failed_iters.append((iter_idx, bs, diff))
            
            status = "✓" if dim_invariant else "✗"
            if dim_invariant:
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            else:
                fail_summary = [(it, bs, f"{d:.2e}") for it, bs, d in failed_iters[:5]]
                print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}, failed_BS={sorted(failed_bs_set)}, "
                      f"failures({len(failed_iters)} total)={fail_summary}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff, sorted(failed_bs_set), len(failed_iters)))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={self.test_batch_sizes}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        """Test if output for a row is the same regardless of position in batch."""
        fixed_bs = 512
        all_invariant = True
        all_diffs = []
        failed_configs = []
        
        print(f"\n    Testing {len(self.hidden_dims)} hidden dims × {self.num_iters} iterations:")
        
        for hidden_dim in self.hidden_dims:
            dim_invariant = True
            dim_max_diff = 0.0
            
            for iter_idx in range(self.num_iters):
                weight = torch.randn(hidden_dim, dtype=self.dtype, device=self.device)
                target_x = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device)
                target_res = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device)
                filler_x = torch.randn(fixed_bs, hidden_dim, dtype=self.dtype, device=self.device)
                filler_res = torch.randn(fixed_bs, hidden_dim, dtype=self.dtype, device=self.device)
                
                ref_output = None
                
                for pos in range(fixed_bs):
                    batch_x = filler_x.clone()
                    batch_res = filler_res.clone()
                    batch_x[pos] = target_x[0]
                    batch_res[pos] = target_res[0]
                    
                    out, _ = self.fused_rms_norm_residual(batch_x, batch_res, weight)
                    torch.cuda.synchronize()
                    
                    target_out = out[pos].clone()
                    
                    if ref_output is None:
                        ref_output = target_out
                    else:
                        diff = (ref_output - target_out).abs().max().item()
                        all_diffs.append(diff)
                        
                        if diff > dim_max_diff:
                            dim_max_diff = diff
                        
                        if diff != 0:
                            dim_invariant = False
                            all_invariant = False
            
            status = "✓" if dim_invariant else "✗"
            print(f"      hidden_dim={hidden_dim:5d}: {status} max_diff={dim_max_diff:.2e}")
            
            if not dim_invariant:
                failed_configs.append((hidden_dim, dim_max_diff))
        
        max_diff = max(all_diffs) if all_diffs else 0.0
        details = f"hidden_dims={self.hidden_dims}, BS={fixed_bs}, iters={self.num_iters}, max_diff={max_diff:.2e}"
        if not self.available:
            details += " (fallback impl)"
        if not all_invariant:
            details += f", failed_configs={failed_configs}"
        return all_invariant, details


# =============================================================================
# EMBEDDING OPERATORS
# =============================================================================

class RotaryEmbeddingTester(OperatorTester):
    """Test Rotary Positional Embedding (RoPE) for invariance."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Embedding"
        self.name = "Rotary Embedding (RoPE)"
        self.head_dim = 128
        self.num_heads = 32
        self.max_seq_len = 4096
        self.base = 10000.0
        
        # Precompute freqs
        self._compute_freqs()
        
    def _compute_freqs(self):
        """Precompute rotary frequencies."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.inv_freq = inv_freq.to(self.device)
        
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, self.inv_freq)
        self.cos_cached = freqs.cos().to(self.dtype)
        self.sin_cached = freqs.sin().to(self.dtype)
    
    def apply_rotary_emb(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to input tensor.
        
        x: (batch, seq_len, num_heads, head_dim)
        positions: (batch, seq_len) - position indices
        """
        # Get cos/sin for positions
        cos = self.cos_cached[positions]  # (batch, seq_len, head_dim//2)
        sin = self.sin_cached[positions]
        
        # Reshape for broadcasting: (batch, seq_len, 1, head_dim//2)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        
        # Split x into two halves
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        
        # Apply rotation
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        return torch.cat([out1, out2], dim=-1)
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        seq_len = 64
        
        target_x = torch.randn(1, seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        target_pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        out_bs1 = self.apply_rotary_emb(target_x, target_pos)
        torch.cuda.synchronize()
        
        results = []
        for bs in [2, 4, 8, 16]:
            filler_x = torch.randn(bs - 1, seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            
            batch_x = torch.cat([target_x, filler_x], dim=0)
            batch_pos = target_pos.expand(bs, -1)
            
            out_batch = self.apply_rotary_emb(batch_x, batch_pos)
            torch.cuda.synchronize()
            
            match = self.check_equal(out_bs1[0], out_batch[0])
            results.append((bs, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested BS: {[r[0] for r in results]}"
        return all_match, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        fixed_bs = 16
        seq_len = 64
        
        target_x = torch.randn(1, seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(fixed_bs, -1)
        
        filler_x = torch.randn(fixed_bs, seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        
        results = []
        ref_output = None
        
        for pos in range(fixed_bs):
            batch_x = filler_x.clone()
            batch_x[pos] = target_x[0]
            
            out = self.apply_rotary_emb(batch_x, positions)
            torch.cuda.synchronize()
            
            target_out = out[pos].clone()
            
            if ref_output is None:
                ref_output = target_out
            else:
                match = self.check_equal(ref_output, target_out)
                results.append((pos, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested positions 0-{fixed_bs-1}"
        return all_match, details


# =============================================================================
# ACTIVATION OPERATORS
# =============================================================================

class ActivationTester(OperatorTester):
    """Test activation functions for invariance."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16, 
                 activation_name: str = "SiLU"):
        super().__init__(device, dtype)
        self.category = "Activation"
        self.name = activation_name
        
        self.activations = {
            "SiLU": F.silu,
            "GELU": F.gelu,
            "ReLU": F.relu,
            "GELU_tanh": lambda x: F.gelu(x, approximate='tanh'),
        }
        self.activation_fn = self.activations.get(activation_name, F.silu)
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        
        target_input = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        out_bs1 = self.activation_fn(target_input)
        torch.cuda.synchronize()
        
        results = []
        for bs in self.batch_sizes[1:]:
            filler = torch.randn(bs - 1, self.hidden_dim, dtype=self.dtype, device=self.device)
            batch_input = torch.cat([target_input, filler], dim=0)
            
            out_batch = self.activation_fn(batch_input)
            torch.cuda.synchronize()
            
            match = self.check_equal(out_bs1[0], out_batch[0])
            results.append((bs, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested BS: {[r[0] for r in results]}"
        return all_match, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        
        target_input = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        fixed_bs = 64
        filler = torch.randn(fixed_bs, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        results = []
        ref_output = None
        
        for pos in range(fixed_bs):
            batch = filler.clone()
            batch[pos] = target_input[0]
            
            out = self.activation_fn(batch)
            torch.cuda.synchronize()
            
            target_out = out[pos].clone()
            
            if ref_output is None:
                ref_output = target_out
            else:
                match = self.check_equal(ref_output, target_out)
                results.append((pos, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested positions 0-{fixed_bs-1}"
        return all_match, details


# =============================================================================
# COMMUNICATION OPERATORS (Distributed AllReduce tests)
# =============================================================================

import multiprocessing as mp
import socket


def get_open_port():
    """Get an available port for distributed communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _allreduce_worker(world_size: int, rank: int, port: int, algorithm: str,
                      batch_sizes: List[int], hidden_dim: int, dtype: torch.dtype,
                      test_type: str, result_queue):
    """Worker function for AllReduce tests - runs on each GPU."""
    import torch.distributed as dist
    
    # Set NCCL algorithm BEFORE init_process_group
    if algorithm == "ring":
        os.environ["NCCL_ALGO"] = "allreduce:ring"
    else:
        os.environ["NCCL_ALGO"] = "allreduce:tree"
    
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    
    # Different seed per rank - each GPU has DIFFERENT input
    torch.manual_seed(42 + rank)
    
    if test_type == "batch":
        # Batch invariance test
        target_input = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        
        dist.barrier()
        
        # Reference output with BS=1
        ref_data = target_input.clone()
        dist.all_reduce(ref_data)
        torch.cuda.synchronize()
        ref_output = ref_data[0].clone()
        
        results = []
        for bs in batch_sizes[1:]:  # Skip bs=1
            filler = torch.randn(bs - 1, hidden_dim, dtype=dtype, device=device)
            batch = torch.cat([target_input, filler], dim=0)
            
            dist.all_reduce(batch)
            torch.cuda.synchronize()
            
            match = torch.equal(ref_output, batch[0])
            results.append((bs, match))
        
        if rank == 0:
            all_match = all(r[1] for r in results)
            failed = [r[0] for r in results if not r[1]]
            result_queue.put(("batch", all_match, failed))
    
    elif test_type == "position":
        # Position invariance test
        fixed_bs = 1027
        target_input = torch.randn(1, hidden_dim, dtype=dtype, device=device)
        filler = torch.randn(fixed_bs, hidden_dim, dtype=dtype, device=device)
        
        dist.barrier()
        
        results = []
        ref_output = None
        
        for pos in range(fixed_bs):
            batch = filler.clone()
            batch[pos] = target_input[0]
            
            dist.all_reduce(batch)
            torch.cuda.synchronize()
            
            target_out = batch[pos].clone()
            
            if ref_output is None:
                ref_output = target_out
            else:
                match = torch.equal(ref_output, target_out)
                results.append((pos, match))
        
        if rank == 0:
            all_match = all(r[1] for r in results)
            failed = [r[0] for r in results if not r[1]][:5]
            result_queue.put(("position", all_match, failed, fixed_bs))
    
    dist.barrier()
    dist.destroy_process_group()


class AllReduceTester(OperatorTester):
    """
    Actual distributed AllReduce test using NCCL.
    Requires multiple GPUs to run.
    """
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 algorithm: str = "ring"):
        super().__init__(device, dtype)
        self.category = "Communication"
        self.name = f"{algorithm.capitalize()}-based AllReduce"
        self.algorithm = algorithm
        self.world_size = min(torch.cuda.device_count(), 4)
        self.available = self.world_size >= 2
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        self._batch_result = None
        self._position_result = None
    
    def _run_distributed_test(self, test_type: str) -> Tuple[Optional[bool], str]:
        if not self.available:
            return None, f"Requires 2+ GPUs (found {self.world_size})"
        
        port = get_open_port()
        result_queue = mp.Queue()
        
        procs = []
        for rank in range(self.world_size):
            p = mp.Process(
                target=_allreduce_worker,
                args=(self.world_size, rank, port, self.algorithm,
                      self.batch_sizes, self.hidden_dim, self.dtype,
                      test_type, result_queue)
            )
            p.start()
            procs.append(p)
        
        for p in procs:
            p.join()
        
        # Get result from rank 0
        if not result_queue.empty():
            result = result_queue.get()
            if result[0] == "batch":
                _, all_match, failed = result
                details = f"Tested BS: {self.batch_sizes[1:]}, world_size={self.world_size}"
                if not all_match:
                    details += f", Failed at BS: {failed}"
                return all_match, details
            else:  # position
                _, all_match, failed, fixed_bs = result
                details = f"Tested positions 0-{fixed_bs-1}, world_size={self.world_size}"
                if not all_match:
                    details += f", Failed at positions: {failed}..."
                return all_match, details
        
        return None, "No result returned"
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        return self._run_distributed_test("batch")
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        return self._run_distributed_test("position")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def create_all_testers(device: torch.device, dtype: torch.dtype) -> List[OperatorTester]:
    """Create all operator testers."""
    testers = [
        # Matmul
        GEMMTester(device, dtype),
        FusedMoETester(device, dtype),
        
        # Attention
        FlashAttention3Tester(device, dtype, num_splits=None),  # FA3 default
        FlashAttention3Tester(device, dtype, num_splits=1),     # FA3 num_splits=1
        FlashInferTester(device, dtype, backend="FA3"),
        FlashInferTester(device, dtype, backend="FA2"),
        
        # Normalization
        RMSNormTester(device, dtype),
        FusedRMSNormResidualTester(device, dtype),
        SGLangRMSNormTester(device, dtype),
        SGLangFusedRMSNormResidualTester(device, dtype),
        
        # Embedding
        RotaryEmbeddingTester(device, dtype),
        
        # Activations
        ActivationTester(device, dtype, "SiLU"),
        
        # Communication (actual distributed tests)
        AllReduceTester(device, dtype, "ring"),
        AllReduceTester(device, dtype, "tree"),
    ]
    return testers


def run_tests(testers: List[OperatorTester], categories: Optional[List[str]] = None) -> List[TestResult]:
    """Run tests for all specified testers."""
    results = []
    
    for tester in testers:
        if categories and tester.category.lower() not in [c.lower() for c in categories]:
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing: {tester.name} ({tester.category})")
        print(f"{'='*60}")
        
        result = TestResult(
            operator_name=tester.name,
            category=tester.category,
            batch_invariant=None,
            position_invariant=None,
        )
        
        try:
            # Test batch invariance
            print("  Testing batch invariance...", end=" ", flush=True)
            batch_inv, batch_details = tester.test_batch_invariance()
            result.batch_invariant = batch_inv
            result.batch_inv_details = batch_details
            
            if batch_inv is None:
                print(f"SKIPPED - {batch_details}")
            elif batch_inv:
                print(f"✓ PASS")
            else:
                print(f"✗ FAIL - {batch_details}")
            
            # Test position invariance
            print("  Testing position invariance...", end=" ", flush=True)
            pos_inv, pos_details = tester.test_position_invariance()
            result.position_invariant = pos_inv
            result.pos_inv_details = pos_details
            
            if pos_inv is None:
                print(f"SKIPPED - {pos_details}")
            elif pos_inv:
                print(f"✓ PASS")
            else:
                print(f"✗ FAIL - {pos_details}")
                
        except Exception as e:
            result.error = str(e)
            print(f"  ERROR: {e}")
        
        results.append(result)
    
    return results


def print_summary_table(results: List[TestResult]):
    """Print a summary table of all test results."""
    print(f"\n{'='*90}")
    print("SUMMARY: Operator Invariance Test Results")
    print(f"{'='*90}")
    print(f"{'Category':<15} | {'Operator':<30} | {'Batch-Inv':^12} | {'Pos-Inv':^12}")
    print("-" * 90)
    
    current_category = None
    for r in results:
        if r.category != current_category:
            if current_category is not None:
                print("-" * 90)
            current_category = r.category
        
        def fmt_result(val):
            if val is None:
                return "N/A"
            return "✓" if val else "✗"
        
        print(f"{r.category:<15} | {r.operator_name:<30} | {fmt_result(r.batch_invariant):^12} | {fmt_result(r.position_invariant):^12}")
    
    print("-" * 90)
    
    # Print LaTeX table
    print(f"\n{'='*90}")
    print("LaTeX Table Output:")
    print(f"{'='*90}")
    print(r"""
\begin{table*}[t]
\centering
\caption{Operator Categorization with Invariance Properties}
\label{tab:operator_invariance}
\begin{tabular}{llcc}
\toprule
\textbf{Category} & \textbf{Operator} & \textbf{Batch-Invariant} & \textbf{Position-Invariant} \\
\midrule""")
    
    current_category = None
    category_count = {}
    
    # Count operators per category
    for r in results:
        category_count[r.category] = category_count.get(r.category, 0) + 1
    
    for r in results:
        def latex_mark(val):
            if val is None:
                return r"\textcolor{gray}{--}"
            return r"\cmark" if val else r"\xmark"
        
        if r.category != current_category:
            if current_category is not None:
                print(r"\midrule")
            current_category = r.category
            count = category_count[r.category]
            if count > 1:
                print(f"\\multirow{{{count}}}{{*}}{{{r.category}}}")
                print(f"& {r.operator_name:<25} & {latex_mark(r.batch_invariant)} & {latex_mark(r.position_invariant)} \\\\")
            else:
                print(f"{r.category} & {r.operator_name:<25} & {latex_mark(r.batch_invariant)} & {latex_mark(r.position_invariant)} \\\\")
        else:
            print(f"& {r.operator_name:<25} & {latex_mark(r.batch_invariant)} & {latex_mark(r.position_invariant)} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table*}""")


def main():
    parser = argparse.ArgumentParser(description="Test operator invariance properties")
    parser.add_argument(
        "--category", "-c",
        type=str,
        nargs="+",
        default=None,
        help="Categories to test: matmul, attention, normalization, embedding, activation, communication"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available tests"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type to use for tests"
    )
    args = parser.parse_args()
    
    # Setup multiprocessing for distributed tests
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
    
    # Setup device and dtype
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    device = torch.device("cuda:0")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    print("=" * 70)
    print("Operator Invariance Test Suite")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create testers
    testers = create_all_testers(device, dtype)
    
    if args.list:
        print("\nAvailable tests:")
        for t in testers:
            print(f"  [{t.category}] {t.name}")
        return
    
    # Run tests
    results = run_tests(testers, args.category)
    
    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
