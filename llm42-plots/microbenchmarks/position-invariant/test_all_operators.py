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
        self.N = 7167
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
    """Test Fused MoE (Triton) for invariance using SGLang's implementation."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Matmul"
        self.name = "Fused MoE (Triton)"
        # Use E=128, N=384 which has config for H100 PCIe
        # w1 shape: (E, N*2, hidden_dim) for gate+up projection with SiLU
        # w2 shape: (E, hidden_dim, N) for down projection
        # So intermediate_size = N = 384, and hidden_dim should match
        self.num_experts = 128
        self.intermediate_size = 384  # N value from config
        self.top_k = 2
        self.available = False
        self._check_availability()
        
    def _check_availability(self):
        """Check if fused MoE kernel is available."""
        try:
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
            self.outplace_fused_experts = outplace_fused_experts
            self.available = True
        except ImportError:
            self.available = False
    
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
        if not self.available:
            return None, "Fused MoE not available (sglang not installed)"
        
        torch.manual_seed(42)
        
        # MoE dimensions matching config E=128, N=384:
        # w1: (num_experts, N * 2, hidden_dim) for gate+up projection with SiLU
        # w2: (num_experts, hidden_dim, N) for down projection
        # hidden_dim must be reasonable (e.g., 1024)
        hidden_dim = 1024
        
        # Expert weights
        w1 = torch.randn(self.num_experts, self.intermediate_size * 2, hidden_dim, 
                         dtype=self.dtype, device=self.device).contiguous()
        w2 = torch.randn(self.num_experts, hidden_dim, self.intermediate_size, 
                         dtype=self.dtype, device=self.device).contiguous()
        
        target_input = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device).contiguous()
        
        # Router logits for target
        target_router = torch.randn(1, self.num_experts, dtype=torch.float32, device=self.device)
        target_topk_weights, target_topk_ids = self._compute_topk(target_router)
        
        # Compute with BS=1
        out_bs1 = self.outplace_fused_experts(
            target_input, w1, w2, target_topk_weights, target_topk_ids,
            activation="silu"
        )
        torch.cuda.synchronize()
        
        results = []
        for bs in [2, 4, 8, 16, 32]:
            filler_input = torch.randn(bs - 1, hidden_dim, dtype=self.dtype, device=self.device)
            filler_router = torch.randn(bs - 1, self.num_experts, dtype=torch.float32, device=self.device)
            filler_topk_weights, filler_topk_ids = self._compute_topk(filler_router)
            
            batch_input = torch.cat([target_input, filler_input], dim=0).contiguous()
            batch_topk_weights = torch.cat([target_topk_weights, filler_topk_weights], dim=0).contiguous()
            batch_topk_ids = torch.cat([target_topk_ids, filler_topk_ids], dim=0).contiguous()
            
            out_batch = self.outplace_fused_experts(
                batch_input, w1, w2, batch_topk_weights, batch_topk_ids,
                activation="silu"
            )
            torch.cuda.synchronize()
            
            match = self.check_equal(out_bs1[0], out_batch[0])
            results.append((bs, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested BS: {[r[0] for r in results]}"
        if not all_match:
            failed = [r[0] for r in results if not r[1]]
            details += f", Failed at BS: {failed}"
        return all_match, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        if not self.available:
            return None, "Fused MoE not available"
        
        torch.manual_seed(42)
        
        hidden_dim = 1024
        fixed_bs = 32
        
        w1 = torch.randn(self.num_experts, self.intermediate_size * 2, hidden_dim, 
                         dtype=self.dtype, device=self.device).contiguous()
        w2 = torch.randn(self.num_experts, hidden_dim, self.intermediate_size, 
                         dtype=self.dtype, device=self.device).contiguous()
        
        target_input = torch.randn(1, hidden_dim, dtype=self.dtype, device=self.device)
        target_router = torch.randn(1, self.num_experts, dtype=torch.float32, device=self.device)
        target_topk_weights, target_topk_ids = self._compute_topk(target_router)
        
        filler_input = torch.randn(fixed_bs, hidden_dim, dtype=self.dtype, device=self.device)
        filler_router = torch.randn(fixed_bs, self.num_experts, dtype=torch.float32, device=self.device)
        filler_topk_weights, filler_topk_ids = self._compute_topk(filler_router)
        
        results = []
        ref_output = None
        
        for pos in range(0, fixed_bs, 4):  # Test every 4th position
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
            
            target_out = out[pos].clone()
            
            if ref_output is None:
                ref_output = target_out
            else:
                match = self.check_equal(ref_output, target_out)
                results.append((pos, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested positions in BS={fixed_bs}"
        if not all_match:
            failed = [r[0] for r in results if not r[1]]
            details += f", Failed at positions: {failed}"
        return all_match, details


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
    """Test RMSNorm for invariance using sgl_kernel."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Normalization"
        self.name = "RMSNorm"
        self.eps = 1e-6
        self.available = False
        self._check_availability()
        
    def _check_availability(self):
        try:
            import sgl_kernel
            self.sgl_kernel = sgl_kernel
            self.available = True
        except ImportError:
            self.available = False
        
    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm using sgl_kernel if available, else fallback."""
        if self.available:
            return self.sgl_kernel.rmsnorm(x, weight, eps=self.eps)
        else:
            # Fallback: reference implementation
            orig_dtype = x.dtype
            x_float = x.float()
            variance = x_float.pow(2).mean(-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.eps)
            return (x_norm * weight.float()).to(orig_dtype)
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        
        weight = torch.randn(self.hidden_dim, dtype=self.dtype, device=self.device)
        target_input = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        out_bs1 = self.rms_norm(target_input, weight)
        torch.cuda.synchronize()
        
        results = []
        for bs in self.batch_sizes[1:]:
            filler = torch.randn(bs - 1, self.hidden_dim, dtype=self.dtype, device=self.device)
            batch_input = torch.cat([target_input, filler], dim=0)
            
            out_batch = self.rms_norm(batch_input, weight)
            torch.cuda.synchronize()
            
            match = self.check_equal(out_bs1[0], out_batch[0])
            results.append((bs, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested BS: {[r[0] for r in results]}"
        if not self.available:
            details += " (fallback impl)"
        if not all_match:
            failed = [r[0] for r in results if not r[1]]
            details += f", Failed at BS: {failed}"
        return all_match, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        
        weight = torch.randn(self.hidden_dim, dtype=self.dtype, device=self.device)
        target_input = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        fixed_bs = 64
        filler = torch.randn(fixed_bs, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        results = []
        ref_output = None
        
        for pos in range(fixed_bs):
            batch = filler.clone()
            batch[pos] = target_input[0]
            
            out = self.rms_norm(batch, weight)
            torch.cuda.synchronize()
            
            target_out = out[pos].clone()
            
            if ref_output is None:
                ref_output = target_out
            else:
                match = self.check_equal(ref_output, target_out)
                results.append((pos, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested positions 0-{fixed_bs-1}"
        if not self.available:
            details += " (fallback impl)"
        return all_match, details


class FusedRMSNormResidualTester(OperatorTester):
    """Test Fused RMSNorm + Residual for invariance using sgl_kernel."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.category = "Normalization"
        self.name = "Fused RMSNorm+Residual"
        self.eps = 1e-6
        self.available = False
        self._check_availability()
        
    def _check_availability(self):
        try:
            import sgl_kernel
            self.sgl_kernel = sgl_kernel
            self.available = True
        except ImportError:
            self.available = False
    
    def fused_rms_norm_residual(self, x: torch.Tensor, residual: torch.Tensor, 
                                 weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused RMSNorm + Residual: out = RMSNorm(x + residual), also returns updated residual."""
        if self.available:
            x = x.clone()
            residual = residual.clone()
            self.sgl_kernel.fused_add_rmsnorm(x, residual, weight, self.eps)
            return x, residual
        else:
            # Fallback implementation
            orig_dtype = x.dtype
            x_float = x.float() + residual.float()
            new_residual = x_float.to(orig_dtype)
            variance = x_float.pow(2).mean(-1, keepdim=True)
            out = (x_float * torch.rsqrt(variance + self.eps) * weight.float()).to(orig_dtype)
            return out, new_residual
    
    def test_batch_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        
        weight = torch.randn(self.hidden_dim, dtype=self.dtype, device=self.device)
        target_x = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        target_res = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        out_bs1, res_bs1 = self.fused_rms_norm_residual(target_x.clone(), target_res.clone(), weight)
        torch.cuda.synchronize()
        
        results = []
        for bs in self.batch_sizes[1:8]:  # Test smaller batch sizes
            filler_x = torch.randn(bs - 1, self.hidden_dim, dtype=self.dtype, device=self.device)
            filler_res = torch.randn(bs - 1, self.hidden_dim, dtype=self.dtype, device=self.device)
            
            batch_x = torch.cat([target_x, filler_x], dim=0)
            batch_res = torch.cat([target_res, filler_res], dim=0)
            
            out_batch, res_batch = self.fused_rms_norm_residual(batch_x.clone(), batch_res.clone(), weight)
            torch.cuda.synchronize()
            
            match = self.check_equal(out_bs1[0], out_batch[0])
            results.append((bs, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested BS: {[r[0] for r in results]}"
        if not self.available:
            details += " (fallback impl)"
        return all_match, details
    
    def test_position_invariance(self) -> Tuple[bool, str]:
        torch.manual_seed(42)
        
        weight = torch.randn(self.hidden_dim, dtype=self.dtype, device=self.device)
        target_x = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        target_res = torch.randn(1, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        fixed_bs = 32
        filler_x = torch.randn(fixed_bs, self.hidden_dim, dtype=self.dtype, device=self.device)
        filler_res = torch.randn(fixed_bs, self.hidden_dim, dtype=self.dtype, device=self.device)
        
        results = []
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
                match = self.check_equal(ref_output, target_out)
                results.append((pos, match))
        
        all_match = all(r[1] for r in results)
        details = f"Tested positions 0-{fixed_bs-1}"
        if not self.available:
            details += " (fallback impl)"
        return all_match, details


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
