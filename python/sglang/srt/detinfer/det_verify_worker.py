# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deterministic verification worker that wraps the target worker at scheduler level."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.detinfer.det_verify_info import DetVerifyInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatchOutput, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker

logger = logging.getLogger(__name__)


class FixedSizeVerificationPool:
    """
    Pre-allocated pool of dummy resources for fixed-size verification batches.
    
    When max_det_verify_batch_size is set, verification batches always have exactly
    N requests. This pool provides pre-allocated dummy resources to fill slots
    when there are fewer than N real requests.
    """
    
    DUMMY_TOKEN_ID = 32  # Match DetVerifyInfo.DUMMY_TOKEN_ID
    
    def __init__(
        self,
        fixed_size: int,
        step_size: int,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        device: str = "cuda",
    ):
        """
        Initialize the fixed-size verification pool.
        
        Args:
            fixed_size: Number of requests in each verification batch (N)
            step_size: Verification step size (tokens per request)
            req_to_token_pool: Pool for request-to-token mapping
            token_to_kv_pool_allocator: KV cache allocator
            device: Device for tensors
        """
        self.fixed_size = fixed_size
        self.step_size = step_size
        self.device = device
        self.req_to_token_pool = req_to_token_pool
        
        # Pre-allocate dummy input_ids (N * step_size tokens)
        total_dummy_tokens = fixed_size * step_size
        self.dummy_input_ids = torch.full(
            (total_dummy_tokens,), 
            self.DUMMY_TOKEN_ID, 
            dtype=torch.int64, 
            device=device
        )
        
        # Pre-allocate KV cache slots for dummy requests
        # Each dummy request needs step_size cache slots
        slots_needed = fixed_size * step_size
        
        # Handle page size alignment
        page_size = token_to_kv_pool_allocator.page_size
        if page_size > 1:
            slots_needed = ((slots_needed + page_size - 1) // page_size) * page_size
        
        self.dummy_cache_locs = token_to_kv_pool_allocator.alloc(slots_needed)
        if self.dummy_cache_locs is None or len(self.dummy_cache_locs) == 0:
            raise RuntimeError(
                f"Failed to allocate {slots_needed} KV cache slots for fixed-size verification pool. "
                f"Consider reducing max_det_verify_batch_size or increasing KV cache size."
            )
        
        # CRITICAL FIX: Allocate actual row indices in req_to_token_pool for dummy requests
        # This is necessary because FlashAttention uses req_pool_indices to look up
        # the page_table from req_to_token_pool.req_to_token[req_pool_indices, :]
        allocated_pool_indices = req_to_token_pool.alloc(fixed_size)
        if allocated_pool_indices is None or len(allocated_pool_indices) < fixed_size:
            raise RuntimeError(
                f"Failed to allocate {fixed_size} req_to_token_pool slots for dummy requests. "
                f"Consider reducing max_det_verify_batch_size or increasing pool size."
            )
        
        self.dummy_req_pool_indices = torch.tensor(
            allocated_pool_indices, dtype=torch.int64, device=device
        )
        self._allocated_pool_indices = allocated_pool_indices  # Keep for freeing
        
        # Write dummy cache locations into req_to_token_pool for each dummy request
        # Each dummy request gets step_size consecutive cache slots
        # FlashAttention only reads page_table[i, 0:cache_seqlens[i]], so we only need
        # to fill the first step_size columns (cache_seqlens for dummies = step_size)
        for i, pool_idx in enumerate(allocated_pool_indices):
            start_slot = i * step_size
            end_slot = start_slot + step_size
            req_to_token_pool.req_to_token[pool_idx, :step_size] = self.dummy_cache_locs[start_slot:end_slot].to(torch.int32)
        
        # Pre-allocate dummy sampling tensors (optimization: avoid creating these per-call)
        self.dummy_temps = torch.zeros((total_dummy_tokens, 1), dtype=torch.float32, device=device)
        self.dummy_top_ps = torch.ones(total_dummy_tokens, dtype=torch.float32, device=device)
        self.dummy_top_ks = torch.full((total_dummy_tokens,), -1, dtype=torch.int32, device=device)
        self.dummy_min_ps = torch.zeros(total_dummy_tokens, dtype=torch.float32, device=device)
        self.dummy_seeds = torch.zeros(total_dummy_tokens, dtype=torch.int32, device=device)
        self.dummy_det_indices = torch.ones((total_dummy_tokens, 1), dtype=torch.int64, device=device)
        
        # Pre-allocate dummy prefix_lens and output_lens (all zeros and step_size respectively)
        self.dummy_prefix_lens = torch.zeros(fixed_size, dtype=torch.int64, device=device)
        self.dummy_output_lens = torch.full((fixed_size,), step_size, dtype=torch.int64, device=device)
        
        logger.info(
            f"FixedSizeVerificationPool initialized: fixed_size={fixed_size}, "
            f"step_size={step_size}, dummy_cache_slots={len(self.dummy_cache_locs)}, "
            f"dummy_pool_indices={allocated_pool_indices}"
        )
    
    def get_dummy_data(self, num_dummies: int):
        """
        Get pre-allocated dummy data for the specified number of dummy requests.
        
        Args:
            num_dummies: Number of dummy requests needed
            
        Returns:
            Tuple of (dummy_input_ids, dummy_cache_locs, dummy_req_pool_indices)
        """
        if num_dummies <= 0:
            return None, None, None
        
        if num_dummies > self.fixed_size:
            raise ValueError(
                f"Requested {num_dummies} dummies but pool only has {self.fixed_size}"
            )
        
        tokens_needed = num_dummies * self.step_size
        cache_slots_needed = num_dummies * self.step_size
        
        return (
            self.dummy_input_ids[:tokens_needed],
            self.dummy_cache_locs[:cache_slots_needed],
            self.dummy_req_pool_indices[:num_dummies],
        )
    
    def get_dummy_sampling_tensors(self, num_dummies: int):
        """
        Get pre-allocated dummy sampling tensors.
        
        Args:
            num_dummies: Number of dummy requests
            
        Returns:
            Tuple of (temps, top_ps, top_ks, min_ps, seeds, det_indices, prefix_lens, output_lens)
            Returns None if num_dummies <= 0
        """
        if num_dummies <= 0:
            return None
        
        tokens_needed = num_dummies * self.step_size
        # Return tuple instead of dict to avoid dict creation overhead
        return (
            self.dummy_temps[:tokens_needed],      # 0: temperatures
            self.dummy_top_ps[:tokens_needed],     # 1: top_ps
            self.dummy_top_ks[:tokens_needed],     # 2: top_ks
            self.dummy_min_ps[:tokens_needed],     # 3: min_ps
            self.dummy_seeds[:tokens_needed],      # 4: seeds
            self.dummy_det_indices[:tokens_needed], # 5: det_indices
            self.dummy_prefix_lens[:num_dummies],  # 6: prefix_lens
            self.dummy_output_lens[:num_dummies],  # 7: output_lens
        )
    
    def free(self, token_to_kv_pool_allocator):
        """Free the pre-allocated KV cache slots and req_to_token_pool slots."""
        if self.dummy_cache_locs is not None and len(self.dummy_cache_locs) > 0:
            token_to_kv_pool_allocator.free(self.dummy_cache_locs)
            self.dummy_cache_locs = None
        
        # Also free the req_to_token_pool slots
        if hasattr(self, '_allocated_pool_indices') and self._allocated_pool_indices is not None:
            self.req_to_token_pool.free(self._allocated_pool_indices)
            self._allocated_pool_indices = None


class DeterministicVerificationWorker:
    """
    Wraps a target worker to provide deterministic verification at scheduler level.
    
    Similar to EagleWorker but for deterministic inference validation:
    - Operates on ScheduleBatch objects (not ModelWorkerBatch)
    - Identifies finished deterministic requests after forward pass
    - Re-runs them with TARGET_DET_VERIFY mode
    - Compares outputs to detect any non-determinism
    """

    def __init__(
        self, 
        target_worker: TpModelWorker, 
        always_align: bool = True,
        max_requests_per_verify: Optional[int] = None,
        metrics_collector = None,
        skip_mismatch: float = 100.0,
        req_to_token_pool = None,
        token_to_kv_pool_allocator = None,
        step_size: Optional[int] = None,
    ):
        """
        Initialize the deterministic verification worker.
        
        Args:
            target_worker: The underlying TpModelWorker to wrap
            always_align: If True, pad verification batches to step_size with dummy tokens
                         for finished requests that have fewer unverified tokens than step_size.
                         This ensures consistent batch sizes for verification. Default: True.
            max_requests_per_verify: Maximum number of requests to verify in a single batch.
                         If None, all requests are verified together. If set, requests are
                         verified in chunks of this size (e.g., 20 requests with max=10
                         will be verified as 10+10). When set with allocators, enables
                         fixed-size batches with padding. Default: None.
            metrics_collector: Optional metrics collector for tracking rollback stats.
            skip_mismatch: Mismatch rate percentage (0.0-100.0).
                         100.0 = normal verification (natural mismatches cause rollback).
                         0.0 = force no mismatches (skip all, for measuring overhead).
                         Values in between (e.g., 5.0) = inject mismatch at position to rollback ceil(5% * window_size) tokens.
            req_to_token_pool: Pool for request-to-token mapping (needed for fixed-size batches).
            token_to_kv_pool_allocator: KV cache allocator (needed for fixed-size batches).
            step_size: Verification step size (needed for fixed-size batches).
        """
        self.target_worker = target_worker
        self.always_align = always_align
        self.max_requests_per_verify = max_requests_per_verify
        self.metrics_collector = metrics_collector
        self.skip_mismatch = skip_mismatch
        
        # Initialize fixed-size verification pool if all required params are provided
        self.fixed_pool: Optional[FixedSizeVerificationPool] = None
        self._step_size = step_size  # Store for late initialization
        if (max_requests_per_verify is not None and 
            req_to_token_pool is not None and 
            token_to_kv_pool_allocator is not None and
            step_size is not None):
            self._init_fixed_pool(
                req_to_token_pool, 
                token_to_kv_pool_allocator,
                getattr(target_worker, 'device', 'cuda')
            )
    
    def _init_fixed_pool(
        self, 
        req_to_token_pool, 
        token_to_kv_pool_allocator,
        device: str = "cuda"
    ):
        """
        Initialize the fixed-size verification pool.
        
        Can be called during __init__ or later via init_fixed_pool() if
        allocators weren't available at construction time.
        """
        if self.max_requests_per_verify is None or self._step_size is None:
            return
        
        try:
            self.fixed_pool = FixedSizeVerificationPool(
                fixed_size=self.max_requests_per_verify,
                step_size=self._step_size,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                device=device,
            )
            logger.info(
                f"Fixed-size verification enabled: batch_size={self.max_requests_per_verify}, "
                f"step_size={self._step_size}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize fixed-size verification pool: {e}. "
                f"Falling back to variable-size batches."
            )
    
    def init_fixed_pool(
        self,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        step_size: int,
        device: str = "cuda",
    ):
        """
        Initialize the fixed-size verification pool after construction.
        
        This is called from the scheduler after memory pools are initialized,
        since the worker is created before init_memory_pool_and_cache().
        
        Args:
            req_to_token_pool: Pool for request-to-token mapping
            token_to_kv_pool_allocator: KV cache allocator
            step_size: Verification step size
            device: Device for tensors
        """
        if self.fixed_pool is not None:
            return  # Already initialized
        
        self._step_size = step_size
        self._init_fixed_pool(req_to_token_pool, token_to_kv_pool_allocator, device)

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        skip_sample: bool = False,
    ) -> ForwardBatchOutput:
        """
        Forward pass - just delegates to target worker.
        Verification happens later in process_batch_result_decode.
        
        Args:
            batch: Input ScheduleBatch or ModelWorkerBatch
            
        Returns:
            ForwardBatchOutput object
        """

        return self.target_worker.forward_batch_generation(batch, skip_sample=skip_sample)

    def check_and_verify_deterministic_requests(
        self, 
        batch: Union[ScheduleBatch, ModelWorkerBatch]
    ) -> List[Tuple[Req, int]]:
        """
        Check for deterministic requests that need verification and verify them.
        Should be called AFTER tokens have been appended and check_finished() called.
        
        This is called from process_batch_result_decode, similar to how Eagle
        handles verification after output processing.
        
        When fixed_pool is enabled (max_det_verify_batch_size is set with allocators),
        verification runs with exactly N requests by:
        1. Including all deterministic requests (ready or not, padding partial outputs)
        2. Filling remaining slots with pre-allocated dummy requests
        
        Args:
            batch: Current ScheduleBatch or ModelWorkerBatch after output processing
            
        Returns:
            List of (req, tokens_rolled_back) tuples for requests that had rollback.
            The caller should use this to free KV cache slots.
        """
        if batch.reqs is None:
            return []
        
        # Collect all deterministic requests and identify which ones need verification
        all_det_reqs = []
        at_least_one_ready = False
        
        for req in batch.reqs:
            if not req.is_deterministic:
                continue
            
            # Skip verification if force_deterministic_mode is True
            if getattr(req, 'force_deterministic_mode', False):
                req.det_verified_tokens = len(req.output_ids)
                continue
            
            # Calculate unverified tokens
            output_len = len(req.output_ids)
            unverified_tokens = output_len - req.det_verified_tokens
            
            if unverified_tokens <= 0:
                continue
            
            all_det_reqs.append(req)
            
            # Check if this request triggers verification
            is_finished = req.finished_reason is not None
            if is_finished or (req.det_step_size is not None and unverified_tokens >= req.det_step_size):
                at_least_one_ready = True
        
        if not at_least_one_ready:
            return []
        
        # Fixed-size batch mode: run with exactly fixed_size requests per batch
        if self.fixed_pool is not None:
            fixed_size = self.fixed_pool.fixed_size
            
            # Separate ready and not-ready requests
            ready_reqs = []
            not_ready_reqs = []
            for req in all_det_reqs:
                output_len = len(req.output_ids)
                unverified_tokens = output_len - req.det_verified_tokens
                is_finished = req.finished_reason is not None
                if is_finished or (req.det_step_size is not None and unverified_tokens >= req.det_step_size):
                    ready_reqs.append(req)
                else:
                    not_ready_reqs.append(req)
            
            all_rollback_results = []
            
            # Process all ready requests in batches of exactly fixed_size
            # Each batch: ready requests first, then not-ready to fill, then dummies
            ready_idx = 0
            not_ready_idx = 0
            
            while ready_idx < len(ready_reqs):
                batch_reqs = []
                
                # Add ready requests up to fixed_size
                while len(batch_reqs) < fixed_size and ready_idx < len(ready_reqs):
                    batch_reqs.append(ready_reqs[ready_idx])
                    ready_idx += 1
                
                # Fill remaining slots with not-ready requests
                while len(batch_reqs) < fixed_size and not_ready_idx < len(not_ready_reqs):
                    batch_reqs.append(not_ready_reqs[not_ready_idx])
                    not_ready_idx += 1
                
                # Calculate dummies needed for remaining slots
                num_dummies = max(0, fixed_size - len(batch_reqs))
                
                rollback_results = self._verify_fixed_batch(batch, batch_reqs, num_dummies)
                all_rollback_results.extend(rollback_results)
            
            return all_rollback_results
        
        # Variable-size batch mode: only verify ready requests
        reqs_to_verify = []
        for req in all_det_reqs:
            output_len = len(req.output_ids)
            unverified_tokens = output_len - req.det_verified_tokens
            is_finished = req.finished_reason is not None
            if is_finished or (req.det_step_size is not None and unverified_tokens >= req.det_step_size):
                reqs_to_verify.append(req)
        
        if reqs_to_verify:
            return self._verify_deterministic_requests_batched(
                batch, reqs_to_verify, self.always_align, self.max_requests_per_verify
            )
        return []
    
    def _verify_fixed_batch(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        real_reqs: List[Req],
        num_dummies: int,
    ) -> List[Tuple[Req, int]]:
        """
        Verify requests in a fixed-size batch with dummy padding.
        
        All real requests are included (ready or not), padded to step_size.
        Remaining slots are filled with pre-allocated dummy requests.
        
        Args:
            original_batch: Original batch context
            real_reqs: All deterministic requests to include (must be <= fixed_size)
            num_dummies: Number of dummy requests to add
            
        Returns:
            List of (req, tokens_rolled_back) tuples for requests that had rollback.
        """
        try:
            # Create DetVerifyInfo with force_include_all=True to include not-ready requests
            det_verify_info = DetVerifyInfo.from_requests(
                real_reqs, 
                always_align=self.always_align,
                force_include_all=True,  # Include requests even if not at step_size boundary
            )
            
            # Append dummy entries if needed
            if num_dummies > 0:
                det_verify_info.append_dummy_entries(num_dummies, self.fixed_pool.step_size)
            
            # Allocate temporary KV cache for padding tokens of real requests
            if det_verify_info.total_padding_cache_slots > 0:
                det_verify_info.allocate_padding_kv_cache(
                    original_batch.token_to_kv_pool_allocator
                )
            
            # Prepare verification batch with dummy data and pre-allocated sampling tensors
            dummy_input_ids, dummy_cache_locs, dummy_req_pool_indices = self.fixed_pool.get_dummy_data(num_dummies)
            dummy_sampling_tuple = self.fixed_pool.get_dummy_sampling_tensors(num_dummies) if num_dummies > 0 else None
            verify_batch = det_verify_info.prepare_verify_batch(
                original_batch, 
                real_reqs,
                dummy_input_ids=dummy_input_ids,
                dummy_cache_locs=dummy_cache_locs,
                dummy_req_pool_indices=dummy_req_pool_indices,
                num_dummies=num_dummies,
                step_size=self.fixed_pool.step_size,
                dummy_sampling_tuple=dummy_sampling_tuple,
            )
            
            # Run verification forward pass
            verify_model_worker_batch = verify_batch.get_model_worker_batch()
            verify_output = self.target_worker.forward_batch_generation(verify_model_worker_batch)
            
            # Extract results
            verified_token_ids = verify_output.next_token_ids
            verified_logprobs = verify_output.logits_output.next_token_logprobs
            
            # Compare outputs (only for real requests, dummies are ignored via padding_masks)
            if self.skip_mismatch >= 100.0:
                rollback_info = det_verify_info.verify_and_compare(
                    real_reqs, verified_token_ids, verified_logprobs
                )
            elif self.skip_mismatch <= 0.0:
                rollback_info = [(len(req.output_ids) - req.det_verified_tokens, 0) for req in real_reqs]
            else:
                rollback_info = det_verify_info.verify_and_compare(
                    real_reqs, verified_token_ids, verified_logprobs,
                    mismatch_percentage=self.skip_mismatch
                )
            
            # Collect rollback results
            rollback_results = []
            for req, info in zip(real_reqs, rollback_info):
                if info is not None and info[1] > 0:
                    req.det_num_rollbacks += 1
                    req.det_tokens_rolled_back += info[1]
                    if self.metrics_collector:
                        self.metrics_collector.num_rollbacks_total.labels(**self.metrics_collector.labels).inc()
                        self.metrics_collector.tokens_rolled_back_total.labels(**self.metrics_collector.labels).inc(info[1])
                    rollback_results.append((req, info[1]))
                    req.det_verified_tokens = len(req.output_ids)
                else:
                    req.det_verified_tokens = len(req.output_ids)
            
            # Update batch state if there was rollback
            if rollback_results:
                self._update_batch_state(original_batch)
            
            # Cleanup
            verify_batch.out_cache_loc = None
            verify_batch.input_ids = None
            verify_batch.seq_lens = None
            verify_batch.seq_lens_cpu = None
            verify_batch.req_pool_indices = None
            
            if hasattr(verify_batch, 'sampling_info') and verify_batch.sampling_info is not None:
                verify_batch.sampling_info.temperatures = None
                verify_batch.sampling_info.top_ps = None
                verify_batch.sampling_info.top_ks = None
                verify_batch.sampling_info.min_ps = None
                verify_batch.sampling_info.sampling_seed = None
                verify_batch.sampling_info.deterministic_indices = None
            
            verify_batch.req_to_token_pool = None
            verify_batch.token_to_kv_pool_allocator = None
            
            # Free temporary KV cache for real request padding
            if det_verify_info.total_padding_cache_slots > 0:
                det_verify_info.free_padding_kv_cache(
                    original_batch.token_to_kv_pool_allocator
                )
            
            return rollback_results
            
        except Exception as e:
            if 'det_verify_info' in locals() and det_verify_info.total_padding_cache_slots > 0:
                try:
                    det_verify_info.free_padding_kv_cache(
                        original_batch.token_to_kv_pool_allocator
                    )
                except Exception:
                    pass
            logger.error(f"Error during fixed-batch verification: {e}")
            raise

    def _verify_deterministic_requests_batched(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        reqs: List[Req],
        always_align: bool = True,
        max_requests: Optional[int] = None,
    ) -> List[Tuple[Req, int]]:
        """
        Verify deterministic requests in batches.
        
        If max_requests is set, requests are verified in chunks of that size.
        For example, 22 requests with max_requests=10 will be verified as 10+10+2.
        
        Args:
            original_batch: Original batch context
            reqs: All requests to verify
            always_align: If True, pad to step_size with dummy tokens
            max_requests: Maximum requests per verification batch. None means all at once.
            
        Returns:
            List of (req, tokens_rolled_back) tuples for all requests that had rollback.
        """
        if max_requests is None or len(reqs) <= max_requests:
            # No batching needed, verify all at once
            return self._verify_deterministic_requests(original_batch, reqs, always_align)
        
        # Split into batches and verify each
        all_rollback_results = []
        for i in range(0, len(reqs), max_requests):
            batch_reqs = reqs[i:i + max_requests]
            rollback_results = self._verify_deterministic_requests(
                original_batch, batch_reqs, always_align
            )
            all_rollback_results.extend(rollback_results)
        
        return all_rollback_results


    def _verify_deterministic_requests(
        self, 
        original_batch: Union[ScheduleBatch, ModelWorkerBatch], 
        reqs: List[Req],
        always_align: bool = True,
    ) -> List[Tuple[Req, int]]:
        """
        Verify deterministic requests by re-running them.
        Tokens are already in req.output_ids at this point.
        
        Args:
            original_batch: Original batch context
            reqs: Requests to verify (tokens already appended)
            always_align: If True, pad to step_size with dummy tokens for finished requests
            
        Returns:
            List of (req, tokens_rolled_back) tuples for requests that had rollback.
        """
        try:
            det_verify_info = DetVerifyInfo.from_requests(reqs, always_align=always_align)
            # logger.info(f"[DetVerifyWorker] Verifying {len(reqs)} requests with total padding cache slots {det_verify_info.total_padding_cache_slots}")
            
            # Allocate temporary KV cache for padding tokens (if any)
            if det_verify_info.total_padding_cache_slots > 0:
                det_verify_info.allocate_padding_kv_cache(
                    original_batch.token_to_kv_pool_allocator
                )
            
            verify_batch = det_verify_info.prepare_verify_batch(original_batch, reqs)
            
            # STEP_DEBUG = 1
            # ground_truth_token_ids = None
            # for sd in range(STEP_DEBUG):
                # logger.info(f"[DetVerifyWorker] verify_batch seq_lens step {sd}: {verify_batch.seq_lens}")
                # Run verification forward pass (batch_invariant context is now managed in tp_worker)
            verify_model_worker_batch = verify_batch.get_model_worker_batch()
            verify_output = self.target_worker.forward_batch_generation(verify_model_worker_batch)
                
                
                # Extract results (sampling already done inside forward_batch_generation)
            verified_token_ids = verify_output.next_token_ids
                # if sd == 0:
                #     ground_truth_token_ids = verified_token_ids.clone()
                # else:
                #     # Compare with ground truth tokens from first step
                #     if not torch.equal(verified_token_ids, ground_truth_token_ids):
                #         logger.error(f"[DetVerifyWorker][ERROR] Mismatch in verified tokens at step {sd}")
                #         logger.error(f"Ground truth tokens: {ground_truth_token_ids}")
                #         logger.error(f"Current tokens: {verified_token_ids}")
                #         raise ValueError("Deterministic verification failed: token mismatch across steps")
            verified_logprobs = verify_output.logits_output.next_token_logprobs
            
            # Compare outputs and handle rollback
            # Handle skip_mismatch percentage: 100.0=normal, 0.0=force no mismatches, in-between=inject at calculated position
            if self.skip_mismatch >= 100.0:
                # Normal verification - natural mismatches cause rollback
                rollback_info = det_verify_info.verify_and_compare(
                    reqs, verified_token_ids, verified_logprobs
                )
            elif self.skip_mismatch <= 0.0:
                # Force no mismatches - skip all rollbacks
                rollback_info = [(len(req.output_ids) - req.det_verified_tokens, 0) for req in reqs]
            else:
                # Percentage-based: inject mismatch at position (window - ceil(X% * window))
                rollback_info = det_verify_info.verify_and_compare(
                    reqs, verified_token_ids, verified_logprobs,
                    mismatch_percentage=self.skip_mismatch
                )
            
            # Collect rollback info for KV cache freeing (will be done by caller)
            rollback_results = []
            for req, info in zip(reqs, rollback_info):
                if info is not None and info[1] > 0:
                    # Track per-request stats
                    req.det_num_rollbacks += 1
                    req.det_tokens_rolled_back += info[1]
                    # Track global metrics
                    if self.metrics_collector:
                        self.metrics_collector.num_rollbacks_total.labels(**self.metrics_collector.labels).inc()
                        self.metrics_collector.tokens_rolled_back_total.labels(**self.metrics_collector.labels).inc(info[1])
                    rollback_results.append((req, info[1]))
                    # CRITICAL: Update det_verified_tokens after rollback
                    # After rollback, output_ids has been truncated and corrected token appended
                    # All tokens up to current length are now verified
                    req.det_verified_tokens = len(req.output_ids)
                    # logger.info(f"[DetVerifyWorker] Rollback for req {req.rid[:8]}: "
                    #            f"mismatch_pos={info[0]}, tokens_rolled_back={info[1]}, "
                    #            f"new_output_len={len(req.output_ids)}, det_verified_tokens={req.det_verified_tokens}, "
                    #            f"finished={req.finished_reason is not None}")
                else:
                    # No rollback for this request
                    req.det_verified_tokens = len(req.output_ids)
                    # logger.debug(f"[DetVerifyWorker] No rollback for req {req.rid[:8]}: "
                    #             f"output_len={len(req.output_ids)}, det_verified_tokens={req.det_verified_tokens}")
            
            # Update verified token counts
            # for req in reqs:
            #     req.det_verified_tokens = len(req.output_ids)
            
            # Update batch state if there was any rollback
            # (need to sync seq_lens with the new req.output_ids length)
            if rollback_results:
                #logger.info(f"[DEBUG][DetVerifyWorker] Rollback detected, updating batch state")
                self._update_batch_state(original_batch)
            
            # Clear verification batch references to free GPU memory
            # Critical: verify_batch shares KV cache allocator with original_batch
            # At high batch sizes (BS=63-64), we need to explicitly clear all tensor
            # references to prevent memory accumulation
            verify_batch.out_cache_loc = None
            verify_batch.input_ids = None
            verify_batch.seq_lens = None
            verify_batch.seq_lens_cpu = None
            verify_batch.req_pool_indices = None
            
            # Clear sampling_info tensors if present
            if hasattr(verify_batch, 'sampling_info') and verify_batch.sampling_info is not None:
                verify_batch.sampling_info.temperatures = None
                verify_batch.sampling_info.top_ps = None
                verify_batch.sampling_info.top_ks = None
                verify_batch.sampling_info.min_ps = None
                verify_batch.sampling_info.sampling_seed = None
                verify_batch.sampling_info.deterministic_indices = None
            
            # Clear shared resource references (don't free, just remove refs)
            verify_batch.req_to_token_pool = None
            verify_batch.token_to_kv_pool_allocator = None
            
            # Free temporary KV cache allocated for padding tokens
            if det_verify_info.total_padding_cache_slots > 0:
                det_verify_info.free_padding_kv_cache(
                    original_batch.token_to_kv_pool_allocator
                )
            
            return rollback_results
            
        except Exception as e:
            # Make sure to free padding KV cache even on error
            if 'det_verify_info' in locals() and det_verify_info.total_padding_cache_slots > 0:
                try:
                    det_verify_info.free_padding_kv_cache(
                        original_batch.token_to_kv_pool_allocator
                    )
                except Exception:
                    pass  # Best effort cleanup
            logger.error(f"Error during verification: {e}")
            raise

    def _update_batch_state(self, batch: Union[ScheduleBatch, ModelWorkerBatch]):
        """
        Update batch.output_ids and seq_lens to reflect current req.output_ids.
        Critical for prepare_for_decode() which uses these tensors.
        """
        if not hasattr(batch, 'output_ids') or batch.output_ids is None:
            return
        
        updated_output_ids = []
        updated_seq_lens = []
        
        for req in batch.reqs:
            # Get last generated token or fallback to last input token
            if req.output_ids:
                updated_output_ids.append(req.output_ids[-1])
            else:
                updated_output_ids.append(req.origin_input_ids[-1] if req.origin_input_ids else 0)
            
            # seq_lens should be length BEFORE current token (prepare_for_decode increments by 1)
            current_total_len = len(req.origin_input_ids) + len(req.output_ids)
            updated_seq_lens.append(current_total_len - 1)
        
        # Update batch tensors
        batch.output_ids = torch.tensor(
            updated_output_ids, 
            dtype=torch.int64, 
            device=batch.device
        )
        batch.seq_lens = torch.tensor(
            updated_seq_lens,
            dtype=torch.int32,
            device=batch.device
        )
        batch.seq_lens_cpu = torch.tensor(updated_seq_lens, dtype=torch.int32)
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.seq_lens_sum = sum(updated_seq_lens)

    def __getattr__(self, name):
        """
        Forward all other attributes to target_worker.
        
        This allows DeterministicVerificationWorker to act as a transparent
        wrapper - any method/attribute not defined here is delegated to the
        underlying target_worker.
        """
        return getattr(self.target_worker, name)
