# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ==============================================================================
"""Scheduler-level verification worker for the LLM-42 DVR protocol.

This module provides two classes:

- ``FixedSizeVerificationPool``: Pre-allocated pool of dummy resources that
  ensures every verification batch has exactly the same shape (for
  position-invariant determinism via grouped verification, §4.3).

- ``LLM42Worker``: Transparent wrapper around a
  ``TpModelWorker`` that intercepts decode batches, identifies requests
  needing verification, constructs a TARGET_LLM42_VERIFY batch, compares
  outputs, and applies rollback on mismatch.

See §4 of the LLM-42 paper (arXiv:2601.17768) for design details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.llm42.llm42_info import LLM42Info
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker

logger = logging.getLogger(__name__)


class FixedSizeVerificationPool:
    """Pre-allocated dummy resources for fixed-shape verification batches.

    Grouped verification (§4.3) requires that every verification pass processes
    exactly ``fixed_size`` requests of ``window_size`` tokens each.  When fewer
    real requests are available, this pool supplies pre-allocated dummy entries
    (input IDs, KV-cache slots, req-pool indices, and sampling tensors) to pad
    the batch to the required shape.
    """
    
    DUMMY_TOKEN_ID = 32  # Match LLM42Info.DUMMY_TOKEN_ID
    
    def __init__(
        self,
        fixed_size: int,
        window_size: int,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        device: str = "cuda",
    ):
        """
        Initialize the fixed-size verification pool.
        
        Args:
            fixed_size: Number of requests in each verification batch (N)
            window_size: Verification step size (tokens per request)
            req_to_token_pool: Pool for request-to-token mapping
            token_to_kv_pool_allocator: KV cache allocator
            device: Device for tensors
        """
        self.fixed_size = fixed_size
        self.window_size = window_size
        self.device = device
        self.req_to_token_pool = req_to_token_pool
        
        # Pre-allocate dummy input_ids (N * window_size tokens)
        total_dummy_tokens = fixed_size * window_size
        self.dummy_input_ids = torch.full(
            (total_dummy_tokens,), 
            self.DUMMY_TOKEN_ID, 
            dtype=torch.int64, 
            device=device
        )
        
        # Pre-allocate KV cache slots for dummy requests
        # Each dummy request needs window_size cache slots
        slots_needed = fixed_size * window_size
        
        # Handle page size alignment
        page_size = token_to_kv_pool_allocator.page_size
        if page_size > 1:
            slots_needed = ((slots_needed + page_size - 1) // page_size) * page_size
        
        self.dummy_cache_locs = token_to_kv_pool_allocator.alloc(slots_needed)
        if self.dummy_cache_locs is None or len(self.dummy_cache_locs) == 0:
            raise RuntimeError(
                f"Failed to allocate {slots_needed} KV cache slots for fixed-size verification pool. "
                f"Consider reducing llm42_verify_batch_size or increasing KV cache size."
            )
        
        # CRITICAL FIX: Allocate actual row indices in req_to_token_pool for dummy requests
        # This is necessary because FlashAttention uses req_pool_indices to look up
        # the page_table from req_to_token_pool.req_to_token[req_pool_indices, :]
        # Note: ReqToTokenPool.alloc() expects list[Req] in the new API, so we
        # allocate directly from free_slots for dummy padding requests.
        if len(req_to_token_pool.free_slots) < fixed_size:
            raise RuntimeError(
                f"Failed to allocate {fixed_size} req_to_token_pool slots for dummy requests. "
                f"Consider reducing llm42_verify_batch_size or increasing pool size."
            )
        allocated_pool_indices = req_to_token_pool.free_slots[:fixed_size]
        req_to_token_pool.free_slots = req_to_token_pool.free_slots[fixed_size:]
        
        self.dummy_req_pool_indices = torch.tensor(
            allocated_pool_indices, dtype=torch.int64, device=device
        )
        self._allocated_pool_indices = allocated_pool_indices  # Keep for freeing
        
        # Write dummy cache locations into req_to_token_pool for each dummy request
        # Each dummy request gets window_size consecutive cache slots
        # FlashAttention only reads page_table[i, 0:cache_seqlens[i]], so we only need
        # to fill the first window_size columns (cache_seqlens for dummies = window_size)
        for i, pool_idx in enumerate(allocated_pool_indices):
            start_slot = i * window_size
            end_slot = start_slot + window_size
            req_to_token_pool.req_to_token[pool_idx, :window_size] = self.dummy_cache_locs[start_slot:end_slot].to(torch.int32)
        
        # Pre-allocate dummy sampling tensors (optimization: avoid creating these per-call)
        self.dummy_temps = torch.zeros((total_dummy_tokens, 1), dtype=torch.float32, device=device)
        self.dummy_top_ps = torch.ones(total_dummy_tokens, dtype=torch.float32, device=device)
        self.dummy_top_ks = torch.full((total_dummy_tokens,), -1, dtype=torch.int32, device=device)
        self.dummy_min_ps = torch.zeros(total_dummy_tokens, dtype=torch.float32, device=device)
        self.dummy_seeds = torch.zeros(total_dummy_tokens, dtype=torch.int32, device=device)
        self.dummy_det_indices = torch.ones((total_dummy_tokens, 1), dtype=torch.int64, device=device)
        
        # Pre-allocate dummy prefix_lens and output_lens (all zeros and window_size respectively)
        self.dummy_prefix_lens = torch.zeros(fixed_size, dtype=torch.int64, device=device)
        self.dummy_output_lens = torch.full((fixed_size,), window_size, dtype=torch.int64, device=device)
        
        # Pre-allocate KV cache slots for padding tokens of real requests.
        # Worst case: every real request has 1 unverified token and needs
        # (window_size - 1) padding slots → fixed_size * (window_size - 1).
        max_padding_slots = fixed_size * (window_size - 1)
        if max_padding_slots > 0:
            padding_slots_to_alloc = max_padding_slots
            if page_size > 1:
                padding_slots_to_alloc = (
                    (max_padding_slots + page_size - 1) // page_size
                ) * page_size
            self.padding_cache_pool = token_to_kv_pool_allocator.alloc(padding_slots_to_alloc)
            if self.padding_cache_pool is None or len(self.padding_cache_pool) == 0:
                raise RuntimeError(
                    f"Failed to allocate {padding_slots_to_alloc} KV cache slots for padding pool. "
                    f"Consider reducing llm42_verify_batch_size or increasing KV cache size."
                )
            # Trim to exact amount needed (discard page-alignment surplus)
            if len(self.padding_cache_pool) > max_padding_slots:
                self._padding_cache_pool_allocated = self.padding_cache_pool
                self.padding_cache_pool = self.padding_cache_pool[:max_padding_slots]
            else:
                self._padding_cache_pool_allocated = self.padding_cache_pool
        else:
            self.padding_cache_pool = None
            self._padding_cache_pool_allocated = None
        
        logger.info(
            f"FixedSizeVerificationPool initialized: fixed_size={fixed_size}, "
            f"window_size={window_size}, dummy_cache_slots={len(self.dummy_cache_locs)}, "
            f"padding_cache_slots={max_padding_slots}, "
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
        
        tokens_needed = num_dummies * self.window_size
        cache_slots_needed = num_dummies * self.window_size
        
        return (
            self.dummy_input_ids[:tokens_needed],
            self.dummy_cache_locs[:cache_slots_needed],
            self.dummy_req_pool_indices[:num_dummies],
        )
    
    def get_padding_cache_locs(self, num_slots: int) -> Optional[torch.Tensor]:
        """
        Get a slice of pre-allocated padding KV cache slots.
        
        Args:
            num_slots: Number of padding cache slots needed
            
        Returns:
            Tensor of cache locations, or None if no slots needed or pool unavailable
        """
        if num_slots <= 0:
            return None
        if self.padding_cache_pool is None:
            raise RuntimeError(
                f"Padding cache pool not allocated but {num_slots} slots requested"
            )
        if num_slots > len(self.padding_cache_pool):
            raise RuntimeError(
                f"Requested {num_slots} padding slots but pool only has "
                f"{len(self.padding_cache_pool)}"
            )
        return self.padding_cache_pool[:num_slots]

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
        
        tokens_needed = num_dummies * self.window_size
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
        
        # Free pre-allocated padding cache pool
        if self._padding_cache_pool_allocated is not None and len(self._padding_cache_pool_allocated) > 0:
            token_to_kv_pool_allocator.free(self._padding_cache_pool_allocated)
            self._padding_cache_pool_allocated = None
            self.padding_cache_pool = None
        
        # Also free the req_to_token_pool slots
        if hasattr(self, '_allocated_pool_indices') and self._allocated_pool_indices is not None:
            self.req_to_token_pool.free_slots.extend(self._allocated_pool_indices)
            self._allocated_pool_indices = None


# ======================================================================
# Verification worker
# ======================================================================


class LLM42Worker:
    """Scheduler-level wrapper that adds decode-verify-rollback (DVR) to a
    ``TpModelWorker``.

    The worker is transparent for the decode fast-path: ``forward_batch_generation``
    simply delegates to the underlying worker.  After each decode step, the
    scheduler calls ``check_and_verify_deterministic_requests`` which:

    1. Collects all deterministic requests that have accumulated enough
       unverified tokens (>= ``window_size``) or have finished.
    2. Constructs a TARGET_LLM42_VERIFY batch and runs a verification forward
       pass under fixed-shape reductions.
    3. Compares verification outputs with the decode-phase tokens.
    4. Rolls back on mismatch: truncates ``output_ids``, appends the verifier
       token, and frees excess KV-cache slots.
    """

    def __init__(
        self, 
        target_worker: TpModelWorker, 
        always_align: bool = True,
        fixed_requests_per_verify: int = 16,
        metrics_collector = None,
        skip_mismatch: float = 100.0,
        req_to_token_pool = None,
        token_to_kv_pool_allocator = None,
        window_size: int = 32,
    ):
        """
        Initialize the deterministic verification worker.
        
        Args:
            target_worker: The underlying TpModelWorker to wrap
            always_align: If True, pad verification batches to window_size with dummy tokens
                         for finished requests that have fewer unverified tokens than window_size.
                         This ensures consistent batch sizes for verification. Default: True.
            fixed_requests_per_verify: Fixed number of requests per verification batch (padded
                         with dummies if needed). Requests are verified in batches of exactly
                         this size (e.g., 22 requests with fixed_requests_per_verify=10 will be
                         verified as 10+10+10, where the last batch has 2 real + 8 dummies).
                         Default: 16.
            metrics_collector: Optional metrics collector for tracking rollback stats.
            skip_mismatch: Mismatch rate percentage (0.0-100.0).
                         100.0 = normal verification (natural mismatches cause rollback).
                         0.0 = force no mismatches (skip all, for measuring overhead).
                         Values in between (e.g., 5.0) = inject mismatch at position to rollback ceil(5% * window_size) tokens.
            req_to_token_pool: Pool for request-to-token mapping (needed for fixed-size batches).
            token_to_kv_pool_allocator: KV cache allocator (needed for fixed-size batches).
            window_size: Number of tokens before verification. Default: 32.
        """
        self.target_worker = target_worker
        self.always_align = always_align
        self.fixed_requests_per_verify = fixed_requests_per_verify
        self.metrics_collector = metrics_collector
        self.skip_mismatch = skip_mismatch
        
        # Initialize fixed-size verification pool if allocators are provided
        # (they may be None here, in which case init_fixed_pool() will be called later)
        self.fixed_pool: Optional[FixedSizeVerificationPool] = None
        self._window_size = window_size
        if req_to_token_pool is not None and token_to_kv_pool_allocator is not None:
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
        try:
            self.fixed_pool = FixedSizeVerificationPool(
                fixed_size=self.fixed_requests_per_verify,
                window_size=self._window_size,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                device=device,
            )
            logger.info(
                f"Fixed-size verification enabled: batch_size={self.fixed_requests_per_verify}, "
                f"window_size={self._window_size}"
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
        window_size: int,
        device: str = "cuda",
    ):
        """
        Initialize the fixed-size verification pool after construction.
        
        This is called from the scheduler after memory pools are initialized,
        since the worker is created before init_memory_pool_and_cache().
        
        Args:
            req_to_token_pool: Pool for request-to-token mapping
            token_to_kv_pool_allocator: KV cache allocator
            window_size: Verification step size
            device: Device for tensors
        """
        if self.fixed_pool is not None:
            return  # Already initialized
        
        self._window_size = window_size
        self._init_fixed_pool(req_to_token_pool, token_to_kv_pool_allocator, device)

    def forward_batch_generation(
        self,
        model_worker_batch=None,
        **kwargs,
    ) -> GenerationBatchResult:
        """
        Forward pass - just delegates to target worker.
        Verification happens later in process_batch_result_decode.
        
        Args:
            model_worker_batch: Input ModelWorkerBatch
            
        Returns:
            GenerationBatchResult object
        """

        return self.target_worker.forward_batch_generation(model_worker_batch, **kwargs)

    def check_and_verify_deterministic_requests(
        self, 
        batch: Union[ScheduleBatch, ModelWorkerBatch]
    ) -> List[Tuple[Req, int]]:
        """
        Check for deterministic requests that need verification and verify them.
        Should be called AFTER tokens have been appended and check_finished() called.
        
        This is called from process_batch_result_decode, similar to how Eagle
        handles verification after output processing.
        
        When fixed_pool is enabled (llm42_verify_batch_size is set with allocators),
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
            
            # Calculate unverified tokens
            output_len = len(req.output_ids)
            unverified_tokens = output_len - req.llm42_verified_tokens
            
            if unverified_tokens <= 0:
                continue
            
            all_det_reqs.append(req)
            
            # Check if this request triggers verification
            is_finished = req.finished_reason is not None
            if is_finished or unverified_tokens >= self._window_size:
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
                unverified_tokens = output_len - req.llm42_verified_tokens
                is_finished = req.finished_reason is not None
                if is_finished or unverified_tokens >= self._window_size:
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
            unverified_tokens = output_len - req.llm42_verified_tokens
            is_finished = req.finished_reason is not None
            if is_finished or unverified_tokens >= self._window_size:
                reqs_to_verify.append(req)
        
        if reqs_to_verify:
            return self._verify_deterministic_requests_batched(
                batch, reqs_to_verify, self.always_align, self.fixed_requests_per_verify
            )
        return []
    
    def _verify_fixed_batch(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        real_reqs: List[Req],
        num_dummies: int,
    ) -> List[Tuple[Req, int]]:
        """Verify a fixed-size batch (real requests + dummy padding).

        Thin wrapper around :meth:`_run_verification` that adds dummy entries
        from the pre-allocated :class:`FixedSizeVerificationPool`.
        """
        return self._run_verification(
            original_batch,
            real_reqs,
            force_include_all=True,
            num_dummies=num_dummies,
        )

    def _verify_deterministic_requests_batched(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        reqs: List[Req],
        always_align: bool = True,
        max_requests: Optional[int] = None,
    ) -> List[Tuple[Req, int]]:
        """Verify requests in variable-size chunks of at most ``max_requests``."""
        if max_requests is None or len(reqs) <= max_requests:
            return self._run_verification(original_batch, reqs)

        all_rollback_results = []
        for i in range(0, len(reqs), max_requests):
            all_rollback_results.extend(
                self._run_verification(original_batch, reqs[i:i + max_requests])
            )
        return all_rollback_results

    # ------------------------------------------------------------------
    # Core verification pipeline (shared by fixed-size and variable-size)
    # ------------------------------------------------------------------

    def _run_verification(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        reqs: List[Req],
        force_include_all: bool = False,
        num_dummies: int = 0,
    ) -> List[Tuple[Req, int]]:
        """Run the full verify-and-rollback pipeline for a single batch.

        This is the single implementation behind both ``_verify_fixed_batch``
        (grouped verification with dummies) and the variable-size path.

        Args:
            original_batch: Decode-phase batch (provides KV pools etc.).
            reqs: Real requests to verify.
            force_include_all: If True, include requests even if they have not
                yet accumulated ``window_size`` unverified tokens (used for
                fixed-size batches where partial requests are padded).
            num_dummies: Number of dummy requests to append from the
                :class:`FixedSizeVerificationPool` (0 for variable-size path).

        Returns:
            List of ``(req, tokens_rolled_back)`` for requests that were
            rolled back.
        """
        try:
            # 1. Build verification metadata
            llm42_info = LLM42Info.from_requests(
                reqs,
                always_align=self.always_align,
                force_include_all=force_include_all,
                window_size=self._window_size,
            )

            if num_dummies > 0:
                llm42_info.append_dummy_entries(num_dummies, self.fixed_pool.window_size)

            # 2. Assign pre-allocated padding KV cache slots (or allocate on the fly
            #    for the variable-size path which has no fixed_pool)
            if llm42_info.total_padding_cache_slots > 0:
                if self.fixed_pool is not None:
                    padding_locs = self.fixed_pool.get_padding_cache_locs(
                        llm42_info.total_padding_cache_slots
                    )
                    llm42_info.set_padding_cache_locs(padding_locs)
                else:
                    alloc_ok = llm42_info.allocate_padding_kv_cache(
                        original_batch.token_to_kv_pool_allocator
                    )
                    if not alloc_ok:
                        logger.error(
                            "FATAL: Padding KV cache allocation failed during LLM-42 verification. "
                            "This breaks the fixed-batch-shape invariant required for deterministic inference. "
                            "The server will shut down to prevent silent non-determinism."
                        )
                        import os, signal
                        os.kill(os.getpid(), signal.SIGTERM)

            # 3. Build the verification ScheduleBatch
            if num_dummies > 0:
                dummy_input_ids, dummy_cache_locs, dummy_req_pool_indices = (
                    self.fixed_pool.get_dummy_data(num_dummies)
                )
                dummy_sampling_tuple = self.fixed_pool.get_dummy_sampling_tensors(num_dummies)
                verify_batch = llm42_info.prepare_verify_batch(
                    original_batch,
                    reqs,
                    dummy_input_ids=dummy_input_ids,
                    dummy_cache_locs=dummy_cache_locs,
                    dummy_req_pool_indices=dummy_req_pool_indices,
                    num_dummies=num_dummies,
                    window_size=self.fixed_pool.window_size,
                    dummy_sampling_tuple=dummy_sampling_tuple,
                )
            else:
                verify_batch = llm42_info.prepare_verify_batch(original_batch, reqs)

            # 4. Run verification forward pass
            verify_model_worker_batch = verify_batch.get_model_worker_batch()
            verify_output = self.target_worker.forward_batch_generation(verify_model_worker_batch)
            verified_token_ids = verify_output.next_token_ids
            verified_logprobs = verify_output.logits_output.next_token_logprobs

            # 5. Compare outputs and determine rollbacks
            if self.skip_mismatch >= 100.0:
                rollback_info = llm42_info.verify_and_compare(
                    reqs, verified_token_ids, verified_logprobs
                )
            elif self.skip_mismatch <= 0.0:
                rollback_info = [
                    (len(req.output_ids) - req.llm42_verified_tokens, 0)
                    for req in reqs
                ]
            else:
                rollback_info = llm42_info.verify_and_compare(
                    reqs, verified_token_ids, verified_logprobs,
                    mismatch_percentage=self.skip_mismatch,
                )

            # 6. Collect rollback results and update per-request stats
            rollback_results = []
            for req, info in zip(reqs, rollback_info):
                req.llm42_num_verification_windows += 1
                if info is not None and info[1] > 0:
                    req.llm42_num_rollbacks += 1
                    req.llm42_tokens_rolled_back += info[1]
                    if self.metrics_collector:
                        self.metrics_collector.increment_rollbacks(info[1])
                    rollback_results.append((req, info[1]))
                req.llm42_verified_tokens = len(req.output_ids)

            if rollback_results:
                rolled_back_reqs = {req for req, _ in rollback_results}
                self._update_batch_state(original_batch, rolled_back_reqs)

            # 7. Release verification-batch tensor references
            self._cleanup_verify_batch(verify_batch)

            # 8. Free temporary KV cache for padding (only for variable-size path;
            #    fixed_pool padding is pre-allocated and reused)
            if llm42_info.total_padding_cache_slots > 0 and self.fixed_pool is None:
                llm42_info.free_padding_kv_cache(
                    original_batch.token_to_kv_pool_allocator
                )

            return rollback_results

        except Exception as e:
            if 'llm42_info' in locals() and llm42_info.total_padding_cache_slots > 0 and self.fixed_pool is None:
                try:
                    llm42_info.free_padding_kv_cache(
                        original_batch.token_to_kv_pool_allocator
                    )
                except Exception:
                    pass
            logger.error(f"Error during verification: {e}")
            raise

    @staticmethod
    def _cleanup_verify_batch(verify_batch):
        """Null out tensor references on a verification batch to free GPU memory."""
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

    def _update_batch_state(self, batch: Union[ScheduleBatch, ModelWorkerBatch],
                           rolled_back_reqs: set = None):
        """Sync ``batch.output_ids`` and ``batch.seq_lens`` with the current
        ``req.output_ids`` after a rollback, so that ``prepare_for_decode()``
        picks up the correct state.

        Only updates entries for requests in *rolled_back_reqs*.  Non-rolled-back
        requests keep their current ``batch.seq_lens`` / ``batch.output_ids``
        values so that the next ``prepare_for_decode`` increments from the
        correct position.
        """
        if not hasattr(batch, 'output_ids') or batch.output_ids is None:
            return
        
        # Start from the current batch tensors; only overwrite rolled-back entries
        current_output_ids = batch.output_ids.tolist()
        current_seq_lens = batch.seq_lens.tolist()
        
        for i, req in enumerate(batch.reqs):
            if rolled_back_reqs and req not in rolled_back_reqs:
                continue  # keep existing batch values for non-rolled-back requests
            
            # Get last generated token or fallback to last input token
            if req.output_ids:
                current_output_ids[i] = req.output_ids[-1]
            else:
                current_output_ids[i] = req.origin_input_ids[-1] if req.origin_input_ids else 0
            
            # seq_lens should be length BEFORE current token (prepare_for_decode increments by 1)
            current_total_len = len(req.origin_input_ids) + len(req.output_ids)
            current_seq_lens[i] = current_total_len - 1
        
        # Update batch tensors
        batch.output_ids = torch.tensor(
            current_output_ids, 
            dtype=torch.int64, 
            device=batch.device
        )
        batch.seq_lens = torch.tensor(
            current_seq_lens,
            dtype=torch.int32,
            device=batch.device
        )
        batch.seq_lens_cpu = torch.tensor(current_seq_lens, dtype=torch.int32)
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.seq_lens_sum = sum(current_seq_lens)

    def __getattr__(self, name):
        """Delegate unknown attributes to ``target_worker`` (transparent proxy)."""
        return getattr(self.target_worker, name)
