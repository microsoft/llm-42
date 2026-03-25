# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ==============================================================================
"""Verification batch metadata for the LLM-42 DVR protocol.

This module defines ``LLM42Info``, which holds all the bookkeeping needed to
construct a TARGET_LLM42_VERIFY forward pass, compare its outputs against the
original decode-phase tokens, and apply rollback on mismatch.

Key responsibilities:
  1. Build verification input_ids (context token + decode outputs + padding).
  2. Allocate / free temporary KV-cache slots for padding tokens.
  3. Prepare a ``ScheduleBatch`` in TARGET_LLM42_VERIFY mode.
  4. Token-level comparison and rollback logic.

See §4.2 of the LLM-42 paper (arXiv:2601.17768) for protocol details.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


class LLM42Info:
    """Metadata and logic for a single verification pass.

    An ``LLM42Info`` instance is created per verification batch.  It records the
    original (decode-phase) token IDs, per-request padding masks, and manages
    temporary KV-cache allocations for padded positions.  After the verification
    forward pass completes, :meth:`verify_and_compare` performs token-level
    comparison and applies rollback when a mismatch is detected.
    """

    # Dummy token ID used for padding (typically 0 or pad_token_id)
    DUMMY_TOKEN_ID = 32

    def __init__(
        self,
        original_outputs: torch.Tensor,  # (total_output_tokens,)
        output_lens: List[int],  # per-request actual output lengths (before padding)
        padded_lens: List[int],  # per-request padded lengths (after padding)
        padding_masks: List[List[bool]],  # per-request mask: True for real tokens, False for padding
        padding_counts: List[int],  # per-request number of padding predictions added
        seq_lens: Optional[torch.Tensor] = None,  # (batch_size,) - optional, may not be used
    ):
        self.original_outputs = original_outputs
        self.output_lens = output_lens  # actual unverified tokens count
        self.padded_lens = padded_lens  # length after padding (may equal output_lens if no padding)
        self.padding_masks = padding_masks  # masks to identify real vs padded tokens
        self.padding_counts = padding_counts  # number of padding predictions per request

        # Track number of real vs dummy requests
        self.num_real_requests = len(output_lens)
        self.num_dummy_requests = 0

        # KV-cache slots needed for padding positions.
        # Each request's verification input has ``padded_len`` tokens; the first
        # token reuses the context cache slot, so ``padded_len - 1`` *new* slots
        # are required.
        self.padding_cache_slots = [max(0, padded_len - 1) for padded_len in padded_lens]
        self.total_padding_cache_slots = sum(self.padding_cache_slots)
        
        # Will be set after KV cache allocation for padding
        self.padding_cache_locs: Optional[torch.Tensor] = None
        self.padding_cache_locs_allocated: Optional[torch.Tensor] = None  # Track full allocation for freeing
    
    def append_dummy_entries(self, num_dummies: int, window_size: int):
        """
        Append dummy request entries for fixed-size batch padding.
        
        Dummy requests have no real outputs—their padding masks are all False,
        so their outputs are completely ignored during comparison.
        
        Args:
            num_dummies: Number of dummy requests to add
            window_size: Step size (padded length) for each dummy
        """
        for _ in range(num_dummies):
            self.output_lens.append(0)  # No real outputs
            self.padded_lens.append(window_size)
            self.padding_masks.append([False] * window_size)  # All padding
            self.padding_counts.append(window_size)
        
        self.num_dummy_requests = num_dummies
        
        # Note: We don't update padding_cache_slots or total_padding_cache_slots here
        # because dummy requests use pre-allocated cache from FixedSizeVerificationPool,
        # not the temporary allocation in allocate_padding_kv_cache()

    @classmethod
    def from_requests(
        cls, 
        reqs: List[Req], 
        always_align: bool = True,
        force_include_all: bool = False,
        window_size: Optional[int] = None,
    ) -> LLM42Info:
        """
        Create LLM42Info from a list of finished deterministic requests.
        
        Args:
            reqs: List of requests to verify
            always_align: If True, pad finished requests to window_size with dummy tokens
            force_include_all: If True, include and pad all requests regardless of finished status.
                              Used for fixed-size batches where we need to include not-yet-ready requests.
            window_size: Verification window size for padding. If None, no padding is applied.
            
        Returns:
            LLM42Info instance
        """
        original_outputs = []
        output_lens = []
        padded_lens = []
        padding_masks = []
        padding_counts = []
        
        for req in reqs:
            unverified_output_ids = req.output_ids[req.llm42_verified_tokens:]
            actual_len = len(unverified_output_ids)
            
            # Skip requests with no unverified tokens
            if actual_len == 0:
                continue
            
            # In fixed-batch mode, truncate to window_size to preserve the
            # fixed-shape invariant.  Remaining tokens will be verified in
            # the next round.  On rollback the extra tokens are discarded
            # together with the mismatched ones; on success the caller must
            # only advance llm42_verified_tokens by actual_len (not the
            # full unverified count) so the extra tokens get re-verified.
            if force_include_all and window_size is not None and actual_len > window_size:
                unverified_output_ids = unverified_output_ids[:window_size]
                actual_len = window_size
            
            output_lens.append(actual_len)
            
            # Determine if padding is needed
            # Note: finished_output is a boolean (False initially), so use truthiness check, not "is not None"
            is_finished = req.finished_reason is not None
            
            # Pad if:
            # 1. always_align is True AND window_size is set AND actual_len < window_size
            # 2. AND (request is finished OR force_include_all is True)
            should_pad = (
                always_align and 
                window_size is not None and 
                actual_len < window_size and
                (is_finished or force_include_all)
            )
            
            if should_pad:
                # Pad to window_size with dummy tokens
                padding_needed = window_size - actual_len
                original_outputs.extend(unverified_output_ids)
                original_outputs.extend([cls.DUMMY_TOKEN_ID] * padding_needed)
                padded_lens.append(window_size)
                padding_counts.append(padding_needed)
                # Mask: True for real tokens, False for padding
                padding_masks.append([True] * actual_len + [False] * padding_needed)
            else:
                # No padding needed
                original_outputs.extend(unverified_output_ids)
                padded_lens.append(actual_len)
                padding_counts.append(0)
                padding_masks.append([True] * actual_len)
        
        return cls(
            original_outputs=torch.tensor(original_outputs, dtype=torch.int64),
            output_lens=output_lens,
            padded_lens=padded_lens,
            padding_masks=padding_masks,
            padding_counts=padding_counts,
        )

    def set_padding_cache_locs(self, padding_locs: torch.Tensor):
        """
        Assign pre-allocated padding KV cache slots.
        
        Used when padding cache is managed externally (e.g., by
        FixedSizeVerificationPool) instead of allocated per-pass.
        
        Args:
            padding_locs: Pre-allocated cache location tensor (at least
                ``total_padding_cache_slots`` elements).
        """
        self.padding_cache_locs = padding_locs[:self.total_padding_cache_slots]
        # No allocated tensor to free — the pool owns it
        self.padding_cache_locs_allocated = None

    def allocate_padding_kv_cache(self, token_to_kv_pool_allocator) -> bool:
        """
        Allocate temporary KV cache slots for padding tokens.
        
        This is the fallback path used when no pre-allocated padding pool is
        available (variable-size verification).  When a
        ``FixedSizeVerificationPool`` is in use, call :meth:`set_padding_cache_locs`
        instead.
        
        Args:
            token_to_kv_pool_allocator: The KV cache allocator
            
        Returns:
            True if allocation succeeded (or no padding needed), False otherwise
        """
        if self.total_padding_cache_slots == 0:
            return True
        
        # With paged allocation, we need to allocate in multiples of page_size
        page_size = token_to_kv_pool_allocator.page_size
        if page_size > 1:
            # Round up to nearest page
            slots_to_allocate = ((self.total_padding_cache_slots + page_size - 1) // page_size) * page_size
        else:
            slots_to_allocate = self.total_padding_cache_slots
        
        # Allocate KV cache slots for padding input tokens
        allocated_locs = token_to_kv_pool_allocator.alloc(slots_to_allocate)
        
        # Store full allocation for later freeing
        self.padding_cache_locs_allocated = allocated_locs
        
        # Only use the slots we actually need (first total_padding_cache_slots)
        if allocated_locs is not None and len(allocated_locs) > 0:
            if len(allocated_locs) > self.total_padding_cache_slots:
                self.padding_cache_locs = allocated_locs[:self.total_padding_cache_slots]
            else:
                self.padding_cache_locs = allocated_locs
        else:
            self.padding_cache_locs = None
        
        if self.padding_cache_locs is None or len(self.padding_cache_locs) == 0:
            original_outputs_list = self.original_outputs.tolist()
            new_original_outputs = []
            offset = 0
            for i, padded_len in enumerate(self.padded_lens):
                actual_len = self.output_lens[i]
                # Take only actual tokens, skip padding
                new_original_outputs.extend(original_outputs_list[offset:offset + actual_len])
                offset += padded_len
            
            self.original_outputs = torch.tensor(new_original_outputs, dtype=torch.int64)
            self.padded_lens = self.output_lens.copy()
            self.padding_counts = [0] * len(self.padding_counts)
            self.padding_cache_slots = [0] * len(self.padding_cache_slots)
            self.padding_masks = [[True] * l for l in self.output_lens]
            self.total_padding_cache_slots = 0
            return False
        
        return True

    def free_padding_kv_cache(self, token_to_kv_pool_allocator):
        """
        Free the temporarily allocated KV cache slots for padding.
        
        Should be called after verification is complete.
        
        Args:
            token_to_kv_pool_allocator: The KV cache allocator
        """
        # Free the full allocated tensor, not just the portion we used
        if self.padding_cache_locs_allocated is not None and len(self.padding_cache_locs_allocated) > 0:
            token_to_kv_pool_allocator.free(self.padding_cache_locs_allocated)
            self.padding_cache_locs_allocated = None
            self.padding_cache_locs = None

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def prepare_verify_batch(
        self,
        original_batch: ScheduleBatch,
        reqs_to_verify: List[Req],
        dummy_input_ids: Optional[torch.Tensor] = None,
        dummy_cache_locs: Optional[torch.Tensor] = None,
        dummy_req_pool_indices: Optional[torch.Tensor] = None,
        num_dummies: int = 0,
        window_size: Optional[int] = None,
        dummy_sampling_tuple: Optional[tuple] = None,
    ) -> ScheduleBatch:
        """Build a ``ScheduleBatch`` in TARGET_LLM42_VERIFY mode.

        The batch replays the last ``window_size`` tokens of each request so the
        verifier can compare its outputs against the decode-phase tokens.  When
        grouped verification is used, ``num_dummies`` dummy requests (drawn from
        the pre-allocated :class:`FixedSizeVerificationPool`) are appended so
        that every verification pass has identical shape.

        Args:
            original_batch: Original decode-phase batch (provides KV pools etc.).
            reqs_to_verify: Real requests whose tokens need verification.
            dummy_input_ids: Pre-allocated dummy input tokens (optional).
            dummy_cache_locs: Pre-allocated dummy KV-cache locations (optional).
            dummy_req_pool_indices: Pre-allocated req_pool indices for dummies.
            num_dummies: Number of dummy requests to append (default 0).
            window_size: Verification window size per request (required when
                ``num_dummies > 0``).
            dummy_sampling_tuple: Pre-allocated dummy sampling tensors as tuple
                ``(temps, top_ps, top_ks, min_ps, seeds, det_indices,
                prefix_lens, output_lens)``.

        Returns:
            A ``ScheduleBatch`` ready for the verification forward pass.
        """
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        
        # For fixed-size batches, we include dummy requests in the batch
        # but keep track of the real requests for result processing
        total_batch_size = len(reqs_to_verify) + num_dummies
        
        verify_batch = ScheduleBatch(reqs=reqs_to_verify, batch_is_full=True)
        verify_batch.forward_mode = ForwardMode.TARGET_LLM42_VERIFY
        
        input_ids = []
        req_pool_indices = []
        output_lens = []  # This will store padded lengths for batch construction
        prefix_lens_list = []
        
        for i, req in enumerate(reqs_to_verify):
            actual_unverified = req.output_ids[req.llm42_verified_tokens:]
            padded_len = self.padded_lens[i]
            actual_len = self.output_lens[i]
            
            if padded_len == 0:
                continue
            
            # Get the last verified token as context
            if req.llm42_verified_tokens > 0:
                last_verified_token = req.output_ids[req.llm42_verified_tokens - 1]
            elif req.origin_input_ids:
                last_verified_token = req.origin_input_ids[-1]
            elif req.input_ids:
                last_verified_token = req.input_ids[-1]
            else:
                logger.error(f"Request {req.rid} has no input tokens for verification")
                continue
            
            # Build input_ids for verification:
            # We need padded_len input tokens to get padded_len output predictions.
            # Input: [last_verified_token, token_0, token_1, ..., token_{padded_len-2}]
            # Output predictions: [pred_0, pred_1, ..., pred_{padded_len-1}]
            # We compare pred_i with actual_unverified[i] for i < actual_len
            #
            # For actual tokens: use actual_unverified[:-1] (all but last)
            # For padding: use dummy tokens to fill up to padded_len - 1 tokens after context
            
            if padded_len > actual_len:
                # Padding case
                # Use actual_len - 1 actual tokens, then pad with dummies
                # Input: [context] + [actual_len-1 tokens] + [padding dummies] = padded_len tokens total
                tokens_after_context = list(actual_unverified[:-1]) if actual_len > 1 else []  # all but last actual token
                padding_needed = padded_len - 1 - len(tokens_after_context)  # = padding_count
                if padding_needed > 0:
                    tokens_after_context.extend([self.DUMMY_TOKEN_ID] * padding_needed)
                # Total: 1 context + (actual_len-1) + padding_count = 1 + padded_len - 1 = padded_len
                input_ids.extend([last_verified_token] + tokens_after_context)
            else:
                # No padding: original logic
                # Input: [last_verified, actual_unverified[:-1]]
                input_ids.extend([last_verified_token] + list(actual_unverified[:-1]))
            
            if req.req_pool_idx is None:
                logger.warning(f"Skipping req {req.rid} in verification: req_pool_idx is None (likely retracted)")
                continue

            req_pool_indices.append(req.req_pool_idx)
            output_lens.append(padded_len)
            prefix_lens_list.append(len(req.origin_input_ids) + req.llm42_verified_tokens - 1)
        
        device = original_batch.device
        verify_batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        verify_batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64, device=device)
        
        verify_batch.extend_lens = output_lens
        verify_batch.prefix_lens = prefix_lens_list
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int64, device=device)
        extend_lens_tensor = torch.tensor(output_lens, dtype=torch.int64, device=device)
        total_seq_lens = prefix_lens + extend_lens_tensor
        verify_batch.seq_lens = total_seq_lens
        verify_batch.seq_lens_cpu = total_seq_lens.cpu()
        verify_batch.extend_logprob_start_lens = [1] * len(reqs_to_verify)
        
        verify_batch.extend_num_tokens = len(input_ids)
        verify_batch.seq_lens_sum = total_seq_lens.sum().item()
        verify_batch.orig_seq_lens = total_seq_lens.clone()
        
        verify_batch.return_logprob = True
        verify_batch.top_logprobs_nums = [0] * len(reqs_to_verify)
        verify_batch.token_ids_logprobs = [[] for _ in reqs_to_verify]
        
        verify_batch.req_to_token_pool = original_batch.req_to_token_pool
        verify_batch.token_to_kv_pool_allocator = original_batch.token_to_kv_pool_allocator
        verify_batch.tree_cache = original_batch.tree_cache
        verify_batch.model_config = original_batch.model_config
        verify_batch.device = device
        
        verify_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            verify_batch, original_batch.model_config.vocab_size
        )
        
        # Force pytorch backend for verification to ensure deterministic seeded
        # sampling via multinomial_with_seed (uses (seed, position) hash).
        if verify_batch.sampling_info is not None:
            verify_batch.sampling_info.force_pytorch_backend = True

        if verify_batch.sampling_info is not None:
            tokens_per_request = torch.tensor(output_lens, dtype=torch.int64, device=device)
            
            verify_batch.sampling_info.temperatures = torch.repeat_interleave(
                verify_batch.sampling_info.temperatures, tokens_per_request, dim=0
            )
            verify_batch.sampling_info.top_ks = torch.repeat_interleave(
                verify_batch.sampling_info.top_ks, tokens_per_request, dim=0
            )
            verify_batch.sampling_info.top_ps = torch.repeat_interleave(
                verify_batch.sampling_info.top_ps, tokens_per_request, dim=0
            )
            verify_batch.sampling_info.min_ps = torch.repeat_interleave(
                verify_batch.sampling_info.min_ps, tokens_per_request, dim=0
            )
            if verify_batch.sampling_info.sampling_seed is not None:
                verify_batch.sampling_info.sampling_seed = torch.repeat_interleave(
                    verify_batch.sampling_info.sampling_seed, tokens_per_request, dim=0
                )
            if hasattr(verify_batch.sampling_info, 'deterministic_indices') and verify_batch.sampling_info.deterministic_indices is not None:
                verify_batch.sampling_info.deterministic_indices = torch.repeat_interleave(
                    verify_batch.sampling_info.deterministic_indices, tokens_per_request, dim=0
                )
        out_cache_locs = []
        padding_offset = 0  # Track offset into allocated padding cache locations
        
        for i, req in enumerate(reqs_to_verify):
            # IMPORTANT: current_seq_len is the number of KV positions that have been written.
            # With N output tokens:
            #   - output_ids[0] from prefill (KV at positions 0 to origin_input_len-1)
            #   - output_ids[1..N-1] from N-1 decode steps
            #   - Decode step i writes KV for output_ids[i-1] at position origin_input_len + i - 1
            #   - Last KV written at position: origin_input_len + (N-1) - 1 = origin_input_len + N - 2
            #   - Total KV positions: origin_input_len + N - 1
            # The last output token (output_ids[N-1]) hasn't had its KV written yet!
            current_seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
            context_idx = len(req.origin_input_ids) + req.llm42_verified_tokens - 1
            actual_len = self.output_lens[i]
            padded_len = self.padded_lens[i]
            padding_count = self.padding_counts[i]
            
            if context_idx >= current_seq_len:
                logger.error(
                    f"ERROR: context_idx {context_idx} >= current_seq_len {current_seq_len} "
                    f"for req {req.rid}. This indicates a logic error."
                )
                raise RuntimeError(
                    f"Attempting to access unallocated cache position {context_idx} "
                    f"when only {current_seq_len} positions have been allocated"
                )
            
            # Context token cache location
            context_cache_loc = verify_batch.req_to_token_pool.req_to_token[req.req_pool_idx, context_idx]
            out_cache_locs.append(context_cache_loc.item())

            # For non-padded case: we need actual_len - 1 more cache locations
            # For padded case: we need (actual_len - 1) + padding_count = padded_len - 1 new cache locations
            #
            # Input structure:
            #   Non-padded: [context, token_0, ..., token_{n-2}] = n tokens (where n = actual_len)
            #   Padded:     [context, token_0, ..., token_{n-2}, dummy_0, ..., dummy_{p-1}] 
            #               = 1 + (n-1) + p = padded_len tokens (where n = actual_len, p = padding_count)
            #
            # Note: For actual_len=1, input is [context] + [0 actual tokens] + [window_size-1 dummies]
            
            start_idx = len(req.origin_input_ids) + req.llm42_verified_tokens
            
            # CRITICAL: Always reuse the SAME cache locations that decode wrote to!
            # Verification must overwrite decode KV at the exact same physical slots
            # so that attention (which reads via page_table = req_to_token) sees the
            # verification KV instead of the old decode KV.
            #
            # Previously, padded case used newly allocated padding_cache_locs, which
            # caused verification to write to different slots than page_table reads from.
            
            # Number of cache locations needed after context (padded_len - 1)
            num_cache_after_context = padded_len - 1
            end_idx = min(start_idx + num_cache_after_context, current_seq_len)
            
            if start_idx < end_idx:
                output_cache_locs = verify_batch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]

                # Check for invalid 0 values which indicate unallocated positions.
                # This mirrors the old fork's diagnostic — log error but do NOT
                # replace, because replacing out_cache_loc without also updating
                # the page_table (built from req_to_token in init_forward_metadata)
                # causes attention to read stale KV from slot 0 while the model
                # writes fresh KV to the replacement slot → garbage output.
                output_list = output_cache_locs.tolist()
                for j, loc in enumerate(output_list):
                    if loc == 0:
                        logger.error(
                            f"[LLM42_VERIFY] Found invalid slot 0 at position {start_idx + j} "
                            f"in req_to_token! (req {req.rid})"
                        )
                
                out_cache_locs.extend(output_list)
            
            # For positions beyond current_seq_len (padding for incomplete sequences),
            # we need to allocate new cache slots
            remaining_needed = num_cache_after_context - (end_idx - start_idx)
            if remaining_needed > 0 and self.padding_cache_locs is not None:
                padding_locs = self.padding_cache_locs[padding_offset:padding_offset + remaining_needed]
                out_cache_locs.extend(padding_locs.tolist())
                padding_offset += remaining_needed
        
        # Convert real request data to tensors first
        real_input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        real_cache_locs = torch.tensor(out_cache_locs, dtype=torch.int64, device=device)
        real_req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64, device=device)
        real_prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int64, device=device)
        real_output_lens = torch.tensor(output_lens, dtype=torch.int64, device=device)
        
        # Append dummy request data for fixed-size batches
        if num_dummies > 0 and dummy_input_ids is not None and dummy_cache_locs is not None:
            if window_size is None:
                raise ValueError("window_size required when adding dummy requests")
            
            dummy_tokens_needed = num_dummies * window_size
            
            # Use tensor concatenation instead of list extend + tensor creation
            # This avoids CPU-GPU round trips
            verify_batch.input_ids = torch.cat([
                real_input_ids, 
                dummy_input_ids[:dummy_tokens_needed]
            ], dim=0)
            
            verify_batch.out_cache_loc = torch.cat([
                real_cache_locs,
                dummy_cache_locs[:dummy_tokens_needed]
            ], dim=0)
            
            # Use pre-allocated dummy req_pool_indices (required for correct page_table lookup)
            if dummy_req_pool_indices is not None:
                verify_batch.req_pool_indices = torch.cat([
                    real_req_pool_indices, 
                    dummy_req_pool_indices[:num_dummies]
                ], dim=0)
            else:
                # Fallback: use first real request's index (WARNING: this is incorrect for attention!)
                logger.warning("dummy_req_pool_indices not provided - using first real request's index")
                dummy_pool_idx = req_pool_indices[0] if req_pool_indices else 0
                dummy_pool_indices = torch.full((num_dummies,), dummy_pool_idx, dtype=torch.int64, device=device)
                verify_batch.req_pool_indices = torch.cat([real_req_pool_indices, dummy_pool_indices], dim=0)
            
            # Use pre-allocated tensors if available, otherwise create new ones
            # Tuple order: (temperatures, top_ps, top_ks, min_ps, seeds, det_indices, prefix_lens, output_lens)
            if dummy_sampling_tuple is not None:
                dummy_prefix_lens = dummy_sampling_tuple[6]
                dummy_output_lens = dummy_sampling_tuple[7]
            else:
                dummy_prefix_lens = torch.zeros(num_dummies, dtype=torch.int64, device=device)
                dummy_output_lens = torch.full((num_dummies,), window_size, dtype=torch.int64, device=device)
            
            prefix_lens = torch.cat([real_prefix_lens, dummy_prefix_lens], dim=0)
            extend_lens_tensor = torch.cat([real_output_lens, dummy_output_lens], dim=0)
            total_seq_lens = prefix_lens + extend_lens_tensor
            
            verify_batch.extend_lens = extend_lens_tensor.tolist()
            verify_batch.prefix_lens = prefix_lens.tolist()
            verify_batch.seq_lens = total_seq_lens
            verify_batch.seq_lens_cpu = total_seq_lens.cpu()
            verify_batch.extend_logprob_start_lens = [1] * (len(reqs_to_verify) + num_dummies)
            
            verify_batch.extend_num_tokens = len(verify_batch.input_ids)
            verify_batch.seq_lens_sum = total_seq_lens.sum().item()
            verify_batch.orig_seq_lens = total_seq_lens.clone()
            
            verify_batch.top_logprobs_nums = [0] * (len(reqs_to_verify) + num_dummies)
            verify_batch.token_ids_logprobs = [[] for _ in range(len(reqs_to_verify) + num_dummies)]
            
            # Extend sampling info with dummy values (use pre-allocated if available)
            if verify_batch.sampling_info is not None:
                if dummy_sampling_tuple is not None:
                    # Use pre-allocated tensors - tuple order: (temps, top_ps, top_ks, min_ps, seeds, det_indices, prefix_lens, output_lens)
                    dummy_temps = dummy_sampling_tuple[0].to(verify_batch.sampling_info.temperatures.dtype)
                    dummy_top_ps = dummy_sampling_tuple[1].to(verify_batch.sampling_info.top_ps.dtype)
                    dummy_top_ks = dummy_sampling_tuple[2].to(verify_batch.sampling_info.top_ks.dtype)
                    dummy_min_ps = dummy_sampling_tuple[3].to(verify_batch.sampling_info.min_ps.dtype)
                else:
                    # Fallback: create new tensors
                    dummy_temps = torch.zeros((dummy_tokens_needed, 1), dtype=verify_batch.sampling_info.temperatures.dtype, device=device)
                    dummy_top_ps = torch.ones(dummy_tokens_needed, dtype=verify_batch.sampling_info.top_ps.dtype, device=device)
                    dummy_top_ks = torch.full((dummy_tokens_needed,), -1, dtype=verify_batch.sampling_info.top_ks.dtype, device=device)
                    dummy_min_ps = torch.zeros(dummy_tokens_needed, dtype=verify_batch.sampling_info.min_ps.dtype, device=device)
                
                verify_batch.sampling_info.temperatures = torch.cat([
                    verify_batch.sampling_info.temperatures, dummy_temps
                ], dim=0)
                verify_batch.sampling_info.top_ps = torch.cat([
                    verify_batch.sampling_info.top_ps, dummy_top_ps
                ], dim=0)
                verify_batch.sampling_info.top_ks = torch.cat([
                    verify_batch.sampling_info.top_ks, dummy_top_ks
                ], dim=0)
                verify_batch.sampling_info.min_ps = torch.cat([
                    verify_batch.sampling_info.min_ps, dummy_min_ps
                ], dim=0)
                
                if verify_batch.sampling_info.sampling_seed is not None:
                    if dummy_sampling_tuple is not None:
                        dummy_seeds = dummy_sampling_tuple[4].to(verify_batch.sampling_info.sampling_seed.dtype)
                    else:
                        dummy_seeds = torch.zeros(dummy_tokens_needed, dtype=verify_batch.sampling_info.sampling_seed.dtype, device=device)
                    verify_batch.sampling_info.sampling_seed = torch.cat([
                        verify_batch.sampling_info.sampling_seed, dummy_seeds
                    ], dim=0)
                
                if hasattr(verify_batch.sampling_info, 'deterministic_indices') and verify_batch.sampling_info.deterministic_indices is not None:
                    if dummy_sampling_tuple is not None:
                        dummy_det_indices = dummy_sampling_tuple[5].to(verify_batch.sampling_info.deterministic_indices.dtype)
                    else:
                        dummy_det_indices = torch.ones((dummy_tokens_needed, 1), dtype=verify_batch.sampling_info.deterministic_indices.dtype, device=device)
                    verify_batch.sampling_info.deterministic_indices = torch.cat([
                        verify_batch.sampling_info.deterministic_indices, dummy_det_indices
                    ], dim=0)
        else:
            # No dummies - use real tensors directly
            verify_batch.input_ids = real_input_ids
            verify_batch.out_cache_loc = real_cache_locs
            verify_batch.req_pool_indices = real_req_pool_indices
            
            total_seq_lens = real_prefix_lens + real_output_lens
            verify_batch.extend_lens = output_lens
            verify_batch.prefix_lens = prefix_lens_list
            verify_batch.seq_lens = total_seq_lens
            verify_batch.seq_lens_cpu = total_seq_lens.cpu()
            verify_batch.extend_logprob_start_lens = [1] * len(reqs_to_verify)
            
            verify_batch.extend_num_tokens = len(real_input_ids)
            verify_batch.seq_lens_sum = total_seq_lens.sum().item()
            verify_batch.orig_seq_lens = total_seq_lens.clone()
            
            verify_batch.top_logprobs_nums = [0] * len(reqs_to_verify)
            verify_batch.token_ids_logprobs = [[] for _ in range(len(reqs_to_verify))]
        
        # Verify consistency
        if len(verify_batch.out_cache_loc) != len(verify_batch.input_ids):
            logger.error(
                f"[LLM42] Cache location count mismatch: "
                f"expected={len(verify_batch.input_ids)}, got={len(verify_batch.out_cache_loc)}"
            )
            raise RuntimeError(
                f"Verification batch has {len(verify_batch.input_ids)} input tokens but only "
                f"{len(verify_batch.out_cache_loc)} cache locations. This will cause memory corruption."
            )
        
        return verify_batch

    # ------------------------------------------------------------------
    # Token comparison and rollback
    # ------------------------------------------------------------------

    def first_mismatch_position(
        self,
        original_ids: List[int],
        verified_ids: List[int],
    ) -> int:
        """Return the index of the first divergent token (or ``min_len`` if all match)."""
        min_len = min(len(original_ids), len(verified_ids))
        if min_len == 0:
            return 0
        
        # Use numpy for faster vectorized comparison
        orig_arr = np.asarray(original_ids[:min_len])
        verify_arr = np.asarray(verified_ids[:min_len])
        mismatches = np.where(orig_arr != verify_arr)[0]
        
        if len(mismatches) > 0:
            return int(mismatches[0])
        return min_len

    def verify_and_compare(
        self,
        reqs: List[Req],
        verified_token_ids: torch.Tensor,
        verified_logprobs: Optional[torch.Tensor] = None,
        mismatch_percentage: Optional[float] = None,
    ):
        """Compare decode-phase tokens with verification outputs and apply rollback.

        For each real request (dummy entries are skipped), this method:

        1. Finds the first token that differs between the decode output and the
           verifier output.
        2. Truncates ``req.output_ids`` to discard everything after the mismatch.
        3. Appends the verifier-generated token at the mismatch position,
           guaranteeing at least one new consistent token per pass.

        Padding tokens (identified via ``padding_masks``) are excluded from the
        comparison.

        Args:
            reqs: Real requests that were verified (no dummies).
            verified_token_ids: Token IDs produced by the verification pass.
            verified_logprobs: Corresponding log-probabilities (optional).
            mismatch_percentage: If set (0–100), synthetically inject a mismatch
                to force exactly ``ceil(pct/100 * window_size)`` tokens to be
                rolled back.  Used for benchmarking overhead.

        Returns:
            List of ``(mismatch_position, tokens_rolled_back)`` per request.
        """
        
        verified_token_ids = verified_token_ids.tolist() if isinstance(verified_token_ids, torch.Tensor) else verified_token_ids
        verified_logprobs_list = verified_logprobs.tolist() if isinstance(verified_logprobs, torch.Tensor) else verified_logprobs
        original_ids = self.original_outputs.tolist()
        rollback_info = []
        offset = 0
        
        # Only process real requests (first num_real_requests entries)
        # Dummy entries are skipped - their outputs are ignored
        num_to_process = min(len(reqs), self.num_real_requests)
        
        for i in range(num_to_process):
            req = reqs[i]
            padded_len = self.padded_lens[i]
            actual_len = self.output_lens[i]  # Only compare actual tokens, not padding
            padding_mask = self.padding_masks[i]
            
            # Extract only the real (non-padded) tokens for comparison
            orig_output = [original_ids[offset + j] for j in range(padded_len) if padding_mask[j]]
            verify_output_full = verified_token_ids[offset : offset + padded_len]
            verify_output = [verify_output_full[j] for j in range(padded_len) if padding_mask[j]]
            
            # Also extract logprobs for real tokens only
            if verified_logprobs_list is not None:
                verify_logprobs_full = verified_logprobs_list[offset : offset + padded_len]
                verify_logprobs = [verify_logprobs_full[j] for j in range(padded_len) if padding_mask[j]]
            else:
                verify_logprobs = None
            
            mismatch_pos = self.first_mismatch_position(orig_output, verify_output)
            
            tokens_to_rollback = len(orig_output) - mismatch_pos
            
            # If mismatch_percentage is set, ALWAYS inject mismatch at calculated position
            # This overrides any natural mismatch to ensure exactly X% rollback
            if mismatch_percentage is not None and len(orig_output) > 0:
                # Calculate mismatch position to rollback ceil(X% * window_size) tokens
                window_size = len(orig_output)
                tokens_to_rollback = math.ceil(mismatch_percentage / 100.0 * window_size)
                mismatch_pos = window_size - tokens_to_rollback
            
            if tokens_to_rollback > 0:
                req.output_ids = req.output_ids[:-tokens_to_rollback]
                
                if verify_logprobs is not None and req.output_token_logprobs_val is not None:
                    # Replace logprobs for ALL verified tokens (up to and including mismatch)
                    req.output_token_logprobs_val = req.output_token_logprobs_val[:-actual_len]
                    req.output_token_logprobs_idx = req.output_token_logprobs_idx[:-actual_len]
                    
                    # Add verified logprobs up to mismatch position
                    req.output_token_logprobs_val.extend(verify_logprobs[:mismatch_pos])
                    req.output_token_logprobs_idx.extend(verify_output[:mismatch_pos])
                
                req.finished_reason = None

                if mismatch_pos < len(verify_output):
                    req.output_ids.append(verify_output[mismatch_pos])
                    
                    if verify_logprobs is not None and req.output_token_logprobs_val is not None:
                        req.output_token_logprobs_val.append(verify_logprobs[mismatch_pos])
                        req.output_token_logprobs_idx.append(verify_output[mismatch_pos])
                    
                    tokens_to_rollback -= 1
                
                rollback_info.append((mismatch_pos, tokens_to_rollback))
            else:
                # No rollback needed, but still replace with verified logprobs and token IDs
                if verify_logprobs is not None and req.output_token_logprobs_val is not None:
                    # Replace logprobs for ALL verified tokens
                    req.output_token_logprobs_val = req.output_token_logprobs_val[:-actual_len]
                    req.output_token_logprobs_idx = req.output_token_logprobs_idx[:-actual_len]
                    
                    # Add verified logprobs and token IDs
                    req.output_token_logprobs_val.extend(verify_logprobs)
                    req.output_token_logprobs_idx.extend(verify_output)
                
                rollback_info.append((mismatch_pos, 0))

            # Update send_output_token_logprobs_offset to match the new logprobs length
            # This is critical for streaming output to work correctly after verification
            if verify_logprobs is not None and req.output_token_logprobs_val is not None:
                req.send_output_token_logprobs_offset = len(req.output_token_logprobs_val)
            offset += padded_len  # Advance by padded length
        
        return rollback_info
