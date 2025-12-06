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
"""Deterministic verification info for TARGET_DET_VERIFY mode."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


class DetVerifyInfo:
    """
    Information for deterministic verification.
    
    Similar to EagleVerifyInput but specifically for validating
    deterministic inference outputs.
    """

    # Dummy token ID used for padding (typically 0 or pad_token_id)
    DUMMY_TOKEN_ID = 0

    def __init__(
        self,
        original_outputs: torch.Tensor,  # (total_output_tokens,)
        seq_lens: torch.Tensor,  # (batch_size,)
        output_lens: List[int],  # per-request actual output lengths (before padding)
        padded_lens: List[int],  # per-request padded lengths (after padding)
        padding_masks: List[List[bool]],  # per-request mask: True for real tokens, False for padding
        padding_counts: List[int],  # per-request number of padding predictions added
    ):
        self.original_outputs = original_outputs
        self.seq_lens = seq_lens
        self.output_lens = output_lens  # actual unverified tokens count
        self.padded_lens = padded_lens  # length after padding (may equal output_lens if no padding)
        self.padding_masks = padding_masks  # masks to identify real vs padded tokens
        self.padding_counts = padding_counts  # number of padding predictions per request
        
        # Calculate KV cache slots needed for padding
        # For each padded request: input = [context] + [actual_len tokens] + [padding_count - 1 dummies]
        # So we need (padding_count - 1) cache slots per padded request
        self.padding_cache_slots = [max(0, p - 1) for p in padding_counts]
        self.total_padding_cache_slots = sum(self.padding_cache_slots)
        
        # Will be set after KV cache allocation for padding
        self.padding_cache_locs: Optional[torch.Tensor] = None

    @classmethod
    def from_requests(
        cls, 
        reqs: List[Req], 
        start_idx: int = 0,
        always_align: bool = True,
    ) -> DetVerifyInfo:
        """
        Create DetVerifyInfo from a list of finished deterministic requests.
        
        Args:
            reqs: List of requests to verify
            start_idx: Index from which to start verifying tokens (for incremental verification)
            always_align: If True, pad finished requests to step_size with dummy tokens
            
        Returns:
            DetVerifyInfo instance
        """
        original_outputs = []
        seq_lens = []
        output_lens = []
        padded_lens = []
        padding_masks = []
        padding_counts = []
        
        for req in reqs:
            unverified_output_ids = req.output_ids[req.det_verified_tokens:]
            actual_len = len(unverified_output_ids)
            output_lens.append(actual_len)
            seq_lens.append(len(req.origin_input_ids) + len(req.output_ids))
            
            # Determine if padding is needed
            # Note: finished_output is a boolean (False initially), so use truthiness check, not "is not None"
            is_finished = req.finished_reason is not None
            step_size = getattr(req, 'det_step_size', None)
            
            if always_align and is_finished and step_size is not None and actual_len < step_size:
                # Pad to step_size with dummy tokens
                padding_needed = step_size - actual_len
                original_outputs.extend(unverified_output_ids)
                original_outputs.extend([cls.DUMMY_TOKEN_ID] * padding_needed)
                padded_lens.append(step_size)
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
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            output_lens=output_lens,
            padded_lens=padded_lens,
            padding_masks=padding_masks,
            padding_counts=padding_counts,
        )

    def allocate_padding_kv_cache(self, token_to_kv_pool_allocator) -> bool:
        """
        Allocate temporary KV cache slots for padding tokens.
        
        Args:
            token_to_kv_pool_allocator: The KV cache allocator
            
        Returns:
            True if allocation succeeded (or no padding needed), False otherwise
        """
        if self.total_padding_cache_slots == 0:
            return True
        
        # Allocate KV cache slots for padding input tokens
        self.padding_cache_locs = token_to_kv_pool_allocator.alloc(self.total_padding_cache_slots)
        
        if self.padding_cache_locs is None:
            logger.warning(
                f"[DET_VERIFY] Failed to allocate {self.total_padding_cache_slots} KV cache slots for padding. "
                f"Falling back to unpadded verification."
            )
            # Fallback: adjust padded_lens to match actual lens (no padding)
            # Also need to rebuild original_outputs without padding tokens
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
        if self.padding_cache_locs is not None and len(self.padding_cache_locs) > 0:
            token_to_kv_pool_allocator.free(self.padding_cache_locs)
            self.padding_cache_locs = None

    def prepare_verify_batch(
        self,
        original_batch: ScheduleBatch,
        reqs_to_verify: List[Req],
    ) -> ScheduleBatch:
        """
        Prepare a batch for verification with TARGET_DET_VERIFY mode.
        
        This creates input_ids containing the full sequence (input + output)
        to re-run through the model. When always_align is True, padded tokens
        use dummy values and will be masked out during comparison.
        
        Args:
            original_batch: Original batch context
            reqs_to_verify: Requests to verify
            
        Returns:
            Modified batch ready for verification
        """
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        
        verify_batch = ScheduleBatch(reqs=reqs_to_verify, batch_is_full=False)
        verify_batch.forward_mode = ForwardMode.TARGET_DET_VERIFY
        
        input_ids = []
        req_pool_indices = []
        output_lens = []  # This will store padded lengths for batch construction
        prefix_lens_list = []
        
        for i, req in enumerate(reqs_to_verify):
            actual_unverified = req.output_ids[req.det_verified_tokens:]
            padded_len = self.padded_lens[i]
            actual_len = self.output_lens[i]
            
            if padded_len == 0:
                continue
            
            last_verified_token = (
                req.output_ids[req.det_verified_tokens - 1]
                if req.det_verified_tokens > 0
                else req.origin_input_ids[-1] if req.origin_input_ids else req.input_ids[-1]
            )
            
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
                # actual_unverified has actual_len tokens, we use all of them as input
                # then add dummy tokens to reach padded_len total input tokens
                tokens_after_context = list(actual_unverified)  # all actual tokens
                padding_needed = padded_len - 1 - len(tokens_after_context)
                if padding_needed > 0:
                    tokens_after_context.extend([self.DUMMY_TOKEN_ID] * padding_needed)
                # Take first padded_len - 1 tokens (context token is the first)
                input_ids.extend([last_verified_token] + tokens_after_context[:padded_len - 1])
            else:
                # No padding: original logic
                # Input: [last_verified, actual_unverified[:-1]]
                input_ids.extend([last_verified_token] + list(actual_unverified[:-1]))
            
            req_pool_indices.append(req.req_pool_idx)
            output_lens.append(padded_len)
            prefix_lens_list.append(len(req.origin_input_ids) + req.det_verified_tokens - 1)
        
        device = original_batch.device
        verify_batch.input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        verify_batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=device)
        
        verify_batch.extend_lens = output_lens
        verify_batch.prefix_lens = prefix_lens_list
        
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=device)
        extend_lens_tensor = torch.tensor(output_lens, dtype=torch.int32, device=device)
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
        
        if verify_batch.sampling_info is not None:
            tokens_per_request = torch.tensor(output_lens, dtype=torch.int32, device=device)
            
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
            if verify_batch.sampling_info.deterministic_indices is not None:
                verify_batch.sampling_info.deterministic_indices = torch.repeat_interleave(
                    verify_batch.sampling_info.deterministic_indices, tokens_per_request, dim=0
                )
        
        out_cache_locs = []
        padding_offset = 0  # Track offset into allocated padding cache locations
        
        for i, req in enumerate(reqs_to_verify):
            current_seq_len = len(req.origin_input_ids) + len(req.output_ids)
            context_idx = len(req.origin_input_ids) + req.det_verified_tokens - 1
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
            # For padded case: we need actual_len - 1 more from existing, plus padding_count from allocated
            #
            # Input structure:
            #   Non-padded: [context, token_0, ..., token_{n-2}] = n tokens
            #   Padded:     [context, token_0, ..., token_{n-1}, dummy_0, ..., dummy_{p-1}] = n + p tokens
            #               where n = actual_len, p = padding_count
            #
            # For padded case, we include all actual tokens, so need cache for all of them
            
            start_idx = len(req.origin_input_ids) + req.det_verified_tokens
            
            if padding_count > 0:
                # Padded case: need all actual_len tokens' cache locations (not actual_len - 1)
                # The input includes [context] + [all actual tokens] + [dummies]
                num_actual_cache_needed = actual_len  # all actual tokens
                end_idx = min(start_idx + num_actual_cache_needed, current_seq_len)
                
                if start_idx < end_idx:
                    output_cache_locs = verify_batch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
                    out_cache_locs.extend(output_cache_locs.tolist())
                
                # Add allocated cache locations for padding input tokens
                # Input structure: [context] + [actual_len tokens] + [padding_cache_slots[i] dummies]
                # Total input = 1 + actual_len + padding_cache_slots[i] = padded_len
                padding_cache_needed = self.padding_cache_slots[i]
                if padding_cache_needed > 0 and self.padding_cache_locs is not None:
                    padding_locs = self.padding_cache_locs[padding_offset:padding_offset + padding_cache_needed]
                    out_cache_locs.extend(padding_locs.tolist())
                    padding_offset += padding_cache_needed
            else:
                # Non-padded case: need actual_len - 1 cache locations after context
                num_unverified = actual_len - 1
                end_idx = min(start_idx + num_unverified, current_seq_len)
                
                if start_idx < end_idx:
                    output_cache_locs = verify_batch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
                    out_cache_locs.extend(output_cache_locs.tolist())
        
        if len(out_cache_locs) != len(input_ids):
            logger.error(
                f"[DET_VERIFY] ERROR: Mismatch in cache location count! "
                f"expected={len(input_ids)}, actual={len(out_cache_locs)}, "
                f"input_ids length={len(input_ids)}"
            )
            raise RuntimeError(
                f"Verification batch has {len(input_ids)} input tokens but only "
                f"{len(out_cache_locs)} cache locations. This will cause memory corruption."
            )
        
        verify_batch.out_cache_loc = torch.tensor(out_cache_locs, dtype=torch.int32, device=device)
        
        return verify_batch

    def first_mismatch_position(
        self,
        original_ids: List[int],
        verified_ids: List[int],
    ) -> int:
        """
        Find the first mismatch position between original and verified token IDs.
        
        Args:
            original_ids: Original output token IDs
            verified_ids: Verified output token IDs
        Returns:
            Index of the first mismatch position, or length if all match
        """
        for i, (orig, verify) in enumerate(zip(original_ids, verified_ids)):
            if orig != verify:
                return i
        return min(len(original_ids), len(verified_ids))

    def verify_and_compare(
        self,
        reqs: List[Req],
        verified_token_ids: torch.Tensor,
        verified_logprobs: Optional[torch.Tensor] = None,
    ):
        """
        Compare original outputs with re-generated outputs.
        On mismatch: ROLLBACK and ACCEPT the verified token predicted at mismatch position.
        
        When always_align is used, padding tokens are masked out and not compared.
        
        Args:
            reqs: Requests that were verified
            verified_token_ids: Token IDs from verification run (argmax predictions)
            verified_logprobs: Log probabilities from verification run (optional)
            
        Returns:
            List of (mismatch_position, tokens_rolled_back) tuples for each request
        """
        
        verified_token_ids = verified_token_ids.tolist() if isinstance(verified_token_ids, torch.Tensor) else verified_token_ids
        verified_logprobs_list = verified_logprobs.tolist() if isinstance(verified_logprobs, torch.Tensor) else verified_logprobs
        original_ids = self.original_outputs.tolist()
        rollback_info = []
        offset = 0
        
        for i, req in enumerate(reqs):
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
                # req.finished_output = None
                
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
