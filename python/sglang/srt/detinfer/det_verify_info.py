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

    def __init__(
        self,
        original_outputs: torch.Tensor,  # (total_output_tokens,)
        seq_lens: torch.Tensor,  # (batch_size,)
        output_lens: List[int],  # per-request output lengths
    ):
        self.original_outputs = original_outputs
        self.seq_lens = seq_lens
        self.output_lens = output_lens

    @classmethod
    def from_requests(cls, reqs: List[Req], start_idx: int = 0) -> DetVerifyInfo:
        """
        Create DetVerifyInfo from a list of finished deterministic requests.
        
        Args:
            reqs: List of requests to verify
            start_idx: Index from which to start verifying tokens (for incremental verification)
            
        Returns:
            DetVerifyInfo instance
        """
        # Collect original outputs (only unverified portion)
        original_outputs = []
        seq_lens = []
        output_lens = []
        
        for req in reqs:
            # For incremental verification, only include unverified tokens
            # Get only the unverified tokens
            unverified_output_ids = req.output_ids[req.det_verified_tokens:]
            original_outputs.extend(unverified_output_ids)
            output_lens.append(len(unverified_output_ids))
            # Full sequence length (input + all outputs including verified)
            seq_lens.append(len(req.origin_input_ids) + len(req.output_ids))
        
        return cls(
            original_outputs=torch.tensor(original_outputs, dtype=torch.int64),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            output_lens=output_lens,
        )

    def prepare_verify_batch(
        self,
        original_batch: ScheduleBatch,
        reqs_to_verify: List[Req],
    ) -> ScheduleBatch:
        """
        Prepare a batch for verification with TARGET_DET_VERIFY mode.
        
        This creates input_ids containing the full sequence (input + output)
        to re-run through the model.
        
        Args:
            original_batch: Original batch context
            reqs_to_verify: Requests to verify
            
        Returns:
            Modified batch ready for verification
        """
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        
        # Create a new batch with only the requests to verify
        verify_batch = ScheduleBatch(
            reqs=reqs_to_verify,
            batch_is_full=False,
        )
        
        # Set forward mode
        verify_batch.forward_mode = ForwardMode.TARGET_DET_VERIFY
        
        # Prepare input_ids: only unverified output tokens
        # The input tokens and verified tokens are already in KV cache, we only need to verify new outputs
        input_ids = []
        req_pool_indices = []
        output_lens = []
        prefix_lens_list = []
        
        for req in reqs_to_verify:
            # Get only unverified tokens
            # logger.info(f"[DET_VERIFY] Preparing req {req.rid} for verification")
            # logger.info(f"[DET_VERIFY] origin_input_ids: {req.origin_input_ids}")
            # logger.info(f"[DET_VERIFY] output_ids: {req.output_ids}")
            # logger.info(f"[DET_VERIFY] det_verified_tokens: {req.det_verified_tokens}")
            # logger.info(f"[DET_VERIFY] unverified_tokens: {req.output_ids[req.det_verified_tokens:]}")
            unverified_tokens = req.output_ids[req.det_verified_tokens:]
            
            if not unverified_tokens:
                continue  # Skip if no unverified tokens
            
            # For verification, we need to include the last verified token (or last input token if nothing verified yet)
            # To verify N tokens, we input N-1 of them plus the context token (N tokens total)
            # Example: To verify [u0, u1, u2], input [context, u0, u1] → get predictions [u0, u1, u2]
            if req.det_verified_tokens > 0:
                # Use last verified output token
                last_verified_token = req.output_ids[req.det_verified_tokens - 1]
            else:
                # No tokens verified yet, use last input token
                # FIXME: Actually this is not needed. First token will always be correct. 
                last_verified_token = req.origin_input_ids[-1] if req.origin_input_ids else req.input_ids[-1]
            
            # Input: context + first (N-1) unverified tokens to verify all N tokens
            verification_input = [last_verified_token] + unverified_tokens[:-1]
            input_ids.extend(verification_input)
            
            # Use existing req_pool_idx
            req_pool_indices.append(req.req_pool_idx)
            
            # Track lengths: we input N tokens (1 context + N-1 unverified) to verify N unverified tokens
            output_lens.append(len(unverified_tokens))
            # Prefix length includes: all inputs + verified outputs (if any)
            prefix_lens_list.append(len(req.origin_input_ids) + req.det_verified_tokens - 1)
        
        # Set batch attributes - ensure all tensors are on the correct device
        device = original_batch.device
        verify_batch.input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        verify_batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=device)
        
        # Use extend_lens and prefix_lens which are used by get_model_worker_batch()
        # extend_lens should be the number of tokens we're extending: N tokens (1 context + N-1 unverified)
        extend_lens_with_context = [length for length in output_lens]  # Already N tokens per request
        verify_batch.extend_lens = extend_lens_with_context
        verify_batch.prefix_lens = prefix_lens_list
        
        # seq_lens should be the total sequence length (prefix + extend tokens)
        # FlashInfer uses this to determine where in the KV cache to access
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=device)
        extend_lens_tensor = torch.tensor(extend_lens_with_context, dtype=torch.int32, device=device)
        total_seq_lens = prefix_lens + extend_lens_tensor
        verify_batch.seq_lens = total_seq_lens
        verify_batch.seq_lens_cpu = total_seq_lens.cpu()
        # For verification, we start sampling from position 1 (skip the last input token)
        # This gives us logits for all N output positions
        verify_batch.extend_logprob_start_lens = [1] * len(reqs_to_verify)
        
        verify_batch.extend_num_tokens = len(input_ids)  # Total tokens including last input tokens
        verify_batch.seq_lens_sum = total_seq_lens.sum().item()
        verify_batch.orig_seq_lens = total_seq_lens.clone()
        
        # Enable return_logprob to get logits for all tokens (not just last token per request)
        # This is critical for multi-token verification where we need to sample all tokens
        verify_batch.return_logprob = True
        verify_batch.top_logprobs_nums = [0] * len(reqs_to_verify)  # Don't need top-k logprobs
        verify_batch.token_ids_logprobs = [[] for _ in reqs_to_verify]  # No specific token logprobs needed
        
        # Copy other necessary attributes from original batch
        verify_batch.req_to_token_pool = original_batch.req_to_token_pool
        verify_batch.token_to_kv_pool_allocator = original_batch.token_to_kv_pool_allocator
        verify_batch.tree_cache = original_batch.tree_cache
        verify_batch.model_config = original_batch.model_config
        verify_batch.device = device
        
        # CRITICAL FIX: Create sampling_info specifically for the verify_batch requests
        # instead of copying from original_batch which may have more requests
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        verify_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            verify_batch, original_batch.model_config.vocab_size
        )
        
        # CRITICAL FIX for temperature > 0:
        # Expand sampling_info tensors to match the number of tokens being verified
        # Original sampling_info has shape (num_requests,) but we need (total_tokens,)
        # where total_tokens = sum(output_lens)
        if verify_batch.sampling_info is not None:
            # Calculate how many tokens each request contributes
            tokens_per_request = torch.tensor(output_lens, dtype=torch.int32, device=device)
            
            # Expand each tensor in sampling_info using repeat_interleave
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
        
        # Instead of allocating new cache, reuse original KV cache locations
        # Build out_cache_loc by extracting existing locations from req_to_token_pool
        out_cache_locs = []
        
        for i, req in enumerate(reqs_to_verify):
            # Calculate the actual current position in the sequence
            # This is the total length including input and ALL generated outputs so far
            current_seq_len = len(req.origin_input_ids) + len(req.output_ids)
            
            # logger.info(
            #     f"[DET_VERIFY] Processing req {req.rid}: "
            #     f"origin_input_ids_len={len(req.origin_input_ids)}, "
            #     f"output_ids_len={len(req.output_ids)}, "
            #     f"det_verified_tokens={req.det_verified_tokens}, "
            #     f"current_seq_len={current_seq_len}"
            # )
            
            # For the context token (last verified/input token), get its existing cache location
            # Make sure we don't access beyond what's been written to req_to_token_pool
            context_idx = len(req.origin_input_ids) + req.det_verified_tokens - 1
            
            # Ensure context_idx is within bounds of what's been allocated
            if context_idx >= current_seq_len:
                logger.error(
                    f"ERROR: context_idx {context_idx} >= current_seq_len {current_seq_len} "
                    f"for req {req.rid}. This indicates a logic error."
                )
                raise RuntimeError(
                    f"Attempting to access unallocated cache position {context_idx} "
                    f"when only {current_seq_len} positions have been allocated"
                )
            
            # logger.info(
            #     f"[DET_VERIFY] Reading context_cache_loc from req_to_token_pool at "
            #     f"[{req.req_pool_idx}, {context_idx}]"
            # )
            
            context_cache_loc = verify_batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, context_idx
            ]
            
            # logger.info(f"[DET_VERIFY] context_cache_loc = {context_cache_loc.item()}")
            
            out_cache_locs.append(context_cache_loc.item())
            
            # For unverified output tokens, get their existing cache locations
            # We input N-1 unverified tokens (u0 to u(N-2)) to verify all N tokens
            # So we only need cache positions for the first N-1 unverified tokens
            start_idx = len(req.origin_input_ids) + req.det_verified_tokens
            # Only read cache locations for N-1 unverified tokens (excluding the last one)
            num_unverified = len(req.output_ids) - req.det_verified_tokens
            end_idx = start_idx + num_unverified - 1  # Exclude last unverified token
            
            # However, we need to ensure we're only reading cache locations that were
            # actually written. The last token's cache location might not be in req_to_token_pool
            # yet if it was just generated.
            # current_seq_len = len(input) + len(output_ids), which includes the last token
            # But the last token might have been added to output_ids without its cache being written yet
            max_readable_idx = current_seq_len - 1  # Last position that should have cache written
            
            if end_idx > max_readable_idx:
                # logger.info(
                #     f"[DET_VERIFY] Adjusting end_idx from {end_idx} to {max_readable_idx} "
                #     f"to avoid reading unwritten cache positions for req {req.rid}"
                # )
                end_idx = max_readable_idx
            
            # If after adjustment we have no tokens to read, skip
            if start_idx >= end_idx:
                # logger.info(
                #     f"[DET_VERIFY] No output cache locations to read for req {req.rid} "
                #     f"(start_idx={start_idx}, end_idx={end_idx})"
                # )
                output_cache_locs_list = []
            else:
                # logger.info(
                #     f"[DET_VERIFY] Reading output_cache_locs from req_to_token_pool at "
                #     f"[{req.req_pool_idx}, {start_idx}:{end_idx}] (first {end_idx - start_idx} unverified tokens)"
                # )
                
                output_cache_locs = verify_batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, start_idx:end_idx
                ]
                output_cache_locs_list = output_cache_locs.tolist()
            
            # logger.info(f"[DET_VERIFY] output_cache_locs = {output_cache_locs_list}")
            
            out_cache_locs.extend(output_cache_locs_list)
            
            # Note: We input N tokens [context, u0, ..., u(N-2)] to verify N tokens [u0, ..., u(N-1)]
            # The forward pass generates K/V for all N input tokens
            # Cache positions: [context_pos, u0_pos, ..., u(N-2)_pos]
            # The last unverified token u(N-1) doesn't get K/V generated for it
        
        # logger.info(f"[DET_VERIFY] Final out_cache_locs = {out_cache_locs}")
        
        # Validate that we have the expected number of cache locations
        expected_num_locs = len(input_ids)
        if len(out_cache_locs) != expected_num_locs:
            logger.error(
                f"[DET_VERIFY] ERROR: Mismatch in cache location count! "
                f"expected={expected_num_locs}, actual={len(out_cache_locs)}, "
                f"input_ids length={len(input_ids)}"
            )
            raise RuntimeError(
                f"Verification batch has {len(input_ids)} input tokens but only "
                f"{len(out_cache_locs)} cache locations. This will cause memory corruption."
            )
        
        # Set out_cache_loc to point to existing cache locations
        verify_batch.out_cache_loc = torch.tensor(
            out_cache_locs, dtype=torch.int32, device=device
        )
        
        # logger.info(
        #     f"Reusing {len(out_cache_locs)} existing KV cache locations for in-place verification "
        #     f"(N tokens input = 1 context + N-1 unverified, to verify N unverified tokens)"
        # )
        
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
        len_orig = len(original_ids)
        len_verify = len(verified_ids)
        n = len_orig if len_orig < len_verify else len_verify

        for i in range(n):
            if original_ids[i] != verified_ids[i]:
                return i
        return n

    def verify_and_compare(
        self,
        reqs: List[Req],
        verified_token_ids: torch.Tensor,
        verified_logprobs: Optional[torch.Tensor] = None,
    ):
        """
        Compare original outputs with re-generated outputs.
        On mismatch: ROLLBACK and ACCEPT the verified token predicted at mismatch position.
        
        Args:
            reqs: Requests that were verified
            verified_token_ids: Token IDs from verification run (argmax predictions)
            verified_logprobs: Log probabilities from verification run (optional)
            
        Returns:
            List of (mismatch_position, tokens_rolled_back, tokens_accepted) tuples
            None for each request that passed verification
        """
        # Convert to list for comparison
        if isinstance(verified_token_ids, torch.Tensor):
            verified_token_ids = verified_token_ids.tolist()
        
        # Convert logprobs to list if provided
        verified_logprobs_list = None
        if verified_logprobs is not None:
            if isinstance(verified_logprobs, torch.Tensor):
                verified_logprobs_list = verified_logprobs.tolist()
            else:
                verified_logprobs_list = verified_logprobs
        
        original_ids = self.original_outputs.tolist()
        rollback_info = []
        
        logger.info(f"[DET_DEBUG] verified_token_ids (full): {verified_token_ids[:100]}")  # First 100 tokens
        logger.info(f"[DET_DEBUG] original_ids (full): {original_ids[:100]}")  # First 100 tokens
        logger.info(f"[DET_DEBUG] output_lens: {self.output_lens}")
        
        # Compare per-request
        offset = 0
        for i, req in enumerate(reqs):
            output_len = self.output_lens[i]
            orig_output = original_ids[offset : offset + output_len]
            verify_output = verified_token_ids[offset : offset + output_len]

            logger.info(
                f"Request {req.rid}: offset={offset}, output_len={output_len}"
            )
            logger.info(
                f"Request {req.rid}: Original output IDs: {orig_output}"
            )
            logger.info(
                f"Request {req.rid}: Verified output IDs: {verify_output}"
            )
            
            # Find FIRST mismatch position
            mismatch_pos = self.first_mismatch_position(orig_output, verify_output)
            
            logger.info(
                f"[DET_DEBUG] Request {req.rid}: mismatch_pos={mismatch_pos}, "
                f"output_len={output_len}, orig_len={len(orig_output)}, verify_len={len(verify_output)}"
            )

            mismatch_pos = min(4, output_len)  # testing

            # mismatch_pos = min(len(orig_output), 3)  # For debugging
            tokens_to_rollback = len(orig_output) - mismatch_pos
            
            if tokens_to_rollback > 0:
                # Truncate to mismatch position (removes wrong tokens from mismatch onwards)
                req.output_ids = req.output_ids[:-tokens_to_rollback]
                
                # Update logprobs with verified values
                if verified_logprobs_list is not None and hasattr(req, 'output_token_logprobs') and req.output_token_logprobs is not None:
                    # Truncate existing logprobs to mismatch position
                    req.output_token_logprobs = req.output_token_logprobs[:-tokens_to_rollback]
                    
                    # Append verified logprobs up to mismatch position
                    verified_logprobs_for_req = verified_logprobs_list[offset : offset + mismatch_pos]
                    req.output_token_logprobs.extend(verified_logprobs_for_req)
                
                # logger.info(
                #     f"Request {req.rid}: Rolled back {tokens_to_rollback} tokens from position {mismatch_pos} onwards. "
                #     f" req.output_ids: {req.output_ids}"
                # )
                
                # Clear finished state after rollback
                # logger.info(
                #     f"Request {req.rid}: Clearing finished_reason after rollback "
                #     f"(was: {req.finished_reason}). Request will continue generation."
                # )
                req.finished_reason = None
                req.finished_output = None
                
                # ACCEPT the verified token at mismatch position!
                # It was predicted with correct KV context from the previous correct token
                if mismatch_pos < len(verify_output):
                    verified_token_at_mismatch = verify_output[mismatch_pos]
                    req.output_ids.append(verified_token_at_mismatch)
                    
                    # Also append the logprob for the accepted token
                    if verified_logprobs_list is not None and hasattr(req, 'output_token_logprobs') and req.output_token_logprobs is not None:
                        accepted_token_logprob = verified_logprobs_list[offset + mismatch_pos]
                        req.output_token_logprobs.append(accepted_token_logprob)
                
                    logger.info(
                        f"After adding accepted token at mismatch, req.output_ids: {req.output_ids}, "
                        f"finished_reason: {req.finished_reason}"
                    )

                    tokens_to_rollback -= 1  # Since we added one accepted token
                
                rollback_info.append((mismatch_pos, tokens_to_rollback))
            else:
                # No mismatch - all tokens verified correctly
                # Update logprobs with verified values for all tokens
                if verified_logprobs_list is not None and hasattr(req, 'output_token_logprobs') and req.output_token_logprobs is not None:
                    # Replace the logprobs for the verified portion
                    verified_logprobs_for_req = verified_logprobs_list[offset : offset + output_len]
                    # Update the last output_len logprobs with verified values
                    start_idx = len(req.output_token_logprobs) - output_len
                    if start_idx >= 0:
                        req.output_token_logprobs[start_idx:] = verified_logprobs_for_req
                    else:
                        # Edge case: if output_token_logprobs is shorter than expected
                        req.output_token_logprobs = verified_logprobs_for_req[-len(req.output_token_logprobs):]
                
                rollback_info.append((mismatch_pos, 0))
            
            # Mark for future deterministic generation
            # req.force_deterministic_mode = True
            
            # CRITICAL: Move offset inside the loop so each request gets the correct slice
            offset += output_len
        
        return rollback_info
