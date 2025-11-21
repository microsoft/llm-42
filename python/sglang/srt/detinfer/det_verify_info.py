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
from typing import TYPE_CHECKING, List

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
            logger.info(f"[DET_VERIFY] Preparing req {req.rid} for verification")
            logger.info(f"[DET_VERIFY] origin_input_ids: {req.origin_input_ids}")
            logger.info(f"[DET_VERIFY] output_ids: {req.output_ids}")
            logger.info(f"[DET_VERIFY] det_verified_tokens: {req.det_verified_tokens}")
            logger.info(f"[DET_VERIFY] unverified_tokens: {req.output_ids[req.det_verified_tokens:]}")
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
        
        # Copy other necessary attributes from original batch
        verify_batch.req_to_token_pool = original_batch.req_to_token_pool
        verify_batch.token_to_kv_pool_allocator = original_batch.token_to_kv_pool_allocator
        verify_batch.tree_cache = original_batch.tree_cache
        verify_batch.model_config = original_batch.model_config
        verify_batch.sampling_info = original_batch.sampling_info
        verify_batch.device = device
        
        # Instead of allocating new cache, reuse original KV cache locations
        # Build out_cache_loc by extracting existing locations from req_to_token_pool
        out_cache_locs = []
        
        for i, req in enumerate(reqs_to_verify):
            # Calculate the actual current position in the sequence
            # This is the total length including input and ALL generated outputs so far
            current_seq_len = len(req.origin_input_ids) + len(req.output_ids)
            
            logger.info(
                f"[DET_VERIFY] Processing req {req.rid}: "
                f"origin_input_ids_len={len(req.origin_input_ids)}, "
                f"output_ids_len={len(req.output_ids)}, "
                f"det_verified_tokens={req.det_verified_tokens}, "
                f"current_seq_len={current_seq_len}"
            )
            
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
            
            logger.info(
                f"[DET_VERIFY] Reading context_cache_loc from req_to_token_pool at "
                f"[{req.req_pool_idx}, {context_idx}]"
            )
            
            context_cache_loc = verify_batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, context_idx
            ]
            
            logger.info(f"[DET_VERIFY] context_cache_loc = {context_cache_loc.item()}")
            
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
                logger.info(
                    f"[DET_VERIFY] Adjusting end_idx from {end_idx} to {max_readable_idx} "
                    f"to avoid reading unwritten cache positions for req {req.rid}"
                )
                end_idx = max_readable_idx
            
            # If after adjustment we have no tokens to read, skip
            if start_idx >= end_idx:
                logger.info(
                    f"[DET_VERIFY] No output cache locations to read for req {req.rid} "
                    f"(start_idx={start_idx}, end_idx={end_idx})"
                )
                output_cache_locs_list = []
            else:
                logger.info(
                    f"[DET_VERIFY] Reading output_cache_locs from req_to_token_pool at "
                    f"[{req.req_pool_idx}, {start_idx}:{end_idx}] (first {end_idx - start_idx} unverified tokens)"
                )
                
                output_cache_locs = verify_batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, start_idx:end_idx
                ]
                output_cache_locs_list = output_cache_locs.tolist()
            
            logger.info(f"[DET_VERIFY] output_cache_locs = {output_cache_locs_list}")
            
            logger.info(f"[DET_VERIFY] output_cache_locs = {output_cache_locs_list}")
            
            out_cache_locs.extend(output_cache_locs_list)
            
            # Note: We input N tokens [context, u0, ..., u(N-2)] to verify N tokens [u0, ..., u(N-1)]
            # The forward pass generates K/V for all N input tokens
            # Cache positions: [context_pos, u0_pos, ..., u(N-2)_pos]
            # The last unverified token u(N-1) doesn't get K/V generated for it
        
        logger.info(f"[DET_VERIFY] Final out_cache_locs = {out_cache_locs}")
        
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
        
        logger.info(
            f"Reusing {len(out_cache_locs)} existing KV cache locations for in-place verification "
            f"(N tokens input = 1 context + N-1 unverified, to verify N unverified tokens)"
        )
        
        return verify_batch

    def verify_and_compare(
        self,
        reqs: List[Req],
        verified_token_ids: torch.Tensor,
    ):
        """
        Compare original outputs with re-generated outputs.
        On mismatch: ROLLBACK and ACCEPT the verified token predicted at mismatch position.
        
        KEY INSIGHT: During verification, we feed [last_verified, unverified0, unverified1, ...]
        - If unverified_i matches its prediction → it's CORRECT, its KV is correct
        - The logits AFTER processing unverified_i predict what token_(i+1) SHOULD be
        - These logits were computed with CORRECT KV context (from correct unverified_i)
        
        Example: tokens [x0, x1, x2, x3, x4, x5, x6] generate predictions [y1, y2, y3, y4, y5, y6, y7]
        If x1==y1, x2==y2, x3==y3, x4==y4, but x5!=y5 (mismatch at position 5):
        
        - Process x0 → predict y1, x1==y1 ✓ → x1 is correct! KV[0] from embedding(x0) is correct
        - Process x1 → predict y2, x2==y2 ✓ → x2 is correct! KV[1] from embedding(x1) is correct
        - Process x2 → predict y3, x3==y3 ✓ → x3 is correct! KV[2] from embedding(x2) is correct
        - Process x3 → predict y4, x4==y4 ✓ → x4 is correct! KV[3] from embedding(x3) is correct
        - Process x4 → predict y5, x5!=y5 ✗ → MISMATCH! But:
          * x4 was correct, so KV[4] from embedding(x4) is correct!
          * Logits after x4 predict y5 with CORRECT KV context
          * We CAN accept y5! ✓
        - Process x5 (wrong) → KV[5] from embedding(x5_WRONG) is wrong ✗
        
        Result: Accept [x0, x1, x2, x3, x4, y5], discard [x6, x7]
        
        Args:
            reqs: Requests that were verified
            verified_token_ids: Token IDs from verification run (argmax predictions)
            
        Returns:
            List of (mismatch_position, tokens_rolled_back, tokens_accepted) tuples
            None for each request that passed verification
        """
        # Convert to list for comparison
        if isinstance(verified_token_ids, torch.Tensor):
            verified_token_ids = verified_token_ids.tolist()
        
        original_ids = self.original_outputs.tolist()
        rollback_info = []
        
        # Compare per-request
        offset = 0
        for i, req in enumerate(reqs):
            output_len = self.output_lens[i]
            orig_output = original_ids[offset : offset + output_len]
            verify_output = verified_token_ids[offset : offset + output_len]
            
            # Check if there's an extra token generated (bonus token!)
            # The verification generates N+1 logits for N tokens input
            # So we have one extra prediction beyond the unverified tokens
            extra_token_idx = offset + output_len
            has_extra_token = extra_token_idx < len(verified_token_ids)
            
        #     if orig_output == verify_output:
        #         # Verification passed - all tokens matched!
        #         req.det_verified = True
        #         req.det_mismatch = False
                
        #         # Accept the extra token generated by verification (if available)
        #         # if has_extra_token:
        #         #     extra_token = verified_token_ids[extra_token_idx]
        #         #     req.output_ids.append(extra_token)
        #         #     # Signal that we accepted an extra token (no rollback, but 1 token accepted)
        #         #     # Format: (position, tokens_rolled_back, tokens_accepted)
        #         #     rollback_info.append((output_len, 0, 1))
        #         #     logger.info(
        #         #         f"Request {req.rid}: Deterministic verification PASSED! "
        #         #         f"All {output_len} tokens matched. "
        #         #         f"BONUS: Accepted 1 extra generated token: {extra_token}"
        #         #     )
        #         # else:
        #         #     rollback_info.append(None)
        #         #     logger.info(
        #         #         f"Request {req.rid}: Deterministic verification PASSED! "
        #         #         f"All {output_len} tokens matched."
        #         #     )
        #         rollback_info.append(None)
        #         logger.info(f"[DET_DEBUG] Request {req.rid}: Setting det_verified=True after verification PASSED")
        #     else:
            # Mismatch detected - ROLLBACK ALL incorrect tokens
            req.det_verified = True
            req.det_mismatch = True
            
            # Find FIRST mismatch position
            mismatch_pos = next(
                (j for j in range(min(len(orig_output), len(verify_output))) 
                    if orig_output[j] != verify_output[j]),
                min(len(orig_output), len(verify_output))
            )

            mismatch_pos = 8  # For debugging
            
            logger.error(
                f"Request {req.rid}: Deterministic verification FAILED at position {mismatch_pos}\n"
                f"  Original tokens:  {orig_output}\n"
                f"  Verified tokens:  {verify_output}\n"
                f"  Divergence at position {mismatch_pos}: "
                f"original={orig_output[mismatch_pos] if mismatch_pos < len(orig_output) else 'N/A'}, "
                f"verified={verify_output[mismatch_pos] if mismatch_pos < len(verify_output) else 'N/A'}"
            )
            
            # ROLLBACK: Remove tokens from mismatch position onwards
            # Tokens BEFORE mismatch are verified correct (argmax matched)
            # Token AT mismatch: The PREVIOUS token was correct!
            #   - Previous token (mismatch_pos - 1) matched, so its KV is correct
            #   - Logits AFTER processing previous token predict what mismatch_pos SHOULD be
            #   - These logits were computed with CORRECT KV context!
            #   - So we CAN accept the verified token at mismatch position!
            # Tokens AFTER mismatch: Their KV was computed from wrong token at mismatch
            tokens_to_rollback = len(orig_output) - mismatch_pos
            
            if tokens_to_rollback > 0:
                # Truncate to mismatch position (removes wrong tokens from mismatch onwards)
                req.output_ids = req.output_ids[:-tokens_to_rollback]
                
                # ACCEPT the verified token at mismatch position!
                # It was predicted with correct KV context from the previous correct token
                if mismatch_pos < len(verify_output):
                    verified_token_at_mismatch = verify_output[mismatch_pos]
                    req.output_ids.append(verified_token_at_mismatch)
                    tokens_accepted = 1
                else:
                    tokens_accepted = 0
                
                rollback_info.append((mismatch_pos, tokens_to_rollback, tokens_accepted))
                
                logger.info(
                    f"Request {req.rid}: ROLLBACK complete\n"
                    f"  - Mismatch at position {mismatch_pos}\n"
                    f"  - Rolled back {tokens_to_rollback} tokens (from position {mismatch_pos} onwards)\n"
                    f"  - Kept {mismatch_pos} verified correct tokens (positions 0-{mismatch_pos-1})\n"
                    f"  - ACCEPTED {tokens_accepted} verified token at mismatch: {verified_token_at_mismatch if tokens_accepted else 'N/A'}\n"
                    f"  - (Previous token was correct, so mismatch prediction has correct KV context!)\n"
                    f"  - New output_ids length: {len(req.output_ids)}\n"
                    f"  - KV cache up to position {len(req.origin_input_ids) + mismatch_pos + tokens_accepted - 1} is CORRECT\n"
                    f"  - KV cache positions {len(req.origin_input_ids) + mismatch_pos + tokens_accepted}+ need to be freed\n"
                    f"  - Will regenerate from position {mismatch_pos + tokens_accepted} in DETERMINISTIC mode"
                )
            else:
                # Edge case: no tokens to rollback
                rollback_info.append((mismatch_pos, 0, 0))
            
            # Mark for future deterministic generation
            req.force_deterministic_mode = True
            logger.info(f"[DET_DEBUG] Request {req.rid}: Setting det_verified=True after verification FAILED")
        
        offset += output_len
        
        return rollback_info
