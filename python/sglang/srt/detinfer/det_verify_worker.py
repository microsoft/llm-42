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
from typing import TYPE_CHECKING, List, Union

import torch

from sglang.srt.detinfer.det_verify_info import DetVerifyInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatchOutput, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker

logger = logging.getLogger(__name__)


class DeterministicVerificationWorker:
    """
    Wraps a target worker to provide deterministic verification at scheduler level.
    
    Similar to EagleWorker but for deterministic inference validation:
    - Operates on ScheduleBatch objects (not ModelWorkerBatch)
    - Identifies finished deterministic requests after forward pass
    - Re-runs them with TARGET_DET_VERIFY mode
    - Compares outputs to detect any non-determinism
    """

    def __init__(self, target_worker: TpModelWorker, det_step_size: int = None):
        """
        Initialize the deterministic verification worker.
        
        Args:
            target_worker: The underlying TpModelWorker to wrap
            det_step_size: Number of tokens to generate before verification.
                          If None, verify only at the end. If set, verify every N tokens.
        """
        self.target_worker = target_worker
        self.det_step_size = det_step_size
        self.stats = {
            "total_verified": 0,
            "total_mismatches": 0,
        }

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
    ):
        """
        Check for deterministic requests that need verification and verify them.
        Should be called AFTER tokens have been appended and check_finished() called.
        
        This is called from process_batch_result_decode, similar to how Eagle
        handles verification after output processing.
        
        Args:
            batch: Current ScheduleBatch or ModelWorkerBatch after output processing
        """
        logger.info(f"[DET_DEBUG] check_and_verify_deterministic_requests called with {len(batch.reqs) if batch.reqs else 0} requests")
        
        if batch.reqs is None:
            logger.info("[DET_DEBUG] batch.reqs is None, returning")
            return
        
        reqs_to_verify = []
        
        for req in batch.reqs:
            logger.info(
                f"[DET_DEBUG] Checking req {req.rid}: is_deterministic={req.is_deterministic}, "
                f"det_verified={req.det_verified}, finished={req.finished()}, "
                f"finished_reason={req.finished_reason}, "
                f"output_ids_len={len(req.output_ids)}, det_verified_tokens={req.det_verified_tokens}"
            )
            
            if not req.is_deterministic:
                logger.info(f"[DET_DEBUG] req {req.rid} is NOT deterministic, skipping")
                continue
            
            # At this point, tokens are already appended and check_finished() was called
            # Check if request is finished
            if req.finished():
                # For finished requests, check if there are unverified tokens
                unverified_tokens = len(req.output_ids) - req.det_verified_tokens
                if unverified_tokens > 0:
                    logger.info(
                        f"[DET_DEBUG] req {req.rid} is finished with {unverified_tokens} unverified tokens, "
                        f"adding to verify list"
                    )
                    reqs_to_verify.append(req)
                else:
                    logger.info(
                        f"[DET_DEBUG] req {req.rid} is finished and all {req.det_verified_tokens} tokens "
                        f"already verified, skipping"
                    )
                continue
            else:
                logger.info(f"[DET_DEBUG] req {req.rid} is NOT finished yet")
            
            # Check for incremental verification at det_step_size boundary
            if self.det_step_size is not None:
                total_tokens = len(req.output_ids)
                unverified_tokens = total_tokens - req.det_verified_tokens
                
                logger.info(
                    f"[DET_DEBUG] rid={req.rid}, output_ids_len={total_tokens}, "
                    f"verified={req.det_verified_tokens}, unverified={unverified_tokens}, "
                    f"det_step_size={self.det_step_size}"
                )
                
                if unverified_tokens >= self.det_step_size:
                    reqs_to_verify.append(req)
        
        if reqs_to_verify:
            logger.info(
                f"Found {len(reqs_to_verify)} deterministic requests to verify"
            )
            self._verify_deterministic_requests(batch, reqs_to_verify)


    def _verify_deterministic_requests(
        self, 
        original_batch: Union[ScheduleBatch, ModelWorkerBatch], 
        reqs: List[Req]
    ):
        """
        Verify deterministic requests by re-running them.
        Tokens are already in req.output_ids at this point.
        
        Args:
            original_batch: Original batch context
            reqs: Requests to verify (tokens already appended)
        """
        try:
            # Create DetVerifyInfo from requests
            det_verify_info = DetVerifyInfo.from_requests(reqs)
            
            # Prepare verification batch
            verify_batch = det_verify_info.prepare_verify_batch(original_batch, reqs)
            
            logger.info(
                f"Running deterministic verification for {len(reqs)} requests "
                f"with {verify_batch.extend_num_tokens} tokens"
            )
            
            # Convert to ModelWorkerBatch and run verification forward pass
            verify_model_worker_batch = verify_batch.get_model_worker_batch()
            verify_output = self.target_worker.forward_batch_generation(
                verify_model_worker_batch
            )

            # For TARGET_DET_VERIFY, we need to sample using the SAME method as original generation.
            # We use standard deterministic sampling with the same parameters (temperature, top_k, top_p).
            # The model_runner.sample() method handles this correctly:
            # - Applies temperature, top_k, top_p filtering
            # - Uses deterministic sampling with the request's sampling seed + position
            # - For greedy (temperature=0), uses argmax
            # - For non-greedy, samples deterministically based on seed
            
            # Get logits from verification output and sample using standard deterministic sampling
            # Use model_runner.sample() which applies proper deterministic sampling
            # The verify_model_worker_batch already has the correct sampling_info with seeds
            verified_token_ids = self.target_worker.model_runner.sample(
                verify_output.logits_output,
                verify_model_worker_batch
            )
            logger.info(f"[DET_DEBUG] verify_model_worker_batch sampling_info: {verify_model_worker_batch.sampling_info}")
            logger.info(f"[DET_DEBUG] reqs[0].output_ids: {reqs[0].output_ids}")
            logger.info(f"[DET_DEBUG] verify_output.logits_output: {verify_output.logits_output}")
            logger.info(f"[DET_DEBUG] verified_token_ids: {verified_token_ids}")
            
            # Compare outputs and handle rollback
            rollback_info = det_verify_info.verify_and_compare(reqs, verified_token_ids)
            
            # Handle KV cache for rolled-back tokens
            # rollback_info is a list where each element is either:
            #   None (verification passed, no rollback)
            #   (mismatch_pos, tokens_rolled_back, tokens_accepted) tuple
            self._handle_kv_cache_rollback(original_batch, reqs, rollback_info, det_verify_info)
            
            # Note: For requests with rollback:
            # - output_ids has been updated (rolled back + accepted verified tokens)
            # - KV cache has been partially freed (for rolled-back tokens)
            # - The verified tokens' KV cache is already in place from verification forward pass
            # - No need to copy - verification wrote directly to original locations
            logger.info(
                f"Verification complete for {len(reqs)} requests. "
                f"Rollbacks handled, KV cache updated."
            )
            
            # Update det_verified_tokens counter for incremental verification
            # rollback_info is (mismatch_pos, tokens_rolled_back, tokens_accepted) or None
            for i, (req, info) in enumerate(zip(reqs, rollback_info)):
                if info is None:
                    # Verification passed, all tokens verified
                    req.det_verified_tokens = len(req.output_ids)
                else:
                    # Rollback occurred - tokens up to mismatch are verified, plus we accepted the corrected token
                    mismatch_pos, tokens_rolled_back, tokens_accepted = info
                    # After rollback + accept, output_ids contains verified tokens + 1 accepted corrected token
                    req.det_verified_tokens = len(req.output_ids)
                    logger.info(
                        f"Request {req.rid}: Updated det_verified_tokens to {req.det_verified_tokens} "
                        f"(after rollback of {tokens_rolled_back} tokens and acceptance of {tokens_accepted} corrected token)"
                    )
            
            # Clean up verification batch resources
            # Since we manually set out_cache_loc to reuse existing locations,
            # we need to ensure no state is tracked as "newly allocated"
            # The verification batch should not affect the allocator's free list
            
            # Clear verification batch references to help with cleanup
            verify_batch.out_cache_loc = None
            verify_batch.input_ids = None
            verify_model_worker_batch = None
            verify_output = None
            verified_token_ids = None
            
            # Update statistics
            self.stats["total_verified"] += len(reqs)
            mismatches = sum(1 for req in reqs if req.det_mismatch)
            self.stats["total_mismatches"] += mismatches
            
            # Update batch.output_ids AND seq_lens to reflect the modified req.output_ids after verification
            # This is crucial because prepare_for_decode() uses batch.output_ids and seq_lens
            if hasattr(original_batch, 'output_ids') and original_batch.output_ids is not None:
                # Rebuild output_ids tensor from the updated req.output_ids lists
                # Each request contributes its last generated token
                updated_output_ids = []
                updated_seq_lens = []
                for req in original_batch.reqs:
                    if len(req.output_ids) > 0:
                        updated_output_ids.append(req.output_ids[-1])
                    else:
                        # Fallback to last input token if no outputs yet
                        updated_output_ids.append(req.origin_input_ids[-1] if req.origin_input_ids else 0)
                    # Update seq_lens to match actual sequence length after rollback
                    updated_seq_lens.append(len(req.origin_input_ids) + len(req.output_ids))
                
                # Update the batch tensors
                original_batch.output_ids = torch.tensor(
                    updated_output_ids, 
                    dtype=torch.int64, 
                    device=original_batch.device
                )
                
                # CRITICAL: Update seq_lens to reflect rolled-back state
                # Otherwise prepare_for_decode() will use stale seq_lens and allocate wrong cache locations
                original_batch.seq_lens = torch.tensor(
                    updated_seq_lens,
                    dtype=torch.int32,
                    device=original_batch.device
                )
                original_batch.seq_lens_cpu = torch.tensor(
                    updated_seq_lens,
                    dtype=torch.int32
                )
                original_batch.orig_seq_lens = original_batch.seq_lens.clone()
                original_batch.seq_lens_sum = sum(updated_seq_lens)
                
                logger.info(
                    f"Updated batch state after verification: "
                    f"output_ids={original_batch.output_ids}, "
                    f"seq_lens={original_batch.seq_lens}, "
                    f"seq_lens_sum={original_batch.seq_lens_sum}"
                )
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            raise
        
        logger.info(
            f"Verification complete. Total verified: {self.stats['total_verified']}, "
            f"Total mismatches: {self.stats['total_mismatches']}"
        )

    def _handle_kv_cache_rollback(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        reqs: List[Req],
        rollback_info: List,
        det_verify_info,
    ):
        """
        Handle KV cache state after rollback.
        
        CRITICAL: We MUST free the KV cache for rolled-back tokens!
        Unlike ngram/EAGLE which allocate new cache for drafts, detinfer reuses
        existing cache locations. When we rollback:
        - The rolled-back tokens are removed from req.output_ids
        - BUT their cache locations remain marked as allocated in req_to_token_pool
        - This causes a memory leak - those locations are never freed!
        
        Solution: Explicitly free the cache locations for rolled-back tokens,
        just like ngram/EAGLE do in their verify functions.
        
        IMPORTANT: When tokens_accepted=1, we accept ONE verified token after rollback.
        That token REUSES the cache location of the first rolled-back token.
        So we only free (tokens_rolled_back - tokens_accepted) cache locations.
        """
        for i, (req, info) in enumerate(zip(reqs, rollback_info)):
            if info is None:
                # No rollback needed for this request
                continue
            
            mismatch_pos, tokens_rolled_back, tokens_accepted = info
            
            if tokens_rolled_back == 0:
                # No tokens rolled back
                continue
            
            # FREE the KV cache for rolled-back tokens!
            # After rollback, req.output_ids has been truncated and possibly had 1 token re-added.
            # 
            # Example: Rollback 3 tokens, accept 1 verified token
            # - Before: output_ids = [t0, t1, t2, t3, t4] (mismatch at pos 2)
            # - After rollback: output_ids = [t0, t1]
            # - After accept: output_ids = [t0, t1, t2'] (accepted verified token)
            # - Tokens to free: OLD t2, t3, t4's cache (but t2' reuses t2's cache)
            # - So free: t3 and t4's cache = (3 rolled back - 1 accepted) = 2 tokens
            
            # The accepted token (if any) REUSES the first rolled-back token's cache position
            # So we only need to free the remaining rolled-back tokens
            num_tokens_to_actually_free = tokens_rolled_back - tokens_accepted
            
            if num_tokens_to_actually_free <= 0:
                logger.info(
                    f"[KV_ROLLBACK] Request {req.rid}: "
                    f"Rolled back {tokens_rolled_back} tokens, accepted {tokens_accepted}, "
                    f"no cache to free (accepted token reuses rolled-back cache)"
                )
                continue
            
            # Calculate positions: Start after the current output_ids (which includes accepted token if any)
            start_free_pos = len(req.origin_input_ids) + len(req.output_ids)
            end_free_pos = start_free_pos + num_tokens_to_actually_free
            
            # Get the cache locations to free from req_to_token_pool
            cache_locs_to_free = original_batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_free_pos:end_free_pos
            ]
            
            # Free them from the allocator
            original_batch.token_to_kv_pool_allocator.free(cache_locs_to_free)
            
            logger.info(
                f"[KV_ROLLBACK] Request {req.rid}: "
                f"Mismatch at position {mismatch_pos}, "
                f"rolled back {tokens_rolled_back} tokens, "
                f"accepted {tokens_accepted} verified token. "
                f"Freed {num_tokens_to_actually_free} cache locations: {cache_locs_to_free.tolist()}"
            )

    def __getattr__(self, name):
        """
        Forward all other attributes to target_worker.
        
        This allows DeterministicVerificationWorker to act as a transparent
        wrapper - any method/attribute not defined here is delegated to the
        underlying target_worker.
        """
        return getattr(self.target_worker, name)
