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
        # Convert to ModelWorkerBatch and run normal forward pass
        if hasattr(batch, "get_model_worker_batch"):
            model_worker_batch = batch.get_model_worker_batch()
        else:
            model_worker_batch = batch

        output = self.target_worker.forward_batch_generation(model_worker_batch, skip_sample=skip_sample)
        
        return output

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
            rollback_info = det_verify_info.verify_and_compare(reqs, verified_token_ids, verify_output.logits_output)
            
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
            
            # Update statistics
            self.stats["total_verified"] += len(reqs)
            mismatches = sum(1 for req in reqs if req.det_mismatch)
            self.stats["total_mismatches"] += mismatches
            
            # Update batch.output_ids to reflect the modified req.output_ids after verification
            # This is crucial because prepare_for_decode() uses batch.output_ids, not req.output_ids
            if hasattr(original_batch, 'output_ids') and original_batch.output_ids is not None:
                # Rebuild output_ids tensor from the updated req.output_ids lists
                # Each request contributes its last generated token
                updated_output_ids = []
                for req in original_batch.reqs:
                    if len(req.output_ids) > 0:
                        updated_output_ids.append(req.output_ids[-1])
                    else:
                        # Fallback to last input token if no outputs yet
                        updated_output_ids.append(req.origin_input_ids[-1] if req.origin_input_ids else 0)
                
                # Update the batch tensor
                original_batch.output_ids = torch.tensor(
                    updated_output_ids, 
                    dtype=torch.int64, 
                    device=original_batch.device
                )
                logger.info(
                    f"Updated batch.output_ids tensor after verification: {original_batch.output_ids}"
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
        Handle KV cache cleanup for rolled-back tokens.
        
        FINAL CORRECT UNDERSTANDING:
        During verification, we feed [last_verified, unverified0, unverified1, ...]
        - If unverified_i matches its prediction → it's CORRECT, KV from embedding(unverified_i) is correct
        - Logits AFTER processing unverified_i predict what unverified_(i+1) SHOULD be
        - These logits are computed with attention to correct KV[0...i]
        
        If mismatch at position 5 (comparing unverified5 vs predicted_token5):
        - Tokens 0-4: Matched their predictions → CORRECT! KV cache is CORRECT
        - Token 4 was correct, so KV[4] from embedding(token4_correct) is correct
        - Logits AFTER processing token4 predict token5 with CORRECT KV context!
        - Token 5: unverified5 != predicted5 → MISMATCH
          * But the PREDICTION (predicted5) was computed with correct KV[0-4] context
          * We CAN accept predicted5! It's the correct deterministic token!
          * We do NOT store KV for predicted5 yet (we haven't processed it)
        - Tokens 6-9: Were computed after feeding wrong unverified5 → must discard
        
        We ACCEPT the verified token at mismatch (it was predicted with correct context).
        We FREE KV cache from (mismatch + 1) onwards (tokens processed after wrong token).
        
        Args:
            original_batch: Original batch context
            reqs: Requests that were verified
            rollback_info: List of (mismatch_pos, tokens_rolled_back, tokens_accepted) or None
            det_verify_info: DetVerifyInfo containing verification metadata
        """
        for i, (req, info) in enumerate(zip(reqs, rollback_info)):
            if info is None:
                # No rollback needed for this request
                continue
            
            mismatch_pos, tokens_rolled_back, tokens_accepted = info
            
            if tokens_rolled_back == 0 and tokens_accepted == 0:
                # No tokens to rollback or accept
                continue
            
            if tokens_rolled_back == 0 and tokens_accepted > 0:
                # Extra token accepted with no rollback (verification passed with bonus token)
                # The KV cache was already allocated and written during verification
                # req_to_token_pool already has the correct cache location
                logger.info(
                    f"[KV_ACCEPT] Request {req.rid}: "
                    f"Accepted {tokens_accepted} extra token(s) from verification. "
                    f"KV cache already allocated and in place."
                )
                continue
            
            logger.info(
                f"[KV_ROLLBACK] Request {req.rid}: "
                f"Mismatch at position {mismatch_pos}, accepted {tokens_accepted} verified token, "
                f"need to FREE {tokens_rolled_back - tokens_accepted} KV cache slots"
            )
            
            # Tokens before mismatch_pos are verified correct - keep their KV cache
            # Token at mismatch_pos: We ACCEPTED the verified prediction (it had correct KV context)
            #   - But we haven't PROCESSED it yet, so no KV cache entry for it yet!
            #   - Next generation will process it and create its KV cache entry
            # Tokens after mismatch_pos: Were computed with wrong KV - FREE them!
            
            # Calculate which cache slots to free
            # Start freeing from (mismatch_pos + tokens_accepted) onwards
            # Since we accepted the verified token but haven't processed it yet, no KV for it
            start_free_idx = len(req.origin_input_ids) + mismatch_pos + tokens_accepted
            end_free_idx = len(req.origin_input_ids) + mismatch_pos + tokens_rolled_back
            
            # Get the cache locations from req_to_token_pool
            cache_locs_to_free = original_batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_free_idx:end_free_idx
            ]
            
            if len(cache_locs_to_free) > 0:
                # Free the KV cache slots using the batch's allocator
                original_batch.token_to_kv_pool_allocator.free(cache_locs_to_free)
                
                logger.info(
                    f"[KV_ROLLBACK] Request {req.rid}: "
                    f"Freed {len(cache_locs_to_free)} KV cache slots (positions {start_free_idx}:{end_free_idx})"
                )
                
                # Update req_to_token_pool to mark these slots as invalid (-1)
                original_batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, start_free_idx:end_free_idx
                ] = -1
            
            logger.info(
                f"[KV_ROLLBACK] Request {req.rid}: "
                f"Rollback complete. Kept {mismatch_pos + tokens_accepted} tokens with correct KV cache, "
                f"will continue generation from position {mismatch_pos + tokens_accepted} in DETERMINISTIC mode"
            )

    def __getattr__(self, name):
        """
        Forward all other attributes to target_worker.
        
        This allows DeterministicVerificationWorker to act as a transparent
        wrapper - any method/attribute not defined here is delegated to the
        underlying target_worker.
        """
        return getattr(self.target_worker, name)
