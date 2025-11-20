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
            
            # For TARGET_DET_VERIFY, we need ALL predicted tokens
            # The model outputs logits for each input position
            # When we input [t0, t1, ..., t18], we get logits predicting [t1, t2, ..., t19]
            # But we want to verify [t0, t1, ..., t18], not predict t19
            # So we need to apply argmax to logits at positions [0:19] to get predictions for [t1:t20]
            # Then shift by 1: predictions_for_[t0:t18] = argmax(logits[shifted positions])
            #
            # Actually, the standard transformer behavior is:
            # Input tokens at positions [0, 1, ..., N-1] → Logits at [0, 1, ..., N-1] → Predict tokens at [1, 2, ..., N]
            # For verification: we want to compare original tokens [t0, t1, ..., t18]
            # We need logits that predict these tokens, which come from positions [-1, 0, 1, ..., 17]
            # But position -1 doesn't exist! So we can only verify tokens [t1, t2, ..., t18] (skip t0)
            #
            # Wait - let me reconsider. In TARGET_VERIFY for speculative decoding:
            # Draft tokens [d0, d1, ..., dN] are input, model outputs logits, then compares
            # So with extend_logprob_start_lens=[0], we should get ALL logits back
            #
            # Let's just use ALL the logits from the verification output
            if hasattr(verify_output, 'logits_output') and verify_output.logits_output:
                # Get the input logprob logits which should contain all verification positions
                if hasattr(verify_output.logits_output, 'input_token_logprobs') and verify_output.logits_output.input_token_logprobs is not None:
                    # input_token_logprobs should have logits for all positions we requested
                    verified_token_ids = torch.argmax(verify_output.logits_output.input_token_logprobs, dim=-1)
                elif hasattr(verify_output.logits_output, 'next_token_logits') and verify_output.logits_output.next_token_logits is not None:
                    # Fall back to next_token_logits 
                    verified_token_ids = torch.argmax(verify_output.logits_output.next_token_logits, dim=-1)
                else:
                    verified_token_ids = verify_output.next_token_ids
            else:
                verified_token_ids = verify_output.next_token_ids
            
            # Compare outputs
            det_verify_info.verify_and_compare(reqs, verified_token_ids, verify_output.logits_output)
            
            # Update det_verified_tokens counter for incremental verification
            # At this point, output_ids has the new token appended (we did it above)
            for req in reqs:
                if not req.det_mismatch:
                    # Only update if verification passed
                    req.det_verified_tokens = len(req.output_ids)
            
            # Free the allocated verification cache to prevent memory leak
            if verify_batch.out_cache_loc is not None:
                verify_batch.token_to_kv_pool_allocator.free(verify_batch.out_cache_loc)
            
            # Update statistics
            self.stats["total_verified"] += len(reqs)
            mismatches = sum(1 for req in reqs if req.det_mismatch)
            self.stats["total_mismatches"] += mismatches
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            raise
        
        logger.info(
            f"Verification complete. Total verified: {self.stats['total_verified']}, "
            f"Total mismatches: {self.stats['total_mismatches']}"
        )

    def __getattr__(self, name):
        """
        Forward all other attributes to target_worker.
        
        This allows DeterministicVerificationWorker to act as a transparent
        wrapper - any method/attribute not defined here is delegated to the
        underlying target_worker.
        """
        return getattr(self.target_worker, name)
