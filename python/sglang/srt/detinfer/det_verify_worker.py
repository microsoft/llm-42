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
        # Track which requests have been logged as non-deterministic to avoid spam
        self._logged_non_deterministic = set()
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
        # logger.info(f"[DET_DEBUG] check_and_verify_deterministic_requests called with {len(batch.reqs) if batch.reqs else 0} requests")
        
        if batch.reqs is None:
            logger.info("[DET_DEBUG] batch.reqs is None, returning")
            return
        
        reqs_to_verify = []
        
        for req in batch.reqs:
            # logger.info(
            #     f"[DET_DEBUG] Checking req {req.rid}: is_deterministic={req.is_deterministic}, "
            #     f"det_verified={req.det_verified}, finished={req.finished()}, "
            #     f"finished_reason={req.finished_reason}, "
            #     f"output_ids_len={len(req.output_ids)}, det_verified_tokens={req.det_verified_tokens}"
            # )
            
            if not req.is_deterministic:
                # Only log once per request to avoid spam
                if req.rid not in self._logged_non_deterministic:
                    logger.debug(f"[DET_DEBUG] req {req.rid} is NOT deterministic, skipping")
                    self._logged_non_deterministic.add(req.rid)
                continue
            
            # Skip verification if force_deterministic_mode is True
            # When this flag is set, batch-invariant mode is forced on, so no verification is needed
            if getattr(req, 'force_deterministic_mode', False):
                # Mark all tokens as verified since we're running with forced determinism
                req.det_verified_tokens = len(req.output_ids)
                logger.debug(f"[DET_DEBUG] req {req.rid} has force_deterministic_mode=True, skipping verification")
                continue
            
            # At this point, tokens are already appended and check_finished() was called
            # Check if request is finished
            if req.finished_output is not None or req.finished_reason is not None:
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
                # logger.info(f"[DET_DEBUG] req {req.rid} is NOT finished yet")
                pass
            
            # Check for incremental verification at det_step_size boundary
            if self.det_step_size is not None:
                total_tokens = len(req.output_ids)
                unverified_tokens = total_tokens - req.det_verified_tokens
                
                # logger.info(
                #     f"[DET_DEBUG] rid={req.rid}, output_ids_len={total_tokens}, "
                #     f"verified={req.det_verified_tokens}, unverified={unverified_tokens}, "
                #     f"det_step_size={self.det_step_size}"
                # )
                
                if unverified_tokens >= self.det_step_size:
                    reqs_to_verify.append(req)
        
        if reqs_to_verify:
            # logger.info(
            #     f"Found {len(reqs_to_verify)} deterministic requests to verify"
            # )
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
            logger.info(f"[DET_DEBUG] verify_model_worker_batch prepared with {len(verify_model_worker_batch.reqs) if verify_model_worker_batch.reqs else 0} requests")
            verify_output = self.target_worker.forward_batch_generation(
                verify_model_worker_batch
            )
            logger.info(f"[DET_DEBUG] verify_model_worker_batch forward pass complete")
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
            
            # Extract verified logprobs from the logits_output (available after sampling)
            verified_logprobs = verify_output.logits_output.next_token_logprobs
            
            # Compare outputs and handle rollback, passing verified logprobs
            rollback_info = det_verify_info.verify_and_compare(
                reqs, verified_token_ids, verified_logprobs
            )
            
            # Handle KV cache for rolled-back tokens
            self._handle_kv_cache_rollback(original_batch, reqs, rollback_info)
            for i, (req, info) in enumerate(zip(reqs, rollback_info)):
                req.det_verified_tokens = len(req.output_ids)
                # logger.info(
                #     f"Request {req.rid}: Updated det_verified_tokens={req.det_verified_tokens} after verification"
                # )
            
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
            
            # Update batch.output_ids AND seq_lens to reflect the modified req.output_ids after verification
            # This is crucial because prepare_for_decode() uses batch.output_ids and seq_lens
            if hasattr(original_batch, 'output_ids') and original_batch.output_ids is not None:
                # Rebuild output_ids tensor from the updated req.output_ids lists
                # Each request contributes its last generated token
                updated_output_ids = []
                updated_seq_lens = []
                
                logger.info(f"[DET_DEBUG] Updating batch state for {len(original_batch.reqs)} requests")
                
                for req in original_batch.reqs:
                    logger.info(
                        f"[DET_DEBUG] req {req.rid}: output_ids={req.output_ids}, "
                        f"len={len(req.output_ids)}, last_token={req.output_ids[-1] if req.output_ids else None}"
                    )
                    if len(req.output_ids) > 0:
                        updated_output_ids.append(req.output_ids[-1])
                    else:
                        # Fallback to last input token if no outputs yet
                        updated_output_ids.append(req.origin_input_ids[-1] if req.origin_input_ids else 0)
                    # Update seq_lens to match actual sequence length after rollback
                    # IMPORTANT: seq_lens should be length BEFORE the current token
                    # because prepare_for_decode() will increment it by 1
                    # So we use: input_len + output_len - 1
                    current_total_len = len(req.origin_input_ids) + len(req.output_ids)
                    updated_seq_lens.append(current_total_len - 1)
                
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
        
        # logger.info(
        #     f"Verification complete. Total verified: {self.stats['total_verified']}, "
        #     f"Total mismatches: {self.stats['total_mismatches']}"
        # )

    def _handle_kv_cache_rollback(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        reqs: List[Req],
        rollback_info: List,
    ):
        """
        Handle KV cache state after rollback.
        """
        for i, (req, info) in enumerate(zip(reqs, rollback_info)):
            if info is None:
                # No rollback needed for this request
                continue
            
            mismatch_pos, tokens_rolled_back = info
            
            if tokens_rolled_back == 0:
                # No tokens rolled back
                continue
            
            # Calculate which KV cache slots need to be freed
            num_slots_to_free = tokens_rolled_back
            
            if num_slots_to_free > 0:
                # Get the KV cache indices for the slots to free
                # These are AFTER the current output_ids length (which includes accepted token)
                start_free_pos = len(req.origin_input_ids) + len(req.output_ids) - 1
                end_free_pos = start_free_pos + num_slots_to_free + 1
                
                kv_indices_to_free_raw = original_batch.req_to_token_pool.req_to_token[
                    req.req_pool_idx, start_free_pos:end_free_pos
                ]
                
                # Filter out zero indices (slot 0 is padding, should never be freed)
                kv_indices_to_free = kv_indices_to_free_raw[kv_indices_to_free_raw != 0]

                logger.info(
                    f"original_batch.req_to_token_pool state for req {req.rid}: "
                    f"kv_indices={original_batch.req_to_token_pool.req_to_token[req.req_pool_idx].tolist()}"
                )
                
                # logger.info(
                #     f"[KV_ROLLBACK] Request {req.rid}: "
                #     f"rolled back {tokens_rolled_back} tokens, "
                #     f"Current output_ids length: {len(req.output_ids)} "
                #     f"(input={len(req.origin_input_ids)} + output={len(req.output_ids)}). "
                #     f"Freeing {len(kv_indices_to_free)} KV cache slots (out of {num_slots_to_free}) "
                #     f"from positions [{start_free_pos}:{end_free_pos}]: "
                #     f"{kv_indices_to_free.tolist()}"
                # )
                
                # FREE the KV cache slots for rolled-back tokens (but not the accepted token!)
                if kv_indices_to_free.numel() > 0:
                    original_batch.token_to_kv_pool_allocator.free(kv_indices_to_free)
                else:
                    # logger.info(f"[KV_ROLLBACK] No valid indices to free for req {req.rid}")
                    pass
            else:
                # logger.info(
                #     f"[KV_ROLLBACK] Request {req.rid}: "
                #     f"no slots to free (accepted token reuses all rolled-back cache)"
                # )
                pass

    def __getattr__(self, name):
        """
        Forward all other attributes to target_worker.
        
        This allows DeterministicVerificationWorker to act as a transparent
        wrapper - any method/attribute not defined here is delegated to the
        underlying target_worker.
        """
        return getattr(self.target_worker, name)
