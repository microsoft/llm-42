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

    def __init__(self, target_worker: TpModelWorker):
        """
        Initialize the deterministic verification worker.
        
        Args:
            target_worker: The underlying TpModelWorker to wrap
        """
        self.target_worker = target_worker
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
        Forward pass with deterministic verification at scheduler level.
        
        For batches containing finished deterministic requests:
        1. Run normal forward pass
        2. Identify finished deterministic requests
        3. Create verification batch with TARGET_DET_VERIFY
        4. Run verification forward pass
        5. Compare outputs
        
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
        
        # Update batch output_ids for verification check
        batch.output_ids = output.next_token_ids
        
        # Check if we have finished deterministic requests to verify
        finished_det_reqs = self._get_finished_deterministic_requests(batch, output.next_token_ids)
        
        if finished_det_reqs:
            logger.info(
                f"Found {len(finished_det_reqs)} finished deterministic requests to verify"
            )
            self._verify_deterministic_requests(batch, finished_det_reqs)
        
        return output

    def _get_finished_deterministic_requests(
        self, 
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        next_token_ids: torch.Tensor = None
    ) -> List[Req]:
        """
        Extract finished deterministic requests from batch.
        
        Args:
            batch: Current ScheduleBatch or ModelWorkerBatch
            next_token_ids: The next token ids generated in this step
            
        Returns:
            List of finished deterministic requests
        """
        finished_det_reqs = []
        
        if batch.reqs is None:
            return []

        if next_token_ids is not None:
            next_token_ids_cpu = next_token_ids.cpu().tolist()
            
            for i, req in enumerate(batch.reqs):
                if not req.is_deterministic or req.det_verified:
                    continue
                
                # Check if already finished
                if req.finished():
                    finished_det_reqs.append(req)
                    continue

                new_token_id = next_token_ids_cpu[i]
                
                # Check max tokens
                if len(req.output_ids) + 1 >= req.sampling_params.max_new_tokens:
                    finished_det_reqs.append(req)
                    continue
                
                # Check EOS
                if not req.sampling_params.ignore_eos:
                    # Check stop_token_ids from sampling_params
                    if req.sampling_params.stop_token_ids and new_token_id in req.sampling_params.stop_token_ids:
                        finished_det_reqs.append(req)
                        continue
                    # Check eos_token_ids from request (req.eos_token_ids is set during tokenization)
                    if req.eos_token_ids and new_token_id in req.eos_token_ids:
                        finished_det_reqs.append(req)
                        continue
        else:
            for req in batch.reqs:
                # Check if request is:
                # 1. Deterministic (is_deterministic flag)
                # 2. Finished (reached stop condition or max tokens)
                # 3. Not already verified
                if (
                    req.is_deterministic
                    and req.finished()
                    and not req.det_verified
                ):
                    finished_det_reqs.append(req)
        
        return finished_det_reqs

    def _verify_deterministic_requests(
        self, original_batch: Union[ScheduleBatch, ModelWorkerBatch], reqs: List[Req]
    ):
        """
        Verify deterministic requests by re-running them.
        
        Args:
            original_batch: Original batch context
            reqs: Requests to verify
        """
        # Create DetVerifyInfo from finished requests
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
        
        # Free the allocated verification cache to prevent memory leak
        if verify_batch.out_cache_loc is not None:
            verify_batch.token_to_kv_pool_allocator.free(verify_batch.out_cache_loc)
        
        # Update statistics
        self.stats["total_verified"] += len(reqs)
        mismatches = sum(1 for req in reqs if req.det_mismatch)
        self.stats["total_mismatches"] += mismatches
        
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
