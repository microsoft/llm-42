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
        self.metrics_collector = getattr(target_worker, "metrics_collector", None)

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
        if batch.reqs is None:
            return
        
        reqs_to_verify = []
        
        for req in batch.reqs:
            if not req.is_deterministic:
                continue
            
            # Skip verification if force_deterministic_mode is True
            if getattr(req, 'force_deterministic_mode', False):
                req.det_verified_tokens = len(req.output_ids)
                continue
            
            # Calculate unverified tokens once
            output_len = len(req.output_ids)
            unverified_tokens = output_len - req.det_verified_tokens
            
            if unverified_tokens <= 0:
                continue
            
            # Verify finished requests with unverified tokens
            is_finished = req.finished_output is not None or req.finished_reason is not None
            
            # Verify if: finished OR reached step size boundary
            if is_finished or (req.det_step_size is not None and unverified_tokens >= req.det_step_size):
                reqs_to_verify.append(req)
        
        if reqs_to_verify:
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
            # logger.info(f"[DEBUG][DetVerifyWorker] Starting verification for {len(reqs)} requests")
            
            # Check allocator state before verification
            available_before = original_batch.token_to_kv_pool_allocator.available_size()
                    # logger.info(f"[DEBUG][DetVerifyWorker] KV cache available BEFORE verification: {available_before}")
            
            det_verify_info = DetVerifyInfo.from_requests(reqs)
            verify_batch = det_verify_info.prepare_verify_batch(original_batch, reqs)
            
            # logger.info(f"[DEBUG][DetVerifyWorker] Verification batch prepared with {len(verify_batch.input_ids)} input tokens")
            #logger.info(f"[DEBUG][DetVerifyWorker] out_cache_loc: {verify_batch.out_cache_loc.tolist() if verify_batch.out_cache_loc is not None else None}")
            
            # Run verification forward pass (batch_invariant context is now managed in tp_worker)
            verify_model_worker_batch = verify_batch.get_model_worker_batch()
            # logger.info(f"[DEBUG][DetVerifyWorker] Running forward pass with forward_mode={verify_model_worker_batch.forward_mode}")
            verify_output = self.target_worker.forward_batch_generation(verify_model_worker_batch)
            
            # Check allocator state after verification
            available_after = original_batch.token_to_kv_pool_allocator.available_size()
            # logger.info(f"[DEBUG][DetVerifyWorker] KV cache available AFTER verification: {available_after}")
            # logger.info(f"[DEBUG][DetVerifyWorker] KV cache delta: {available_before - available_after}")
            
            # Extract results (sampling already done inside forward_batch_generation)
            verified_token_ids = verify_output.next_token_ids
            verified_logprobs = verify_output.logits_output.next_token_logprobs
            
            # Compare outputs and handle rollback
            rollback_info = det_verify_info.verify_and_compare(
                reqs, verified_token_ids, verified_logprobs
            )
            
            self._handle_kv_cache_rollback(original_batch, reqs, rollback_info)
            
            # Update verified token counts
            for req in reqs:
                req.det_verified_tokens = len(req.output_ids)
            
            # Update batch state if there was any rollback
            # (need to sync seq_lens with the new req.output_ids length)
            had_rollback = any(info[1] > 0 for info in rollback_info if info is not None)
            if had_rollback:
                #logger.info(f"[DEBUG][DetVerifyWorker] Rollback detected, updating batch state")
                self._update_batch_state(original_batch)
            
            # Clear verification batch references to free GPU memory
            # Critical: verify_batch shares KV cache allocator with original_batch
            # At high batch sizes (BS=63-64), we need to explicitly clear all tensor
            # references to prevent memory accumulation
            verify_batch.out_cache_loc = None
            verify_batch.input_ids = None
            verify_batch.seq_lens = None
            verify_batch.seq_lens_cpu = None
            verify_batch.req_pool_indices = None
            
            # Clear sampling_info tensors if present
            if hasattr(verify_batch, 'sampling_info') and verify_batch.sampling_info is not None:
                verify_batch.sampling_info.temperatures = None
                verify_batch.sampling_info.top_ps = None
                verify_batch.sampling_info.top_ks = None
                verify_batch.sampling_info.min_ps = None
                verify_batch.sampling_info.sampling_seed = None
                verify_batch.sampling_info.deterministic_indices = None
            
            # Clear shared resource references (don't free, just remove refs)
            verify_batch.req_to_token_pool = None
            verify_batch.token_to_kv_pool_allocator = None
            
            # Force GPU memory cleanup - critical for high batch sizes
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            
            # NOTE: We don't update original_batch state here because:
            # 1. The original batch is already done processing
            # 2. The scheduler will create a new batch for the next decode step
            # 3. Modifying seq_lens here might confuse KV cache tracking
            # self._update_batch_state(original_batch)
            
            # Check allocator state after cleanup
            available_final = original_batch.token_to_kv_pool_allocator.available_size()
            # logger.info(f"[DEBUG][DetVerifyWorker] KV cache available AFTER cleanup: {available_final}")
            # logger.info(f"[DEBUG][DetVerifyWorker] KV cache total delta: {available_before - available_final}")
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            raise

    def _update_batch_state(self, batch: Union[ScheduleBatch, ModelWorkerBatch]):
        """
        Update batch.output_ids and seq_lens to reflect current req.output_ids.
        Critical for prepare_for_decode() which uses these tensors.
        """
        if not hasattr(batch, 'output_ids') or batch.output_ids is None:
            return
        
        updated_output_ids = []
        updated_seq_lens = []
        
        for req in batch.reqs:
            # Get last generated token or fallback to last input token
            if req.output_ids:
                updated_output_ids.append(req.output_ids[-1])
            else:
                updated_output_ids.append(req.origin_input_ids[-1] if req.origin_input_ids else 0)
            
            # seq_lens should be length BEFORE current token (prepare_for_decode increments by 1)
            current_total_len = len(req.origin_input_ids) + len(req.output_ids)
            updated_seq_lens.append(current_total_len - 1)
        
        # Update batch tensors
        batch.output_ids = torch.tensor(
            updated_output_ids, 
            dtype=torch.int64, 
            device=batch.device
        )
        batch.seq_lens = torch.tensor(
            updated_seq_lens,
            dtype=torch.int32,
            device=batch.device
        )
        batch.seq_lens_cpu = torch.tensor(updated_seq_lens, dtype=torch.int32)
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.seq_lens_sum = sum(updated_seq_lens)

    def _handle_kv_cache_rollback(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        reqs: List[Req],
        rollback_info: List,
    ):
        """
        Handle KV cache state after rollback.
        """
        for req, info in zip(reqs, rollback_info):
            if info is None or info[1] == 0:
                continue
            
            # Track metrics
            if self.metrics_collector:
                self.metrics_collector.num_rollbacks_total.labels(**self.metrics_collector.labels).inc()
                self.metrics_collector.tokens_rolled_back_total.labels(**self.metrics_collector.labels).inc(info[1])
            
            mismatch_pos, tokens_rolled_back = info
            
            # Get the KV cache indices for the slots to free
            start_free_pos = len(req.origin_input_ids) + len(req.output_ids) - 1
            end_free_pos = start_free_pos + tokens_rolled_back + 1
            
            kv_indices_to_free_raw = original_batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_free_pos:end_free_pos
            ]
            
            # Filter out zero indices (slot 0 is padding, should never be freed)
            kv_indices_to_free = kv_indices_to_free_raw[kv_indices_to_free_raw != 0]
            
            if kv_indices_to_free.numel() > 0:
                original_batch.token_to_kv_pool_allocator.free(kv_indices_to_free)

    def __getattr__(self, name):
        """
        Forward all other attributes to target_worker.
        
        This allows DeterministicVerificationWorker to act as a transparent
        wrapper - any method/attribute not defined here is delegated to the
        underlying target_worker.
        """
        return getattr(self.target_worker, name)
