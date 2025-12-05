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
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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

    def __init__(
        self, 
        target_worker: TpModelWorker, 
        always_align: bool = True,
        max_requests_per_verify: Optional[int] = None,
    ):
        """
        Initialize the deterministic verification worker.
        
        Args:
            target_worker: The underlying TpModelWorker to wrap
            always_align: If True, pad verification batches to step_size with dummy tokens
                         for finished requests that have fewer unverified tokens than step_size.
                         This ensures consistent batch sizes for verification. Default: True.
            max_requests_per_verify: Maximum number of requests to verify in a single batch.
                         If None, all requests are verified together. If set, requests are
                         verified in chunks of this size (e.g., 20 requests with max=10
                         will be verified as 10+10). Default: None.
        """
        self.target_worker = target_worker
        self.always_align = always_align
        self.max_requests_per_verify = max_requests_per_verify
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
    ) -> List[Tuple[Req, int]]:
        """
        Check for deterministic requests that need verification and verify them.
        Should be called AFTER tokens have been appended and check_finished() called.
        
        This is called from process_batch_result_decode, similar to how Eagle
        handles verification after output processing.
        
        Args:
            batch: Current ScheduleBatch or ModelWorkerBatch after output processing
            
        Returns:
            List of (req, tokens_rolled_back) tuples for requests that had rollback.
            The caller should use this to free KV cache slots.
        """
        if batch.reqs is None:
            return []
        
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
            return self._verify_deterministic_requests_batched(
                batch, reqs_to_verify, self.always_align, self.max_requests_per_verify
            )
        return []

    def _verify_deterministic_requests_batched(
        self,
        original_batch: Union[ScheduleBatch, ModelWorkerBatch],
        reqs: List[Req],
        always_align: bool = True,
        max_requests: Optional[int] = None,
    ) -> List[Tuple[Req, int]]:
        """
        Verify deterministic requests in batches.
        
        If max_requests is set, requests are verified in chunks of that size.
        For example, 22 requests with max_requests=10 will be verified as 10+10+2.
        
        Args:
            original_batch: Original batch context
            reqs: All requests to verify
            always_align: If True, pad to step_size with dummy tokens
            max_requests: Maximum requests per verification batch. None means all at once.
            
        Returns:
            List of (req, tokens_rolled_back) tuples for all requests that had rollback.
        """
        if max_requests is None or len(reqs) <= max_requests:
            # No batching needed, verify all at once
            return self._verify_deterministic_requests(original_batch, reqs, always_align)
        
        # Split into batches and verify each
        all_rollback_results = []
        for i in range(0, len(reqs), max_requests):
            batch_reqs = reqs[i:i + max_requests]
            rollback_results = self._verify_deterministic_requests(
                original_batch, batch_reqs, always_align
            )
            all_rollback_results.extend(rollback_results)
        
        return all_rollback_results


    def _verify_deterministic_requests(
        self, 
        original_batch: Union[ScheduleBatch, ModelWorkerBatch], 
        reqs: List[Req],
        always_align: bool = True,
    ) -> List[Tuple[Req, int]]:
        """
        Verify deterministic requests by re-running them.
        Tokens are already in req.output_ids at this point.
        
        Args:
            original_batch: Original batch context
            reqs: Requests to verify (tokens already appended)
            always_align: If True, pad to step_size with dummy tokens for finished requests
            
        Returns:
            List of (req, tokens_rolled_back) tuples for requests that had rollback.
        """
        try:
            det_verify_info = DetVerifyInfo.from_requests(reqs, always_align=always_align)
            # logger.info(f"[DetVerifyWorker] Verifying {len(reqs)} requests with total padding cache slots {det_verify_info.total_padding_cache_slots}")
            
            # Allocate temporary KV cache for padding tokens (if any)
            if det_verify_info.total_padding_cache_slots > 0:
                det_verify_info.allocate_padding_kv_cache(
                    original_batch.token_to_kv_pool_allocator
                )
            
            verify_batch = det_verify_info.prepare_verify_batch(original_batch, reqs)
            
            # Run verification forward pass (batch_invariant context is now managed in tp_worker)
            verify_model_worker_batch = verify_batch.get_model_worker_batch()
            verify_output = self.target_worker.forward_batch_generation(verify_model_worker_batch)
            
            # Extract results (sampling already done inside forward_batch_generation)
            verified_token_ids = verify_output.next_token_ids
            verified_logprobs = verify_output.logits_output.next_token_logprobs
            
            # Compare outputs and handle rollback
            rollback_info = det_verify_info.verify_and_compare(
                reqs, verified_token_ids, verified_logprobs
            )
            
            # Collect rollback info for KV cache freeing (will be done by caller)
            rollback_results = []
            for req, info in zip(reqs, rollback_info):
                if info is not None and info[1] > 0:
                    # Track per-request stats
                    req.det_num_rollbacks += 1
                    req.det_tokens_rolled_back += info[1]
                    # Track global metrics
                    if self.metrics_collector:
                        self.metrics_collector.num_rollbacks_total.labels(**self.metrics_collector.labels).inc()
                        self.metrics_collector.tokens_rolled_back_total.labels(**self.metrics_collector.labels).inc(info[1])
                    rollback_results.append((req, info[1]))
            
            # Update verified token counts
            for req in reqs:
                req.det_verified_tokens = len(req.output_ids)
            
            # Update batch state if there was any rollback
            # (need to sync seq_lens with the new req.output_ids length)
            if rollback_results:
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
            
            # Free temporary KV cache allocated for padding tokens
            if det_verify_info.total_padding_cache_slots > 0:
                det_verify_info.free_padding_kv_cache(
                    original_batch.token_to_kv_pool_allocator
                )
            
            return rollback_results
            
        except Exception as e:
            # Make sure to free padding KV cache even on error
            if 'det_verify_info' in locals() and det_verify_info.total_padding_cache_slots > 0:
                try:
                    det_verify_info.free_padding_kv_cache(
                        original_batch.token_to_kv_pool_allocator
                    )
                except Exception:
                    pass  # Best effort cleanup
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

    def __getattr__(self, name):
        """
        Forward all other attributes to target_worker.
        
        This allows DeterministicVerificationWorker to act as a transparent
        wrapper - any method/attribute not defined here is delegated to the
        underlying target_worker.
        """
        return getattr(self.target_worker, name)
