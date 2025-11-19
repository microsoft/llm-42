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
"""Deterministic verification worker that wraps the target worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

from sglang.srt.detinfer.det_verify_info import DetVerifyInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

logger = logging.getLogger(__name__)


class DeterministicVerificationWorker:
    """
    Wraps a target worker to provide deterministic verification.
    
    Similar to EagleWorker but for deterministic inference validation:
    - Intercepts forward_batch_generation calls
    - Identifies finished deterministic requests
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
        batch: ScheduleBatch,
        skip_sample: bool = False,
    ) -> Tuple[LogitsProcessorOutput, bool]:
        """
        Forward pass with deterministic verification.
        
        For batches containing finished deterministic requests:
        1. Run normal forward pass
        2. Identify finished deterministic requests
        3. Create verification batch with TARGET_DET_VERIFY
        4. Run verification forward pass
        5. Compare outputs
        
        Args:
            batch: Input batch
            skip_sample: Whether to skip sampling
            
        Returns:
            Tuple of (logits_output, finished_flag)
        """
        # Run normal forward pass
        logits_output, finished = self.target_worker.forward_batch_generation(
            batch, skip_sample=skip_sample
        )
        
        # Check if we have finished deterministic requests to verify
        finished_det_reqs = self._get_finished_deterministic_requests(batch)
        
        if finished_det_reqs:
            logger.info(
                f"Found {len(finished_det_reqs)} finished deterministic requests to verify"
            )
            self._verify_deterministic_requests(batch, finished_det_reqs)
        
        return logits_output, finished

    def _get_finished_deterministic_requests(
        self, batch: ScheduleBatch
    ) -> List[Req]:
        """
        Extract finished deterministic requests from batch.
        
        Args:
            batch: Current batch
            
        Returns:
            List of finished deterministic requests
        """
        finished_det_reqs = []
        
        for req in batch.reqs:
            # Check if request is:
            # 1. Deterministic (temperature=0 or is_deterministic flag)
            # 2. Finished (reached stop condition or max tokens)
            # 3. Not already verified
            if (
                getattr(req, "is_deterministic", False)
                and req.finished()
                and not getattr(req, "det_verified", False)
            ):
                finished_det_reqs.append(req)
        
        return finished_det_reqs

    def _verify_deterministic_requests(
        self, original_batch: ScheduleBatch, reqs: List[Req]
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
        
        # Run verification forward pass (skip sampling - we just need logits)
        verify_logits_output, _ = self.target_worker.forward_batch_generation(
            verify_batch, skip_sample=True
        )
        
        # Get verified token IDs from logits
        verified_token_ids = verify_logits_output.next_token_ids
        
        # Compare outputs
        det_verify_info.verify_and_compare(reqs, verified_token_ids, verify_logits_output)
        
        # Update statistics
        self.stats["total_verified"] += len(reqs)
        mismatches = sum(1 for req in reqs if getattr(req, "det_mismatch", False))
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
