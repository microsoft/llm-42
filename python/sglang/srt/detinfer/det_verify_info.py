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
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

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
    def from_requests(cls, reqs: List[Req]) -> DetVerifyInfo:
        """
        Create DetVerifyInfo from a list of finished deterministic requests.
        
        Args:
            reqs: List of requests to verify
            
        Returns:
            DetVerifyInfo instance
        """
        # Collect original outputs
        original_outputs = []
        seq_lens = []
        output_lens = []
        
        for req in reqs:
            # Store the original generated output
            original_outputs.extend(req.output_ids)
            output_lens.append(len(req.output_ids))
            # Full sequence length (input + output)
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
        
        # Prepare input_ids: full sequence for each request
        input_ids = []
        req_pool_indices = []
        extend_seq_lens = []
        extend_prefix_lens = []
        
        for req in reqs_to_verify:
            # Full sequence: original input + generated output
            full_seq = req.origin_input_ids + req.output_ids
            input_ids.extend(full_seq)
            
            # Use existing req_pool_idx
            req_pool_indices.append(req.req_pool_idx)
            
            # Extend lens: prefix is input, extend is output
            extend_prefix_lens.append(len(req.origin_input_ids))
            extend_seq_lens.append(len(req.output_ids))
        
        # Set batch attributes
        verify_batch.input_ids = torch.tensor(input_ids, dtype=torch.int32)
        verify_batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32)
        verify_batch.seq_lens = self.seq_lens.clone()
        verify_batch.extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int32)
        verify_batch.extend_prefix_lens = torch.tensor(extend_prefix_lens, dtype=torch.int32)
        verify_batch.extend_num_tokens = len(input_ids)
        
        # Copy other necessary attributes from original batch
        verify_batch.req_to_token_pool = original_batch.req_to_token_pool
        verify_batch.token_to_kv_pool_allocator = original_batch.token_to_kv_pool_allocator
        verify_batch.tree_cache = original_batch.tree_cache
        verify_batch.model_config = original_batch.model_config
        
        return verify_batch

    def verify_and_compare(
        self,
        reqs: List[Req],
        verified_token_ids: torch.Tensor,
        logits_output: LogitsProcessorOutput,
    ):
        """
        Compare original outputs with re-generated outputs.
        
        Args:
            reqs: Requests that were verified
            verified_token_ids: Token IDs from verification run
            logits_output: Logits from verification run
        """
        # Convert to list for comparison
        if isinstance(verified_token_ids, torch.Tensor):
            verified_token_ids = verified_token_ids.tolist()
        
        original_ids = self.original_outputs.tolist()
        
        # Compare per-request
        offset = 0
        for i, req in enumerate(reqs):
            output_len = self.output_lens[i]
            orig_output = original_ids[offset : offset + output_len]
            verify_output = verified_token_ids[offset : offset + output_len]
            
            if orig_output == verify_output:
                # Verification passed
                req.det_verified = True
                req.det_mismatch = False
                logger.info(f"Request {req.rid}: Deterministic verification PASSED")
            else:
                # Mismatch detected
                req.det_verified = True
                req.det_mismatch = True
                logger.error(
                    f"Request {req.rid}: Deterministic verification FAILED\n"
                    f"  Original:  {orig_output}\n"
                    f"  Verified:  {verify_output}\n"
                    f"  Mismatch at positions: {[j for j in range(min(len(orig_output), len(verify_output))) if orig_output[j] != verify_output[j]]}"
                )
            
            offset += output_len
