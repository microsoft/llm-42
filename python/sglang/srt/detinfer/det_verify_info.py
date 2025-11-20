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
            unverified_tokens = req.output_ids[req.det_verified_tokens:]
            
            if not unverified_tokens:
                continue  # Skip if no unverified tokens
            
            # For verification, we need to include the last verified token (or last input token if nothing verified yet)
            # This way we get logits that predict each unverified token
            # Example: input_ids=[last_verified_token, unverified0, unverified1, ...] → logits predict [unverified0, unverified1, unverified2, ...]
            if req.det_verified_tokens > 0:
                # Use last verified output token
                last_verified_token = req.output_ids[req.det_verified_tokens - 1]
            else:
                # No tokens verified yet, use last input token
                last_verified_token = req.origin_input_ids[-1] if req.origin_input_ids else req.input_ids[-1]
            
            verification_input = [last_verified_token] + unverified_tokens
            input_ids.extend(verification_input)
            
            # Use existing req_pool_idx
            req_pool_indices.append(req.req_pool_idx)
            
            # Track lengths: we input (1 last_verified + N unverified) tokens
            # But we only want to verify the N unverified tokens
            output_lens.append(len(unverified_tokens))
            # Prefix length includes: all inputs + verified outputs (if any)
            prefix_lens_list.append(len(req.origin_input_ids) + req.det_verified_tokens - 1)
        
        # Set batch attributes - ensure all tensors are on the correct device
        device = original_batch.device if hasattr(original_batch, 'device') else 'cuda'
        verify_batch.input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        verify_batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=device)
        
        # Use extend_lens and prefix_lens which are used by get_model_worker_batch()
        # extend_lens should be the number of tokens we're extending: 1 (last input) + N (outputs)
        extend_lens_with_last_input = [length + 1 for length in output_lens]
        verify_batch.extend_lens = extend_lens_with_last_input
        verify_batch.prefix_lens = prefix_lens_list
        
        # seq_lens should be the total sequence length (prefix + 1 last_input + output tokens being verified)
        # FlashInfer uses this to determine where in the KV cache to access
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=device)
        extend_lens_tensor = torch.tensor(extend_lens_with_last_input, dtype=torch.int32, device=device)
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
            # For the context token (last verified/input token), get its existing cache location
            context_idx = len(req.origin_input_ids) + req.det_verified_tokens - 1
            context_cache_loc = verify_batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, context_idx
            ]
            out_cache_locs.append(context_cache_loc)
            
            # For unverified output tokens, get their existing cache locations
            # These will be OVERWRITTEN during verification (in-place update)
            start_idx = len(req.origin_input_ids) + req.det_verified_tokens
            end_idx = len(req.origin_input_ids) + len(req.output_ids)
            output_cache_locs = verify_batch.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_idx:end_idx
            ]
            out_cache_locs.extend(output_cache_locs.tolist())
        
        # Set out_cache_loc to point to existing cache locations
        verify_batch.out_cache_loc = torch.tensor(
            out_cache_locs, dtype=torch.int32, device=device
        )
        
        logger.info(
            f"Reusing {len(out_cache_locs)} existing KV cache locations for in-place verification "
            f"(no temporary allocation needed)"
        )
        
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
                logger.info(f"[DET_DEBUG] Request {req.rid}: Setting det_verified=True after verification PASSED")
                logger.info(f"Request {req.rid}: Deterministic verification PASSED")
            else:
                # Mismatch detected
                req.det_verified = True
                req.det_mismatch = True
                logger.info(f"[DET_DEBUG] Request {req.rid}: Setting det_verified=True after verification FAILED")
                logger.error(
                    f"Request {req.rid}: Deterministic verification FAILED\n"
                    f"  Original:  {orig_output}\n"
                    f"  Verified:  {verify_output}\n"
                    f"  Mismatch at positions: {[j for j in range(min(len(orig_output), len(verify_output))) if orig_output[j] != verify_output[j]]}"
                )
            
            offset += output_len
