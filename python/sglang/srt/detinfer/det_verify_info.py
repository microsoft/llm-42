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
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

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
        original_outputs = []
        seq_lens = []
        output_lens = []
        
        for req in reqs:
            unverified_output_ids = req.output_ids[req.det_verified_tokens:]
            original_outputs.extend(unverified_output_ids)
            output_lens.append(len(unverified_output_ids))
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
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        
        verify_batch = ScheduleBatch(reqs=reqs_to_verify, batch_is_full=False)
        verify_batch.forward_mode = ForwardMode.TARGET_DET_VERIFY
        
        input_ids = []
        req_pool_indices = []
        output_lens = []
        prefix_lens_list = []
        
        for req in reqs_to_verify:
            unverified_tokens = req.output_ids[req.det_verified_tokens:]
            if not unverified_tokens:
                continue
            
            last_verified_token = (
                req.output_ids[req.det_verified_tokens - 1]
                if req.det_verified_tokens > 0
                else req.origin_input_ids[-1] if req.origin_input_ids else req.input_ids[-1]
            )
            
            input_ids.extend([last_verified_token] + unverified_tokens[:-1])
            req_pool_indices.append(req.req_pool_idx)
            output_lens.append(len(unverified_tokens))
            prefix_lens_list.append(len(req.origin_input_ids) + req.det_verified_tokens - 1)
        
        device = original_batch.device
        verify_batch.input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        verify_batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=device)
        
        verify_batch.extend_lens = output_lens
        verify_batch.prefix_lens = prefix_lens_list
        
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=device)
        extend_lens_tensor = torch.tensor(output_lens, dtype=torch.int32, device=device)
        total_seq_lens = prefix_lens + extend_lens_tensor
        verify_batch.seq_lens = total_seq_lens
        verify_batch.seq_lens_cpu = total_seq_lens.cpu()
        verify_batch.extend_logprob_start_lens = [1] * len(reqs_to_verify)
        
        verify_batch.extend_num_tokens = len(input_ids)
        verify_batch.seq_lens_sum = total_seq_lens.sum().item()
        verify_batch.orig_seq_lens = total_seq_lens.clone()
        
        verify_batch.return_logprob = True
        verify_batch.top_logprobs_nums = [0] * len(reqs_to_verify)
        verify_batch.token_ids_logprobs = [[] for _ in reqs_to_verify]
        
        verify_batch.req_to_token_pool = original_batch.req_to_token_pool
        verify_batch.token_to_kv_pool_allocator = original_batch.token_to_kv_pool_allocator
        verify_batch.tree_cache = original_batch.tree_cache
        verify_batch.model_config = original_batch.model_config
        verify_batch.device = device
        
        verify_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            verify_batch, original_batch.model_config.vocab_size
        )
        
        if verify_batch.sampling_info is not None:
            tokens_per_request = torch.tensor(output_lens, dtype=torch.int32, device=device)
            
            verify_batch.sampling_info.temperatures = torch.repeat_interleave(
                verify_batch.sampling_info.temperatures, tokens_per_request, dim=0
            )
            verify_batch.sampling_info.top_ks = torch.repeat_interleave(
                verify_batch.sampling_info.top_ks, tokens_per_request, dim=0
            )
            verify_batch.sampling_info.top_ps = torch.repeat_interleave(
                verify_batch.sampling_info.top_ps, tokens_per_request, dim=0
            )
            verify_batch.sampling_info.min_ps = torch.repeat_interleave(
                verify_batch.sampling_info.min_ps, tokens_per_request, dim=0
            )
            if verify_batch.sampling_info.sampling_seed is not None:
                verify_batch.sampling_info.sampling_seed = torch.repeat_interleave(
                    verify_batch.sampling_info.sampling_seed, tokens_per_request, dim=0
                )
            if verify_batch.sampling_info.deterministic_indices is not None:
                verify_batch.sampling_info.deterministic_indices = torch.repeat_interleave(
                    verify_batch.sampling_info.deterministic_indices, tokens_per_request, dim=0
                )
        
        out_cache_locs = []
        
        for req in reqs_to_verify:
            current_seq_len = len(req.origin_input_ids) + len(req.output_ids)
            context_idx = len(req.origin_input_ids) + req.det_verified_tokens - 1
            
            if context_idx >= current_seq_len:
                logger.error(
                    f"ERROR: context_idx {context_idx} >= current_seq_len {current_seq_len} "
                    f"for req {req.rid}. This indicates a logic error."
                )
                raise RuntimeError(
                    f"Attempting to access unallocated cache position {context_idx} "
                    f"when only {current_seq_len} positions have been allocated"
                )
            
            context_cache_loc = verify_batch.req_to_token_pool.req_to_token[req.req_pool_idx, context_idx]
            out_cache_locs.append(context_cache_loc.item())
            
            start_idx = len(req.origin_input_ids) + req.det_verified_tokens
            num_unverified = len(req.output_ids) - req.det_verified_tokens
            end_idx = min(start_idx + num_unverified - 1, current_seq_len - 1)
            
            if start_idx < end_idx:
                output_cache_locs = verify_batch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
                out_cache_locs.extend(output_cache_locs.tolist())
        
        if len(out_cache_locs) != len(input_ids):
            logger.error(
                f"[DET_VERIFY] ERROR: Mismatch in cache location count! "
                f"expected={len(input_ids)}, actual={len(out_cache_locs)}, "
                f"input_ids length={len(input_ids)}"
            )
            raise RuntimeError(
                f"Verification batch has {len(input_ids)} input tokens but only "
                f"{len(out_cache_locs)} cache locations. This will cause memory corruption."
            )
        
        verify_batch.out_cache_loc = torch.tensor(out_cache_locs, dtype=torch.int32, device=device)
        
        return verify_batch

    def first_mismatch_position(
        self,
        original_ids: List[int],
        verified_ids: List[int],
    ) -> int:
        """
        Find the first mismatch position between original and verified token IDs.
        
        Args:
            original_ids: Original output token IDs
            verified_ids: Verified output token IDs
        Returns:
            Index of the first mismatch position, or length if all match
        """
        for i, (orig, verify) in enumerate(zip(original_ids, verified_ids)):
            if orig != verify:
                return i
        return min(len(original_ids), len(verified_ids))

    def verify_and_compare(
        self,
        reqs: List[Req],
        verified_token_ids: torch.Tensor,
        verified_logprobs: Optional[torch.Tensor] = None,
    ):
        """
        Compare original outputs with re-generated outputs.
        On mismatch: ROLLBACK and ACCEPT the verified token predicted at mismatch position.
        
        Args:
            reqs: Requests that were verified
            verified_token_ids: Token IDs from verification run (argmax predictions)
            verified_logprobs: Log probabilities from verification run (optional)
            
        Returns:
            List of (mismatch_position, tokens_rolled_back) tuples for each request
        """
        
        verified_token_ids = verified_token_ids.tolist() if isinstance(verified_token_ids, torch.Tensor) else verified_token_ids
        verified_logprobs_list = verified_logprobs.tolist() if isinstance(verified_logprobs, torch.Tensor) else verified_logprobs
        original_ids = self.original_outputs.tolist()
        rollback_info = []
        offset = 0
        
        for i, req in enumerate(reqs):
            output_len = self.output_lens[i]
            orig_output = original_ids[offset : offset + output_len]
            verify_output = verified_token_ids[offset : offset + output_len]
            
            mismatch_pos = self.first_mismatch_position(orig_output, verify_output)
            # mismatch_pos = min(5, output_len)  # testing
            
            tokens_to_rollback = len(orig_output) - mismatch_pos
            
            if tokens_to_rollback > 0:
                req.output_ids = req.output_ids[:-tokens_to_rollback]
                
                if verified_logprobs_list is not None and req.output_token_logprobs_val is not None:
                    # Replace logprobs for ALL verified tokens (up to and including mismatch)
                    req.output_token_logprobs_val = req.output_token_logprobs_val[:-output_len]
                    req.output_token_logprobs_idx = req.output_token_logprobs_idx[:-output_len]
                    
                    # Add verified logprobs up to mismatch position
                    req.output_token_logprobs_val.extend(verified_logprobs_list[offset : offset + mismatch_pos])
                    req.output_token_logprobs_idx.extend(verify_output[:mismatch_pos])
                
                req.finished_reason = None
                req.finished_output = None
                
                if mismatch_pos < len(verify_output):
                    req.output_ids.append(verify_output[mismatch_pos])
                    
                    if verified_logprobs_list is not None and req.output_token_logprobs_val is not None:
                        req.output_token_logprobs_val.append(verified_logprobs_list[offset + mismatch_pos])
                        req.output_token_logprobs_idx.append(verify_output[mismatch_pos])
                    
                    tokens_to_rollback -= 1
                
                rollback_info.append((mismatch_pos, tokens_to_rollback))
            else:
                # No rollback needed, but still replace with verified logprobs and token IDs
                if verified_logprobs_list is not None and req.output_token_logprobs_val is not None:
                    # Replace logprobs for ALL verified tokens
                    req.output_token_logprobs_val = req.output_token_logprobs_val[:-output_len]
                    req.output_token_logprobs_idx = req.output_token_logprobs_idx[:-output_len]
                    
                    # Add verified logprobs and token IDs
                    req.output_token_logprobs_val.extend(verified_logprobs_list[offset : offset + output_len])
                    req.output_token_logprobs_idx.extend(verify_output[:output_len])
                
                rollback_info.append((mismatch_pos, 0))

            # Update send_output_token_logprobs_offset to match the new logprobs length
            # This is critical for streaming output to work correctly after verification
            if verified_logprobs_list is not None and req.output_token_logprobs_val is not None:
                req.send_output_token_logprobs_offset = len(req.output_token_logprobs_val)
            offset += output_len
        
        return rollback_info
