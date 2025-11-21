from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size

    # NOTE (csy): this is to determine if a cache has prefix matching feature.
    # Chunk cache always return True to indicate no prefix matching.
    # TODO (csy): Using a prefix cache trait to replace this
    @property
    def disable(self):
        return True

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
        )

    def cache_finished_req(self, req: Req, insert: bool = True):
        import logging
        logger = logging.getLogger(__name__)
        
        # For deterministic requests with verification:
        # We need to free KV cache for all tokens that were actually generated.
        # With proper rollback handling, req_to_token_pool should only contain
        # valid cache locations up to the current sequence length.
        
        current_seq_len = len(req.origin_input_ids) + len(req.output_ids)
        
        if hasattr(req, 'is_deterministic') and req.is_deterministic:
            # For deterministic requests, we free cache for all tokens up to current_seq_len
            # Verification may have written to some of these positions, but they should all
            # be contiguous from 0 to current_seq_len-1 (excluding the last output token's KV
            # which is never computed).
            
            # Free input + (outputs - 1) since the last output token's KV is never computed
            num_tokens_to_free = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            
            logger.info(
                f"[CACHE_FREE] Deterministic request {req.rid}: "
                f"current_seq_len={current_seq_len}, "
                f"input_len={len(req.origin_input_ids)}, "
                f"output_len={len(req.output_ids)}, "
                f"freeing {num_tokens_to_free} tokens"
            )
            
            # Read the cache locations for tokens that have KV cache
            if num_tokens_to_free > 0:
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx,
                    :num_tokens_to_free,
                ]
                
                logger.info(
                    f"[CACHE_FREE] Freeing {num_tokens_to_free} tokens for deterministic request {req.rid}: "
                    f"kv_indices={kv_indices.tolist()[:10]}..." if num_tokens_to_free > 10 
                    else f"kv_indices={kv_indices.tolist()}"
                )
            else:
                kv_indices = None
        else:
            # For non-deterministic requests, use the standard logic
            # Free input + (outputs - 1) since the last output token's KV is never computed
            num_tokens_to_free = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            
            if num_tokens_to_free > 0:
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx,
                    :num_tokens_to_free,
                ]
                logger.info(
                    f"[CACHE_FREE] Freeing {num_tokens_to_free} tokens for request {req.rid}: "
                    f"kv_indices={kv_indices.tolist()[:10]}..." if num_tokens_to_free > 10 
                    else f"kv_indices={kv_indices.tolist()}"
                )
            else:
                kv_indices = None
        
        if num_tokens_to_free > 0 and kv_indices is not None:
            self.req_to_token_pool.free(req.req_pool_idx)
            self.token_to_kv_pool_allocator.free(kv_indices)
        else:
            # Just free the req slot if no KV cache to free
            logger.info(f"[CACHE_FREE] No KV cache to free for request {req.rid}")
            self.req_to_token_pool.free(req.req_pool_idx)

    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: SWATokenToKVPoolAllocator,
        page_size: int,
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool_allocator, page_size)
        assert isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        if prelen >= req.evicted_seqlen_local + attention_chunk_size:
            new_evicted_seqlen_local = attention_chunk_size * (
                prelen // attention_chunk_size
            )
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass
