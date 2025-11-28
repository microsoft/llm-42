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
        # logger.info(
        #     f"Free For request {req.rid}: "
        #     f"input_ids len={len(req.origin_input_ids)}, "
        #     f"output_ids len={len(req.output_ids)}"
        #     f"Total tokens to free: {len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)}"
        # )
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            # The last output token is the one being generated, so it doesn't have KV cache yet.
            # Use max(..., 0) to handle the case when output_ids is empty.
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        # logger.info(f"len(kv_indices) to free: {len(kv_indices)}")
        # logger.info(f"Freeing KV cache indices: {kv_indices}")
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)
        # logger.info(f"KV cache available AFTER cleaning: {self.token_to_kv_pool_allocator.available_size()}")

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
