# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ==============================================================================
"""LLM-42: Deterministic inference via decode-verify-rollback (DVR).

This module implements the core DVR protocol described in the LLM-42 paper
(arXiv:2601.17768). It provides:

- ``LLM42Info``: Metadata for a verification batch — tracks original outputs,
  padding masks, and KV cache allocations, and implements the verify-and-compare
  logic (including rollback on mismatch).

- ``LLM42Worker``: Scheduler-level wrapper around a
  TpModelWorker that intercepts finished deterministic requests, replays them
  under a fixed-shape reduction schedule (TARGET_LLM42_VERIFY), and commits or
  rolls back tokens based on the verification result.

Typical usage (from the scheduler)::

    worker = LLM42Worker(target_worker, ...)
    output = worker.forward_batch_generation(batch)
    rollbacks = worker.check_and_verify_deterministic_requests(batch)
"""

from sglang.srt.llm42.llm42_info import LLM42Info
from sglang.srt.llm42.llm42_worker import LLM42Worker

__all__ = [
    "LLM42Info",
    "LLM42Worker",
]
