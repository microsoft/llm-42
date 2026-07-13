"""Shared utilities for TP matmul microbenchmarks.

Re-exports helpers from the top-level bench_utils and adds TP-aware
model dimension configs.
"""

import os
import sys

# Re-export from the local (vendored) bench_utils
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import Tee, get_short_gpu_name, setup_run_dir, tee_stdout

__all__ = [
    "get_short_gpu_name",
    "setup_run_dir",
    "tee_stdout",
    "Tee",
    "get_model_ops_config",
    "get_tp_sharded_shape",
]


# ── Model dimension configs ──────────────────────────────────────────────────

MODEL_CONFIGS = {
    "llama3-8b": {
        "HIDDEN_SIZE": 4096,
        "INTERMEDIATE_SIZE": 14336,
        "NUM_HEADS": 32,
        "NUM_KV_HEADS": 8,
        "HEAD_DIM": 128,
    },
    "llama3-70b": {
        "HIDDEN_SIZE": 8192,
        "INTERMEDIATE_SIZE": 28672,
        "NUM_HEADS": 64,
        "NUM_KV_HEADS": 8,
        "HEAD_DIM": 128,
    },
}


def get_model_ops_config(model_name: str = "llama3-8b"):
    """Return per-layer operation configs with (M_func, K, N) for full (TP=1) dims."""
    cfg = MODEL_CONFIGS[model_name]
    H = cfg["HIDDEN_SIZE"]
    I = cfg["INTERMEDIATE_SIZE"]
    NH = cfg["NUM_HEADS"]
    NKV = cfg["NUM_KV_HEADS"]
    HD = cfg["HEAD_DIM"]
    return {
        "preproj": {
            "K": H, "N": (NH + 2 * NKV) * HD,
            "tp_shard": "col",  # column-parallel: N is sharded
            "desc": "Attention QKV Projection (fused)",
        },
        "postproj": {
            "K": H, "N": H,
            "tp_shard": "row",  # row-parallel: K is sharded
            "desc": "Attention O Projection",
        },
        "mlp_up": {
            "K": H, "N": I * 2,  # gate_up_proj fused
            "tp_shard": "col",
            "desc": "MLP Gate+Up Projection",
        },
        "mlp_down": {
            "K": I, "N": H,
            "tp_shard": "row",
            "desc": "MLP Down Projection",
        },
    }


def get_tp_sharded_shape(op_cfg: dict, tp_size: int):
    """Return (K, N) after TP sharding for a given operation config.

    Column-parallel (q/k/v/gate_up): weight is [K, N/TP]  → N shrinks
    Row-parallel (o/down):           weight is [K/TP, N]  → K shrinks
    """
    K, N = op_cfg["K"], op_cfg["N"]
    if op_cfg["tp_shard"] == "col":
        return K, N // tp_size
    else:  # row
        return K // tp_size, N
