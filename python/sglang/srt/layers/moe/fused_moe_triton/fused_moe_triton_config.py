from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton

from sglang.srt.utils import get_device_name, is_hip

# Import global_server_args_dict for deterministic inference check
# This dict is populated early in server startup and remains valid during CUDA graph capture
def _get_deterministic_flags():
    """Get deterministic inference flags from global_server_args_dict.
    
    Returns (enable_deterministic_inference, enable_det_infer) tuple.
    Falls back to (0, 0) if dict is not available (e.g., in tuning scripts).
    """
    try:
        from sglang.srt.managers.schedule_batch import global_server_args_dict
        enable_det_infer = global_server_args_dict.get("enable_det_infer", 0) or 0
        enable_deterministic_inference = global_server_args_dict.get("enable_deterministic_inference", 0) or 0
        return enable_deterministic_inference, enable_det_infer
    except (ImportError, KeyError):
        return 0, 0

logger = logging.getLogger(__name__)
_is_hip = is_hip()


def get_config_file_name(
    E: int,
    N: int,
    dtype: Optional[str],
    block_shape: Optional[List[int]] = None,
    per_channel_quant: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    per_channel_quant_selector = ",per_channel_quant=True" if per_channel_quant else ""
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}{per_channel_quant_selector}.json"


@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = 0,
    block_k: Optional[int] = 0,
    per_channel_quant: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    # Check for deterministic inference - use default config for batch invariance
    # Mode 3 (enable_det_infer=3) uses non-batch-invariant kernels, so skip this check
    enable_deterministic_inference, enable_det_infer = _get_deterministic_flags()
    logger.info(f"Checking deterministic inference settings in get_moe_configs...")
    logger.info(f"  enable_deterministic_inference: {enable_deterministic_inference}, enable_det_infer: {enable_det_infer}")
    if enable_deterministic_inference > 0 or (enable_det_infer > 0 and enable_det_infer != 3):
        logger.warning(
            "Deterministic inference is enabled (enable_deterministic_inference=%d, enable_det_infer=%d), "
            "using default MoE kernel config for batch invariance.",
            enable_deterministic_inference, enable_det_infer
        )
        return None

    # Supported Triton versions, should be sorted from the newest to the oldest
    supported_triton_versions = ["3.4.0", "3.3.1", "3.2.0", "3.1.0"]

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype, [block_n, block_k])

    # We found that using the fused_moe_kernel config from Triton 3.1.0 with Triton 3.2.0 results in negative performance gains,
    # so we also include the Triton version as a key for finding the fused_moe_kernel config to achieve the best performance.
    config_dir = os.environ.get(
        "SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_file_path = os.path.join(
        config_dir,
        "configs",
        version_dir,
        json_file_name,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # Please note that although we find the config files, performance might still be suboptimal.
            # This is because the tuning environment might differ from your current environment.
            # For example, updating the Triton version might cause all old configs to become suboptimal.
            # To achieve the best performance, consider re-tuning the Triton fused MOE kernel in your environment.
            # For the tuning method, refer to: https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            logger.info(f"Using MoE kernel config from {config_file_path}.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # Searching for other triton versions that supports the same config
    for try_triton_version in supported_triton_versions:
        if try_triton_version == triton_version:
            continue
        try_config_file_path = os.path.join(
            config_dir,
            "configs",
            f"triton_{try_triton_version.replace('.', '_')}",
            json_file_name,
        )
        if os.path.exists(try_config_file_path):
            with open(try_config_file_path) as f:
                logger.warning(
                    f"Config file not found at {config_file_path}. Fallback to triton version {try_triton_version} and use MoE kernel config from {try_config_file_path}. Performance might be sub-optimal!",
                )
                # If a configuration has been found, return it
                return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default MoE kernel config. Performance might be sub-optimal! "
            "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
        ),
        config_file_path,
    )
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    # Check for deterministic inference - use fixed config for batch invariance
    # Mode 3 (enable_det_infer=3) uses non-batch-invariant kernels, so skip this check
    enable_deterministic_inference, enable_det_infer = _get_deterministic_flags()
    if enable_deterministic_inference > 0 or (enable_det_infer > 0 and enable_det_infer != 3):
        # Return fixed config regardless of M, E, dtype for batch invariance
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

    if dtype == "fp8_w8a8":
        if block_shape is None:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 2 if _is_hip else 4,
            }
            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2 if _is_hip else 4,
                }
        else:
            # Block-wise quant: BLOCK_SIZE_K must be divisible by block_shape[1]
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2 if _is_hip else 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
):
    from sglang.srt.layers.moe.fused_moe_triton import get_config

    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(
                M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape
            )
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None
