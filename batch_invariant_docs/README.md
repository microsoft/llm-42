# Deterministic Inference Guide

This document describes the deterministic inference system in SGLang, which provides reproducible, batch-invariant operations for model inference.

## Overview

SGLang supports three independent deterministic inference modes, each using simple number-based configuration:

1. **Global Deterministic Inference** (`--enable-deterministic-inference`)
2. **Forward-Mode-Based Switching** (`--enable-det-infer`)
3. **Batch-Composition-Based Switching** (`--enable-selective-determinism`)

## Mode Values

All three flags use the same numbering system:

- **`0`**: Disabled (default)
- **`1`**: Use `bi_kernel` (CUTLASS matmul) + `vllm_rmsnorm`
- **`2`**: Use `batch_invariant` (Triton matmul) + `native_rmsnorm`

## Flag Descriptions

### 1. Global Deterministic Inference

```bash
--enable-deterministic-inference [0|1|2]
```

**Purpose**: Enables batch-invariant operations globally and statically throughout the entire inference process.

**Behavior**:
- When set to `1` or `2`, batch-invariant mode is always active
- Applies to all forward passes regardless of forward mode or batch composition
- Sets the CUDA matmul implementation and RMSNorm variant

**Use Case**: When you need consistent deterministic behavior for all requests.

**Example**:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-deterministic-inference 2
```

---

### 2. Forward-Mode-Based Switching

```bash
--enable-det-infer [0|1|2]
```

**Purpose**: Dynamically switches batch-invariant operations based on the forward mode.

**Behavior**:
- **DECODE mode**: Batch-invariant is **disabled** (uses faster non-deterministic operations)
- **EXTEND/VERIFY modes**: Batch-invariant is **enabled**
- Optimizes for performance during decode while maintaining determinism during prefill
- Uses standard CUDA graphs (no dual graphs needed - switching is based on forward mode)

**Use Case**: Deterministic verification workflows where you want to trade off performance during decode for determinism during verification.

**Example**:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-det-infer 2
```

**Forward Modes**:
- `DECODE`: Single-token generation (fast path)
- `EXTEND`: Prefill with cached prefix
- `TARGET_VERIFY`: Verification pass for speculative decoding
- `TARGET_DET_VERIFY`: Deterministic verification pass

---

### 3. Batch-Composition-Based Switching (Selective Determinism)

```bash
--enable-selective-determinism [0|1|2]
```

**Purpose**: Dynamically switches batch-invariant operations based on whether ANY request in the batch requires deterministic behavior.

**Behavior**:
- Checks the `is_any_deterministic` flag for each batch
- If **ALL** requests are non-deterministic (temperature > 0): Batch-invariant **disabled**
- If **ANY** request is deterministic (temperature = 0, greedy sampling): Batch-invariant **enabled**
- Uses **dual CUDA graphs**: Captures two sets of graphs (one deterministic, one not) to enable efficient runtime switching

**Use Case**: Mixed workloads where some requests need deterministic behavior (temperature=0) and others don't (temperature>0).

**Example**:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-selective-determinism 2
```

**Note**: This mode doubles CUDA graph memory usage and capture time due to dual graph support.

---

## Implementation Details

### Kernel Selection

#### Mode 1: `bi_kernel` + `vllm_rmsnorm`
- **Matmul**: CUTLASS-based batch-invariant kernel (`bf16_batch_invariant_mm`)
- **RMSNorm**: vLLM's fused RMSNorm implementation
- **Performance**: Faster than mode 2, good determinism
- **Use**: Recommended for most deterministic workloads

#### Mode 2: `batch_invariant` + `native_rmsnorm`
- **Matmul**: Triton persistent matmul kernel (`matmul_persistent`)
- **RMSNorm**: Native PyTorch implementation (float32 accumulation)
- **Performance**: Slower but maximum determinism guarantees
- **Use**: When strictest determinism is required

### Attention Backend Configuration

When any deterministic mode is enabled:
- **FlashInfer**: Uses fixed split tile sizes, tensor cores for decode, disables KV split for CUDA graphs
- **Triton**: Uses fixed split tile size (256)
- **FlashAttention**: Forces `num_splits=1`

### Environment Variables

The following environment variables are automatically set:

```bash
SGLANG_ENABLE_DETERMINISTIC_INFERENCE=<mode>  # From --enable-deterministic-inference or --enable-det-infer
SGLANG_ENABLE_SELECTIVE_DETERMINISM=<mode>    # From --enable-selective-determinism
```

You can also manually configure:

```bash
SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096  # Prefill split size
SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=2048   # Decode split size
SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=256        # Triton decode split size
```

---

## Combining Flags

The three flags are **independent** and can be combined:

### Example 1: Global + Selective
```bash
--enable-deterministic-inference 1 \
--enable-selective-determinism 2
```
- Global mode provides baseline determinism
- Selective determinism can further optimize when batch is fully non-deterministic
- ⚠️ Generally not recommended (conflicting goals)

### Example 2: Forward-Mode + Selective
```bash
--enable-det-infer 2 \
--enable-selective-determinism 1
```
- Forward-mode handles DECODE/EXTEND switching
- Selective determinism handles batch composition
- Both systems work together with dual CUDA graphs

### Recommendation
**Use only ONE flag at a time** for clearest behavior:
- Production with all deterministic: `--enable-deterministic-inference 1`
- Verification workflows: `--enable-det-infer 2`
- Mixed workloads: `--enable-selective-determinism 1`

---

## Architecture

### Key Components

1. **`batch_invariant_ops.py`**: Core kernel implementations
   - `enable_batch_invariant_mode(mode)`: Activates batch-invariant kernels
   - `disable_batch_invariant_mode()`: Deactivates batch-invariant kernels
   - `is_batch_invariant_mode_enabled()`: Checks current state

2. **`model_runner.py`**: Forward pass orchestration
   - Manages batch-invariant mode enabling/disabling
   - Handles forward-mode-based and selective determinism logic
   - Tracks statistics for deterministic vs non-deterministic passes

3. **Attention Backends**: Configure deterministic settings
   - `flashinfer_backend.py`: Dual CUDA graph support, split tile configuration
   - `triton_backend.py`: Fixed split tile for determinism
   - `flashattention_backend.py`: Forces single split

4. **`layernorm.py`**: RMSNorm implementations
   - Switches between vLLM and native based on mode
   - Supports dynamic switching for selective determinism

### CUDA Graph Support

**Standard Graph Mode** (Global, Forward-Mode):
- One set of CUDA graphs per batch size and forward mode
- Storage key: `(batch_size,)` for global deterministic, separate graphs per forward mode for `enable_det_infer`
- Forward-mode switching uses different graph captures (DECODE graphs vs EXTEND graphs)

**Dual Graph Mode** (Selective Determinism only):
- Two sets of CUDA graphs per batch size (within same forward mode)
- Storage key: `(batch_size, is_deterministic_graph)`
- Automatically selects correct graph based on `is_any_deterministic` flag
- Required because switching happens dynamically within the same forward mode

---

## Performance Considerations

### Memory Usage
- **Mode 1**: Standard memory usage
- **Mode 2**: Slightly higher (float32 accumulation in RMSNorm)
- **Selective Determinism only**: 2x CUDA graph memory (dual graphs within same forward mode)
- **Forward-Mode Switching**: Standard memory (separate graphs per forward mode, not duplicated)

### Latency Impact
- **Mode 1**: ~5-10% slower than non-deterministic
- **Mode 2**: ~10-20% slower than non-deterministic
- **Forward-Mode Switching**: Only extends/verifies are slower, decode remains fast
- **Selective Determinism**: Near-zero overhead when switching (pre-captured graphs)

### Throughput
- Deterministic modes reduce parallelism slightly
- Selective determinism provides best throughput for mixed workloads

---

## Troubleshooting

### Issue: Non-deterministic results despite flag being set

**Check**:
1. Verify sampling backend is set to `pytorch` (automatic when deterministic mode enabled)
2. Ensure attention backend is supported (`flashinfer`, `fa3`, or `triton`)
3. Check that radix cache is not disabled (required for deterministic inference)

### Issue: CUDA OOM with selective determinism

**Cause**: Dual CUDA graphs double memory usage

**Solutions**:
- Use `--enable-det-infer` instead (single graph)
- Reduce `--max-total-tokens`
- Use mode 1 instead of mode 2 (lower memory overhead)

### Issue: Slower than expected performance

**Check**:
1. Mode 2 is slower than mode 1 - consider switching
2. Selective determinism requires dual graphs - verify it's needed
3. Check if `enable-deterministic-inference` is set when you meant `enable-det-infer`

---

## Migration from Old Bitwise System

If you have old configurations using bitwise flags:

### Old System (Deprecated)
```bash
--enable-deterministic-inference 1    # Matmul only
--enable-deterministic-inference 513  # Matmul (1) + Selective (512)
--enable-deterministic-inference 130  # Matmul (2) + Skip FlashInfer (128)
```

### New System
```bash
--enable-deterministic-inference 1        # Mode 1: bi_kernel
--enable-selective-determinism 1          # Separate flag for selective
--enable-det-infer 2                      # Forward-mode switching
```

**Key Changes**:
- No more bitwise operations (`&`, `|`)
- Simple 0/1/2 values for each flag
- Three independent flags instead of one combined flag
- Clearer naming: `selective_determinism` instead of `temperature_based_switching`

---

## Examples

### Example 1: Simple Deterministic Server
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-deterministic-inference 1
```

### Example 2: Verification with Det-Infer
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-det-infer 2
```

### Example 3: Mixed Workload with Selective Determinism
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-selective-determinism 1
```

### Example 4: Testing Both Modes
```python
# Test mode 1
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "What is 2+2?",
    "temperature": 0,
    "max_tokens": 100
  }'

# Test mode 2
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "What is 2+2?",
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

---

## Summary

- **Three independent flags**: `enable-deterministic-inference`, `enable-det-infer`, `enable-selective-determinism`
- **Two modes per flag**: `1` (bi_kernel+vllm) or `2` (batch_invariant+native)
- **Use one flag at a time** for clearest behavior
- **Mode 1 recommended** for most use cases (good balance of speed and determinism)
- **Selective determinism** for mixed temperature workloads
- **Forward-mode switching** for verification workflows
