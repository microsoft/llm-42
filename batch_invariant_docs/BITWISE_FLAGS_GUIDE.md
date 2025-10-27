# Complete Guide: enable_deterministic_inference Bitwise Flags

## Overview

The `--enable-deterministic-inference` flag uses a **bitwise system** where different bits control different deterministic behaviors across the SGLang codebase. This allows fine-grained control over which deterministic features are enabled.

## Bit Assignments

| Bit(s) | Value | Component | File | Purpose |
|--------|-------|-----------|------|---------|
| 1, 2, 4 | 1, 2, 4 | Batch-invariant matmul | batch_invariant_ops.py | Matmul kernel selection |
| 32 | 32 | (reserved) | (reserved) | Reserved for matmul behavior |
| 64 | 64 | Layernorm | layernorm.py | Layernorm deterministic behavior |
| 128 | 128 | Flashinfer attention | flashinfer_backend.py | Flashinfer deterministic mode |
| 256 | 256 | vLLM RMSNorm | layernorm.py | vLLM RMSNorm mode |
| 512 | 512 | Temperature-based switching | model_runner.py | Dynamic batch-invariant control |

## Detailed Descriptions

### Bits 1, 2, 4: Batch-Invariant Matmul Modes

Controlled in: `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`

- **Bit 1 (value 1)**: ThinkingMachine kernel
  - Uses existing batch-invariant matmul implementation
  - Ensures deterministic matrix multiplication

- **Bit 2 (value 2)**: CUTLASS kernel
  - Uses CUDA-based CUTLASS kernel for matmul
  - Higher performance with determinism

- **Bit 4 (value 4)**: 25% split CUTLASS kernel
  - Uses CUTLASS with 25% work splitting
  - Optimized for specific workloads

**Note**: Only one of bits 1, 2, or 4 should be set at a time for matmul mode selection.

### Bit 512: Temperature-Based Dynamic Switching (NEW)

Controlled in: `python/sglang/srt/model_executor/model_runner.py`

- **Enables**: Dynamic batch-invariant mode based on request temperatures
- **Behavior**:
  - When ANY request has `temperature == 0`: Use batch-invariant for the batch
  - When ALL requests have `temperature > 0`: Use default (non-deterministic) implementation
- **Default**: OFF (opt-in feature)

### Bit 64: Layernorm Deterministic Behavior

Controlled in: `python/sglang/srt/layers/layernorm.py`

- **When SET**: Disables certain layernorm optimizations for determinism
- **Check**: `if deterministic and not (deterministic & 64)`
- **Effect**: Uses native forward method for layernorm

### Bit 128: Flashinfer Deterministic Mode

Controlled in: `python/sglang/srt/layers/attention/flashinfer_backend.py`

- **When SET**: Disables flashinfer deterministic mode
- **Check**: `not (enable_deterministic_inference & 128)`
- **Effect**: Enables tensor cores for decode, sets split tile sizes

### Bit 256: vLLM RMSNorm Mode

Controlled in: `python/sglang/srt/layers/layernorm.py`

- **When SET**: Disables vLLM RMSNorm mode
- **Check**: `if deterministic and not (deterministic & 256)`
- **Effect**: Uses vLLM fused RMSNorm with batch size 256

## Usage Examples

### Basic Static Modes

```bash
# Mode 1 only (always batch-invariant)
--enable-deterministic-inference 1

# Mode 2 only (always batch-invariant)
--enable-deterministic-inference 2
```

### Temperature-Based Dynamic Modes

```bash
# Dynamic mode with matmul mode 1 (512 + 1 = 513)
--enable-deterministic-inference 513

# Dynamic mode with matmul mode 2 (512 + 2 = 514)
--enable-deterministic-inference 514
```

### Combining Multiple Features

```bash
# Dynamic mode 1 + layernorm deterministic
# 512 (temp-based) + 1 (mode 1) + 64 (layernorm) = 577
--enable-deterministic-inference 577

# Dynamic mode 1 + flashinfer deterministic
# 512 (temp-based) + 1 (mode 1) + 128 (flashinfer) = 641
--enable-deterministic-inference 641

# All deterministic features (except vLLM RMSNorm)
# 512 + 1 + 64 + 128 = 705
--enable-deterministic-inference 705

# Everything including vLLM RMSNorm
# 512 + 1 + 64 + 128 + 256 = 961
--enable-deterministic-inference 961
```

## How Components Check Flags

Each component checks for its specific bit(s):

```python
# In batch_invariant_ops.py
if mode & 1:
    # Use mode 1
elif mode & 2:
    # Use mode 2
elif mode & 4:
    # Use mode 4

# In model_runner.py (NEW)
if enable_deterministic_inference & 512:
    # Enable temperature-based switching

# In layernorm.py
if deterministic and not (deterministic & 64):
    # Use native layernorm

# In flashinfer_backend.py
if enable_deterministic_inference and not (enable_deterministic_inference & 128):
    # Enable flashinfer deterministic mode

# In layernorm.py
if deterministic and not (deterministic & 256):
    # Use vLLM RMSNorm mode
```

## Calculating Flag Values

To calculate the flag value, add the bit values you want to enable:

1. Choose a matmul mode: 1, 2, or 4
2. Add 512 if you want temperature-based switching (NEW)
3. Add 64 if you want layernorm deterministic
4. Add 128 if you want flashinfer deterministic
5. Add 256 if you want vLLM RMSNorm mode

Example:
- Want: Dynamic mode with CUTLASS + flashinfer deterministic
- Calculation: 2 (CUTLASS) + 512 (dynamic) + 128 (flashinfer) = 642
- Command: `--enable-deterministic-inference 642`

## Backward Compatibility

All existing flag values work without modification:

- `--enable-deterministic-inference 1` → Still works (static mode 1)
- `--enable-deterministic-inference 2` → Still works (static mode 2)
- `--enable-deterministic-inference 4` → Still works (static mode 4)

New features (bit 32) are opt-in and don't affect existing behavior.

## Common Configurations

| Use Case | Value | Bits | Description |
|----------|-------|------|-------------|
| Basic deterministic | 1 | 1 | Static batch-invariant mode 1 |
| Dynamic deterministic | 513 | 512+1 | Temp-based with mode 1 |
| Full deterministic (no vLLM) | 705 | 512+1+64+128 | All features except vLLM |
| Complete deterministic | 961 | 512+1+64+128+256 | All deterministic features |

## Notes

- Bits can be combined using bitwise OR (addition works for non-overlapping bits)
- Each component independently checks for its relevant bit(s)
- Setting a bit to 1 enables that feature (except for "disable" bits like 64, 128, 256)
- The system is extensible: new bits can be added without affecting existing ones
