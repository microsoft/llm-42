# Temperature-Based Dynamic Batch-Invariant Mode

## Overview

This feature adds an **optional temperature-based dynamic switching layer** on top of the existing batch-invariant modes in SGLang. It does NOT modify or replace the existing batch-invariant behavior.

### Existing Behavior (Unchanged)

The existing batch-invariant modes work exactly as before:
- `--enable-deterministic-inference 1`: Always uses batch-invariant with mode 1 (ThinkingMachine kernel)
- `--enable-deterministic-inference 2`: Always uses batch-invariant with mode 2 (CUTLASS kernel)
- `--enable-deterministic-inference 4`: Always uses batch-invariant with mode 4 (25% split CUTLASS)

### New Feature: Temperature-Based Dynamic Switching

By setting **bit 512** (value 512), you enable an **additional layer** that dynamically switches batch-invariant mode based on request temperatures:

1. **temperature == 0**: Use batch-invariant (deterministic) implementation
2. **temperature > 0**: Use non-deterministic (standard PyTorch) implementation
3. **Mixed batch**: If at least one request has temperature == 0, use batch-invariant for the entire batch
4. **All non-deterministic**: If all requests have temperature > 0, use default implementation

## Usage

### Option 1: Existing Static Batch-Invariant Mode (Unchanged)

Use WITHOUT bit 32 for static batch-invariant behavior (always on):

```bash
# Always use batch-invariant mode 1 (existing behavior)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 1

# Always use batch-invariant mode 2 (existing behavior)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 2
```

### Option 2: New Temperature-Based Dynamic Mode

Use WITH bit 512 to enable temperature-based switching:

```bash
# Enable dynamic mode with mode 1 (512 + 1 = 513)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 513

# Enable dynamic mode with mode 2 (512 + 2 = 514)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 514

# Enable dynamic mode with mode 4 (512 + 4 = 516)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 516
```

### Flag Breakdown

The `--enable-deterministic-inference` flag uses a **bitwise system** where different bits control different deterministic behaviors across the codebase:

| Bit(s) | Value | Component | Purpose |
|--------|-------|-----------|---------|
| 1, 2, 4 | 1, 2, 4 | batch_invariant_ops | Matmul kernel mode selection |
| 32 | 32 | (reserved) | Reserved for matmul behavior |
| 64 | 64 | layernorm | Layernorm behavior control |
| 128 | 128 | flashinfer_backend | Flashinfer deterministic mode |
| 256 | 256 | layernorm | vLLM RMSNorm mode |
| 512 | 512 | model_runner | Temperature-based switching (NEW) |

**Bits 1, 2, 4** (Matmul modes):
- `1`: ThinkingMachine kernel
- `2`: CUTLASS kernel
- `4`: 25% split CUTLASS kernel

**Bit 512** (Temperature-based switching - NEW FEATURE):
- Enables dynamic control based on request temperatures
- Opt-in feature that adds a layer on top of static modes

**Other bits**: Control other deterministic behaviors (layernorm, flashinfer, etc.)

### Combining Flags

You can combine multiple bits using bitwise OR (addition works for non-overlapping bits):

```bash
# Static mode 1 only
--enable-deterministic-inference 1

# Dynamic mode with mode 1 (512 + 1 = 513)
--enable-deterministic-inference 513

# Dynamic mode with mode 1 + layernorm deterministic (512 + 1 + 64 = 577)
--enable-deterministic-inference 577

# Dynamic mode with mode 2 + flashinfer deterministic (512 + 2 + 128 = 642)
--enable-deterministic-inference 642
```

### Without Bit 32 (Existing Behavior - Unchanged)

### Without Bit 32 (Existing Behavior - Unchanged)

The existing behavior is preserved when bit 32 is NOT set:

```bash
# Static mode: Always use batch-invariant mode 1 (EXISTING BEHAVIOR)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 1

# Static mode: Always use batch-invariant mode 2 (EXISTING BEHAVIOR)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 2
```

**Important**: All existing deployments continue to work exactly as before. This is a purely additive feature.

## Implementation Details

### Code Changes

This feature adds a new layer of control WITHOUT modifying the existing batch-invariant behavior:

1. **model_runner.py**: Enhanced initialization and forward pass
   - Preserves existing static batch-invariant mode behavior when bit 32 is NOT set
   - Adds temperature-based checking only when bit 32 IS set
   - Uses `self.enable_temperature_based_switching` flag to control the new feature
   - Stores `self.temperature_based_mode_value` for the mode to use when dynamic switching

2. **Initialization Logic**:
   ```python
   # Existing behavior (unchanged)
   if enable_deterministic_inference and not (enable_deterministic_inference & 512):
       enable_batch_invariant_mode(enable_deterministic_inference)
   
   # New feature (additive)
   self.enable_temperature_based_switching = bool(enable_deterministic_inference & 512)
   ```

3. **Forward Pass Logic** (only active when bit 512 is set):
   ```python
   if self.enable_temperature_based_switching:
       # Check temperatures in batch
       use_batch_invariant = torch.any(temperatures < 1e-6).item()
       # Apply context manager for dynamic switching
       with set_batch_invariant_mode(enabled=use_batch_invariant):
           # Execute forward pass
   ```

### How It Works

1. **Without Bit 512 (Existing Behavior)**:
   - Batch-invariant mode is enabled statically at initialization
   - Remains on for all forward passes
   - No temperature checking occurs
   - **This is the default and existing behavior**

2. **With Bit 512 (New Feature)**:
   - Server starts WITHOUT batch-invariant mode enabled by default
   - During each forward pass:
     - Check temperatures in the batch
     - If any temp == 0: Enable batch-invariant mode for this pass
     - If all temp > 0: Use default (non-batch-invariant) implementation
   - Clean up state after forward pass

3. **Performance Impact**:
   - Without bit 512: Zero overhead (existing code path)
   - With bit 512: Single tensor comparison per batch (~microseconds)
   - Benefit: Avoids batch-invariant overhead for non-deterministic batches

## Examples

### Basic Usage

```bash
# Static mode 1: Always batch-invariant (EXISTING)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 1

# Dynamic mode 1: Temperature-based with mode 1 (NEW)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 513
```

### Advanced: Combining Multiple Deterministic Features

```bash
# Dynamic mode 1 + layernorm deterministic (513 + 64 = 577)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 577

# Dynamic mode 2 + flashinfer deterministic (514 + 128 = 642)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 642

# All deterministic features: mode 1 + temp-based + layernorm + flashinfer
# (1 + 512 + 64 + 128 = 705)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 705
```

## Benefits

1. **Flexibility**: Switch between deterministic and non-deterministic based on need
2. **Performance**: Only use batch-invariant when necessary
3. **Correctness**: Ensures deterministic results when temperature == 0
4. **Simplicity**: User-controlled via standard temperature parameter

## Testing

Run the test script to verify the implementation:

```bash
# Start server with dynamic mode
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 33

# Run tests
python test_temperature_based_deterministic.py
```

## Notes

- Temperature threshold is `1e-6` to account for floating-point precision
- Batch-invariant mode applies to the entire batch if any request needs it
- This ensures consistent results within a batch
- The mode value (1, 2, 4, etc.) is preserved from initialization
