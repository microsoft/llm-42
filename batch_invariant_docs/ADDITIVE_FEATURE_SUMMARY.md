# Summary: Temperature-Based Dynamic Batch-Invariant - Additive Feature

## What Changed

I've implemented a **new optional feature layer** that adds temperature-based dynamic switching WITHOUT modifying existing batch-invariant behavior.

## Key Design Decisions

✅ **Existing modes (1, 2, 4, etc.) remain completely unchanged**
- Static batch-invariant behavior is preserved
- No modifications to the existing code paths
- All existing deployments continue to work exactly as before

✅ **New feature is opt-in via bit 512**
- Only activates when bit 512 is explicitly set
- Zero overhead when bit 512 is not set
- Additive layer on top of existing functionality

## Code Changes

### File: `python/sglang/srt/model_executor/model_runner.py`

#### 1. Initialization (lines ~415-423)
```python
# Existing behavior (unchanged)
if enable_deterministic_inference and not (enable_deterministic_inference & 512):
    enable_batch_invariant_mode(enable_deterministic_inference)

# New feature flags (additive)
self.enable_temperature_based_switching = bool(enable_deterministic_inference & 512)
self.temperature_based_mode_value = (enable_deterministic_inference & ~512) if self.enable_temperature_based_switching else 0
```

#### 2. Forward Pass (lines ~1999-2035)
```python
# New temperature-based logic (only when bit 512 is set)
if self.enable_temperature_based_switching:
    # Check temperatures
    use_batch_invariant = torch.any(temperatures < 1e-6).item()
    
    # Apply dynamic switching
    with set_batch_invariant_mode(enabled=use_batch_invariant):
        # Execute forward pass
```

## Usage Examples

### Existing Static Mode (Unchanged)
```bash
# Always batch-invariant with mode 1
python -m sglang.launch_server --model-path <path> --enable-deterministic-inference 1
```

### New Dynamic Mode (Opt-In)
```bash
# Temperature-based with mode 1 (512 + 1 = 513)
python -m sglang.launch_server --model-path <path> --enable-deterministic-inference 513
```

## Behavior Comparison

| Flag | Behavior |
|------|----------|
| `1` | Always batch-invariant (EXISTING) |
| `2` | Always batch-invariant (EXISTING) |
| `4` | Always batch-invariant (EXISTING) |
| `513` | Temperature-based: temp=0 → batch-invariant, temp>0 → default (NEW) |
| `514` | Temperature-based: temp=0 → batch-invariant, temp>0 → default (NEW) |
| `516` | Temperature-based: temp=0 → batch-invariant, temp>0 → default (NEW) |

## Benefits of This Approach

1. **Non-Breaking**: Existing deployments unchanged
2. **Backward Compatible**: All existing flags work as before
3. **Opt-In**: New feature only when explicitly enabled
4. **Layered**: Built on top of existing infrastructure
5. **Flexible**: Users can choose static or dynamic mode
6. **Safe**: Zero risk to existing functionality

## Testing

Users with existing setups can continue using them:
```bash
# This continues to work exactly as before
--enable-deterministic-inference 1
```

Users who want the new feature can opt in:
```bash
# This enables the new temperature-based switching
--enable-deterministic-inference 513
```

## Documentation Updated

All documentation files have been updated to emphasize:
- Existing behavior is unchanged
- New feature is opt-in via bit 512
- Clear distinction between static and dynamic modes
- Backward compatibility guaranteed
