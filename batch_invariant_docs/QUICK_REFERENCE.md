# Quick Reference: Temperature-Based Batch-Invariant Mode

## TL;DR

This is an **optional new feature** (bit 512) that adds temperature-based switching ON TOP of existing batch-invariant modes.

**Existing behavior (unchanged)**:
- `--enable-deterministic-inference 1` → Always batch-invariant

**New feature (opt-in)**:
- `--enable-deterministic-inference 513` (512 + 1) → Temperature-based: temp=0 uses batch-invariant, temp>0 uses default

## Setup

### Option 1: Existing Static Mode (Unchanged)
```bash
# Always use batch-invariant (EXISTING BEHAVIOR)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 1
```

### Option 2: New Temperature-Based Dynamic Mode
```bash
# Add bit 512 for temperature-based switching (NEW FEATURE)
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-deterministic-inference 513
```

## API Usage

```python
# Deterministic request (uses batch-invariant)
completion = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.0  # ← Triggers batch-invariant mode
)

# Non-deterministic request (uses default)
completion = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    temperature=0.8  # ← Uses standard PyTorch
)
```

## Flag Values

The `--enable-deterministic-inference` flag uses **bitwise flags** to control multiple deterministic features:

### Matmul Modes (Bits 1, 2, 4)
| Bit | Value | Description |
|-----|-------|-------------|
| 1 | 1 | ThinkingMachine kernel |
| 2 | 2 | CUTLASS kernel |
| 4 | 4 | 25% split CUTLASS |

### Additional Features
| Bit | Value | Feature | Controlled In |
|-----|-------|---------|---------------|
| 32 | 32 | Temperature-based switching (NEW) | model_runner.py |
| 64 | 64 | Layernorm deterministic | layernorm.py |
| 128 | 128 | Flashinfer deterministic | flashinfer_backend.py |
| 256 | 256 | vLLM RMSNorm mode | layernorm.py |

### Common Flag Values

| Value | Meaning |
|-------|---------|
| `1` | Static batch-invariant (ThinkingMachine matmul) |
| `2` | Static batch-invariant (CUTLASS matmul) |
| `513` | Temperature-based dynamic (512) + ThinkingMachine matmul (1) |
| `514` | Temperature-based dynamic (512) + CUTLASS matmul (2) |
| `577` | Temperature-based (512) + ThinkingMachine matmul (1) + LayerNorm (64) |
| `705` | Temperature-based (512) + ThinkingMachine matmul (1) + LayerNorm (64) + FlashInfer (128) |
| `961` | Temperature-based (512) + ThinkingMachine matmul (1) + LayerNorm (64) + FlashInfer (128) + vLLM RMSNorm (256) |

## Batch Behavior (When Bit 32 is Set)

### Batch Behavior Summary

| Batch Composition | Flag Value 1 (Static) | Flag Value 513 (Temp-based) |
|-------------------|----------------------|---------------------------|
| All temp=0 | Batch-invariant | Batch-invariant |
| All temp>0 | Batch-invariant | Non-deterministic (default) |
| Mixed temps | Batch-invariant | Batch-invariant |

**Rule**: Batch-invariant is used if **any** request has temp=0

**Note**: This only applies when bit 32 is set. Without bit 32, static mode is always used.

## Benefits

✅ Automatic switching based on temperature  
✅ No performance penalty for non-deterministic batches  
✅ Guaranteed determinism when temp=0  
✅ Works with existing SGLang API (no changes needed)

## Testing

```bash
# Terminal 1: Start server
python -m sglang.launch_server --model-path <path> --enable-deterministic-inference 33

# Terminal 2: Run tests
python test_temperature_based_deterministic.py
```

## Example Output

```
# Request with temp=0.0 (run twice)
Response 1: "4"
Response 2: "4"  ← Same output (deterministic)

# Request with temp=0.8 (run twice)
Response 1: "Four"
Response 2: "The answer is 4."  ← Different outputs (non-deterministic)
```
