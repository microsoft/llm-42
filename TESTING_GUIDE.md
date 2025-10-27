# Testing Temperature-Based Batch-Invariant Switching

## Quick Start

### 1. Start the Server

Open a terminal and run:

```bash
./launch_temperature_test.sh
```

Or manually:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1 \
    --enable-deterministic-inference 513
```

**Flag 513 = 512 + 1:**
- Bit 512: Enables temperature-based switching
- Bit 1: Uses ThinkingMachine matmul kernel

Wait for the server to fully start (you'll see "Uvicorn running on http://0.0.0.0:30000").

### 2. Run the Tests

Open a **new terminal** and run:

```bash
python test_temperature_switching.py
```

Or if on a different machine:

```bash
python test_temperature_switching.py http://your-server-ip:30000
```

## What to Expect

### Test Output

The test script will:
1. Send 5 requests with different temperatures (0, 0.8, 0, 1.0, 0)
2. Check deterministic consistency with temperature=0
3. Display results for each test

Example output:
```
================================================================================
Testing Temperature-Based Batch-Invariant Switching
================================================================================

✓ Server is running at http://localhost:30000

Running test cases...

Test 1: Temperature = 0 (should use batch-invariant)
  Temperature: 0.0
  Expected mode: batch-invariant
  Response: The answer is 4....
  ✓ Request successful

Test 2: Temperature = 0.8 (should use non-deterministic)
  Temperature: 0.8
  Expected mode: non-deterministic
  Response: Let me tell you a story about......
  ✓ Request successful
...
```

### Server Logs

In the **server terminal**, look for these logs:

1. **Initialization:**
   ```
   Enabling batch invariant mode with existing kernels...
   ```

2. **Statistics (every 100 forward passes):**
   ```
   Temperature-based switching stats: batch_invariant=75, non_deterministic=25, total=100
   ```

The statistics show:
- `batch_invariant`: Number of forward passes using deterministic mode
- `non_deterministic`: Number of forward passes using non-deterministic mode
- `total`: Total forward passes

## Manual Testing

You can also test manually using curl:

### Temperature = 0 (Deterministic)

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.0,
    "max_tokens": 50
  }'
```

Run this multiple times - responses should be identical.

### Temperature > 0 (Non-Deterministic)

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Tell me a story."}],
    "temperature": 0.8,
    "max_tokens": 50
  }'
```

Run this multiple times - responses may vary.

## Testing Different Modes

### Static Batch-Invariant (Always On)

```bash
# Start server with flag 1 (no temperature switching)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-deterministic-inference 1
```

Behavior: Always uses batch-invariant, regardless of temperature.

### Temperature-Based with CUTLASS Kernel

```bash
# Start server with flag 514 (512 + 2)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-deterministic-inference 514
```

Behavior: Temperature-based switching with CUTLASS matmul kernel.

### Temperature-Based with All Features

```bash
# Start server with flag 705 (512 + 1 + 64 + 128)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-deterministic-inference 705
```

Behavior: Temperature-based + matmul + LayerNorm + FlashInfer deterministic.

## Verification Checklist

- [ ] Server starts successfully with flag 513
- [ ] Server logs show "Enabling batch invariant mode with existing kernels..."
- [ ] Test script connects to server successfully
- [ ] Requests with temperature=0 work correctly
- [ ] Requests with temperature>0 work correctly
- [ ] Server logs show statistics after 100 forward passes
- [ ] Temperature=0 requests produce identical outputs (deterministic)
- [ ] Temperature>0 requests may produce varied outputs

## Troubleshooting

### Server won't start

- Check if port 30000 is already in use: `lsof -i :30000`
- Check if model is downloaded: The server will download it automatically on first run
- Check GPU availability: `nvidia-smi`

### Tests fail to connect

- Verify server is running: `curl http://localhost:30000/health`
- Check firewall settings
- Try with explicit IP: `python test_temperature_switching.py http://127.0.0.1:30000`

### No statistics in logs

- Statistics only appear after 100 forward passes
- Send more requests or lower `_stats_log_interval` in model_runner.py
- Check if flag 512 is set (e.g., use 513, 514, not 1, 2)

### Not seeing deterministic behavior

- Ensure temperature is exactly 0.0
- Check that flag includes bit 512 (513, 514, etc.)
- Verify batch-invariant mode initialized in server logs

## Next Steps

After successful testing:
1. Review the statistics in server logs
2. Test with your actual workload
3. Adjust `_stats_log_interval` if needed (default: 100)
4. Monitor performance impact
5. Compare with static mode (flag 1) for your use case
