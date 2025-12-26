# Verification Overhead Benchmark

This benchmark measures the **pure verification overhead** of deterministic inference with different step sizes. By enabling `--det-infer-skip-mismatch`, all servers skip the mismatch detection and rollback, so we measure only the cost of running the verification forward pass.

## Overview

- **Goal**: Measure how much overhead verification adds at different step sizes
- **Key setting**: `--det-infer-skip-mismatch` (no rollback/recomputation)
- **Step sizes**: 32, 128, 512, 1024
- **Dataset configs**: Various input/output length combinations

## Usage

### 1. Start the servers

```bash
cd bench_online_verification_overhead
./launch_servers_parallel.sh
```

This starts 4 servers on GPUs 0-3, each with a different step size:
- GPU 0 (port 30010): step_size=32
- GPU 1 (port 30011): step_size=128
- GPU 2 (port 30012): step_size=512
- GPU 3 (port 30013): step_size=1024

All servers have `--det-infer-skip-mismatch` enabled.

### 2. Run the benchmark

In a separate terminal:

```bash
./run_all_verification_overhead.sh
```

This runs all 4 dataset configurations:
- 512 input, 1024 output
- 1024 input, 1024 output
- 512 input, 2048 output
- 1024 input, 2048 output

Or run a single configuration:

```bash
RANDOM_INPUT_LEN=512 RANDOM_OUTPUT_LEN=1024 ./run_benchmark.sh
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QPS` | 12 | Queries per second |
| `NUM_PROMPTS` | 4096 | Number of prompts to run |
| `BASE_URLS` | localhost:30010-30013 | Server URLs |
| `CONFIG_NAMES` | step sizes 32,128,512,1024 | Config names |

## Output

Results are saved in directories like:
- `results_in512_out1024_qps12_n4096/`
- `results_in1024_out1024_qps12_n4096/`
- etc.

Each directory contains:
- `summary.json`: Latency statistics per step size
- `latency_data.json`: Raw latency data for plotting
- `*.jsonl`: Raw request/response data
- `mismatch_heatmap.pdf`: Comparison heatmap
- `ttft_bars.pdf`, `tpot_bars.pdf`, `e2e_bars.pdf`: Individual metric bar charts
- `latency_combined.pdf`: All three metrics side by side
- `latency_mean_comparison.pdf`: Mean latency comparison
- `ttft_cdf.pdf`, `tpot_cdf.pdf`, `e2e_cdf.pdf`: CDF distributions

## What This Measures

With `--det-infer-skip-mismatch`:
- ✅ Verification forward pass runs
- ✅ KV cache allocation for padding
- ✅ Batch preparation and token collection
- ❌ No mismatch detection
- ❌ No rollback (recomputation = 0)

This isolates the **verification infrastructure cost** from the variable cost of rollback.
