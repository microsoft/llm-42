# Verification Percentage Benchmark

This benchmark compares the **latency and output quality** across different determinism modes and mismatch percentages.

## Overview

- **Goal**: Compare latency overhead across determinism modes
- **Configurations**:
  - `default`: Non-deterministic baseline (standard SGLang)
  - `global`: Global deterministic mode (`--enable-deterministic-inference 2`)
  - `detinfer_512_0pct`: DetInfer with step_size=512, 0% mismatches (no rollback)
  - `detinfer_512_5pct`: DetInfer with step_size=512, 5% forced mismatches
- **Dataset configs**: Various input/output length combinations

## Usage

### 1. Start the servers

```bash
cd bench_online_verification_percentage
./launch_servers_parallel.sh
```

This starts 4 servers on GPUs 0-3:
- GPU 0 (port 30010): `default` - Non-deterministic baseline
- GPU 1 (port 30011): `global` - Global deterministic mode
- GPU 2 (port 30012): `detinfer_512_0pct` - DetInfer, 0% mismatches
- GPU 3 (port 30013): `detinfer_512_5pct` - DetInfer, 5% mismatches

### 2. Run the benchmark

In a separate terminal:

```bash
./run_all_verification_percentage.sh
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
| `CONFIG_NAMES` | default,global,detinfer_512_0pct,detinfer_512_5pct | Config names |

## Output

Results are saved in directories like:
- `results_in512_out1025_qps12_n4096/`
- `results_in1024_out1025_qps12_n4096/`
- etc.

Each directory contains:
- `summary.json`: Latency statistics per configuration
- `latency_data.json`: Raw latency data for plotting
- `*.jsonl`: Raw request/response data
- `mismatch_heatmap.pdf`: Comparison heatmap
- `ttft_bars.pdf`, `tpot_bars.pdf`, `e2e_bars.pdf`: Individual metric bar charts
- `latency_combined.pdf`: All three metrics side by side
- `latency_mean_comparison.pdf`: Mean latency comparison
- `ttft_cdf.pdf`, `tpot_cdf.pdf`, `e2e_cdf.pdf`: CDF distributions

## What This Measures

| Config | Deterministic | Verification | Rollback |
|--------|---------------|--------------|----------|
| `default` | ❌ No | ❌ No | ❌ No |
| `global` | ✅ Yes (always) | ❌ No | ❌ No |
| `detinfer_512_0pct` | ✅ Yes (on-demand) | ✅ Yes | ❌ No (0%) |
| `detinfer_512_5pct` | ✅ Yes (on-demand) | ✅ Yes | ✅ Yes (5%) |

This benchmark helps understand:
1. **Baseline overhead**: How much does global determinism cost vs non-deterministic?
2. **Verification overhead**: What's the cost of running verification passes?
3. **Rollback cost**: How much does 5% mismatch rate affect latency?

## Mismatch Percentage Explained

The `--det-skip-mismatch` parameter controls forced mismatch injection:
- **100%** (default): Normal behavior (natural mismatches)
- **0%**: Force no mismatches (skip all rollbacks)
- **5%**: Inject mismatch at position `window - ceil(5% × window)` to rollback exactly `ceil(5% × window)` tokens

For step_size=512 and 5% mismatch:
- `tokens_to_rollback = ceil(0.05 × 512) = 26`
- `mismatch_pos = 512 - 26 = 486`
