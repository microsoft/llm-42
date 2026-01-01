# Offline Multi-Config Throughput Benchmark

This directory contains scripts to benchmark offline throughput across different server configurations for DetInfer comparison.

## Overview

The benchmark measures offline throughput for multiple configurations:
- **Non-Deterministic**: Standard SGLang without determinism
- **Global Deterministic**: SGLang with `--enable-deterministic-inference 2`
- **DetInfer-ws32-bs16**: DetInfer with window size 32 and verify batch size 16
- **DetInfer-ws32-bs32**: DetInfer with window size 32 and verify batch size 32

## Scripts

| Script | Description |
|--------|-------------|
| `launch_servers_parallel.sh` | Launch multiple servers (one per GPU) with different configs |
| `run_offline_benchmark.sh` | Run benchmarks against pre-launched servers |
| `run_all_dataset_configs.sh` | Run benchmarks for all 4 dataset configs |
| `plot_throughput.py` | Generate throughput plots for single result file |
| `plot_throughput_comparison.py` | Generate comparison plot across multiple datasets |

## Quick Start (Recommended)

```bash
# Step 1: Launch servers (one per GPU, different configs)
./launch_servers_parallel.sh

# Step 2: In another terminal, run all dataset configs
./run_all_dataset_configs.sh

# Or run single dataset with ShareGPT (default)
./run_offline_benchmark.sh

# Or with random dataset
DATASET_NAME=random RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=128 ./run_offline_benchmark.sh
```

## Alternative: Standalone Mode

```bash
# Run benchmarks that launch servers internally (slower, sequential)
./run_offline_throughput.sh

# Or customize parameters
NUM_PROMPTS=500 INPUT_LEN=512 OUTPUT_LEN=256 ./run_offline_throughput.sh
```

## Environment Variables

### Server Launch (`launch_servers_parallel.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | `4` | Number of GPUs available |
| `SGLANG_TEST_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model path |
| `SGLANG_BASE_PORT` | `30005` | Starting port number |
| `SGLANG_TP_SIZE` | `1` | Tensor parallelism size |
| `SGLANG_ATTENTION_BACKEND` | `flashinfer` | Attention backend |
| `CONFIG_NAMES` | `sglang_non_deterministic,...` | Comma-separated config names |

### Benchmark (`run_offline_benchmark.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_URLS` | `http://127.0.0.1:30005,...` | Comma-separated server URLs |
| `CONFIG_NAMES` | `sglang_non_deterministic,...` | Comma-separated config names |
| `NUM_PROMPTS` | `16384` | Number of prompts to benchmark |
| `DATASET_NAME` | `sharegpt` | Dataset: `sharegpt` or `random` |
| `DATASET_PATH` | (auto) | Path to ShareGPT JSON file |
| `SHAREGPT_CONTEXT_LEN` | `16384` | Max context length for ShareGPT |
| `RANDOM_INPUT_LEN` | `1024` | Input length (when DATASET_NAME=random) |
| `RANDOM_OUTPUT_LEN` | `128` | Output length (when DATASET_NAME=random) |
| `DETERMINISTIC_RATIOS` | `0.0 0.1 0.2 0.5 1.0` | Space-separated ratios to test |

### Standalone Benchmark (`run_offline_throughput.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_TEST_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model path |
| `SGLANG_TP_SIZE` | `1` | Tensor parallelism size |
| `SGLANG_ATTENTION_BACKEND` | `flashinfer` | Attention backend |
| `NUM_PROMPTS` | `1000` | Number of prompts to benchmark |
| `INPUT_LEN` | `1024` | Input sequence length |
| `OUTPUT_LEN` | `128` | Output sequence length |
| `DETERMINISTIC_RATIOS` | `0.0 0.1 0.2 0.5 1.0` | Space-separated ratios to test |

## Output

Results are saved to `results_<timestamp>/`:
- `benchmark_results.jsonl` - Raw benchmark results
- `throughput_bars.pdf` - Grouped bar chart comparing configurations
- `throughput_line.pdf` - Line plot of throughput vs deterministic ratio
- `throughput_results.csv` - CSV summary of results

## Plotting Only

To regenerate plots from existing results:

```bash
python plot_throughput.py --results-file results_*/benchmark_results.jsonl --output-dir plots/
```

### Plot Options

```bash
python plot_throughput.py \
    --results-file results.jsonl \
    --output-dir output/ \
    --metric output_throughput  # or total_throughput
```

## Example Output

The throughput bar chart shows grouped bars for each configuration at different deterministic ratios, making it easy to compare:
- How throughput changes with deterministic request ratio
- Overhead of deterministic inference vs non-deterministic baseline
- Comparison between global deterministic and DetInfer approaches

## Notes

- Benchmarks run sequentially (not parallel) to avoid GPU contention
- Each configuration is tested across all specified deterministic ratios
- Results include both output throughput and total throughput metrics
