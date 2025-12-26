# Offline Benchmarks

This directory contains scripts for benchmarking offline throughput with different deterministic inference configurations.

## Configurations Tested

1. **Default (Baseline)**: Standard SGLang configuration with TP=1
2. **Deterministic Inference Mode 2**: Global deterministic inference with `--enable-deterministic-inference 2`
3. **Det-Infer Mode 3**: Forward-mode-based deterministic inference with `--enable-det-infer 3` and varying `--det-infer-window-size` (16, 64, 128)

## Parameters Varied

- **Input Lengths**: 512, 1024, 2048 tokens
- **Output Lengths**: 128, 256, 512, 1024 tokens
- All requests use `is_deterministic=True`

## Usage

### Run Benchmarks

```bash
# Make script executable
chmod +x run_offline_benchmarks.sh

# Run with default model
./run_offline_benchmarks.sh

# Run with custom model
SGLANG_TEST_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct ./run_offline_benchmarks.sh

# Customize number of prompts
NUM_PROMPTS=500 ./run_offline_benchmarks.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_TEST_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Model path |
| `SGLANG_TP_SIZE` | `1` | Tensor parallelism |
| `SGLANG_ATTENTION_BACKEND` | `flashinfer` | Attention backend |
| `NUM_PROMPTS` | `100` | Number of prompts per configuration |

### Plot Results Separately

```bash
python plot_results.py results/benchmark_results_TIMESTAMP.jsonl --output-dir plots/
```

## Output

Results are saved to `results/` directory:
- `benchmark_results_TIMESTAMP.jsonl`: Raw results in JSONL format
- `plots_TIMESTAMP/`: Generated plots
  - `throughput_by_input_len_out*.png`: Throughput vs input length
  - `throughput_by_output_len_in*.png`: Throughput vs output length
  - `heatmap_*.png`: Throughput heatmaps per configuration
  - `overhead_comparison.png`: Overhead compared to baseline
  - `step_size_comparison.png`: Effect of det-infer-window-size
  - `summary.txt`: Text summary of all results
