# Online Benchmarks

Measure TTFT, TPOT, and E2E latency for online serving workloads.

## Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| **random** | Synthetic prompts | Built-in |
| **sharegpt** | Conversational | Auto-download |
| **arxiv** | Summarization (long input) | `ccdv/arxiv-summarization` HuggingFace |

## Quick Start

```bash
# Start server first
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000

# Run all benchmarks
./run_online_benchmarks.sh

# Or run arxiv-specific benchmark
python run_arxiv_benchmark.py --base-url http://localhost:30000 --num-prompts 500
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_BASE_URL` | `http://localhost:30000` | Server URL |
| `SGLANG_TEST_MODEL` | `Meta-Llama-3.1-8B-Instruct` | Model path |
| `NUM_PROMPTS` | `500` | Prompts per config |

## Output

Results saved to `results/`:
- `results_TIMESTAMP.jsonl` - Raw metrics
- `plots_TIMESTAMP/` - Visualizations

## Metrics

- **TTFT**: Time to first token (prefill latency)
- **TPOT**: Time per output token (decode latency)  
- **E2E**: End-to-end request latency
