# Multi-Percentage Deterministic Mode Testing

## Quick Start

```bash
# Test with multiple percentages
./run_tests.sh --trace-type arxiv --temp0-pcts 0,1,5,10

# Plot results
./plot_results.sh --input-dir llama-3.1-8b_arxiv_summarization_pct_0_1_5_10
```

## How It Works

**Test Strategy:**
- **Baseline, Mode 66, Mode 257**: Run once at 100% (all deterministic)
- **Mode 578**: Runs at each specified percentage (0%, 1%, 5%, 10%, etc.)

**Total tests** = 3 + number_of_percentages

## Trace Types

| Type | Dataset |
|------|---------|
| `arxiv` | ArXiv summarization (default) |
| `sharegpt` | ShareGPT 8K conversations |
| `lmsys` | LMSYS Chat 1M conversations |

## Examples

```bash
# ArXiv with 0%, 1%, 5%, 10% (7 tests total)
./run_tests.sh --trace-type arxiv --temp0-pcts 0,1,5,10

# ShareGPT with custom percentages
./run_tests.sh --trace-type sharegpt --temp0-pcts 0,5,10,100

# Custom trace file
./run_tests.sh \
  --trace-file /path/to/trace.csv \
  --trace-name my_dataset \
  --temp0-pcts 0,1,5,10
```

## Output Structure

```
llama-3.1-8b_arxiv_pct_0_1_5_10/
├── pct_100/              # Baseline, Mode 66, Mode 257 (all @ 100%)
│   ├── baseline_nondet/
│   ├── det_mode_66/
│   └── det_mode_257/
├── pct_0/                # Mode 578 @ 0%
│   └── det_mode_578/
├── pct_1/                # Mode 578 @ 1%
│   └── det_mode_578/
└── ...
```

## Plotting

Plots are generated in two formats:

1. **Per-percentage plots**: `pct_X/` directories
2. **Unified comparison**: `unified_comparison/` directory
   - Shows all modes together
   - Baseline/66/257 @ 100% vs Mode 578 @ all percentages

## Options

```bash
--trace-type TYPE         # arxiv | sharegpt | lmsys
--trace-file FILE         # Custom trace file path
--temp0-pcts PCTS         # Comma-separated percentages (e.g., 0,1,5,10)
--model MODEL             # Model path
--max-requests N          # Number of requests
--qps QPS                 # Queries per second
--assignment-mode MODE    # random | fixed
--seed SEED               # Random seed
```

## Test Modes

| Mode | Description | Percentage |
|------|-------------|------------|
| **baseline** | Non-deterministic baseline | Always 100% |
| **mode 66** | Batch-invariant (vllm-rmsnorm + cutlass) | Always 100% |
| **mode 257** | Batch-invariant (native-rmsnorm + TM) | Always 100% |
| **mode 578** | Temperature-based switching | Varies (0,1,5,10...) |

## Quick Test Scripts

```bash
# Test single trace
./quick_test_multi_pct.sh arxiv    # or sharegpt, lmsys

# Test all three traces
./test_all_traces.sh
```

## Help

```bash
./run_tests.sh --help
./plot_results.sh --help
```
