# Rollback Statistics - Quick Guide

## Usage

```bash
# 1. Add metrics (one-time)
python add_rollback_metrics.py

# 2. Start server with metrics
python -m sglang.launch_server --enable-metrics --enable-deterministic-inference 1 ...

# 3. Collect stats while benchmarking
python collect_rollback_stats.py --duration 300 &

# 4. Run benchmark
python your_benchmark.py

# 5. Plot results
python plot_rollback_stats.py
```

## Files

- `add_rollback_metrics.py` - Adds tracking code
- `collect_rollback_stats.py` - Collects metrics from running server
- `plot_rollback_stats.py` - Generates plots

## Parameters to Vary

- `--min-det-step-size` (1, 5, 10, 20, 50)
- Batch size
- Request rate
- Temperature

## Example Comparison

```bash
# Collect for different configs
python collect_rollback_stats.py --output step_1.json &
# ... run benchmark with step_size=1 ...

python collect_rollback_stats.py --output step_10.json &
# ... run benchmark with step_size=10 ...

# Compare
python plot_rollback_stats.py --compare step_1.json step_10.json --labels "step=1" "step=10"
```
