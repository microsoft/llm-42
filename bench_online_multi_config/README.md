#### Multi-Config Performance


```bash
DATASET_NAME=random RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=1 ./run_compare_mismatches_multi_config.sh
```

```bash
python plot_prefill_batch_sizes.py --results-dirs results_* --qps 12
```