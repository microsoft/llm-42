# LLM-42: Enabling Determinism in LLM Inference with Verified Speculation

> Raja Gond¹, Aditya K Kamath², Ramachandran Ramjee¹, and Ashish Panwar¹
>
> ¹Microsoft Research India &nbsp;&nbsp; ²University of Washington

This directory is the artifact for the paper. It reproduces the main
experimental figures using the LLM-42 build of SGLang from the parent
repository. Each numbered sub-folder corresponds to one figure (or figure
group) in the paper and contains a self-contained `run.sh`. A top-level
`run_all.sh` runs every benchmark in order.

## Contents

| Folder            | Paper figure | What it measures |
|-------------------|--------------|------------------|
| `1-figure4/`      | Figure 4     | Performance of batch-invariant matmul kernels vs baseline torch.mm |
| `2-figure5/`      | Figure 5     | Impact of determinism on decode throughput (microbenchmark) |
| `3-figure6/`      | Figure 6     | CDF of consistent-span lengths (6a: 8B, 6b: 70B) |
| `4-figure9a/`     | Figure 9a    | Forward-pass latency of a single iteration vs. number of tokens (8B & 70B) |
| `5-figure9b/`     | Figure 9b    | Rollback / recompute cost (8B & 70B) |
| `6-figure10/`     | Figure 10    | Offline inference throughput |
| `7-figure11-12/`  | Figures 11–12 | Online (QPS-driven) latency CDFs and TTFT |
| `8-figure13/`     | Figure 13    | Rollback-statistics heatmaps (recompute & throughput) across LLM-42 configs |

All generated PDFs are collected in `llm42-plots/`, together with the paper's
reference figures under `llm42-plots/reference/` and a `view-artifact.html`
viewer (see [Viewing the results](#viewing-the-results)). Each benchmark also
keeps its raw results and server logs under `<benchmark>/runs/`.

## Requirements and Installation

- **GPUs (NVIDIA H100).**
  - The **8B** model runs on a **single GPU**.
  - The **70B** model is evaluated on **8 GPUs** in the paper. The scripts detect
    the number of visible GPUs and **automatically fall back to 4 GPUs** (tensor
    parallelism TP-8 → TP-4) when only 4 are available. **Fewer than 4 GPUs is
    not supported for the 70B model** — it is skipped automatically, and the 8B
    results are still produced.

- **Model access.** The scripts pull `meta-llama/Llama-3.1-8B-Instruct` and
  `meta-llama/Llama-3.3-70B-Instruct` from Hugging Face. These are gated models,
  so make sure you have accepted their licenses before running the
  `huggingface-cli login` step above.

- **Environment.** The benchmarks run inside the LLM-42 SGLang container. From
  the repository root (the parent of this directory):

  ```bash
  # Create and attach to a GPU-enabled Docker container (uses lmsysorg/sglang:v0.5.4)
  ./run_container.sh create && ./run_container.sh attach

  # Inside the container: workspace is mounted at /workspace
  cd /workspace
  apt update; apt upgrade -y
  git config --global --add safe.directory /workspace

  # Build sgl-kernel and install sglang in editable mode
  ./build_all.sh

  # Authenticate with Hugging Face to download gated models (e.g., Llama)
  huggingface-cli login --token <HF_TOKEN>

  # Move into the artifact directory to run the benchmarks
  cd /workspace/sosp-artifact
  ```

## Running the full artifact

The easiest way to reproduce everything is the orchestrator:

```bash
./run_all.sh [--duration quick|full] [--models 8b|70b|8b,70b] [--force]
```

It runs benchmarks 1–13 in order, printing a per-benchmark timing summary to
`run_all_timings.log` and writing all figures to `llm42-plots/`.

### Configuration options

| Option       | Values                 | Default | Effect |
|--------------|------------------------|---------|--------|
| `--duration` | `quick`, `full`        | `quick` | Workload size for the long-running benchmarks. `quick` is a fast smoke test; `full` reproduces the paper. |
| `--models`   | `8b`, `70b`, `8b,70b`  | `8b`    | Which model(s) to run for the long-running benchmarks. |
| `--force`    | (flag)                 | off     | Re-run and overwrite existing results (see *Resume* below). |

`--duration` and `--models` are forwarded only to the three long-running
benchmarks that accept them (`6-figure10`, `7-figure11-12`, `8-figure13`). The
micro-benchmarks (`1-figure4`–`5-figure9b`) have no such flags and run with
their own defaults — most of them cover both the 8B and 70B models (auto-skipping
the 70B model when fewer than 4 GPUs are present), except `2-figure5`, which is
8B-only.

### Examples

```bash
# Default: quick smoke test, 8B only. Good for a first end-to-end check.
./run_all.sh

# Full paper reproduction, both models (needs 8 GPUs, or 4 with auto TP fallback).
./run_all.sh --duration full --models 8b,70b

# Full run, 8B only.
./run_all.sh --duration full --models 8b

# Force everything to re-run from scratch.
./run_all.sh --duration full --models 8b,70b --force
```

### Resume / skip

By default every benchmark **resumes**: any experiment whose results already
exist on disk is skipped, and only its plot is refreshed. This makes it safe to
re-run `run_all.sh` after an interruption. Pass `--force` to ignore existing
results and recompute everything.

> **Note:** do not edit a `run.sh` while `run_all.sh` is executing it — bash
> reads scripts as they run, so an in-place edit can corrupt the current run.

### Approximate durations

Times below assume **8 GPUs**; with fewer GPUs they scale up proportionally.

| Benchmark        | quick           | full        |
|------------------|-----------------|-------------|
| `1-figure4`      | ~1 min          | ~1 min      |
| `2-figure5`      | ~2 min          | ~2 min      |
| `3-figure6`      | ~30 min         | ~30 min     |
| `4-figure9a`     | ~2 min          | ~2 min      |
| `5-figure9b`     | ~20 min         | ~20 min     |
| `6-figure10`     | ~15 min         | > 24 hours  |
| `7-figure11-12`  | ~15 min         | > 12 hours  |
| `8-figure13`     | ~10 min         | ~6 hours    |

## Running individual benchmarks

Every benchmark can also be run on its own. Just enter its folder and invoke
`run.sh`:

```bash
cd 4-figure9a && ./run.sh                       # forward-pass latency (8B + 70B)

cd 6-figure10 && ./run.sh --models 8b --duration quick   # quick offline-throughput run
cd 8-figure13 && ./run.sh --models 8b,70b --duration full   # full rollback heatmaps

cd 3-figure6  && ./run.sh --force               # re-run even if results exist
```

- The long-running benchmarks (`6-figure10`, `7-figure11-12`, `8-figure13`)
  accept `--duration`, `--models`, and `--force`.
- The micro-benchmarks (`1-figure4`–`5-figure9b`) accept `--force`; `1-figure4`
  and `4-figure9a` additionally accept `--plot-only` to regenerate plots from
  existing data.
- Each script's header comment documents its full set of environment-variable
  overrides (e.g. `NUM_GPUS`, `NUM_PROMPTS`, `MODEL`, `TP_SIZE`).

## Viewing the results

Every benchmark writes its paper figure(s) into `llm42-plots/`; the paper's own
reference figures are committed alongside them under `llm42-plots/reference/`.

Once your runs are done, copy the whole `llm42-plots/` directory to your local
machine and open `llm42-plots/view-artifact.html` in a web browser. The viewer
lays out each figure with the paper's **reference** plot on the left and your
**reproduced** plot on the right, so you can compare them side by side. Figures
that were not generated (e.g. the 70B panels of an 8B-only run) show a
"not generated yet" placeholder.

```bash
# Example: copy the folder off the GPU host, then open the viewer locally
scp -r <user>@<gpu-host>:/path/to/sosp-artifact/llm42-plots ./llm42-plots
# open ./llm42-plots/view-artifact.html in your browser
```