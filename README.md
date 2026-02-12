# LLM-42: Enabling Determinism in LLM Inference with Verified Speculation

[![arXiv](https://img.shields.io/badge/arXiv-2601.17768-b31b1b.svg)](https://arxiv.org/abs/2601.17768)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**LLM-42** is a scheduling-based approach to enable determinism in LLM inference, inspired by speculative decoding. It is implemented on top of the [SGLang](https://github.com/sgl-project/sglang) (v0.5.3/v0.5.4) serving framework.

> **Authors:** Raja Gond‡, Aditya K Kamath†, Ramachandran Ramjee‡, Ashish Panwar‡  
> ‡Microsoft Research &nbsp;&nbsp; †University of Washington

---

## Overview

In LLM inference, the same prompt may yield different outputs across runs. This non-determinism arises from floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size. The standard approach—batch-invariant computation—eliminates non-determinism but at a steep cost: up to **56% throughput degradation** and the need to rewrite every core kernel.

**LLM-42** takes a different approach via a **decode–verify–rollback (DVR)** protocol:

1. **Decode** — Tokens are generated using the standard fast path with dynamic batching (no overhead).
2. **Verify** — A lightweight verifier periodically replays a fixed-size window of recently generated tokens under a fixed-shape reduction schedule, guaranteeing deterministic output.
3. **Rollback** — On mismatch, the sequence is rolled back to the last verified-consistent token and decoding resumes.

### Key Properties

- **Selective determinism** — Only requests that need determinism incur verification overhead (`is_deterministic=True|False` per request).
- **Grouped verification** — Verifies small windows from multiple requests together to amortize verification cost while keeping rollback cost low.
- **No new kernels required** — Reuses existing optimized GEMM, RMSNorm, and FusedMoE kernels unchanged; determinism is enforced at the scheduling level.
- **Guaranteed forward progress** — Each verification pass produces at least one new consistent output token.

### Performance Highlights

| Metric | SGLang-Det | LLM-42 (100% det) | LLM-42 (10% det) |
|---|---|---|---|
| Offline throughput vs. non-det baseline | Up to −56% | Within −6% (worst) | Within −1% |
| Online P50 latency @ 12 QPS (ShareGPT) | 4.64s | Comparable | +3% vs non-det |
| Recomputation overhead (ShareGPT, 100% det) | N/A | 0.32% | ≈0% |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   LLM-42 Server                      │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌────────────────┐  │
│  │  Prefill  │───▶│  Decode   │───▶│   Verify &     │  │
│  │(determin.)│    │(fast path)│    │   Rollback     │  │
│  └──────────┘    └──────────┘    └────────────────┘  │
│       │               │                │             │
│       ▼               ▼                ▼             │
│  ┌──────────────────────────────────────────────┐    │
│  │              KV Cache (shared)                │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  Batch-invariant ops (used in verification only):    │
│  • Triton persistent matmul                          │
│  • Triton / CUDA RMSNorm                             │
│  • FlashAttention-3 (num_splits=1)                   │
└──────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA 12.x (tested on H100 PCIe)
- Docker (recommended) or a local environment with PyTorch 2.x

### Docker Setup

```bash
# Pull the image and create a GPU-enabled container
./run_container.sh create

# Attach to the running container
./run_container.sh attach

# Other commands: stop, restart, status
./run_container.sh status
```

### Build

```bash
# Build everything (sgl-kernel + sglang)
./build_all.sh

# Build only the custom CUDA/Triton kernels
./build_all.sh --kernel-only

# Install only sglang (editable mode)
./build_all.sh --sglang-only

# Clean rebuild
./build_all.sh --clean
```

---

## Usage

### Launching a Server

```bash
# Non-deterministic baseline (standard SGLang)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct

# Global deterministic mode (batch-invariant kernels everywhere)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-deterministic-inference 2

# LLM-42: Decode-Verify-Rollback mode (recommended)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-llm-42 3 \
    --llm-42-window-size 64 \
    --llm-42-verify-batch-size 8
```

### Server Configuration

| Argument | Default | Description |
|---|---|---|
| `--enable-deterministic-inference` | `0` | Global batch-invariant mode: `1`=det matmul+rmsnorm, `2`=batch-invariant+native rmsnorm |
| `--enable-llm-42` | `0` | DVR verification mode: `0`=off, `1`–`3`= different kernel configurations |
| `--llm-42-window-size` | `32` | Number of tokens decoded before verification |
| `--llm-42-verify-batch-size` | `16` | Number of requests verified together (grouped verification) |
| `--llm-42-skip-mismatch` | `100.0` | Mismatch skip rate (for testing/debugging) |

### Per-Request Determinism

LLM-42 supports **selective determinism** via a per-request API flag:

```python
import openai

client = openai.Client(base_url="http://localhost:30000/v1", api_key="EMPTY")

# Only this request is verified for determinism
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"is_deterministic": True},
)
```

### Offline Benchmarks (Makefile)

```bash
# Non-deterministic baseline
make run_offline

# Deterministic mode (batch-invariant kernels)
make run_offline_det MODE=1

# Run all deterministic mode combinations
make run_test_all_modes

# Microbenchmarks
make figure_matmul       # GEMM kernel comparison
make figure_rmsnorm      # RMSNorm kernel comparison
make figure_prefill      # FlashInfer attention benchmark
```

#### Deterministic Mode Bitmask

Modes can be combined via bitwise OR to control which operators use deterministic vs. non-deterministic kernels:

| Bit | Value | Meaning |
|---|---|---|
| 0 | `1` | Deterministic defaults (det matmul, det rmsnorm, det attention) |
| 1 | `2` | Use kernel matmul |
| 2 | `4` | Use split-stream matmul |
| 5 | `32` | Use non-det matmul |
| 6 | `64` | Use non-det rmsnorm |
| 7 | `128` | Use non-det attention |

**Examples:** `96` = non-det matmul + rmsnorm; `224` = all non-deterministic; `3` = det with kernel matmul.

---

## Repository Structure

```
.
├── build_all.sh                      # Build script (sgl-kernel + sglang)
├── run_container.sh                  # Docker container management
├── Makefile                          # Benchmark and experiment targets
│
├── python/sglang/                    # Modified SGLang source
│   └── srt/
│       └── batch_invariant_ops/      # Batch-invariant kernels
│           ├── persistent_matmul.py  # Triton persistent-kernel matmul
│           ├── rms_norm.py           # Triton batch-invariant RMSNorm
│           └── ...                   # CUDA fused matmul kernels
│
├── sgl-kernel/                       # Custom CUDA/Triton kernel package
│
├── test_batch_invariance/            # Correctness tests
│   ├── test_llm42/                # DVR verification tests
│   │   ├── sglang_test.py            # Batch-size invariance tests
│   │   ├── verify_batch_invariance.py# BS=1 vs BS=N comparison
│   │   └── test_det_verification.py  # Verify-and-rollback system tests
│   ├── test_llm42_sharegpt/       # ShareGPT dataset tests
│   ├── test_moe_llm42/            # MoE model tests
│   ├── test_fused_add_rmsnorm.py     # RMSNorm batch invariance test
│   └── test_amd_and_nccl_all_reduce.py  # NCCL AllReduce tests
│
├── llm42-plots/                      # Paper figure generation
│   ├── bench_cover_figure/           # Throughput comparison plots
│   ├── bench_offline_eval/           # Offline throughput evaluation
│   ├── bench_online_qps/            # Online latency at various QPS
│   ├── bench_online_ablation/        # Window size × batch size sweep
│   ├── bench_verification_cost/      # Verification overhead analysis
│   ├── microbenchmarks/              # Kernel-level benchmarks
│   │   ├── matmul/                   # GEMM benchmarks
│   │   ├── rms_norm/                 # RMSNorm benchmarks
│   │   └── attention/                # Attention benchmarks
│   ├── mismatch-indices/             # Mismatch position analysis
│   └── multi_gpu_offline/            # Multi-GPU experiments
│
├── llm42_benchmarks/              # End-to-end benchmarks
│   ├── bench_online_multi_qps/       # Online benchmarks across QPS levels
│   └── bench_online_multi_step_size/ # Varying verification step sizes
│
├── benchmark/                        # Additional benchmark suites
├── examples/                         # Usage examples
├── docs/                             # Documentation
└── output/                           # Benchmark output data
```

---

## Running Tests

### Batch Invariance Correctness

```bash
# Test that outputs are identical across different batch sizes
cd test_batch_invariance/test_llm42
bash launch_batch_invariance_test.sh

# Run the BS=1 vs BS=N verification
python verify_batch_invariance.py

# Test the DVR verify-and-rollback system
python test_det_verification.py
```

### MoE Model Tests

```bash
cd test_batch_invariance/test_moe_llm42
# Tests batch invariance for Mixture-of-Experts models
```

---

## Reproducing Paper Results

The experiments from the paper can be reproduced using the Makefile targets and scripts in `llm42-plots/`:

| Paper Figure | Command |
|---|---|
| Fig. 1 (Intro throughput) | `make figure_1_sgl && make plot_figure_1` |
| Fig. 4a (GEMM comparison) | `make figure_matmul && make plot_matmul` |
| Fig. 4b (RMSNorm comparison) | `make figure_rmsnorm && make plot_rmsnorm` |
| Fig. 5 (Decode throughput) | See `llm42-plots/bench_cover_figure/` |
| Fig. 10 (Offline throughput) | See `llm42-plots/bench_offline_eval/` |
| Fig. 11 (Online latency CDFs) | See `llm42-plots/bench_online_qps/` |
| Fig. 12 (Ablation study) | See `llm42-plots/bench_online_ablation/` |

**Hardware used in the paper:** 4× NVIDIA H100 PCIe (80 GB HBM3), 64-core CPU, ~1.65 TB DRAM.  
**Primary model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Additional models tested for correctness:** Qwen-4B-Instruct-2507, Qwen3-14B, Qwen3-30B-A3B-Instruct-2507 (1–4 GPUs with tensor parallelism).

---

## Citation

```bibtex
@article{gond2025llm42,
  title     = {{LLM-42}: Enabling Determinism in {LLM} Inference with Verified Speculation},
  author    = {Gond, Raja and Kamath, Aditya K and Ramjee, Ramachandran and Panwar, Ashish},
  journal   = {arXiv preprint arXiv:2601.17768},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.17768}
}
```

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.
