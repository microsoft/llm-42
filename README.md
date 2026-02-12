# LLM-42: Enabling Determinism in LLM Inference with Verified Speculation

[![arXiv](https://img.shields.io/badge/arXiv-2601.17768-b31b1b.svg)](https://arxiv.org/abs/2601.17768)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**LLM-42** enables deterministic LLM inference via a **decode–verify–rollback** protocol, without rewriting GPU kernels. Built on [SGLang](https://github.com/sgl-project/sglang) v0.5.3.

> Raja Gond‡, Aditya K Kamath†, Ramachandran Ramjee‡, Ashish Panwar‡  
> ‡Microsoft Research &nbsp; †University of Washington

## How it works

Standard LLM serving is non-deterministic: dynamic batching changes GPU reduction orders, producing different outputs across runs. LLM-42 fixes this with a lightweight verify-rollback loop:

1. **Decode** — generate tokens using fast, unmodified kernels with dynamic batching.
2. **Verify** — replay a window of tokens under a fixed-shape schedule to check consistency.
3. **Rollback** — on mismatch, discard inconsistent tokens and resume from the last verified position.

Only requests marked `is_deterministic=True` incur verification; the rest run at full speed.

## Quick start

```bash
# Create and attach to a GPU-enabled Docker container (uses lmsysorg/sglang:v0.5.4)
./run_container.sh create && ./run_container.sh attach

# Inside the container: workspace is mounted at /workspace
cd /workspace
apt update; apt upgrade -y
git config --global --add safe.directory /workspace

# Build sgl-kernel (custom CUDA/Triton kernels) and install sglang in editable mode
./build_all.sh

# Authenticate with Hugging Face to download gated models (e.g., Llama)
huggingface-cli login --token <HF_TOKEN>

# Launch the server with LLM-42 decode-verify-rollback enabled
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-llm42 3 \
    --llm42-window-size 64 \
    --llm42-verify-batch-size 8 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --random-seed 42
```

## Configuration

| Flag | Default | Description |
|---|---|---|
| `--enable-llm42` | `0` | Enable LLM-42 DVR (`3` = recommended) |
| `--llm42-window-size` | `64` | Tokens decoded before verification |
| `--llm42-verify-batch-size` | `8` | Requests per verification batch (grouped verification) |

Additional flags for benchmarking: `--enable-deterministic-inference` (global batch-invariant baseline), `--llm42-skip-mismatch` (synthetic mismatch injection).

Per-request control via the OpenAI-compatible API:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"is_deterministic": True},
)
```

## Hardware

4× NVIDIA H100 PCIe (80 GB HBM3), 64-core CPU, ~1.65 TB DRAM.

## Citation

```bibtex
@article{gond2025llm42,
  title   = {{LLM-42}: Enabling Determinism in {LLM} Inference with Verified Speculation},
  author  = {Gond, Raja and Kamath, Aditya K and Ramjee, Ramachandran and Panwar, Ashish},
  journal = {arXiv preprint arXiv:2601.17768},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.17768}
}
```

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.
