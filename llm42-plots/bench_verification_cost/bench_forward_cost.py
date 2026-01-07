"""
Benchmark the forward pass (prefill) cost for Llama 3.1-8B Instruct.

Measures latency of a single forward pass for different token counts:
16, 32, 64, 128, 256, 512

Usage:
    python bench_forward_cost.py --model-path meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import csv
import time
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import suppress_other_loggers
from sglang.srt.utils.hf_transformers_utils import get_tokenizer


def load_model(server_args, port_args, tp_rank):
    """Load the model and tokenizer."""
    suppress_other_loggers()
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_synthetic_inputs(batch_size: int, input_len: int) -> List[Req]:
    """Prepare synthetic input requests for benchmarking."""
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(temperature=0, max_new_tokens=1)

    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs


@torch.no_grad()
def run_forward_pass(reqs: List[Req], model_runner: ModelRunner):
    """Run a single prefill/extend forward pass."""
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


def benchmark_input_len(
    model_runner: ModelRunner,
    input_len: int,
    batch_size: int,
    warmup_iters: int,
    bench_iters: int,
    device: str,
) -> dict:
    """Benchmark forward pass for a specific input length."""
    print(f"  Benchmarking input_len={input_len}...")

    # Warmup
    for _ in range(warmup_iters):
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        reqs = prepare_synthetic_inputs(batch_size, input_len)
        run_forward_pass(reqs, model_runner)

    # Benchmark
    latencies = []
    for _ in range(bench_iters):
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        reqs = prepare_synthetic_inputs(batch_size, input_len)

        torch.cuda.synchronize()
        tic = time.perf_counter()
        run_forward_pass(reqs, model_runner)
        torch.cuda.synchronize()
        latency = time.perf_counter() - tic
        latencies.append(latency)

    latencies = np.array(latencies)
    avg_latency_ms = np.mean(latencies) * 1000
    std_latency_ms = np.std(latencies) * 1000
    latency_per_token_ms = avg_latency_ms / input_len
    std_latency_per_token_ms = std_latency_ms / input_len

    result = {
        "input_len": input_len,
        "avg_latency_ms": avg_latency_ms,
        "std_latency_ms": std_latency_ms,
        "latency_per_token_ms": latency_per_token_ms,
        "std_latency_per_token_ms": std_latency_per_token_ms,
    }

    print(f"    Latency: {avg_latency_ms:.3f} ± {std_latency_ms:.3f} ms")
    print(f"    Per-token: {latency_per_token_ms:.4f} ± {std_latency_per_token_ms:.4f} ms/token")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark forward pass cost")
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="Input lengths to benchmark",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=50,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="forward_cost_results.csv",
        help="Output CSV file for results",
    )

    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    # Set up environment and load model
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    print("Loading model...")
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank=0)
    print("Model loaded.\n")

    # Run benchmarks
    batch_size = 1
    results = []

    print(f"Running benchmarks (batch_size={batch_size}):")
    for input_len in args.input_lens:
        result = benchmark_input_len(
            model_runner=model_runner,
            input_len=input_len,
            batch_size=batch_size,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            device=server_args.device,
        )
        results.append(result)

    # Save results to CSV
    print(f"\nSaving results to {args.result_file}...")
    with open(args.result_file, "w", newline="") as f:
        fieldnames = [
            "input_len",
            "avg_latency_ms",
            "std_latency_ms",
            "latency_per_token_ms",
            "std_latency_per_token_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Done!")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Input Len':<12} {'Latency (ms)':<20} {'Per-token (ms/tok)':<20}")
    print("=" * 70)
    for r in results:
        print(
            f"{r['input_len']:<12} "
            f"{r['avg_latency_ms']:.3f} ± {r['std_latency_ms']:.3f}".ljust(20) + " "
            f"{r['latency_per_token_ms']:.4f} ± {r['std_latency_per_token_ms']:.4f}"
        )
    print("=" * 70)

    # Cleanup
    if server_args.tp_size > 1:
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
