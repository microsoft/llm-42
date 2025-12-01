# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
Rollback statistics collection with min_det_step_size sweep.

This script runs workloads and collects rollback metrics from the server
to analyze how min_det_step_size affects rollback counts and tokens.

Environment variables:
  - SGLANG_TEST_MODEL: served model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)
  - SGLANG_TP_SIZE: tensor parallelism size (e.g., 4)
  - SGLANG_HOST: server host (default: 127.0.0.1)
  - SGLANG_PORT: server port (default: 30000)
  - SGLANG_ATTENTION_BACKEND: backend name (default: flashinfer)
  - SGLANG_TEST_SEED: random seed (default: 12345)

Usage:
    # Start server with metrics enabled:
    python -m sglang.launch_server --model-path <model> --enable-metrics --enable-det-infer 1 --min-det-step-size <N>
    
    # Run this script:
    python vllm_online_batch_invariance_multitest.py
"""

import os
import random
import re
import sys
import time
import json
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

import requests
from utils import _random_prompt, resolve_model_name

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")


@dataclass
class RollbackStats:
    """Container for rollback statistics."""
    num_rollbacks: int = 0
    tokens_rolled_back: int = 0
    num_requests: int = 0
    
    @property
    def rollback_rate(self) -> float:
        return self.num_rollbacks / self.num_requests if self.num_requests > 0 else 0.0
    
    @property
    def avg_tokens_per_rollback(self) -> float:
        return self.tokens_rolled_back / self.num_rollbacks if self.num_rollbacks > 0 else 0.0


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus metrics text."""
    metrics = {}
    for name in ["sglang:num_rollbacks_total", "sglang:tokens_rolled_back_total", "sglang:num_requests_total"]:
        if m := re.search(rf'{re.escape(name)}\{{[^}}]*\}}\s+([\d.]+)', text):
            metrics[name] = float(m.group(1))
    return metrics


def get_rollback_stats(base_url: str) -> RollbackStats | None:
    """Fetch current rollback stats from server metrics endpoint."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        if r.status_code == 200:
            m = parse_metrics(r.text)
            return RollbackStats(
                num_rollbacks=int(m.get("sglang:num_rollbacks_total", 0)),
                tokens_rolled_back=int(m.get("sglang:tokens_rolled_back_total", 0)),
                num_requests=int(m.get("sglang:num_requests_total", 0)),
            )
    except Exception as e:
        print(f"Warning: Could not fetch metrics: {e}")
    return None


def _request_completion(
    base_url: str,
    model: str,
    prompt: Any,
    sp: dict[str, Any],
    max_retries: int = 3,
    retry_backoff: float = 0.5,
    verbose: bool = False,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(sp)
    # Enable deterministic inference
    payload["is_deterministic"] = True

    for attempt in range(max_retries + 1):
        try:
            if verbose and attempt > 0:
                print(f"      Retry attempt {attempt}/{max_retries}...")
            response = requests.post(
                f"{base_url}/v1/completions",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:  # pragma: no cover
            if attempt < max_retries:
                if verbose:
                    print(f"      Request failed: {e}, retrying...")
                time.sleep(retry_backoff * (2**attempt))
                continue
            sys.stderr.write(f"Error: {e}\n")
            return None
    return None


def run_workload(
    base_url: str,
    model_name: str,
    prompts: list[str],
    batch_size: int,
    max_tokens: int,
    temperature: float = 0.0,
    verbose: bool = False,
    num_batches: int = 1,
) -> int:
    """
    Run a workload and return the number of requests completed.
    
    Sends batched requests to generate tokens, which will trigger rollbacks
    when deterministic inference is enabled.
    
    Args:
        prompts: Pool of prompts to sample from
        batch_size: Number of prompts per batch (all sent together in one request)
        num_batches: Number of batched requests to send
    """
    sp_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 42,
    }
    
    completed = 0
    
    for i in range(num_batches):
        # Select batch_size prompts for this batch
        start_idx = (i * batch_size) % len(prompts)
        batch_prompts = []
        for j in range(batch_size):
            batch_prompts.append(prompts[(start_idx + j) % len(prompts)])
        
        if verbose:
            print(f"    Batch {i+1}/{num_batches} ({len(batch_prompts)} prompts in single request)... ", end="", flush=True)
        
        # Send all prompts in ONE batched request
        resp = _request_completion(base_url, model_name, batch_prompts, sp_kwargs, verbose=verbose)
        
        if resp and resp.get("choices"):
            completed += len(resp["choices"])
            if verbose:
                print("✓")
        else:
            if verbose:
                print("✗")
    
    return completed


def collect_rollback_data(
    base_url: str,
    model_name: str,
    prompts: list[str],
    batch_size: int,
    max_tokens: int,
    temperature: float = 0.0,
    verbose: bool = False,
    num_batches: int = 1,
) -> dict[str, Any]:
    """
    Run workload and collect rollback statistics.
    
    Args:
        prompts: Pool of prompts to sample from
        batch_size: Number of prompts per batched request
        num_batches: Number of batched requests to send
    
    Returns dict with:
        - num_rollbacks: number of rollback events
        - tokens_rolled_back: total tokens rolled back
        - num_requests: requests processed
        - rollback_rate: rollbacks per request
    """
    # Get initial stats
    stats_before = get_rollback_stats(base_url)
    if stats_before is None:
        print("Warning: Could not get initial metrics. Make sure server has --enable-metrics")
        stats_before = RollbackStats()
    
    # Run workload
    completed = run_workload(
        base_url=base_url,
        model_name=model_name,
        prompts=prompts,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
        num_batches=num_batches,
    )
    
    # Get final stats
    stats_after = get_rollback_stats(base_url)
    if stats_after is None:
        stats_after = RollbackStats()
    
    # Calculate delta
    delta_rollbacks = stats_after.num_rollbacks - stats_before.num_rollbacks
    delta_tokens = stats_after.tokens_rolled_back - stats_before.tokens_rolled_back
    delta_requests = stats_after.num_requests - stats_before.num_requests
    
    return {
        "num_rollbacks": delta_rollbacks,
        "tokens_rolled_back": delta_tokens,
        "num_requests": delta_requests,
        "completed": completed,
        "rollback_rate": delta_rollbacks / delta_requests if delta_requests > 0 else 0.0,
        "avg_tokens_per_rollback": delta_tokens / delta_rollbacks if delta_rollbacks > 0 else 0.0,
    }


def plot_rollback_results(
    results: dict[str, list[dict[str, Any]]],
    x_values: list[int],
    x_label: str,
    title: str,
    output_path: str,
    legend_key: str = "label",
):
    """
    Create rollback plots.
    
    Args:
        results: Dict mapping label -> list of result dicts (one per x_value)
        x_values: Values for x-axis (e.g., min_det_step_size values)
        x_label: Label for x-axis
        title: Plot title
        output_path: Path to save plot
        legend_key: Key name for legend entries
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot (matplotlib not installed): {output_path}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of rollbacks
    ax1 = axes[0]
    for label, data_list in results.items():
        rollbacks = [d["num_rollbacks"] for d in data_list]
        ax1.plot(x_values, rollbacks, marker='o', label=label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Number of Rollbacks")
    ax1.set_title(f"{title}\nRollback Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tokens rolled back
    ax2 = axes[1]
    for label, data_list in results.items():
        tokens = [d["tokens_rolled_back"] for d in data_list]
        ax2.plot(x_values, tokens, marker='s', label=label)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Tokens Rolled Back")
    ax2.set_title(f"{title}\nTokens Rolled Back")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def run_sweep_vary_batch_size(
    base_url: str,
    model_name: str,
    min_det_step_sizes: list[int],
    batch_sizes: list[int],
    fixed_max_tokens: int,
    n_prompts: int,
    temperature: float = 0.8,
    output_dir: str = "rollback_results",
) -> dict[str, Any]:
    """
    Sweep over min_det_step_size, varying batch_size with fixed max_tokens.
    
    NOTE: This assumes you restart the server with different --min-det-step-size
    values between runs. This function collects data for a SINGLE min_det_step_size.
    
    For automated sweeps, use the run_full_sweep() function or run_sweep.sh script.
    """
    print(f"\n{'='*80}")
    print(f"SWEEP: Vary Batch Size (fixed max_tokens={fixed_max_tokens})")
    print(f"{'='*80}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prompts per run: {n_prompts}")
    print(f"Temperature: {temperature}")
    print(f"{'='*80}\n")
    
    random.seed(int(os.getenv("SGLANG_TEST_SEED", "12345")))
    prompts = [_random_prompt(10, 50) for _ in range(n_prompts)]
    
    results = {}
    
    for batch_size in batch_sizes:
        label = f"BS={batch_size}"
        print(f"\nTesting {label}...")
        
        data = collect_rollback_data(
            base_url=base_url,
            model_name=model_name,
            prompts=prompts,
            batch_size=batch_size,
            max_tokens=fixed_max_tokens,
            temperature=temperature,
            verbose=True,
        )
        
        results[label] = data
        print(f"  Rollbacks: {data['num_rollbacks']}, Tokens: {data['tokens_rolled_back']}")
    
    return results


def run_sweep_vary_max_tokens(
    base_url: str,
    model_name: str,
    min_det_step_sizes: list[int],
    max_tokens_list: list[int],
    fixed_batch_size: int,
    n_prompts: int,
    temperature: float = 0.8,
    output_dir: str = "rollback_results",
) -> dict[str, Any]:
    """
    Sweep over min_det_step_size, varying max_tokens with fixed batch_size.
    """
    print(f"\n{'='*80}")
    print(f"SWEEP: Vary Max Tokens (fixed batch_size={fixed_batch_size})")
    print(f"{'='*80}")
    print(f"Max tokens: {max_tokens_list}")
    print(f"Prompts per run: {n_prompts}")
    print(f"Temperature: {temperature}")
    print(f"{'='*80}\n")
    
    random.seed(int(os.getenv("SGLANG_TEST_SEED", "12345")))
    prompts = [_random_prompt(10, 50) for _ in range(n_prompts)]
    
    results = {}
    
    for max_tokens in max_tokens_list:
        label = f"max_tokens={max_tokens}"
        print(f"\nTesting {label}...")
        
        data = collect_rollback_data(
            base_url=base_url,
            model_name=model_name,
            prompts=prompts,
            batch_size=fixed_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=True,
        )
        
        results[label] = data
        print(f"  Rollbacks: {data['num_rollbacks']}, Tokens: {data['tokens_rolled_back']}")
    
    return results


def run_single_config_and_save(
    base_url: str,
    model_name: str,
    min_det_step_size: int,
    batch_sizes: list[int],
    max_tokens_list: list[int],
    fixed_max_tokens: int,
    fixed_batch_size: int,
    n_prompts: int,
    temperature: float,
    output_dir: str,
    num_batches: int = 1,
):
    """
    Run tests for a single min_det_step_size and save results to JSON.
    
    This is designed to be called after starting the server with a specific
    --min-det-step-size value.
    
    Args:
        num_batches: Number of batched requests to send per configuration.
                     Each batch sends batch_size prompts in a single request.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f"# min_det_step_size = {min_det_step_size}")
    print(f"{'#'*80}")
    
    random.seed(int(os.getenv("SGLANG_TEST_SEED", "12345")))
    prompts = [_random_prompt(10, 50) for _ in range(n_prompts)]
    
    # Sweep 1: Vary batch size, fixed max_tokens
    print(f"\n--- Vary Batch Size (max_tokens={fixed_max_tokens}, {num_batches} batch(es) per config) ---")
    batch_size_results = {}
    for batch_size in batch_sizes:
        label = f"BS={batch_size}"
        print(f"  {label}... ", end="", flush=True)
        
        data = collect_rollback_data(
            base_url=base_url,
            model_name=model_name,
            prompts=prompts,
            batch_size=batch_size,
            max_tokens=fixed_max_tokens,
            temperature=temperature,
            verbose=False,
            num_batches=num_batches,
        )
        batch_size_results[label] = data
        print(f"rollbacks={data['num_rollbacks']}, tokens={data['tokens_rolled_back']}")
    
    # Sweep 2: Vary max_tokens, fixed batch_size
    print(f"\n--- Vary Max Tokens (batch_size={fixed_batch_size}, {num_batches} batch(es) per config) ---")
    max_tokens_results = {}
    for max_tokens in max_tokens_list:
        label = f"max_tokens={max_tokens}"
        print(f"  {label}... ", end="", flush=True)
        
        data = collect_rollback_data(
            base_url=base_url,
            model_name=model_name,
            prompts=prompts,
            batch_size=fixed_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
            num_batches=num_batches,
        )
        max_tokens_results[label] = data
        print(f"rollbacks={data['num_rollbacks']}, tokens={data['tokens_rolled_back']}")
    
    # Save results
    results = {
        "min_det_step_size": min_det_step_size,
        "config": {
            "batch_sizes": batch_sizes,
            "max_tokens_list": max_tokens_list,
            "fixed_max_tokens": fixed_max_tokens,
            "fixed_batch_size": fixed_batch_size,
            "n_prompts": n_prompts,
            "temperature": temperature,
        },
        "vary_batch_size": batch_size_results,
        "vary_max_tokens": max_tokens_results,
    }
    
    output_file = Path(output_dir) / f"step_{min_det_step_size}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")
    
    return results


def load_and_plot_sweep_results(
    output_dir: str,
    min_det_step_sizes: list[int],
):
    """
    Load results from multiple min_det_step_size runs and create plots.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Cannot create plots.")
        return
    
    output_path = Path(output_dir)
    
    # Load all results
    all_results = {}
    for step_size in min_det_step_sizes:
        result_file = output_path / f"step_{step_size}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results[step_size] = json.load(f)
        else:
            print(f"Warning: Missing results for step_size={step_size}")
    
    if not all_results:
        print("No results found to plot.")
        return
    
    # Get config from first result
    first_result = list(all_results.values())[0]
    batch_sizes = first_result["config"]["batch_sizes"]
    max_tokens_list = first_result["config"]["max_tokens_list"]
    
    x_values = sorted(all_results.keys())
    
    # ===== Plot 1: Vary Batch Size =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for bs in batch_sizes:
        label = f"BS={bs}"
        rollbacks = []
        tokens = []
        for step_size in x_values:
            data = all_results[step_size]["vary_batch_size"].get(label, {})
            rollbacks.append(data.get("num_rollbacks", 0))
            tokens.append(data.get("tokens_rolled_back", 0))
        
        axes[0].plot(x_values, rollbacks, marker='o', label=label)
        axes[1].plot(x_values, tokens, marker='s', label=label)
    
    axes[0].set_xlabel("min_det_step_size")
    axes[0].set_ylabel("Number of Rollbacks")
    axes[0].set_title(f"Rollbacks vs min_det_step_size\n(Vary Batch Size, fixed max_tokens={first_result['config']['fixed_max_tokens']})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("min_det_step_size")
    axes[1].set_ylabel("Tokens Rolled Back")
    axes[1].set_title(f"Tokens Rolled Back vs min_det_step_size\n(Vary Batch Size)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / "plot_vary_batch_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
    
    # ===== Plot 2: Vary Max Tokens =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for mt in max_tokens_list:
        label = f"max_tokens={mt}"
        rollbacks = []
        tokens = []
        for step_size in x_values:
            data = all_results[step_size]["vary_max_tokens"].get(label, {})
            rollbacks.append(data.get("num_rollbacks", 0))
            tokens.append(data.get("tokens_rolled_back", 0))
        
        axes[0].plot(x_values, rollbacks, marker='o', label=label)
        axes[1].plot(x_values, tokens, marker='s', label=label)
    
    axes[0].set_xlabel("min_det_step_size")
    axes[0].set_ylabel("Number of Rollbacks")
    axes[0].set_title(f"Rollbacks vs min_det_step_size\n(Vary max_tokens, fixed batch_size={first_result['config']['fixed_batch_size']})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("min_det_step_size")
    axes[1].set_ylabel("Tokens Rolled Back")
    axes[1].set_title(f"Tokens Rolled Back vs min_det_step_size\n(Vary max_tokens)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / "plot_vary_max_tokens.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'step_size':>12} | {'config':>20} | {'rollbacks':>10} | {'tokens':>10}")
    print("-" * 60)
    for step_size in x_values:
        for label, data in all_results[step_size]["vary_batch_size"].items():
            print(f"{step_size:>12} | {label:>20} | {data['num_rollbacks']:>10} | {data['tokens_rolled_back']:>10}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rollback statistics collection with min_det_step_size sweep")
    parser.add_argument("--host", default=os.getenv("SGLANG_HOST", "127.0.0.1"), help="Server host")
    parser.add_argument("--port", type=int, default=int(os.getenv("SGLANG_PORT", "30003")), help="Server port")
    parser.add_argument("--min-det-step-size", type=int, default=None,
                        help="Current min_det_step_size (for single run). If not set, will only plot existing results.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="Batch sizes to test")
    parser.add_argument("--max-tokens-list", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Max tokens values to test")
    parser.add_argument("--fixed-max-tokens", type=int, default=128,
                        help="Fixed max_tokens for batch size sweep")
    parser.add_argument("--fixed-batch-size", type=int, default=32,
                        help="Fixed batch size for max_tokens sweep")
    parser.add_argument("--n-prompts", type=int, default=64,
                        help="Number of prompts in the prompt pool (reused across batches)")
    parser.add_argument("--num-batches", type=int, default=1,
                        help="Number of batched requests per config. Each batch sends batch_size prompts in ONE request.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--output-dir", default="rollback_results",
                        help="Output directory for results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only plot existing results, don't run tests")
    parser.add_argument("--step-sizes-to-plot", type=int, nargs="+", default=[1, 5, 10, 20, 50],
                        help="Step sizes to include in plots")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    backend = os.getenv("SGLANG_ATTENTION_BACKEND", "flashinfer")
    
    print("\n" + "="*80)
    print("ROLLBACK STATISTICS COLLECTION")
    print("="*80)
    print(f"Server: {base_url}")
    print(f"Backend: {backend}")
    print(f"Output dir: {args.output_dir}")
    print("="*80 + "\n")
    
    if args.plot_only:
        # Only create plots from existing data
        load_and_plot_sweep_results(args.output_dir, args.step_sizes_to_plot)
    elif args.min_det_step_size is not None:
        # Run tests for a single min_det_step_size
        
        # Check server health
        print("Checking server health... ", end="", flush=True)
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"FAILED - Server returned status {response.status_code}")
                sys.exit(1)
            print("✓")
        except requests.exceptions.RequestException as e:
            print(f"FAILED - {e}")
            print(f"\nError: Server not running at {base_url}")
            print("Start server with:")
            print(f"  python -m sglang.launch_server --model-path <model> --enable-metrics --enable-det-infer 1 --min-det-step-size {args.min_det_step_size}")
            sys.exit(1)
        
        # Check metrics endpoint
        stats = get_rollback_stats(base_url)
        if stats is None:
            print("Warning: Could not fetch metrics. Make sure --enable-metrics is set.")
        else:
            print(f"Initial metrics: rollbacks={stats.num_rollbacks}, tokens={stats.tokens_rolled_back}")
        
        model_name = resolve_model_name(backend)
        
        run_single_config_and_save(
            base_url=base_url,
            model_name=model_name,
            min_det_step_size=args.min_det_step_size,
            batch_sizes=args.batch_sizes,
            max_tokens_list=args.max_tokens_list,
            fixed_max_tokens=args.fixed_max_tokens,
            fixed_batch_size=args.fixed_batch_size,
            n_prompts=args.n_prompts,
            temperature=args.temperature,
            output_dir=args.output_dir,
            num_batches=args.num_batches,
        )
        
        print("\nTo create plots after running all step sizes:")
        print(f"  python {sys.argv[0]} --plot-only --output-dir {args.output_dir}")
    else:
        print("Usage:")
        print("  1. Start server with specific --min-det-step-size")
        print("  2. Run: python vllm_online_batch_invariance_multitest.py --min-det-step-size <N>")
        print("  3. Repeat for different step sizes")
        print("  4. Plot: python vllm_online_batch_invariance_multitest.py --plot-only")
        print()
        print("Or use run_sweep.sh to automate the full sweep.")

