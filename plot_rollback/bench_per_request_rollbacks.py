#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
Benchmark for analyzing per-request rollback statistics.

This script sends batched deterministic inference requests and parses the server logs
to extract per-request rollback stats (rollbacks, tokens_rolled_back per request).

Each API call sends batch_size prompts together (batched request), matching the
behavior of vllm_online_batch_invariance_multitest.py.

Usage:
    # Start server with logging enabled:
    python -m sglang.launch_server --model-path <model> --enable-det-infer 1 \\
        --min-det-step-size 10 2>&1 | tee server.log
    
    # Run benchmark (sends 2 batched requests, each with 32 prompts = 64 total prompts):
    python bench_per_request_rollbacks.py --num-requests 2 --batch-size 32 --log-file server.log
    
    # Or run without log file (stats will be printed in server output):
    python bench_per_request_rollbacks.py --num-requests 100 --batch-size 1

Environment variables:
  - SGLANG_TEST_MODEL: served model name
  - SGLANG_HOST: server host (default: 127.0.0.1)
  - SGLANG_PORT: server port (default: 30000)
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, List, Dict
from collections import defaultdict

import requests

from utils import _random_prompt, resolve_model_name


@dataclass
class RequestRollbackStats:
    """Per-request rollback statistics."""
    rid: str
    num_rollbacks: int
    tokens_rolled_back: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PrometheusStats:
    """Prometheus metrics snapshot."""
    num_rollbacks: int = 0
    tokens_rolled_back: int = 0
    num_requests: int = 0


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    num_requests: int = 1  # Number of batched API calls
    batch_size: int = 32  # Number of prompts per batched API call (matches old script default)
    max_tokens: int = 128
    temperature: float = 0.0
    seed: int = 12345  # Match vllm_online_batch_invariance_multitest.py
    min_prompt_words: int = 10  # Match vllm_online_batch_invariance_multitest.py
    max_prompt_words: int = 50  # Match vllm_online_batch_invariance_multitest.py
    step_size: int = 0  # det_step_size used (0 = unknown)


@dataclass 
class BenchmarkResults:
    """Aggregated benchmark results."""
    config: dict
    total_requests: int = 0
    requests_with_rollbacks: int = 0
    total_rollbacks: int = 0
    total_tokens_rolled_back: int = 0
    per_request_stats: List[RequestRollbackStats] = field(default_factory=list)
    # Prometheus metrics (for comparison)
    prometheus_stats: Optional[Dict[str, int]] = None
    
    @property
    def rollback_rate(self) -> float:
        """Fraction of requests that had at least one rollback."""
        return self.requests_with_rollbacks / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def avg_rollbacks_per_request(self) -> float:
        """Average number of rollbacks per request (across all requests)."""
        return self.total_rollbacks / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def avg_tokens_per_rollback(self) -> float:
        """Average tokens rolled back per rollback event."""
        return self.total_tokens_rolled_back / self.total_rollbacks if self.total_rollbacks > 0 else 0.0
    
    @property
    def avg_tokens_rolled_back_per_request(self) -> float:
        """Average tokens rolled back per request."""
        return self.total_tokens_rolled_back / self.total_requests if self.total_requests > 0 else 0.0
    
    def to_dict(self) -> dict:
        result = {
            "config": self.config,
            "summary": {
                "total_requests": self.total_requests,
                "requests_with_rollbacks": self.requests_with_rollbacks,
                "total_rollbacks": self.total_rollbacks,
                "total_tokens_rolled_back": self.total_tokens_rolled_back,
                "rollback_rate": self.rollback_rate,
                "avg_rollbacks_per_request": self.avg_rollbacks_per_request,
                "avg_tokens_per_rollback": self.avg_tokens_per_rollback,
                "avg_tokens_rolled_back_per_request": self.avg_tokens_rolled_back_per_request,
            },
            "per_request_stats": [s.to_dict() for s in self.per_request_stats],
        }
        if self.prometheus_stats:
            result["prometheus_stats"] = self.prometheus_stats
        return result
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nConfiguration:")
        for k, v in self.config.items():
            print(f"  {k}: {v}")
        
        print(f"\nSummary:")
        print(f"  Total requests:              {self.total_requests}")
        print(f"  Requests with rollbacks:     {self.requests_with_rollbacks} ({self.rollback_rate*100:.1f}%)")
        print(f"  Total rollback events:       {self.total_rollbacks}")
        print(f"  Total tokens rolled back:    {self.total_tokens_rolled_back}")
        print(f"  Avg rollbacks/request:       {self.avg_rollbacks_per_request:.2f}")
        print(f"  Avg tokens/rollback:         {self.avg_tokens_per_rollback:.2f}")
        print(f"  Avg tokens rolled/request:   {self.avg_tokens_rolled_back_per_request:.2f}")
        
        if self.per_request_stats:
            # Distribution analysis
            rollback_counts = [s.num_rollbacks for s in self.per_request_stats]
            tokens_counts = [s.tokens_rolled_back for s in self.per_request_stats]
            
            print(f"\nDistribution of rollbacks per request:")
            print(f"  Min: {min(rollback_counts)}, Max: {max(rollback_counts)}, Median: {sorted(rollback_counts)[len(rollback_counts)//2]}")
            
            # Histogram of rollback counts
            hist = defaultdict(int)
            for c in rollback_counts:
                hist[c] += 1
            print(f"\n  Rollback count histogram:")
            for count in sorted(hist.keys()):
                bar = "█" * min(hist[count], 50)
                print(f"    {count:3d} rollbacks: {hist[count]:4d} requests {bar}")
        
        # Print Prometheus comparison if available
        if self.prometheus_stats:
            print(f"\n  Prometheus /metrics comparison:")
            print(f"    Prometheus rollbacks:       {self.prometheus_stats.get('num_rollbacks', 'N/A')}")
            print(f"    Prometheus tokens rolled:   {self.prometheus_stats.get('tokens_rolled_back', 'N/A')}")
            print(f"    Per-request log rollbacks:  {self.total_rollbacks}")
            print(f"    Per-request log tokens:     {self.total_tokens_rolled_back}")
            prom_rb = self.prometheus_stats.get('num_rollbacks', 0)
            prom_tok = self.prometheus_stats.get('tokens_rolled_back', 0)
            if prom_rb == self.total_rollbacks and prom_tok == self.total_tokens_rolled_back:
                print(f"    ✓ MATCH!")
            else:
                print(f"    ✗ MISMATCH (delta: rollbacks={prom_rb - self.total_rollbacks}, tokens={prom_tok - self.total_tokens_rolled_back})")
        
        print("=" * 60)


def parse_prometheus_metrics(text: str) -> PrometheusStats:
    """Parse Prometheus metrics text."""
    stats = PrometheusStats()
    for name, attr in [
        ("sglang:num_rollbacks_total", "num_rollbacks"),
        ("sglang:tokens_rolled_back_total", "tokens_rolled_back"),
        ("sglang:num_requests_total", "num_requests"),
    ]:
        if m := re.search(rf'{re.escape(name)}\{{[^}}]*\}}\s+([\d.]+)', text):
            setattr(stats, attr, int(float(m.group(1))))
    return stats


def get_prometheus_stats(base_url: str) -> Optional[PrometheusStats]:
    """Fetch current rollback stats from server Prometheus metrics endpoint."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        if r.status_code == 200:
            return parse_prometheus_metrics(r.text)
    except Exception as e:
        print(f"Warning: Could not fetch Prometheus metrics: {e}")
    return None


def parse_log_for_rollback_stats(log_file: str) -> Dict[str, RequestRollbackStats]:
    """
    Parse server log file to extract per-request rollback stats.
    
    Looks for lines like:
    Det Rollback Stats(rid=abc123): rollbacks=3, tokens_rolled_back=15
    """
    pattern = re.compile(
        r'Det Rollback Stats\(rid=([^)]+)\): rollbacks=(\d+), tokens_rolled_back=(\d+)'
    )
    
    stats = {}
    with open(log_file, 'r') as f:
        for line in f:
            if match := pattern.search(line):
                rid = match.group(1)
                num_rollbacks = int(match.group(2))
                tokens_rolled_back = int(match.group(3))
                stats[rid] = RequestRollbackStats(
                    rid=rid,
                    num_rollbacks=num_rollbacks,
                    tokens_rolled_back=tokens_rolled_back,
                )
    return stats


def send_request(
    base_url: str,
    model: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout: int = 300,
) -> Optional[dict]:
    """Send a completion request and return the response."""
    payload = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "is_deterministic": True,
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return None


def run_benchmark(
    base_url: str,
    model_name: str,
    config: BenchmarkConfig,
    verbose: bool = False,
) -> tuple[List[str], Optional[Dict[str, int]]]:
    """
    Run the benchmark and return list of request IDs and Prometheus delta.
    
    Sends num_requests batched API calls, each containing batch_size prompts.
    This matches the behavior of vllm_online_batch_invariance_multitest.py.
    
    Total prompts generated = num_requests * batch_size
    
    Returns:
        Tuple of (request_ids, prometheus_delta)
        - request_ids: List of request IDs that were sent
        - prometheus_delta: Dict with num_rollbacks and tokens_rolled_back delta
    """
    request_ids = []
    
    # Get Prometheus stats BEFORE benchmark
    stats_before = get_prometheus_stats(base_url)
    if stats_before:
        print(f"  Prometheus before: rollbacks={stats_before.num_rollbacks}, tokens={stats_before.tokens_rolled_back}")
    
    # Generate all prompts upfront
    total_prompts = config.num_requests * config.batch_size
    prompts = [
        _random_prompt(config.min_prompt_words, config.max_prompt_words)
        for _ in range(total_prompts)
    ]
    
    print(f"\nRunning benchmark: {config.num_requests} batched requests, "
          f"batch_size={config.batch_size} prompts/request, "
          f"max_tokens={config.max_tokens}")
    print(f"  Total prompts: {total_prompts}")
    
    for i in range(config.num_requests):
        if verbose or (i + 1) % 10 == 0 or config.num_requests <= 10:
            print(f"  Batch {i+1}/{config.num_requests} ({config.batch_size} prompts)...", end=" ", flush=True)
        
        # Get batch_size prompts for this batched request
        start_idx = i * config.batch_size
        batch_prompts = prompts[start_idx:start_idx + config.batch_size]
        
        # Send ALL batch_size prompts in ONE API call (batched request)
        resp = send_request(
            base_url=base_url,
            model=model_name,
            prompts=batch_prompts,  # List of batch_size prompts
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            seed=config.seed,  # Use consistent seed like vllm script
        )
        
        if resp:
            # Extract request IDs from response - each prompt gets its own request ID
            if "choices" in resp:
                for choice in resp["choices"]:
                    if "meta_info" in choice and "id" in choice["meta_info"]:
                        request_ids.append(choice["meta_info"]["id"])
            
            if verbose or (i + 1) % 10 == 0 or config.num_requests <= 10:
                num_choices = len(resp.get("choices", []))
                print(f"✓ ({num_choices} completions)")
        else:
            if verbose or (i + 1) % 10 == 0 or config.num_requests <= 10:
                print("✗")
    
    # Get Prometheus stats AFTER benchmark
    prometheus_delta = None
    stats_after = get_prometheus_stats(base_url)
    if stats_after:
        print(f"  Prometheus after:  rollbacks={stats_after.num_rollbacks}, tokens={stats_after.tokens_rolled_back}")
        if stats_before:
            prometheus_delta = {
                "num_rollbacks": stats_after.num_rollbacks - stats_before.num_rollbacks,
                "tokens_rolled_back": stats_after.tokens_rolled_back - stats_before.tokens_rolled_back,
                "num_requests": stats_after.num_requests - stats_before.num_requests,
            }
            print(f"  Prometheus DELTA:  rollbacks={prometheus_delta['num_rollbacks']}, tokens={prometheus_delta['tokens_rolled_back']}")
    
    return request_ids, prometheus_delta


def analyze_log_file(
    log_file: str,
    config: BenchmarkConfig,
    request_ids: Optional[List[str]] = None,
) -> BenchmarkResults:
    """
    Analyze log file for per-request rollback stats.
    
    Args:
        log_file: Path to server log file
        config: Benchmark configuration
        request_ids: Optional list of request IDs to filter by
    """
    stats = parse_log_for_rollback_stats(log_file)
    
    # Filter by request IDs if provided
    if request_ids:
        stats = {rid: s for rid, s in stats.items() if rid in request_ids}
    
    results = BenchmarkResults(
        config=asdict(config),
        total_requests=len(stats),
        per_request_stats=list(stats.values()),
    )
    
    for s in stats.values():
        if s.num_rollbacks > 0:
            results.requests_with_rollbacks += 1
        results.total_rollbacks += s.num_rollbacks
        results.total_tokens_rolled_back += s.tokens_rolled_back
    
    return results


def wait_for_server(base_url: str, timeout: int = 60) -> bool:
    """Wait for server to be ready."""
    print(f"Waiting for server at {base_url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print("Server is ready!")
                return True
        except:
            pass
        time.sleep(2)
    print("Server not ready after timeout")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark for analyzing per-request rollback statistics"
    )
    
    # Server options
    parser.add_argument("--host", default=os.getenv("SGLANG_HOST", "127.0.0.1"),
                        help="Server host")
    parser.add_argument("--port", type=int, default=int(os.getenv("SGLANG_PORT", "30000")),
                        help="Server port")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect or SGLANG_TEST_MODEL)")
    
    # Benchmark options
    parser.add_argument("--num-requests", type=int, default=1,
                        help="Number of batched API calls to send")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of prompts per batched API call (all sent together)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed (default matches vllm_online_batch_invariance_multitest.py)")
    parser.add_argument("--min-prompt-words", type=int, default=10,
                        help="Minimum prompt length in words (default matches vllm script)")
    parser.add_argument("--max-prompt-words", type=int, default=50,
                        help="Maximum prompt length in words (default matches vllm script)")
    parser.add_argument("--step-size", type=int, default=0,
                        help="det_step_size used (for labeling in results)")
    
    # Log analysis options
    parser.add_argument("--log-file", type=str, default=None,
                        help="Server log file to parse for stats (if not provided, just runs requests)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze log file, don't send requests")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    config = BenchmarkConfig(
        num_requests=args.num_requests,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        min_prompt_words=args.min_prompt_words,
        max_prompt_words=args.max_prompt_words,
        step_size=args.step_size,
    )
    
    # Set random seed
    random.seed(args.seed)
    
    request_ids = []
    prometheus_delta = None
    
    if args.analyze_only:
        if not args.log_file:
            print("Error: --log-file required with --analyze-only", file=sys.stderr)
            sys.exit(1)
    else:
        # Run benchmark
        if not wait_for_server(base_url):
            sys.exit(1)
        
        # Resolve model name
        model_name = args.model
        if not model_name:
            # Try to get from server
            try:
                r = requests.get(f"{base_url}/v1/models", timeout=5)
                if r.status_code == 200:
                    models = r.json().get("data", [])
                    if models:
                        model_name = models[0].get("id")
            except:
                pass
        
        if not model_name:
            model_name = resolve_model_name("flashinfer")
        
        print(f"Using model: {model_name}")
        
        request_ids, prometheus_delta = run_benchmark(
            base_url=base_url,
            model_name=model_name,
            config=config,
            verbose=args.verbose,
        )
        
        print(f"\nSent {len(request_ids)} request IDs collected")
        
        # Give server time to flush logs
        if args.log_file:
            print("Waiting for server to flush logs...")
            time.sleep(2)
    
    # Analyze log file if provided
    if args.log_file:
        if not Path(args.log_file).exists():
            print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
            sys.exit(1)
        
        print(f"\nAnalyzing log file: {args.log_file}")
        results = analyze_log_file(
            log_file=args.log_file,
            config=config,
            request_ids=request_ids if request_ids else None,
        )
        
        # Attach Prometheus stats for comparison
        if prometheus_delta:
            results.prometheus_stats = prometheus_delta
        
        results.print_summary()
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nNote: To see per-request rollback stats, provide --log-file with server log")
        print("The server will log lines like:")
        print('  Det Rollback Stats(rid=xxx): rollbacks=N, tokens_rolled_back=M')


if __name__ == "__main__":
    main()
