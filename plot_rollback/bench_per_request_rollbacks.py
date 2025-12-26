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
        --det-infer-window-size 10 2>&1 | tee server.log
    
    # Run benchmark (sends 2 batched requests, each with 32 prompts = 64 total prompts):
    python bench_per_request_rollbacks.py --num-requests 2 --batch-size 32 --log-file server.log
    
    # Run with QPS rate limiting (async requests):
    python bench_per_request_rollbacks.py --num-requests 10 --batch-size 8 --qps 4 --log-file server.log
    
    # Or run without log file (stats will be printed in server output):
    python bench_per_request_rollbacks.py --num-requests 100 --batch-size 1

Environment variables:
  - SGLANG_TEST_MODEL: served model name
  - SGLANG_HOST: server host (default: 127.0.0.1)
  - SGLANG_PORT: server port (default: 30000)
"""

import argparse
import asyncio
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

import aiohttp
import requests
from tqdm import tqdm

from utils import _random_prompt, resolve_model_name

# ShareGPT dataset URL
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

def download_and_cache_file(url: str, filename: Optional[str] = None) -> str:
    """Download and cache a file from URL."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])
    
    if os.path.isfile(filename):
        try:
            with open(filename) as f:
                json.load(f)
            print(f"Using cached file: {filename}")
            return filename
        except json.JSONDecodeError:
            pass
    
    print(f"Downloading from {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024
    
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    
    return filename


@dataclass
class ShareGPTSample:
    """A single sample from ShareGPT dataset with prompt and expected output length."""
    prompt: str
    output_len: int  # Length of assistant response in tokens


def load_sharegpt_prompts(
    num_prompts: int,
    dataset_path: Optional[str] = None,
    max_prompt_len: int = 8192,
    max_output_len: int = 2048,
    tokenizer: Optional[Any] = None,
) -> List[ShareGPTSample]:
    """
    Load prompts from ShareGPT dataset.
    
    Args:
        num_prompts: Number of prompts to load
        dataset_path: Path to local ShareGPT JSON file (downloads if not provided)
        max_prompt_len: Maximum prompt length in tokens
        max_output_len: Maximum output length in tokens
        tokenizer: HuggingFace tokenizer for accurate token counting (if None, uses char/4 heuristic)
    
    Returns:
        List of ShareGPTSample with prompt and expected output length
    """
    # Download if needed
    if dataset_path is None or not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)
    
    print(f"Loading ShareGPT dataset from: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Filter conversations with at least 2 turns (user + assistant)
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    
    # Helper function to count tokens
    def count_tokens(text: str) -> int:
        if tokenizer is not None:
            return len(tokenizer.encode(text))
        else:
            # Fallback: rough heuristic (~4 chars per token)
            return len(text) // 4
    
    if tokenizer is not None:
        print(f"  Using tokenizer: {tokenizer.name_or_path}")
    else:
        print(f"  Warning: No tokenizer provided, using char/4 heuristic for token counts")
    
    # Extract first user message and assistant response from each conversation
    samples = []
    for data in dataset:
        convs = data.get("conversations", data.get("conversation", []))
        if len(convs) >= 2:
            prompt = convs[0].get("value", "")
            response = convs[1].get("value", "")
            
            prompt_tokens = count_tokens(prompt)
            output_tokens = count_tokens(response)
            
            # Filter by token length
            if (10 < prompt_tokens < max_prompt_len and 
                1 < output_tokens < max_output_len):
                samples.append(ShareGPTSample(
                    prompt=prompt,
                    output_len=output_tokens,
                ))
    
    # Shuffle and take num_prompts
    random.shuffle(samples)
    samples = samples[:num_prompts]
    
    if samples:
        output_lens = [s.output_len for s in samples]
        avg_output_len = sum(output_lens) / len(output_lens)
        print(f"Loaded {len(samples)} samples from ShareGPT")
        print(f"  Output length stats: min={min(output_lens)}, max={max(output_lens)}, avg={avg_output_len:.1f} tokens")
    else:
        print(f"Loaded 0 samples from ShareGPT")
    
    return samples


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
    step_size: int = 0  # det_infer_window_size used (0 = unknown)
    # Dataset options
    dataset: str = "random"  # "random" or "sharegpt"
    dataset_path: Optional[str] = None  # Path to dataset file (for sharegpt)
    # QPS options
    qps: float = 0.0  # Requests per second (0 = sync/no rate limit)


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


async def async_send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    seed: int,
    pbar: Optional[tqdm] = None,
    request_timeout: int = 300,  # 5 min per-request timeout
    request_idx: int = 0,  # For tracking/debugging
) -> Optional[dict]:
    """Send an async completion request and return the response."""
    payload = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "is_deterministic": True,
    }
    
    # Calculate prompt size for debugging
    prompt_chars = sum(len(p) for p in prompts) if isinstance(prompts, list) else len(prompts[0]) if prompts else 0
    
    try:
        # Use per-request timeout
        async with asyncio.timeout(request_timeout):
            async with session.post(
                f"{base_url}/v1/completions",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if pbar:
                        pbar.update(1)
                    return result
                else:
                    error_text = await response.text()
                    print(f"\n[Request {request_idx}] Failed with status {response.status}: {error_text[:200]}", file=sys.stderr)
                    if pbar:
                        pbar.update(1)
                    return None
    except asyncio.TimeoutError:
        print(f"\n[Request {request_idx}] Timeout after {request_timeout}s (prompt_chars={prompt_chars}, max_tokens={max_tokens})", file=sys.stderr)
        if pbar:
            pbar.update(1)
        return None
    except aiohttp.ClientError as e:
        print(f"\n[Request {request_idx}] ClientError: {type(e).__name__}: {e} (prompt_chars={prompt_chars})", file=sys.stderr)
        if pbar:
            pbar.update(1)
        return None
    except Exception as e:
        print(f"\n[Request {request_idx}] Exception: {type(e).__name__}: {e} (prompt_chars={prompt_chars})", file=sys.stderr)
        if pbar:
            pbar.update(1)
        return None


async def run_benchmark_async(
    base_url: str,
    model_name: str,
    config: BenchmarkConfig,
    prompts: List[str],
    prompt_max_tokens: List[int],
    verbose: bool = False,
) -> List[str]:
    """
    Run benchmark with QPS rate limiting using async requests.
    
    Args:
        base_url: Server URL
        model_name: Model name
        config: Benchmark configuration
        prompts: List of all prompts (pre-generated)
        prompt_max_tokens: List of max_tokens per prompt
        verbose: Verbose output
    
    Returns:
        List of request IDs
    """
    request_ids = []
    
    # Create batches (prompts and their max_tokens)
    batches = []
    batch_max_tokens_list = []
    for i in range(config.num_requests):
        start_idx = i * config.batch_size
        batch_prompts = prompts[start_idx:start_idx + config.batch_size]
        batch_max_tokens = prompt_max_tokens[start_idx:start_idx + config.batch_size]
        batches.append(batch_prompts)
        # Use max of batch output lengths (API only accepts one max_tokens per request)
        batch_max_tokens_list.append(max(batch_max_tokens) if batch_max_tokens else config.max_tokens)
    
    # Calculate delay between requests for target QPS
    delay = 1.0 / config.qps if config.qps > 0 else 0
    
    print(f"  Running async with QPS={config.qps}, delay={delay:.3f}s between requests")
    
    # Create aiohttp session with longer timeout and higher connection limit
    timeout = aiohttp.ClientTimeout(total=1800)  # 30 min for long-running requests
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)  # No connection limit
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        task_indices = []  # Track which request index each task corresponds to
        pbar = tqdm(total=config.num_requests, desc="Requests")
        
        for i, batch_prompts in enumerate(batches):
            # Create task with request index for tracking
            task = asyncio.create_task(
                async_send_request(
                    session=session,
                    base_url=base_url,
                    model=model_name,
                    prompts=batch_prompts,
                    max_tokens=batch_max_tokens_list[i],
                    temperature=config.temperature,
                    seed=config.seed,
                    pbar=pbar,
                    request_idx=i,
                )
            )
            tasks.append(task)
            task_indices.append(i)
            
            # Rate limit: wait before sending next request
            if delay > 0 and i < len(batches) - 1:
                await asyncio.sleep(delay)
        
        # Wait for all tasks to complete (with overall timeout)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=1800  # 30 min overall timeout for all remaining tasks
            )
        except asyncio.TimeoutError:
            print(f"\nWarning: Timed out waiting for all requests to complete", file=sys.stderr)
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Gather completed results
            results = []
            for idx, task in enumerate(tasks):
                if task.done() and not task.cancelled():
                    try:
                        results.append(task.result())
                    except Exception as e:
                        results.append(e)
                else:
                    print(f"\n[Request {task_indices[idx]}] Did not complete (cancelled or stuck)", file=sys.stderr)
                    results.append(None)
        pbar.close()
    
    # Count successes and failures, track which failed
    success_count = 0
    failure_count = 0
    failed_indices = []
    for idx, result in enumerate(results):
        if isinstance(result, dict) and "choices" in result:
            success_count += 1
        elif isinstance(result, Exception):
            failure_count += 1
            failed_indices.append(task_indices[idx])
            print(f"  [Request {task_indices[idx]}] Exception in result: {type(result).__name__}: {result}", file=sys.stderr)
        elif result is None:
            failure_count += 1
            failed_indices.append(task_indices[idx])
    
    if failure_count > 0:
        print(f"\n  Completed: {success_count} success, {failure_count} failed out of {len(results)} total")
        print(f"  Failed request indices: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")
    
    # Extract request IDs from results
    for result in results:
        if isinstance(result, dict) and "choices" in result:
            for choice in result["choices"]:
                if "meta_info" in choice and "id" in choice["meta_info"]:
                    request_ids.append(choice["meta_info"]["id"])
    
    return request_ids


def run_benchmark(
    base_url: str,
    model_name: str,
    config: BenchmarkConfig,
    tokenizer: Optional[Any] = None,
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
    
    # Track per-prompt max_tokens (for ShareGPT, use dataset output lengths)
    prompt_max_tokens: List[int] = []
    
    if config.dataset == "sharegpt":
        print(f"  Using ShareGPT dataset")
        samples = load_sharegpt_prompts(
            num_prompts=total_prompts,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
        )
        prompts = [s.prompt for s in samples]
        prompt_max_tokens = [s.output_len for s in samples]
        
        # Pad with random prompts if ShareGPT doesn't have enough
        if len(prompts) < total_prompts:
            print(f"  Warning: Only {len(prompts)} ShareGPT prompts available, padding with random")
            for _ in range(total_prompts - len(prompts)):
                prompts.append(_random_prompt(config.min_prompt_words, config.max_prompt_words))
                prompt_max_tokens.append(config.max_tokens)  # Use default for random
    else:
        print(f"  Using random prompts ({config.min_prompt_words}-{config.max_prompt_words} words)")
        prompts = [
            _random_prompt(config.min_prompt_words, config.max_prompt_words)
            for _ in range(total_prompts)
        ]
        prompt_max_tokens = [config.max_tokens] * total_prompts
    
    print(f"\nRunning benchmark: {config.num_requests} batched requests, "
          f"batch_size={config.batch_size} prompts/request")
    if config.dataset == "sharegpt":
        avg_max_tokens = sum(prompt_max_tokens) / len(prompt_max_tokens) if prompt_max_tokens else config.max_tokens
        print(f"  max_tokens: per-prompt from dataset (avg={avg_max_tokens:.1f})")
    else:
        print(f"  max_tokens: {config.max_tokens} (fixed)")
    print(f"  Total prompts: {total_prompts}, dataset: {config.dataset}")
    
    # Use async if QPS is set, otherwise sync
    if config.qps > 0:
        print(f"  QPS mode: {config.qps} requests/second")
        request_ids = asyncio.run(
            run_benchmark_async(
                base_url=base_url,
                model_name=model_name,
                config=config,
                prompts=prompts,
                prompt_max_tokens=prompt_max_tokens,
                verbose=verbose,
            )
        )
    else:
        # Synchronous mode
        for i in range(config.num_requests):
            if verbose or (i + 1) % 10 == 0 or config.num_requests <= 10:
                print(f"  Batch {i+1}/{config.num_requests} ({config.batch_size} prompts)...", end=" ", flush=True)
            
            # Get batch_size prompts for this batched request
            start_idx = i * config.batch_size
            batch_prompts = prompts[start_idx:start_idx + config.batch_size]
            batch_max_tokens = prompt_max_tokens[start_idx:start_idx + config.batch_size]
            
            # Use max of batch output lengths (API only accepts one max_tokens per request)
            batch_max_token = max(batch_max_tokens) if batch_max_tokens else config.max_tokens
            
            # Send ALL batch_size prompts in ONE API call (batched request)
            resp = send_request(
                base_url=base_url,
                model=model_name,
                prompts=batch_prompts,  # List of batch_size prompts
                max_tokens=batch_max_token,
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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of prompts per batched API call (all sent together)")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default matches vllm_online_batch_invariance_multitest.py)")
    parser.add_argument("--min-prompt-words", type=int, default=0,
                        help="Minimum prompt length in words (for random dataset)")
    parser.add_argument("--max-prompt-words", type=int, default=0,
                        help="Maximum prompt length in words (for random dataset)")
    parser.add_argument("--step-size", type=int, default=0,
                        help="det_infer_window_size used (for labeling in results)")
    parser.add_argument("--qps", type=float, default=0.0,
                        help="Requests per second (0 = synchronous, no rate limit)")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, default="random",
                        choices=["random", "sharegpt"],
                        help="Dataset to use for prompts: 'random' or 'sharegpt'")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to dataset file (for sharegpt, downloads if not provided)")
    
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
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        qps=args.qps,
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
        
        # Load tokenizer for accurate token counting (for ShareGPT dataset)
        tokenizer = None
        if config.dataset == "sharegpt":
            try:
                from transformers import AutoTokenizer
                print(f"Loading tokenizer for {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")
            except Exception as e:
                print(f"  Warning: Could not load tokenizer: {e}")
                print(f"  Falling back to char/4 heuristic for token counts")
        
        request_ids, prometheus_delta = run_benchmark(
            base_url=base_url,
            model_name=model_name,
            config=config,
            tokenizer=tokenizer,
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
