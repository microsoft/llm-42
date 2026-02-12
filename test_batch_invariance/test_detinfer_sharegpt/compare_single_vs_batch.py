#!/usr/bin/env python3
"""
Compare a single prompt run vs batch runs at a configurable QPS.

This script:
1. Runs a single prompt once as a baseline (sequential, no QPS pressure)
2. Runs the same prompt N times at a configurable QPS
3. Compares if all N outputs match the baseline token-by-token
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np


@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: int
    prompt: str
    output_text: str
    output_tokens: List[int]
    latency: float
    success: bool
    error: Optional[str] = None


def get_tokenizer(model_id: str):
    """Load tokenizer for the model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_request_body: Dict[str, Any],
    request_id: int,
    seed: int,
) -> RequestResult:
    """Send a single request to the server and return the result."""
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "is_deterministic": True,
        "seed": seed,
        **extra_request_body,
    }
    
    start_time = time.perf_counter()
    output_text = ""
    error = None
    success = False
    
    try:
        async with session.post(api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                output_text = data["choices"][0]["message"]["content"]
                success = True
            else:
                error = f"HTTP {response.status}: {await response.text()}"
    except Exception as e:
        error = str(e)
    
    latency = time.perf_counter() - start_time
    
    return RequestResult(
        request_id=request_id,
        prompt=prompt,
        output_text=output_text,
        output_tokens=[],  # Will be filled later
        latency=latency,
        success=success,
        error=error,
    )


async def run_single_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_request_body: Dict[str, Any],
    seed: int,
) -> RequestResult:
    """Run a single request (baseline)."""
    api_url = f"{base_url}/v1/chat/completions"
    
    timeout = aiohttp.ClientTimeout(total=6000)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await send_request(
            session, api_url, model, prompt, max_tokens, extra_request_body, request_id=0, seed=seed
        )
    
    return result


async def run_batch_requests(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_request_body: Dict[str, Any],
    num_repeats: int,
    qps: float,
    seed: int,
) -> List[RequestResult]:
    """Run multiple requests at a specified QPS."""
    api_url = f"{base_url}/v1/chat/completions"
    
    timeout = aiohttp.ClientTimeout(total=6000)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        interval = 1.0 / qps if qps > 0 else 0
        
        start_time = time.perf_counter()
        
        for i in range(num_repeats):
            # Calculate when this request should be sent
            expected_time = i * interval
            current_time = time.perf_counter() - start_time
            
            # Wait if we're ahead of schedule
            if current_time < expected_time:
                await asyncio.sleep(expected_time - current_time)
            
            # Create the task
            task = asyncio.create_task(
                send_request(
                    session, api_url, model, prompt, max_tokens, extra_request_body, request_id=i, seed=seed
                )
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
    
    return list(results)


def tokenize_results(results: List[RequestResult], tokenizer) -> List[RequestResult]:
    """Tokenize the output text for all results."""
    for result in results:
        if result.success:
            result.output_tokens = tokenizer.encode(result.output_text, add_special_tokens=False)
    return results


def first_mismatch(a: List[int], b: List[int]) -> int:
    """Return first mismatch index; treat length difference as a mismatch.
    
    If sequences are identical and equal length, return the length.
    """
    len_a, len_b = len(a), len(b)
    limit = min(len_a, len_b)
    
    for i in range(limit):
        if a[i] != b[i]:
            return i
    
    # If we got here, common prefix matches
    if len_a != len_b:
        return limit
    
    return len_a


def compare_outputs(baseline: RequestResult, batch_results: List[RequestResult]) -> Dict[str, Any]:
    """Compare all batch results against the baseline."""
    comparisons = []
    
    baseline_tokens = baseline.output_tokens
    baseline_text = baseline.output_text
    
    match_count = 0
    mismatch_count = 0
    
    for result in batch_results:
        if not result.success:
            comparisons.append({
                "request_id": result.request_id,
                "match": False,
                "error": result.error,
                "first_mismatch_index": -1,
            })
            mismatch_count += 1
            continue
        
        first_mm = first_mismatch(baseline_tokens, result.output_tokens)
        is_match = (first_mm == len(baseline_tokens)) and (first_mm == len(result.output_tokens))
        
        if is_match:
            match_count += 1
        else:
            mismatch_count += 1
        
        comparisons.append({
            "request_id": result.request_id,
            "match": is_match,
            "first_mismatch_index": first_mm if not is_match else -1,
            "baseline_length": len(baseline_tokens),
            "batch_length": len(result.output_tokens),
            "baseline_text": baseline_text,
            "batch_text": result.output_text,
            "baseline_tokens": baseline_tokens,
            "batch_tokens": result.output_tokens,
        })
    
    return {
        "total": len(batch_results),
        "matches": match_count,
        "mismatches": mismatch_count,
        "match_rate": match_count / len(batch_results) if batch_results else 0,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare single prompt vs batch runs")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--base-url", required=True, help="Server URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer (default: same as model)")
    parser.add_argument("--prompt", required=True, help="Prompt to test")
    parser.add_argument("--num-repeats", type=int, default=10, help="Number of times to repeat prompt in batch")
    parser.add_argument("--qps", type=float, default=4.0, help="Requests per second for batch run")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("single_vs_batch_out"))
    parser.add_argument("--extra-request-body", default=None, help="Extra JSON body for requests")

    cli_args = parser.parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse extra request body
    extra_body = {}
    if cli_args.extra_request_body:
        extra_body = json.loads(cli_args.extra_request_body)

    print(f"Configuration:")
    print(f"  Base URL:    {cli_args.base_url}")
    print(f"  Model:       {cli_args.model}")
    print(f"  Prompt:      {cli_args.prompt[:50]}...")
    print(f"  Num Repeats: {cli_args.num_repeats}")
    print(f"  QPS:         {cli_args.qps}")
    print(f"  Max Tokens:  {cli_args.max_tokens}")
    print()

    # Load tokenizer
    tokenizer_id = cli_args.tokenizer or cli_args.model
    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = get_tokenizer(tokenizer_id)
    print()

    # Step 1: Run single baseline request
    print("=" * 50)
    print("Step 1: Running single baseline request...")
    print("=" * 50)
    
    baseline_result = asyncio.run(
        run_single_request(
            cli_args.base_url,
            cli_args.model,
            cli_args.prompt,
            cli_args.max_tokens,
            extra_body,
            cli_args.seed,
        )
    )
    
    if not baseline_result.success:
        print(f"ERROR: Baseline request failed: {baseline_result.error}")
        return 1
    
    baseline_result = tokenize_results([baseline_result], tokenizer)[0]
    print(f"  Baseline completed in {baseline_result.latency:.2f}s")
    print(f"  Output length: {len(baseline_result.output_tokens)} tokens")
    print(f"  Output text (first 100 chars): {baseline_result.output_text[:100]}...")
    print()

    # Step 2: Run batch requests at specified QPS
    print("=" * 50)
    print(f"Step 2: Running {cli_args.num_repeats} requests at {cli_args.qps} QPS...")
    print("=" * 50)
    
    batch_results = asyncio.run(
        run_batch_requests(
            cli_args.base_url,
            cli_args.model,
            cli_args.prompt,
            cli_args.max_tokens,
            extra_body,
            cli_args.num_repeats,
            cli_args.qps,
            cli_args.seed,
        )
    )
    
    batch_results = tokenize_results(batch_results, tokenizer)
    
    successful = sum(1 for r in batch_results if r.success)
    failed = len(batch_results) - successful
    avg_latency = np.mean([r.latency for r in batch_results if r.success])
    
    print(f"  Completed: {successful}/{len(batch_results)} successful")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"  Average latency: {avg_latency:.2f}s")
    print()

    # Step 3: Compare results
    print("=" * 50)
    print("Step 3: Comparing outputs...")
    print("=" * 50)
    
    comparison = compare_outputs(baseline_result, batch_results)
    
    print(f"  Total requests: {comparison['total']}")
    print(f"  Matches:        {comparison['matches']} ({comparison['match_rate'] * 100:.1f}%)")
    print(f"  Mismatches:     {comparison['mismatches']} ({(1 - comparison['match_rate']) * 100:.1f}%)")
    print()

    # Show details of mismatches
    if comparison['mismatches'] > 0:
        print("Mismatch details:")
        for comp in comparison['comparisons']:
            if not comp['match']:
                if 'error' in comp:
                    print(f"  Request {comp['request_id']}: Error - {comp['error']}")
                else:
                    print(f"  Request {comp['request_id']}: First mismatch at token {comp['first_mismatch_index']}")
                    print(f"    Baseline length: {comp['baseline_length']}, Batch length: {comp['batch_length']}")
        print()

    # Save results
    results_file = cli_args.output_dir / "results.json"
    with results_file.open("w") as f:
        json.dump({
            "config": {
                "base_url": cli_args.base_url,
                "model": cli_args.model,
                "prompt": cli_args.prompt,
                "num_repeats": cli_args.num_repeats,
                "qps": cli_args.qps,
                "max_tokens": cli_args.max_tokens,
                "seed": cli_args.seed,
            },
            "baseline": {
                "text": baseline_result.output_text,
                "tokens": baseline_result.output_tokens,
                "latency": baseline_result.latency,
            },
            "comparison": {
                "total": comparison['total'],
                "matches": comparison['matches'],
                "mismatches": comparison['mismatches'],
                "match_rate": comparison['match_rate'],
            },
            "details": comparison['comparisons'],
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Return exit code based on whether all matched
    if comparison['mismatches'] > 0:
        print()
        print("WARNING: Not all outputs matched the baseline!")
        return 0  # Still return 0, but with warning
    else:
        print()
        print("SUCCESS: All outputs matched the baseline!")
        return 0


if __name__ == "__main__":
    exit(main())
