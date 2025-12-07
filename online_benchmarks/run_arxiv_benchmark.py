#!/usr/bin/env python3
"""
Arxiv-based online benchmark using ccdv/arxiv-summarization from HuggingFace.
Measures TTFT, TPOT, and E2E latency.

Usage:
    python run_arxiv_benchmark.py --base-url http://localhost:30000 --num-prompts 500 --model meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List

import aiohttp
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int


@dataclass
class RequestMetrics:
    ttft: float = 0.0
    e2e: float = 0.0
    output_tokens: int = 0
    
    @property
    def tpot(self) -> float:
        return (self.e2e - self.ttft) / max(1, self.output_tokens - 1) if self.output_tokens > 1 else 0.0


@dataclass
class BenchmarkResults:
    metrics: list[RequestMetrics] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    def summary(self) -> dict:
        if not self.metrics:
            return {}
        ttfts = [m.ttft * 1000 for m in self.metrics]
        tpots = [m.tpot * 1000 for m in self.metrics if m.tpot > 0]
        e2es = [m.e2e * 1000 for m in self.metrics]
        total_tokens = sum(m.output_tokens for m in self.metrics)
        duration = self.end_time - self.start_time
        
        return {
            "num_requests": len(self.metrics),
            "total_time": duration,
            "output_throughput": total_tokens / duration if duration > 0 else 0,
            "request_throughput": len(self.metrics) / duration if duration > 0 else 0,
            "mean_ttft_ms": np.mean(ttfts),
            "median_ttft_ms": np.median(ttfts),
            "p99_ttft_ms": np.percentile(ttfts, 99),
            "mean_tpot_ms": np.mean(tpots) if tpots else 0,
            "median_tpot_ms": np.median(tpots) if tpots else 0,
            "p99_tpot_ms": np.percentile(tpots, 99) if tpots else 0,
            "mean_e2e_latency_ms": np.mean(e2es),
            "median_e2e_latency_ms": np.median(e2es),
            "p99_e2e_latency_ms": np.percentile(e2es, 99),
        }


def load_arxiv_dataset(
    tokenizer,
    num_samples: int,
    context_len: int = 8192,
) -> List[DatasetRow]:
    """Load arxiv articles, tokenize prompts and abstracts to get actual lengths."""
    print("Loading ccdv/arxiv-summarization from HuggingFace...")
    ds = load_dataset("ccdv/arxiv-summarization", split="test", trust_remote_code=True)
    
    samples = []
    for item in ds:
        if len(samples) >= num_samples:
            break
        
        article = item['article']
        abstract = item['abstract']
        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
        
        # Tokenize to get actual token counts
        prompt_token_ids = tokenizer.encode(prompt)
        abstract_token_ids = tokenizer.encode(abstract)
        prompt_len = len(prompt_token_ids)
        output_len = len(abstract_token_ids)
        
        # Skip if too short
        if prompt_len < 10 or output_len < 2:
            continue
        
        # Skip if total exceeds context length (like ShareGPT)
        if context_len and prompt_len + output_len > context_len:
            continue
        
        samples.append(DatasetRow(prompt=prompt, prompt_len=prompt_len, output_len=output_len))
    
    random.shuffle(samples)
    
    total_input = sum(s.prompt_len for s in samples)
    total_output = sum(s.output_len for s in samples)
    avg_input = total_input / len(samples) if samples else 0
    avg_output = total_output / len(samples) if samples else 0
    print(f"Loaded {len(samples)} samples")
    print(f"#Input tokens: {total_input} (avg: {avg_input:.0f})")
    print(f"#Output tokens: {total_output} (avg: {avg_output:.0f})")
    return samples


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    max_tokens: int,
    is_deterministic: bool,
) -> RequestMetrics:
    """Send single request and measure TTFT, E2E, output tokens."""
    url = f"{base_url}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0, "ignore_eos": True},
        "stream": True,
    }
    if is_deterministic:
        payload["is_deterministic"] = True
    
    metrics = RequestMetrics()
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    
    try:
        async with session.post(url, json=payload) as resp:
            async for chunk in resp.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    metrics.ttft = first_token_time - start
                # Count tokens from streamed response
                try:
                    data = json.loads(chunk.decode().strip('data: \n'))
                    if 'meta_info' in data:
                        metrics.output_tokens = data['meta_info'].get('completion_tokens', 0)
                except:
                    pass
            
            metrics.e2e = time.perf_counter() - start
    except Exception as e:
        print(f"Request error: {e}")
    
    return metrics


async def run_benchmark(
    base_url: str,
    data: List[DatasetRow],
    request_rate: float,
    deterministic_ratio: float,
) -> BenchmarkResults:
    """Run benchmark with Poisson arrival rate and mixed deterministic requests."""
    results = BenchmarkResults()
    results.start_time = time.perf_counter()
    
    # Determine which requests are deterministic
    num_det = int(len(data) * deterministic_ratio)
    det_indices = set(random.sample(range(len(data)), num_det))
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, row in enumerate(data):
            is_det = i in det_indices
            tasks.append(send_request(session, base_url, row.prompt, row.output_len, is_det))
            
            # Poisson inter-arrival time
            if request_rate < float('inf'):
                await asyncio.sleep(random.expovariate(request_rate))
        
        results.metrics = await asyncio.gather(*tasks)
    
    results.end_time = time.perf_counter()
    return results


def main():
    parser = argparse.ArgumentParser(description='Arxiv online benchmark (ccdv/arxiv-summarization)')
    parser.add_argument('--base-url', default='http://localhost:30000')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model for tokenizer')
    parser.add_argument('--num-prompts', type=int, default=500)
    parser.add_argument('--request-rate', type=float, default=8.0, help='Requests per second (QPS)')
    parser.add_argument('--deterministic-ratio', type=float, default=0.0, help='Fraction of requests that are deterministic (0.0-1.0)')
    parser.add_argument('--output-file')
    parser.add_argument('--context-len', type=int, default=8192, help='Max context length (prompt + output)')
    args = parser.parse_args()
    
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    data = load_arxiv_dataset(tokenizer, args.num_prompts, args.context_len)
    
    print(f"Running benchmark: rate={args.request_rate} QPS, det_ratio={args.deterministic_ratio}")
    results = asyncio.run(run_benchmark(
        args.base_url, data, args.request_rate, args.deterministic_ratio
    ))
    
    summary = results.summary()
    summary.update({
        "dataset": "arxiv",
        "rate": args.request_rate,
        "det_ratio": args.deterministic_ratio,
    })
    
    print("\n" + "=" * 50)
    print(f"TTFT: mean={summary['mean_ttft_ms']:.1f}ms, p50={summary['median_ttft_ms']:.1f}ms, p99={summary['p99_ttft_ms']:.1f}ms")
    print(f"TPOT: mean={summary['mean_tpot_ms']:.1f}ms, p50={summary['median_tpot_ms']:.1f}ms, p99={summary['p99_tpot_ms']:.1f}ms")
    print(f"E2E:  mean={summary['mean_e2e_latency_ms']:.1f}ms, p50={summary['median_e2e_latency_ms']:.1f}ms, p99={summary['p99_e2e_latency_ms']:.1f}ms")
    print(f"Throughput: {summary['output_throughput']:.1f} tokens/s, {summary['request_throughput']:.2f} req/s")
    
    if args.output_file:
        from pathlib import Path
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        print(f"Saved to {args.output_file}")


if __name__ == '__main__':
    main()
