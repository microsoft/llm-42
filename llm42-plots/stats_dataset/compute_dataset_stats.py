#!/usr/bin/env python3
"""
Compute statistics (mean, median, std deviation) for arxiv and sharegpt datasets.

This script analyzes both input and output token lengths separately for:
- ShareGPT dataset (conversation-based)
- Arxiv dataset (summarization-based)

Usage:
    python compute_dataset_stats.py --model meta-llama/Llama-3.1-8B-Instruct
    python compute_dataset_stats.py --model meta-llama/Llama-3.1-8B-Instruct --num-samples 5000
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Try to import tokenizer
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("Warning: transformers not installed, using char/4 heuristic for token counts")

# ShareGPT dataset URL
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


@dataclass
class DatasetStats:
    """Statistics for a dataset."""
    name: str
    input_mean: float
    input_median: float
    input_std: float
    input_min: int
    input_max: int
    output_mean: float
    output_median: float
    output_std: float
    output_min: int
    output_max: int
    num_samples: int


def download_and_cache_file(url: str, filename: Optional[str] = None) -> str:
    """Download and cache a file from URL."""
    cache_dir = Path.home() / ".cache" / "sglang"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = url.split("/")[-1]
    
    cache_path = cache_dir / filename
    
    if cache_path.exists():
        print(f"Using cached file: {cache_path}")
        return str(cache_path)
    
    print(f"Downloading {url} to {cache_path}...")
    import urllib.request
    urllib.request.urlretrieve(url, cache_path)
    print(f"Downloaded to {cache_path}")
    
    return str(cache_path)


def load_sharegpt_data(
    num_samples: int,
    tokenizer=None,
    dataset_path: Optional[str] = None,
    context_len: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int], int]:
    """
    Load input and output token lengths from ShareGPT dataset.
    
    Returns:
        Tuple of (input_lengths, output_lengths, num_filtered)
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
    
    # Shuffle for random sampling
    random.seed(seed)
    random.shuffle(dataset)
    
    # Helper function to count tokens (matches bench_serving.py)
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    
    print(f"  Using tokenizer: {tokenizer.name_or_path}")
    
    # Extract input/output lengths
    input_lengths = []
    output_lengths = []
    num_filtered = 0
    
    for data in dataset:
        if num_samples is not None and len(input_lengths) >= num_samples:
            break
            
        convs = data.get("conversations", data.get("conversation", []))
        if len(convs) >= 2:
            prompt = convs[0].get("value", "")
            response = convs[1].get("value", "")
            
            prompt_tokens = count_tokens(prompt)
            output_tokens = count_tokens(response)
            
            # Filter out very short samples (matches bench_serving.py)
            if prompt_tokens < 2 or output_tokens < 2:
                num_filtered += 1
                continue
            
            # Filter by context length (combined input + output)
            if context_len is not None and prompt_tokens + output_tokens > context_len:
                num_filtered += 1
                continue
            
            input_lengths.append(prompt_tokens)
            output_lengths.append(output_tokens)
    
    print(f"  Loaded {len(input_lengths)} ShareGPT samples (filtered: {num_filtered})")
    return input_lengths, output_lengths, num_filtered


def load_arxiv_data(
    num_samples: int,
    tokenizer=None,
    context_len: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int], int]:
    """
    Load input and output token lengths from arxiv-summarization dataset.
    
    Returns:
        Tuple of (input_lengths, output_lengths, num_filtered) where:
        - input_lengths: lengths of the article prompts
        - output_lengths: lengths of the abstracts (summaries)
        - num_filtered: number of samples filtered out
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' package to use the arxiv dataset: "
            "pip install datasets"
        )
    
    print("Loading ccdv/arxiv-summarization from HuggingFace...")
    ds = load_dataset("ccdv/arxiv-summarization", split="test")
    
    # Helper function to count tokens (matches bench_serving.py)
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    
    print(f"  Using tokenizer: {tokenizer.name_or_path}")
    
    # Convert to list and shuffle
    dataset_list = list(ds)
    random.seed(seed)
    random.shuffle(dataset_list)
    
    # Extract input/output lengths
    input_lengths = []
    output_lengths = []
    num_filtered = 0
    
    for item in dataset_list:
        if num_samples is not None and len(input_lengths) >= num_samples:
            break
            
        article = item['article']
        abstract = item['abstract']
        
        # Create summarization prompt (to match actual usage)
        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
        
        prompt_tokens = count_tokens(prompt)
        output_tokens = count_tokens(abstract)
        
        # Filter out very short samples
        if prompt_tokens < 10 or output_tokens < 2:
            num_filtered += 1
            continue
        
        # Filter by context length (combined input + output)
        if context_len is not None and prompt_tokens + output_tokens > context_len:
            num_filtered += 1
            continue
        
        input_lengths.append(prompt_tokens)
        output_lengths.append(output_tokens)
    
    print(f"  Loaded {len(input_lengths)} Arxiv samples (filtered: {num_filtered})")
    return input_lengths, output_lengths, num_filtered


def load_etalon_trace(
    trace_path: str,
    num_samples: int = None,
    context_len: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int], int]:
    """
    Load input and output token lengths from Etalon trace CSV file.
    
    Etalon trace format (CSV):
        num_prefill_tokens,num_decode_tokens,num_total_tokens,pd_ratio
    
    Available traces from https://github.com/project-etalon/etalon/tree/main/data/processed_traces:
        - arxiv_summarization_filtered_stats_llama2_tokenizer.csv
        - sharegpt_8k_filtered_stats_llama2_tokenizer.csv
        - lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv
        - bwb_stats_llama2_tokenizer_filtered_v2.csv
    
    Returns:
        Tuple of (input_lengths, output_lengths, num_filtered)
    """
    import csv
    
    # Check if it's a URL or local file
    if trace_path.startswith("http"):
        trace_path = download_and_cache_file(trace_path)
    
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Etalon trace file not found: {trace_path}")
    
    print(f"Loading Etalon trace from: {trace_path}")
    
    # Read all rows
    rows = []
    with open(trace_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Shuffle for random sampling
    random.seed(seed)
    random.shuffle(rows)
    
    # Extract input/output lengths
    input_lengths = []
    output_lengths = []
    num_filtered = 0
    
    for row in rows:
        if num_samples is not None and len(input_lengths) >= num_samples:
            break
        
        prefill_tokens = int(row['num_prefill_tokens'])
        decode_tokens = int(row['num_decode_tokens'])
        
        # Filter out very short samples
        if prefill_tokens < 2 or decode_tokens < 2:
            num_filtered += 1
            continue
        
        # Filter by context length (combined input + output)
        if context_len is not None and prefill_tokens + decode_tokens > context_len:
            num_filtered += 1
            continue
        
        input_lengths.append(prefill_tokens)
        output_lengths.append(decode_tokens)
    
    trace_name = os.path.basename(trace_path)
    print(f"  Loaded {len(input_lengths)} samples from {trace_name} (filtered: {num_filtered})")
    return input_lengths, output_lengths, num_filtered


def compute_stats(name: str, input_lengths: List[int], output_lengths: List[int]) -> DatasetStats:
    """Compute statistics from input and output lengths."""
    return DatasetStats(
        name=name,
        input_mean=np.mean(input_lengths),
        input_median=np.median(input_lengths),
        input_std=np.std(input_lengths),
        input_min=int(np.min(input_lengths)),
        input_max=int(np.max(input_lengths)),
        output_mean=np.mean(output_lengths),
        output_median=np.median(output_lengths),
        output_std=np.std(output_lengths),
        output_min=int(np.min(output_lengths)),
        output_max=int(np.max(output_lengths)),
        num_samples=len(input_lengths),
    )


def print_stats(stats: DatasetStats):
    """Pretty print statistics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {stats.name}")
    print(f"Number of samples: {stats.num_samples}")
    print(f"{'='*60}")
    
    print(f"\nInput Token Statistics:")
    print(f"  Mean:   {stats.input_mean:,.2f}")
    print(f"  Median: {stats.input_median:,.2f}")
    print(f"  Std:    {stats.input_std:,.2f}")
    print(f"  Min:    {stats.input_min:,}")
    print(f"  Max:    {stats.input_max:,}")
    
    print(f"\nOutput Token Statistics:")
    print(f"  Mean:   {stats.output_mean:,.2f}")
    print(f"  Median: {stats.output_median:,.2f}")
    print(f"  Std:    {stats.output_std:,.2f}")
    print(f"  Min:    {stats.output_min:,}")
    print(f"  Max:    {stats.output_max:,}")


def save_stats_to_json(stats_list: List[DatasetStats], output_path: str):
    """Save statistics to JSON file."""
    data = []
    for stats in stats_list:
        data.append({
            "dataset": stats.name,
            "num_samples": stats.num_samples,
            "input": {
                "mean": round(stats.input_mean, 2),
                "median": round(stats.input_median, 2),
                "std": round(stats.input_std, 2),
                "min": stats.input_min,
                "max": stats.input_max,
            },
            "output": {
                "mean": round(stats.output_mean, 2),
                "median": round(stats.output_median, 2),
                "std": round(stats.output_std, 2),
                "min": stats.output_min,
                "max": stats.output_max,
            }
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nStatistics saved to: {output_path}")


def print_comparison_table(stats_list: List[DatasetStats]):
    """Print a comparison table of all datasets."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    
    # Header
    header = f"{'Dataset':<15} {'Samples':>10} | {'Input Mean':>12} {'Input Med':>12} {'Input Std':>12} | {'Output Mean':>12} {'Output Med':>12} {'Output Std':>12}"
    print(header)
    print("-" * len(header))
    
    for stats in stats_list:
        row = f"{stats.name:<15} {stats.num_samples:>10,} | {stats.input_mean:>12,.1f} {stats.input_median:>12,.1f} {stats.input_std:>12,.1f} | {stats.output_mean:>12,.1f} {stats.output_median:>12,.1f} {stats.output_std:>12,.1f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics for arxiv and sharegpt datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path for tokenizer. Default: meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze per dataset. Default: all samples",
    )
    parser.add_argument(
        "--sharegpt-path",
        type=str,
        default=None,
        help="Path to local ShareGPT JSON file (will download if not provided)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path to save statistics",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["sharegpt", "arxiv"],
        choices=["sharegpt", "arxiv", "etalon"],
        help="Datasets to analyze. Default: sharegpt arxiv. Use 'etalon' with --etalon-trace.",
    )
    parser.add_argument(
        "--etalon-trace",
        type=str,
        default=None,
        help="Path or URL to Etalon trace CSV file. Required when using 'etalon' dataset.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Maximum context length (input + output). Samples exceeding this are filtered out.",
    )
    
    args = parser.parse_args()
    
    # Print filtering info
    if args.context_len:
        print(f"\nFiltering: context_len (input + output) <= {args.context_len:,} tokens")
    
    # Check if we need a tokenizer (etalon traces have pre-computed token counts)
    needs_tokenizer = "sharegpt" in args.datasets or "arxiv" in args.datasets
    
    # Initialize tokenizer if needed
    tokenizer = None
    if needs_tokenizer:
        if not HAS_TOKENIZER:
            raise ImportError(
                "transformers package is required for tokenization. "
                "Install with: pip install transformers"
            )
        print(f"Loading tokenizer: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    all_stats = []
    
    # Load and compute stats for ShareGPT
    if "sharegpt" in args.datasets:
        print("\n" + "="*60)
        print("Processing ShareGPT dataset...")
        print("="*60)
        sharegpt_inputs, sharegpt_outputs, _ = load_sharegpt_data(
            num_samples=args.num_samples,
            tokenizer=tokenizer,
            dataset_path=args.sharegpt_path,
            context_len=args.context_len,
            seed=args.seed,
        )
        sharegpt_stats = compute_stats("ShareGPT", sharegpt_inputs, sharegpt_outputs)
        print_stats(sharegpt_stats)
        all_stats.append(sharegpt_stats)
    
    # Load and compute stats for Arxiv
    if "arxiv" in args.datasets:
        print("\n" + "="*60)
        print("Processing Arxiv dataset...")
        print("="*60)
        arxiv_inputs, arxiv_outputs, _ = load_arxiv_data(
            num_samples=args.num_samples,
            tokenizer=tokenizer,
            context_len=args.context_len,
            seed=args.seed,
        )
        arxiv_stats = compute_stats("Arxiv", arxiv_inputs, arxiv_outputs)
        print_stats(arxiv_stats)
        all_stats.append(arxiv_stats)
    
    # Load and compute stats for Etalon trace
    if "etalon" in args.datasets:
        if args.etalon_trace is None:
            raise ValueError("--etalon-trace is required when using 'etalon' dataset")
        print("\n" + "="*60)
        print("Processing Etalon trace...")
        print("="*60)
        etalon_inputs, etalon_outputs, _ = load_etalon_trace(
            trace_path=args.etalon_trace,
            num_samples=args.num_samples,
            context_len=args.context_len,
            seed=args.seed,
        )
        trace_name = os.path.basename(args.etalon_trace).replace('.csv', '').replace('_', ' ').title()
        etalon_stats = compute_stats(f"Etalon ({trace_name})", etalon_inputs, etalon_outputs)
        print_stats(etalon_stats)
        all_stats.append(etalon_stats)
    
    # Print comparison table
    if len(all_stats) > 1:
        print_comparison_table(all_stats)
    
    # Save to JSON if requested
    if args.output:
        save_stats_to_json(all_stats, args.output)
    else:
        # Default output path
        script_dir = Path(__file__).parent
        default_output = script_dir / "dataset_stats.json"
        save_stats_to_json(all_stats, str(default_output))


if __name__ == "__main__":
    main()
