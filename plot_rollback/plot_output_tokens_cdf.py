#!/usr/bin/env python3
"""
Plot CDF of output tokens from ShareGPT dataset samples.

This script loads ShareGPT prompts and plots the distribution of 
expected output token lengths (from assistant responses in the dataset).

Usage:
    python plot_output_tokens_cdf.py --num-samples 1000 --output output_tokens_cdf.png
"""

import argparse
import json
import os
import random
from typing import List, Optional

import matplotlib.pyplot as plt
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


def download_and_cache_file(url: str, filename: Optional[str] = None) -> str:
    """Download and cache a file from URL."""
    import requests
    from tqdm import tqdm
    
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])
    
    if os.path.isfile(filename):
        print(f"Using cached file: {filename}")
        return filename
    
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


def load_sharegpt_output_lengths(
    num_samples: int,
    dataset_path: Optional[str] = None,
    max_prompt_len: int = 8192,
    max_output_len: int = 2048,
    tokenizer=None,
    seed: int = 42,
) -> List[int]:
    """
    Load output token lengths from ShareGPT dataset.
    
    Returns:
        List of output token lengths
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
        print(f"  Using char/4 heuristic for token counts")
    
    # Extract output lengths
    output_lengths = []
    prompt_lengths = []
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
                output_lengths.append(output_tokens)
                prompt_lengths.append(prompt_tokens)
    
    # Shuffle and take num_samples
    random.seed(seed)
    indices = list(range(len(output_lengths)))
    random.shuffle(indices)
    indices = indices[:num_samples]
    
    output_lengths = [output_lengths[i] for i in indices]
    prompt_lengths = [prompt_lengths[i] for i in indices]
    
    print(f"Loaded {len(output_lengths)} samples from ShareGPT")
    print(f"  Output tokens: min={min(output_lengths)}, max={max(output_lengths)}, "
          f"avg={np.mean(output_lengths):.1f}, median={np.median(output_lengths):.1f}")
    print(f"  Prompt tokens: min={min(prompt_lengths)}, max={max(prompt_lengths)}, "
          f"avg={np.mean(prompt_lengths):.1f}, median={np.median(prompt_lengths):.1f}")
    
    return output_lengths, prompt_lengths


def plot_cdf(data: List[int], title: str, xlabel: str, output_path: str, color: str = 'blue'):
    """Plot CDF of the data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_data, cdf, color=color, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add percentile annotations
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(sorted_data, p)
        ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'P{p}: {val:.0f}', xy=(val, p/100), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_cdf(output_lengths: List[int], prompt_lengths: List[int], output_path: str):
    """Plot combined CDF of output and prompt tokens."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Output tokens CDF
    sorted_output = np.sort(output_lengths)
    cdf_output = np.arange(1, len(sorted_output) + 1) / len(sorted_output)
    axes[0].plot(sorted_output, cdf_output, color='blue', linewidth=2)
    axes[0].set_xlabel('Output Tokens', fontsize=12)
    axes[0].set_ylabel('CDF', fontsize=12)
    axes[0].set_title(f'Output Token Distribution (n={len(output_lengths)})', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Add stats
    stats_text = f'Mean: {np.mean(output_lengths):.1f}\nMedian: {np.median(output_lengths):.1f}\nP95: {np.percentile(output_lengths, 95):.0f}\nP99: {np.percentile(output_lengths, 99):.0f}'
    axes[0].text(0.95, 0.05, stats_text, transform=axes[0].transAxes, 
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Prompt tokens CDF
    sorted_prompt = np.sort(prompt_lengths)
    cdf_prompt = np.arange(1, len(sorted_prompt) + 1) / len(sorted_prompt)
    axes[1].plot(sorted_prompt, cdf_prompt, color='green', linewidth=2)
    axes[1].set_xlabel('Prompt Tokens', fontsize=12)
    axes[1].set_ylabel('CDF', fontsize=12)
    axes[1].set_title(f'Prompt Token Distribution (n={len(prompt_lengths)})', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Add stats
    stats_text = f'Mean: {np.mean(prompt_lengths):.1f}\nMedian: {np.median(prompt_lengths):.1f}\nP95: {np.percentile(prompt_lengths, 95):.0f}\nP99: {np.percentile(prompt_lengths, 99):.0f}'
    axes[1].text(0.95, 0.05, stats_text, transform=axes[1].transAxes, 
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_histogram(output_lengths: List[int], prompt_lengths: List[int], output_path: str):
    """Plot histogram of token distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Output tokens histogram
    axes[0].hist(output_lengths, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Output Tokens', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Output Token Distribution (n={len(output_lengths)})', fontsize=14)
    axes[0].axvline(np.mean(output_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(output_lengths):.1f}')
    axes[0].axvline(np.median(output_lengths), color='orange', linestyle='--', label=f'Median: {np.median(output_lengths):.1f}')
    axes[0].legend()
    
    # Prompt tokens histogram
    axes[1].hist(prompt_lengths, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Prompt Tokens', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title(f'Prompt Token Distribution (n={len(prompt_lengths)})', fontsize=14)
    axes[1].axvline(np.mean(prompt_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(prompt_lengths):.1f}')
    axes[1].axvline(np.median(prompt_lengths), color='orange', linestyle='--', label=f'Median: {np.median(prompt_lengths):.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot CDF of output tokens from ShareGPT dataset")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to load")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to ShareGPT JSON file")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        help="Model name for tokenizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", type=str, default="output_tokens_cdf.png", 
                        help="Output file path")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--no-tokenizer", action="store_true", help="Don't use tokenizer, use char/4 heuristic")
    
    args = parser.parse_args()
    
    # Load tokenizer if available
    tokenizer = None
    if HAS_TOKENIZER and not args.no_tokenizer:
        try:
            print(f"Loading tokenizer: {args.model}")
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
    
    # Load output lengths
    output_lengths, prompt_lengths = load_sharegpt_output_lengths(
        num_samples=args.num_samples,
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot CDFs
    output_base = os.path.splitext(args.output)[0]
    
    plot_cdf(
        output_lengths, 
        f'CDF of Output Tokens (ShareGPT, n={len(output_lengths)})',
        'Output Tokens',
        os.path.join(args.output_dir, f'{output_base}_output.png'),
        color='blue'
    )
    
    plot_cdf(
        prompt_lengths,
        f'CDF of Prompt Tokens (ShareGPT, n={len(prompt_lengths)})',
        'Prompt Tokens', 
        os.path.join(args.output_dir, f'{output_base}_prompt.png'),
        color='green'
    )
    
    plot_combined_cdf(
        output_lengths, 
        prompt_lengths,
        os.path.join(args.output_dir, f'{output_base}_combined.png')
    )
    
    plot_histogram(
        output_lengths,
        prompt_lengths,
        os.path.join(args.output_dir, f'{output_base}_histogram.png')
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nOutput Tokens:")
    print(f"  Count:  {len(output_lengths)}")
    print(f"  Min:    {min(output_lengths)}")
    print(f"  Max:    {max(output_lengths)}")
    print(f"  Mean:   {np.mean(output_lengths):.2f}")
    print(f"  Median: {np.median(output_lengths):.2f}")
    print(f"  Std:    {np.std(output_lengths):.2f}")
    print(f"  P50:    {np.percentile(output_lengths, 50):.0f}")
    print(f"  P90:    {np.percentile(output_lengths, 90):.0f}")
    print(f"  P95:    {np.percentile(output_lengths, 95):.0f}")
    print(f"  P99:    {np.percentile(output_lengths, 99):.0f}")
    
    print(f"\nPrompt Tokens:")
    print(f"  Count:  {len(prompt_lengths)}")
    print(f"  Min:    {min(prompt_lengths)}")
    print(f"  Max:    {max(prompt_lengths)}")
    print(f"  Mean:   {np.mean(prompt_lengths):.2f}")
    print(f"  Median: {np.median(prompt_lengths):.2f}")
    print(f"  Std:    {np.std(prompt_lengths):.2f}")
    print(f"  P50:    {np.percentile(prompt_lengths, 50):.0f}")
    print(f"  P90:    {np.percentile(prompt_lengths, 90):.0f}")
    print(f"  P95:    {np.percentile(prompt_lengths, 95):.0f}")
    print(f"  P99:    {np.percentile(prompt_lengths, 99):.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
