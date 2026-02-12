# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
HTTP-based batch invariance test across multiple batch sizes and max_tokens.
Tests batch invariance by comparing consecutive batch sizes using ShareGPT dataset.

Usage:
    # Start server:
    bash launch_batch_invariance_test.sh
    
    # Run test with default settings:
    python vllm_online_batch_invariance_multitest.py --backend fa3
    
    # Run with custom batch sizes and async mode:
    python vllm_online_batch_invariance_multitest.py --backend fa3 --mode async --batch-sizes 57 103 305
    
    # Use custom ShareGPT file:
    python vllm_online_batch_invariance_multitest.py --backend fa3 --dataset /path/to/ShareGPT.json
    
    # Run with custom host/port:
    python vllm_online_batch_invariance_multitest.py --backend fa3 --host 127.0.0.1 --port 30005
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import aiohttp
import requests
from utils import BACKENDS, resolve_model_name


# ShareGPT dataset URL
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


@dataclass
class ShareGPTSample:
    """A single sample from ShareGPT dataset with prompt and expected output length."""
    prompt: str
    output_len: int
    prompt_len: int = 0


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
    if not dataset_path or not os.path.isfile(dataset_path):
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
    
    # Shuffle dataset for variety
    random.shuffle(dataset)
    
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
                    prompt_len=prompt_tokens
                ))
                
                if len(samples) >= num_prompts:
                    break
    
    if len(samples) < num_prompts:
        print(f"  Warning: Only found {len(samples)} valid samples (requested {num_prompts})")
    
    print(f"  Loaded {len(samples)} ShareGPT samples")
    if samples:
        avg_prompt = sum(s.prompt_len for s in samples) / len(samples)
        avg_output = sum(s.output_len for s in samples) / len(samples)
        print(f"  Average prompt length: {avg_prompt:.1f} tokens")
        print(f"  Average output length: {avg_output:.1f} tokens")
    
    return samples


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
                timeout=1800,
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


async def _request_completion_async(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    sp: dict[str, Any],
    max_retries: int = 3,
    retry_backoff: float = 0.5,
) -> dict[str, Any] | None:
    """Async version of _request_completion for concurrent individual requests."""
    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(sp)
    payload["is_deterministic"] = True

    for attempt in range(max_retries + 1):
        try:
            async with session.post(
                f"{base_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=1800),
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff * (2**attempt))
                continue
            return None
    return None


def _extract_tokens_and_logprobs(
    choice: dict[str, Any],
) -> tuple[list[Any], list[float] | None, str]:
    tokens: list[Any] = []
    token_logprobs: list[float] | None = None
    text: str = choice.get("text", "")
    lp = choice.get("logprobs")
    if lp and isinstance(lp, dict):
        tokens = lp.get("token_ids") or lp.get("tokens") or []
        token_logprobs = lp.get("token_logprobs", None)
    return tokens, token_logprobs, text


async def _compare_two_batch_sizes_async(
    samples: list[ShareGPTSample],
    sp_kwargs_list: list[dict[str, Any]],
    base_url: str,
    model_name: str,
    batch_size_1: int,
    batch_size_2: int,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Compare two batch sizes using async individual requests (for per-prompt max_tokens)."""
    compare_count = min(batch_size_1, batch_size_2)
    
    async with aiohttp.ClientSession() as session:
        # Send batch 1 requests concurrently
        tasks_1 = [
            _request_completion_async(
                session, base_url, model_name, 
                samples[i].prompt, sp_kwargs_list[i]
            )
            for i in range(batch_size_1)
        ]
        responses_1 = await asyncio.gather(*tasks_1)
        
        # Send batch 2 requests concurrently
        tasks_2 = [
            _request_completion_async(
                session, base_url, model_name,
                samples[i].prompt, sp_kwargs_list[i]
            )
            for i in range(batch_size_2)
        ]
        responses_2 = await asyncio.gather(*tasks_2)
    
    # Extract results for batch 1
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    bs1_texts: list[str] = []
    
    for i, resp in enumerate(responses_1[:compare_count]):
        if resp is None or not resp.get("choices"):
            return False, f"BS={batch_size_1} request {i} failed"
        choice = resp["choices"][0]
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            return False, f"BS={batch_size_1} missing logprobs for prompt {i}"
        bs1_tokens_per_prompt.append(list(toks))
        bs1_logprobs_per_prompt.append(list(lps))
        bs1_texts.append(text)
    
    # Extract results for batch 2
    bs2_tokens_per_prompt: list[list[Any]] = []
    bs2_logprobs_per_prompt: list[list[float] | None] = []
    bs2_texts: list[str] = []
    
    for i, resp in enumerate(responses_2[:compare_count]):
        if resp is None or not resp.get("choices"):
            return False, f"BS={batch_size_2} request {i} failed"
        choice = resp["choices"][0]
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            return False, f"BS={batch_size_2} missing logprobs for prompt {i}"
        bs2_tokens_per_prompt.append(list(toks))
        bs2_logprobs_per_prompt.append(list(lps))
        bs2_texts.append(text)
    
    # Compare results
    prompts = [s.prompt for s in samples[:compare_count]]
    for i, (tokens_bs1, tokens_bs2, logprobs_bs1, logprobs_bs2, text_bs1, text_bs2) in enumerate(
        zip(
            bs1_tokens_per_prompt,
            bs2_tokens_per_prompt,
            bs1_logprobs_per_prompt,
            bs2_logprobs_per_prompt,
            bs1_texts,
            bs2_texts,
        ), 1
    ):
        if tokens_bs1 != tokens_bs2:
            mismatch_pos = -1
            for pos, (t1, t2) in enumerate(zip(tokens_bs1, tokens_bs2)):
                if t1 != t2:
                    mismatch_pos = pos
                    break
            
            if mismatch_pos == -1 and len(tokens_bs1) != len(tokens_bs2):
                mismatch_pos = min(len(tokens_bs1), len(tokens_bs2))
            
            error_msg = (
                f"Prompt {i}: Different tokens sampled.\n"
                f"  Prompt: {repr(prompts[i-1][:80])}\n"
                f"  Mismatch at position: {mismatch_pos}\n"
                f"  Token lengths: BS={batch_size_1} has {len(tokens_bs1)} tokens, BS={batch_size_2} has {len(tokens_bs2)} tokens\n"
            )
            
            if mismatch_pos >= 0 and mismatch_pos < len(tokens_bs1) and mismatch_pos < len(tokens_bs2):
                error_msg += (
                    f"  BS={batch_size_1} token[{mismatch_pos}]: {tokens_bs1[mismatch_pos]}\n"
                    f"  BS={batch_size_2} token[{mismatch_pos}]: {tokens_bs2[mismatch_pos]}\n"
                )
            
            error_msg += (
                f"  BS={batch_size_1} output: {repr(text_bs1)}\n"
                f"  BS={batch_size_2} output: {repr(text_bs2)}"
            )
            return False, error_msg
        
        if logprobs_bs1 is None or logprobs_bs2 is None:
            return False, f"Prompt {i}: Missing logprobs in one of the runs"
        
        if len(logprobs_bs1) != len(logprobs_bs2):
            return False, (
                f"Prompt {i}: Different number of steps: "
                f"{len(logprobs_bs1)} (BS={batch_size_1}) vs {len(logprobs_bs2)} (BS={batch_size_2})"
            )
        
        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bs2)):
            if a != b:
                diff = abs(a - b)
                error_msg = (
                    f"Prompt {i} Step {t}: Bitwise logprob mismatch (abs diff={diff:.6e})\n"
                    f"  BS={batch_size_1}: {a:.6e}, BS={batch_size_2}: {b:.6e}"
                    f"  BS={batch_size_1} {logprobs_bs1=}\n"
                    f"  BS={batch_size_2} {logprobs_bs2=}"
                )
                if verbose:
                    error_msg += f"\n  Tokens: {tokens_bs1}"
                return False, error_msg
    
    return True, ""


def _compare_two_batch_sizes(
    samples: list[ShareGPTSample],
    sp_kwargs: dict[str, Any],
    base_url: str,
    model_name: str,
    batch_size_1: int,
    batch_size_2: int,
    verbose: bool = False,
    use_per_prompt_max_tokens: bool = False,
) -> tuple[bool, str]:
    """
    Compare two different batch sizes using the same prompts.
    Compares the first min(batch_size_1, batch_size_2) outputs.
    
    Returns:
        (success, error_message): True if test passes, False otherwise with error message
    """
    compare_count = min(batch_size_1, batch_size_2)
    
    # BS1: Get first batch output
    prompts_1 = [s.prompt for s in samples[:batch_size_1]]
    
    # Try per-prompt max_tokens if requested
    sp_kwargs_1 = sp_kwargs.copy()
    if use_per_prompt_max_tokens:
        sp_kwargs_1["max_tokens"] = [s.output_len + 10 for s in samples[:batch_size_1]]
    
    resp_1 = _request_completion(base_url, model_name, prompts_1, sp_kwargs_1, verbose=False)
    if resp_1 is None or not resp_1.get("choices"):
        return False, f"BS={batch_size_1} batched request failed or returned empty response"
    
    choices_1 = resp_1.get("choices", [])
    if len(choices_1) != batch_size_1:
        return False, f"BS={batch_size_1} returned {len(choices_1)} choices, expected {batch_size_1}"
    
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    bs1_texts: list[str] = []
    
    for idx, choice in enumerate(choices_1[:compare_count]):
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            return False, f"BS={batch_size_1} missing logprobs for prompt {idx}"
        bs1_tokens_per_prompt.append(list(toks))
        bs1_logprobs_per_prompt.append(list(lps))
        bs1_texts.append(text)

    # BS2: Get second batch output
    prompts_2 = [s.prompt for s in samples[:batch_size_2]]
    
    sp_kwargs_2 = sp_kwargs.copy()
    if use_per_prompt_max_tokens:
        sp_kwargs_2["max_tokens"] = [s.output_len + 10 for s in samples[:batch_size_2]]
    
    resp_2 = _request_completion(base_url, model_name, prompts_2, sp_kwargs_2, verbose=False)
    if resp_2 is None or not resp_2.get("choices"):
        return False, f"BS={batch_size_2} batched request failed or returned empty response"
    
    choices_2 = resp_2.get("choices", [])
    if len(choices_2) != batch_size_2:
        return False, f"BS={batch_size_2} returned {len(choices_2)} choices, expected {batch_size_2}"
    
    bs2_tokens_per_prompt: list[list[Any]] = []
    bs2_logprobs_per_prompt: list[list[float] | None] = []
    bs2_texts: list[str] = []
    
    for idx, choice in enumerate(choices_2[:compare_count]):
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            return False, f"BS={batch_size_2} missing logprobs for prompt {idx}"
        bs2_tokens_per_prompt.append(list(toks))
        bs2_logprobs_per_prompt.append(list(lps))
        bs2_texts.append(text)

    # Compare results (only first compare_count outputs)
    prompts = [s.prompt for s in samples[:compare_count]]
    for i, (tokens_bs1, tokens_bs2, logprobs_bs1, logprobs_bs2, text_bs1, text_bs2) in enumerate(
        zip(
            bs1_tokens_per_prompt,
            bs2_tokens_per_prompt,
            bs1_logprobs_per_prompt,
            bs2_logprobs_per_prompt,
            bs1_texts,
            bs2_texts,
        ), 1
    ):
        if tokens_bs1 != tokens_bs2:
            # Find first mismatch position
            mismatch_pos = -1
            for pos, (t1, t2) in enumerate(zip(tokens_bs1, tokens_bs2)):
                if t1 != t2:
                    mismatch_pos = pos
                    break
            
            if mismatch_pos == -1 and len(tokens_bs1) != len(tokens_bs2):
                mismatch_pos = min(len(tokens_bs1), len(tokens_bs2))
            
            error_msg = (
                f"Prompt {i}: Different tokens sampled.\n"
                f"  Prompt: {repr(prompts[i-1][:80])}\n"
                f"  Mismatch at position: {mismatch_pos}\n"
                f"  Token lengths: BS={batch_size_1} has {len(tokens_bs1)} tokens, BS={batch_size_2} has {len(tokens_bs2)} tokens\n"
            )
            
            if mismatch_pos >= 0 and mismatch_pos < len(tokens_bs1) and mismatch_pos < len(tokens_bs2):
                error_msg += (
                    f"  BS={batch_size_1} token[{mismatch_pos}]: {tokens_bs1[mismatch_pos]}\n"
                    f"  BS={batch_size_2} token[{mismatch_pos}]: {tokens_bs2[mismatch_pos]}\n"
                )
            
            error_msg += (
                f"  BS={batch_size_1} output: {repr(text_bs1)}\n"
                f"  BS={batch_size_2} output: {repr(text_bs2)}"
            )
            return False, error_msg
        
        if logprobs_bs1 is None or logprobs_bs2 is None:
            return False, f"Prompt {i}: Missing logprobs in one of the runs"
        
        if len(logprobs_bs1) != len(logprobs_bs2):
            return False, (
                f"Prompt {i}: Different number of steps: "
                f"{len(logprobs_bs1)} (BS={batch_size_1}) vs {len(logprobs_bs2)} (BS={batch_size_2})"
            )
        
        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bs2)):
            if a != b:
                diff = abs(a - b)
                error_msg = (
                    f"Prompt {i} Step {t}: Bitwise logprob mismatch (abs diff={diff:.6e})\n"
                    f"  BS={batch_size_1}: {a:.6e}, BS={batch_size_2}: {b:.6e}"
                    f"  BS={batch_size_1} {logprobs_bs1=}\n"
                    f"  BS={batch_size_2} {logprobs_bs2=}"
                )
                if verbose:
                    error_msg += f"\n  Tokens: {tokens_bs1}"
                return False, error_msg
    
    return True, ""


def test_multi_batch_invariance(
    backend: str,
    base_url: str,
    batch_sizes: list[int],
    max_tokens_list: list[int],
    n_prompts: int = 256,
    mode: str = "batch",
    dataset_path: Optional[str] = None,
    max_prompt_len: int = 16384,
    max_output_len: int = 2048,
    seed: int = 12345,
) -> None:
    """
    Test batch invariance across multiple batch sizes and max_tokens values.
    
    Args:
        backend: Attention backend name
        base_url: Server base URL
        batch_sizes: List of batch sizes to test
        max_tokens_list: List of max_tokens values to test
        n_prompts: Total number of prompts to generate (should be >= max(batch_sizes))
        mode: "batch" (batched requests) or "async" (async individual requests)
        dataset_path: Path to local ShareGPT JSON file (downloads if not provided)
        max_prompt_len: Maximum prompt length in tokens
        max_output_len: Maximum output length in tokens
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    model_name = resolve_model_name(backend)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=1800)
        if response.status_code != 200:
            raise RuntimeError(f"Server not responding at {base_url}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Server not running at {base_url}. "
            f"Start with launch_batch_invariance_test.sh"
        ) from e
    
    # Generate all prompts once
    max_batch_size = max(batch_sizes)
    if n_prompts < max_batch_size:
        n_prompts = max_batch_size
        print(f"Warning: Increasing n_prompts to {max_batch_size} to match largest batch size")
    
    print(f"\n{'='*80}")
    print(f"MULTI-BATCH INVARIANCE TEST")
    print(f"{'='*80}")
    print(f"Server: {base_url}")
    print(f"Backend: {backend}")
    print(f"Model: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max tokens: {max_tokens_list}")
    print(f"Total prompts needed: {n_prompts}")
    print(f"{'='*80}\n")
    
    # Try to load tokenizer for accurate token counting
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Tokenizer loaded: {tokenizer.name_or_path}")
    except Exception as e:
        print(f"  Warning: Could not load tokenizer ({e}), using char/4 heuristic")
    
    print("Loading ShareGPT prompts...")
    samples_all = load_sharegpt_prompts(
        num_prompts=n_prompts,
        dataset_path=dataset_path,
        max_prompt_len=max_prompt_len,
        max_output_len=max_output_len,
        tokenizer=tokenizer,
    )
    
    if len(samples_all) < n_prompts:
        raise RuntimeError(
            f"Not enough ShareGPT samples: needed {n_prompts}, got {len(samples_all)}"
        )
    
    print(f"Loaded {len(samples_all)} ShareGPT samples")
    print(f"{'='*80}\n")
    
    # Test results tracking (pairwise comparisons of consecutive batch sizes)
    total_tests = (len(batch_sizes) - 1) * len(max_tokens_list)
    passed_tests = 0
    failed_tests = 0
    results = []
    sample_idx = 0  # Track which samples we've used
    
    print(f"Running {total_tests} test configurations...\n")
    
    for max_tokens in max_tokens_list:
        print(f"\n{'─'*80}")
        print(f"Testing max_tokens = {max_tokens}")
        print(f"{'─'*80}")
        
        for i in range(len(batch_sizes) - 1):
            batch_size_1 = batch_sizes[i]
            batch_size_2 = batch_sizes[i + 1]
            test_name = f"BS={batch_size_1} vs BS={batch_size_2}, max_tokens={max_tokens}"
            print(f"  [{passed_tests + failed_tests + 1}/{total_tests}] {test_name}... ", end="", flush=True)
            
            sp_kwargs: dict[str, Any] = {
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "seed": 42,
                "logprobs": 1,
                "ignore_eos": True,
            }
            
            try:
                # Use enough samples for the larger batch size
                max_needed = max(batch_size_1, batch_size_2)
                start_idx = sample_idx % len(samples_all)
                end_idx = (sample_idx + max_needed) % len(samples_all)
                
                if end_idx > start_idx:
                    test_samples = samples_all[start_idx:end_idx]
                else:
                    # Handle wrap-around
                    test_samples = samples_all[start_idx:] + samples_all[:end_idx]
                
                # Ensure we have enough samples
                if len(test_samples) < max_needed:
                    test_samples = (samples_all * ((max_needed // len(samples_all)) + 1))[:max_needed]
                
                sample_idx += max_needed
                
                if mode == "async":
                    # Use async mode with per-prompt max_tokens
                    sp_kwargs_list = [
                        {
                            **sp_kwargs,
                            "max_tokens": s.output_len + 10
                        }
                        for s in test_samples
                    ]
                    success, error_msg = asyncio.run(_compare_two_batch_sizes_async(
                        samples=test_samples,
                        sp_kwargs_list=sp_kwargs_list,
                        base_url=base_url,
                        model_name=model_name,
                        batch_size_1=batch_size_1,
                        batch_size_2=batch_size_2,
                        verbose=False,
                    ))
                else:
                    # Use sync batched mode
                    success, error_msg = _compare_two_batch_sizes(
                        samples=test_samples,
                        sp_kwargs=sp_kwargs,
                        base_url=base_url,
                        model_name=model_name,
                        batch_size_1=batch_size_1,
                        batch_size_2=batch_size_2,
                        verbose=False,
                        use_per_prompt_max_tokens=False,
                    )
                
                if success:
                    print("✓ PASS")
                    passed_tests += 1
                    results.append((test_name, "PASS", ""))
                else:
                    print("✗ FAIL")
                    failed_tests += 1
                    results.append((test_name, "FAIL", error_msg))
                    if error_msg:
                        print(f"      Error: {error_msg}")
            
            except Exception as e:
                print("✗ ERROR")
                failed_tests += 1
                error_msg = str(e)
                results.append((test_name, "ERROR", error_msg))
                print(f"      Error: {error_msg}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Failed: {failed_tests} ({100*failed_tests/total_tests:.1f}%)")
    print(f"{'='*80}\n")
    
    if failed_tests > 0:
        print("Failed tests:")
        for test_name, status, error_msg in results:
            if status in ["FAIL", "ERROR"]:
                print(f"  ✗ {test_name}")
                if error_msg:
                    for line in error_msg.split('\n'):
                        print(f"      {line}")
        print()
    
    if failed_tests == 0:
        print(f"{'='*80}")
        print(f"✓✓✓ ALL TESTS PASSED ✓✓✓")
        print(f"{'='*80}\n")
    else:
        print(f"{'='*80}")
        print(f"✗✗✗ SOME TESTS FAILED ✗✗✗")
        print(f"{'='*80}\n")
        raise AssertionError(f"{failed_tests}/{total_tests} tests failed")


if __name__ == "__main__":
    """Run test standalone"""
    parser = argparse.ArgumentParser(
        description="HTTP-based batch invariance test across multiple batch sizes and max_tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python %(prog)s --backend fa3
  
  # Run with async mode and custom batch sizes
  python %(prog)s --backend fa3 --mode async --batch-sizes 57 103 305
  
  # Use custom ShareGPT file
  python %(prog)s --backend fa3 --dataset /path/to/ShareGPT.json
  
  # Run with custom host/port
  python %(prog)s --backend fa3 --host 127.0.0.1 --port 30005
        """
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=BACKENDS,
        help="Attention backend name"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30005,
        help="Server port (default: 30005)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[57, 89, 197],
        help="List of batch sizes to test (default: 57 89 197)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[35, 897, 1783],
        help="List of max_tokens values to test (default: 35 897 1783)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "async"],
        default="async",
        help="Test mode: 'batch' (batched requests) or 'async' (async individual requests with per-prompt max_tokens, default)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to local ShareGPT JSON file (downloads from HuggingFace if not provided)"
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=16384,
        help="Maximum prompt length in tokens (default: 16384)"
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=2048,
        help="Maximum output length in tokens (default: 2048)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MULTI-BATCH INVARIANCE TEST")
    print("="*80)
    
    base_url = f"http://{args.host}:{args.port}"
    n_prompts = max(args.batch_sizes) * 2  # Generate enough prompts for largest batch
    
    print(f"Server: {base_url}")
    print(f"Backend: {args.backend}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Dataset: ShareGPT")
    if args.dataset:
        print(f"Dataset path: {args.dataset}")
    print(f"Total prompts needed: {n_prompts}")
    print(f"Mode: {args.mode}")
    if args.mode == "async":
        print("  Using async individual requests with per-prompt max_tokens")
    else:
        print("  Using standard batched requests")
    print("="*80 + "\n")
    
    # Check server health
    print("Checking server health... ", end="", flush=True)
    try:
        response = requests.get(f"{base_url}/health", timeout=1800)
        if response.status_code != 200:
            print(f"FAILED - Server returned status {response.status_code}")
            sys.exit(1)
        print("✓")
    except requests.exceptions.RequestException as e:
        print(f"FAILED - {e}")
        print(f"\nError: Server not running at {base_url}")
        print("Start server with: bash launch_batch_invariance_test.sh")
        sys.exit(1)
    
    # Run test
    try:
        test_multi_batch_invariance(
            backend=args.backend,
            base_url=base_url,
            batch_sizes=args.batch_sizes,
            max_tokens_list=args.max_tokens,
            n_prompts=n_prompts,
            mode=args.mode,
            dataset_path=args.dataset,
            max_prompt_len=args.max_prompt_len,
            max_output_len=args.max_output_len,
            seed=args.seed,
        )
        sys.exit(0)
    except AssertionError as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗✗✗ TEST ERROR ✗✗✗")
        print(f"{'='*80}")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
