# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
HTTP-based batch invariance test with mixed deterministic/non-deterministic requests.
Tests BS=1 vs BS=N for various N and token lengths, with configurable mix of deterministic
and non-deterministic requests.

Environment variables:
  - SGLANG_TEST_MODEL: served model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)
  - SGLANG_TP_SIZE: tensor parallelism size (e.g., 4)
  - SGLANG_HOST: server host (default: 127.0.0.1)
  - SGLANG_PORT: server port (default: 30000)
  - SGLANG_ATTENTION_BACKEND: backend name (default: flashinfer)
  - SGLANG_TEST_SEED: random seed (default: 12345)

Usage:
    # Start server:
    bash launch_batch_invariance_test.sh
    
    # Run test with 70% deterministic, 30% non-deterministic:
    python vllm_online_batch_invariance_mixed_test.py --deterministic-pct 70
    
    # Run test with default 50/50 split:
    python vllm_online_batch_invariance_mixed_test.py
"""

import argparse
import os
import random
import sys
import time
from typing import Any

import requests
from utils import BACKENDS, _random_prompt, resolve_model_name


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


def _compare_bs1_vs_bsn(
    prompts: list[str],
    sp_kwargs: dict[str, Any],
    base_url: str,
    model_name: str,
    batch_size: int,
    deterministic_pct: float = 50.0,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Compare BS=1 vs BS=N for a given batch size with mixed deterministic/non-deterministic requests.
    
    Args:
        prompts: List of prompts to use
        sp_kwargs: Sampling parameters (base config)
        base_url: Server URL
        model_name: Model name
        batch_size: Batch size to test
        deterministic_pct: Percentage of requests that should be deterministic (0-100)
        verbose: Print verbose output
    
    Returns:
        (success, error_message): True if test passes, False otherwise with error message
    """
    # Determine which requests should be deterministic
    num_deterministic = int(batch_size * deterministic_pct / 100.0)
    num_non_deterministic = batch_size - num_deterministic
    
    # Create list indicating which requests are deterministic
    is_deterministic = [True] * num_deterministic + [False] * num_non_deterministic
    random.shuffle(is_deterministic)  # Randomize order
    
    if verbose:
        print(f"    Testing BS={batch_size}: {num_deterministic} deterministic, {num_non_deterministic} non-deterministic")
    
    # BS=1: Get reference outputs
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    bs1_texts: list[str] = []
    
    for i, (p, is_det) in enumerate(zip(prompts[:batch_size], is_deterministic), 1):
        sp = sp_kwargs.copy()
        sp["is_deterministic"] = is_det
        
        resp = _request_completion(base_url, model_name, p, sp, verbose=False)
        if resp is None or not resp.get("choices"):
            return False, f"BS=1 request {i} failed or returned empty response"
        choice = resp["choices"][0]
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            return False, f"BS=1 request {i} missing logprobs"
        bs1_tokens_per_prompt.append(list(toks))
        bs1_logprobs_per_prompt.append(list(lps))
        bs1_texts.append(text)

    # BS=N: Get batched output with mixed deterministic/non-deterministic
    # Use the new list-based is_deterministic API
    batch_prompts = prompts[:batch_size]
    
    sp_batch = sp_kwargs.copy()
    sp_batch["is_deterministic"] = is_deterministic
    
    resp = _request_completion(base_url, model_name, batch_prompts, sp_batch, verbose=False)
    if resp is None or not resp.get("choices"):
        return False, f"BS={batch_size} batched request failed or returned empty response"
    
    choices = resp.get("choices", [])
    if len(choices) != batch_size:
        return False, f"BS={batch_size} returned {len(choices)} choices, expected {batch_size}"
    
    bsN_tokens_per_prompt: list[list[Any]] = []
    bsN_logprobs_per_prompt: list[list[float] | None] = []
    bsN_texts: list[str] = []
    
    for idx, choice in enumerate(choices):
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            return False, f"BS={batch_size} missing logprobs for prompt {idx}"
        bsN_tokens_per_prompt.append(list(toks))
        bsN_logprobs_per_prompt.append(list(lps))
        bsN_texts.append(text)

    # Compare results - only deterministic requests should match exactly
    for i, (tokens_bs1, tokens_bsN, logprobs_bs1, logprobs_bsN, text_bs1, text_bsN, is_det) in enumerate(
        zip(
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
            bs1_texts,
            bsN_texts,
            is_deterministic,
        ), 1
    ):
        if is_det:
            # Deterministic requests must match exactly
            if tokens_bs1 != tokens_bsN:
                error_msg = (
                    f"Prompt {i} (DETERMINISTIC): Different tokens sampled.\n"
                    f"  Prompt: {repr(batch_prompts[i-1][:80])}\n"
                    f"  BS=1 output: {repr(text_bs1)}\n"
                    f"  BS={batch_size} output: {repr(text_bsN)}"
                )
                return False, error_msg
            
            if logprobs_bs1 is None or logprobs_bsN is None:
                return False, f"Prompt {i} (DETERMINISTIC): Missing logprobs in one of the runs"
            
            if len(logprobs_bs1) != len(logprobs_bsN):
                return False, (
                    f"Prompt {i} (DETERMINISTIC): Different number of steps: "
                    f"{len(logprobs_bs1)} (BS=1) vs {len(logprobs_bsN)} (BS={batch_size})"
                )
            
            for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
                if a != b:
                    diff = abs(a - b)
                    error_msg = (
                        f"Prompt {i} (DETERMINISTIC) Step {t}: Bitwise logprob mismatch (abs diff={diff:.6e})\n"
                        f"  BS=1: {a:.6e}, BS={batch_size}: {b:.6e}"
                    )
                    if verbose:
                        error_msg += f"\n  Tokens: {tokens_bs1}"
                    return False, error_msg
        else:
            # Non-deterministic requests - just verify we got valid output
            if not tokens_bsN:
                return False, f"Prompt {i} (NON-DETERMINISTIC): Empty output in BS={batch_size}"
            if logprobs_bsN is None or len(logprobs_bsN) == 0:
                return False, f"Prompt {i} (NON-DETERMINISTIC): Missing logprobs in BS={batch_size}"
            
            # We expect outputs may differ, so just log this for information
            if verbose and tokens_bs1 != tokens_bsN:
                print(f"      Prompt {i} (NON-DETERMINISTIC): Outputs differ as expected")
    
    return True, ""


def test_multi_batch_invariance(
    backend: str,
    base_url: str,
    batch_sizes: list[int],
    max_tokens_list: list[int],
    deterministic_pct: float = 50.0,
    n_prompts: int = 256,
) -> None:
    """
    Test batch invariance across multiple batch sizes and max_tokens values with mixed
    deterministic and non-deterministic requests.
    
    Args:
        backend: Attention backend name
        base_url: Server base URL
        batch_sizes: List of batch sizes to test
        max_tokens_list: List of max_tokens values to test
        deterministic_pct: Percentage of requests that should be deterministic (0-100)
        n_prompts: Total number of prompts to generate (should be >= max(batch_sizes))
    """
    random.seed(int(os.getenv("SGLANG_TEST_SEED", "12345")))
    model_name = resolve_model_name(backend)
    
    # Validate deterministic_pct
    if not 0 <= deterministic_pct <= 100:
        raise ValueError(f"deterministic_pct must be between 0 and 100, got {deterministic_pct}")
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
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
    print(f"MIXED DETERMINISTIC/NON-DETERMINISTIC BATCH INVARIANCE TEST")
    print(f"{'='*80}")
    print(f"Server: {base_url}")
    print(f"Backend: {backend}")
    print(f"Model: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max tokens: {max_tokens_list}")
    print(f"Deterministic: {deterministic_pct:.0f}% | Non-deterministic: {100-deterministic_pct:.0f}%")
    print(f"Total prompts generated: {n_prompts}")
    print(f"{'='*80}\n")
    
    prompts_all = [_random_prompt(10, 50) for _ in range(n_prompts)]
    
    # Test results tracking
    total_tests = len(batch_sizes) * len(max_tokens_list)
    passed_tests = 0
    failed_tests = 0
    results = []
    
    print(f"Running {total_tests} test configurations...\n")
    
    for max_tokens in max_tokens_list:
        print(f"\n{'─'*80}")
        print(f"Testing max_tokens = {max_tokens}")
        print(f"{'─'*80}")
        
        for batch_size in batch_sizes:
            num_det = int(batch_size * deterministic_pct / 100.0)
            num_non_det = batch_size - num_det
            test_name = f"BS={batch_size} ({num_det}D/{num_non_det}ND), max_tokens={max_tokens}"
            print(f"  [{passed_tests + failed_tests + 1}/{total_tests}] {test_name}... ", end="", flush=True)
            
            sp_kwargs: dict[str, Any] = {
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "seed": 42,
                "logprobs": 5,
            }
            
            try:
                success, error_msg = _compare_bs1_vs_bsn(
                    prompts=prompts_all,
                    sp_kwargs=sp_kwargs,
                    base_url=base_url,
                    model_name=model_name,
                    batch_size=batch_size,
                    deterministic_pct=deterministic_pct,
                    verbose=False,
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
    print(f"Deterministic ratio: {deterministic_pct:.0f}% / {100-deterministic_pct:.0f}%")
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
        description="Batch invariance test with mixed deterministic/non-deterministic requests"
    )
    parser.add_argument(
        "--deterministic-pct",
        type=float,
        default=50.0,
        help="Percentage of requests that should be deterministic (0-100, default: 50)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[2, 8, 32],
        help="List of batch sizes to test (default: 2 8 32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[8, 96],
        help="List of max_tokens values to test (default: 8 96)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (default: from SGLANG_HOST env or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: from SGLANG_PORT env or 30000)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Attention backend (default: from SGLANG_ATTENTION_BACKEND env or flashinfer)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MIXED DETERMINISTIC/NON-DETERMINISTIC BATCH INVARIANCE TEST")
    print("="*80)
    
    # Get configuration from args or environment
    host = args.host or os.getenv("SGLANG_HOST", "127.0.0.1")
    port = args.port or int(os.getenv("SGLANG_PORT", "30000"))
    base_url = f"http://{host}:{port}"
    backend = args.backend or os.getenv("SGLANG_ATTENTION_BACKEND", "flashinfer")
    deterministic_pct = args.deterministic_pct
    batch_sizes = args.batch_sizes
    max_tokens_list = args.max_tokens
    
    n_prompts = max(batch_sizes) * 2  # Generate enough prompts for largest batch
    
    print(f"Server: {base_url}")
    print(f"Backend: {backend}")
    print(f"Deterministic: {deterministic_pct:.0f}% | Non-deterministic: {100-deterministic_pct:.0f}%")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max tokens: {max_tokens_list}")
    print(f"Total prompts: {n_prompts}")
    print("="*80 + "\n")
    
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
        print("Start server with: bash launch_batch_invariance_test.sh")
        sys.exit(1)
    
    # Validate deterministic_pct
    if not 0 <= deterministic_pct <= 100:
        print(f"Error: --deterministic-pct must be between 0 and 100, got {deterministic_pct}")
        sys.exit(1)
    
    # Run test
    try:
        test_multi_batch_invariance(
            backend=backend,
            base_url=base_url,
            batch_sizes=batch_sizes,
            max_tokens_list=max_tokens_list,
            deterministic_pct=deterministic_pct,
            n_prompts=n_prompts,
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
