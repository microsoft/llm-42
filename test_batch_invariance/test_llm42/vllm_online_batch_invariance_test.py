# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
HTTP-based batch invariance test: send requests to a running
SGLang server and compare BS=1 vs BS=N results (tokens and per-step logprobs).

Environment variables:
  - SGLANG_TEST_MODEL: served model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)
  - SGLANG_TP_SIZE: tensor parallelism size (e.g., 4)

"""

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
    # Enable deterministic inference
    payload["is_deterministic"] = True

    for attempt in range(max_retries + 1):
        try:
            if verbose and attempt > 0:
                print(f"    Retry attempt {attempt}/{max_retries}...")
            response = requests.post(
                f"{base_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:  # pragma: no cover
            if attempt < max_retries:
                if verbose:
                    print(f"    Request failed: {e}, retrying...")
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


def _compare_bs1_vs_bsn_single_process(
    prompts: list[str],
    sp_kwargs: dict[str, Any],
    base_url: str,
    model_name: str,
) -> None:
    print(f"\n→ Phase 1: Testing BS=1 (single requests)")
    print(f"  Sending {len(prompts)} individual requests...")
    
    # BS=1
    bs1_tokens_per_prompt: list[list[Any]] = []
    bs1_logprobs_per_prompt: list[list[float] | None] = []
    bs1_texts: list[str] = []
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] Sending request... ", end="", flush=True)
        resp = _request_completion(base_url, model_name, p, sp_kwargs, verbose=False)
        if resp is None or not resp.get("choices"):
            print("FAILED")
            raise AssertionError("BS=1 empty/failed response")
        choice = resp["choices"][0]
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            print("FAILED - no logprobs")
            raise AssertionError(
                "logprobs not returned; ensure server supports 'logprobs'"
            )
        bs1_tokens_per_prompt.append(list(toks))
        bs1_logprobs_per_prompt.append(list(lps))
        bs1_texts.append(text)
        print(f"✓ (tokens: {len(toks)})")

    print(f"\n→ Phase 2: Testing BS={len(prompts)} (batched request)")
    print(f"  Sending batched request with {len(prompts)} prompts... ", end="", flush=True)
    
    # BS=N
    bsN_tokens_per_prompt: list[list[Any]] = [None] * len(prompts)  # type: ignore[list-item]
    bsN_logprobs_per_prompt: list[list[float] | None] = [None] * len(prompts)
    bsN_texts: list[str] = [None] * len(prompts)  # type: ignore[list-item]
    resp = _request_completion(base_url, model_name, prompts, sp_kwargs, verbose=True)
    if resp is None or not resp.get("choices"):
        print("FAILED")
        raise AssertionError("BS=N empty/failed batched response")
    choices = resp.get("choices", [])
    if len(choices) != len(prompts):
        print(f"FAILED - got {len(choices)} choices")
        raise AssertionError(
            f"BS=N choices length {len(choices)} != num prompts {len(prompts)}"
        )
    print(f"✓")
    print(f"  Processing {len(choices)} responses...")
    for idx, choice in enumerate(choices):
        toks, lps, text = _extract_tokens_and_logprobs(choice)
        if lps is None:
            raise AssertionError(f"BS=N missing logprobs for prompt {idx}")
        bsN_tokens_per_prompt[idx] = list(toks)
        bsN_logprobs_per_prompt[idx] = list(lps)
        bsN_texts[idx] = text
    print(f"  ✓ All responses processed")

    # compare
    print(f"\n→ Phase 3: Comparing BS=1 vs BS={len(prompts)} results")
    print(f"  Comparing {len(prompts)} prompt outputs...")
    
    for i, (tokens_bs1, tokens_bsN, logprobs_bs1, logprobs_bsN, text_bs1, text_bsN) in enumerate(
        zip(
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
            bs1_texts,
            bsN_texts,
        ), 1
    ):
        print(f"  [{i}/{len(prompts)}] Comparing prompt {i}... ", end="", flush=True)
        
        if tokens_bs1 != tokens_bsN:
            print(f"FAILED - tokens differ")
            print(f"\n  Prompt: {repr(prompts[i-1])}")
            print(f"  BS=1 output: {repr(text_bs1)}")
            print(f"  BS=N output: {repr(text_bsN)}")
            raise AssertionError(
                f"Prompt {i} (sampling): Different tokens sampled. "
                f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
            )
        
        # Print outputs even when they match
        print(f"✓ (tokens={len(tokens_bs1)})")
        print(f"    Prompt: {repr(prompts[i-1][:60])}{'...' if len(prompts[i-1]) > 60 else ''}")
        print(f"    Output: {repr(text_bs1)}")
        if logprobs_bs1 is None or logprobs_bsN is None:
            print(f"FAILED - missing logprobs")
            raise AssertionError(f"Prompt {i}: Missing logprobs in one of the runs")
        if len(logprobs_bs1) != len(logprobs_bsN):
            print(f"FAILED - different lengths")
            raise AssertionError(
                f"Prompt {i}: Different number of steps: "
                f"{len(logprobs_bs1)} (BS=1) vs {len(logprobs_bsN)} (BS=N)."
            )
        
        # Check logprobs match
        mismatch_found = False
        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
            print(f"    Step {t}: logprob BS=1: {a:.6e}, BS=N: {b:.6e}")
            if a != b:
                diff = abs(a - b)
                print(f"FAILED - logprob mismatch at step {t}")
                raise AssertionError(
                    f"Prompt {i} Step {t}: Bitwise mismatch "
                    f"(abs diff={diff:.6e}). "
                    f"BS=1 tokens: {tokens_bs1} BS=N tokens: {tokens_bsN}"
                )
        
        # Already printed above
        print(f"✓ (tokens={len(tokens_bs1)}, logprobs match)")
    
    print(f"  ✓ All {len(prompts)} prompts match perfectly!")


def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
    backend: str,
    base_url: str,
    n_prompts: int = 8,
) -> None:
    """Test batch invariance with a running SGLang server.
    
    This test assumes a server is already running (e.g., via launch_temperature_test.sh).
    The server should be started with --enable-llm-42 flag.
    
    Usage:
        # Start server:
        bash launch_batch_invariance_test.sh
        
        # Run test:
        python vllm_online_batch_invariance_test.py
    """
    random.seed(int(os.getenv("SGLANG_TEST_SEED", "12345")))
    model_name = resolve_model_name(backend)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"Server not responding at {base_url}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Server not running at {base_url}. Start with launch_batch_invariance_test.sh") from e
    
    prompts_all = [_random_prompt(10, 50) for _ in range(n_prompts)]

    sp_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        # "top_p": 1.0,
        "max_tokens": 8,
        "seed": 42,
        "logprobs": 5,
    }

    print(f"\n{'='*80}")
    print(f"Testing batch invariance for backend: {backend}")
    print(f"Server: {base_url}")
    print(f"Model: {model_name}")
    print(f"Number of prompts: {n_prompts}")
    print(f"{'='*80}\n")
    print(f"Prompts:")
    for i, prompt in enumerate(prompts_all, 1):
        print(f"  [{i}/{len(prompts_all)}] {prompt}")

    _compare_bs1_vs_bsn_single_process(
        prompts=prompts_all,
        sp_kwargs=sp_kwargs,
        base_url=base_url,
        model_name=model_name,
    )
    
    print(f"\n✓ Batch invariance test PASSED for backend: {backend}\n")


if __name__ == "__main__":
    """Run test standalone"""
    print("\n" + "="*80)
    print("BATCH INVARIANCE TEST")
    print("="*80)
    
    # Get configuration from environment
    host = os.getenv("SGLANG_HOST", "127.0.0.1")
    port = int(os.getenv("SGLANG_PORT", "30000"))
    base_url = f"http://{host}:{port}"
    backend = os.getenv("SGLANG_ATTENTION_BACKEND", "flashinfer")
    n_prompts = int(os.getenv("SGLANG_TEST_N_PROMPTS", "8"))
    
    print(f"Server: {base_url}")
    print(f"Backend: {backend}")
    print(f"Number of prompts: {n_prompts}")
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
    
    # Run test for the configured backend
    try:
        test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
            backend=backend,
            base_url=base_url,
            n_prompts=n_prompts,
        )
        print(f"\n{'='*80}")
        print(f"✓✓✓ BATCH INVARIANCE TEST PASSED ✓✓✓")
        print(f"{'='*80}\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n{'='*80}")
        print(f"✗✗✗ BATCH INVARIANCE TEST FAILED ✗✗✗")
        print(f"{'='*80}")
        print(f"Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗✗✗ TEST ERROR ✗✗✗")
        print(f"{'='*80}")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)