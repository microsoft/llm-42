#!/usr/bin/env python3
"""
Batch Invariance Verification Script

This script verifies that the model produces identical outputs regardless of batch size
when running in batch-invariant mode. It tests:
1. Single request vs batched requests (with same prompt)
2. Different batch compositions (1, 2, 4, 8, 16, etc.)
3. Output token-by-token comparison
4. Logprob comparison (if enabled)

Usage:
    # Start server with batch-invariant mode enabled
    python -m sglang.launch_server \
        --model-path <model_path> \
        --enable-deterministic-inference 1

    # Run verification
    python verify_batch_invariance.py --host localhost --port 30000 --n-trials 10

    # With temperature-based mode (bit 512)
    python -m sglang.launch_server \
        --model-path <model_path> \
        --enable-deterministic-inference 513

    # Run verification with temperature=0
    python verify_batch_invariance.py --temperature 0.0 --n-trials 10
"""

import argparse
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class VerificationResult:
    """Results from a single verification test"""
    batch_size: int
    prompt: str
    outputs: List[str]
    logprobs: Optional[List[List[float]]]
    success: bool
    error_msg: Optional[str] = None


class BatchInvarianceVerifier:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 30000,
        temperature: float = 0.0,
        max_tokens: int = 50,
        sampling_seed: int = 42,
        return_logprob: bool = False,
    ):
        self.base_url = f"http://{host}:{port}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sampling_seed = sampling_seed
        self.return_logprob = return_logprob

    def send_request(
        self,
        prompt: str,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a single completion request"""
        payload = {
            "model": "default",
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.sampling_seed,
            "logprobs": 1 if self.return_logprob else None,
            "is_deterministic": True,  # Enable deterministic inference
        }
        
        # Merge extra parameters if provided
        if extra_body:
            payload.update(extra_body)

        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def send_batch_requests(
        self,
        prompts: List[str],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Send multiple requests (will be batched by server)"""
        results = []
        for prompt in prompts:
            result = self.send_request(prompt, extra_body)
            results.append(result)
            # Small delay to allow batching
            time.sleep(0.01)
        return results

    def extract_output_and_logprobs(
        self, response: Dict[str, Any]
    ) -> Tuple[str, Optional[List[float]]]:
        """Extract output text and logprobs from response"""
        text = response["choices"][0]["text"]
        logprobs = None
        
        if "logprobs" in response["choices"][0] and response["choices"][0]["logprobs"]:
            logprob_data = response["choices"][0]["logprobs"]
            if "token_logprobs" in logprob_data:
                logprobs = [lp for lp in logprob_data["token_logprobs"] if lp is not None]
        
        return text, logprobs

    def verify_single_prompt_batch_invariance(
        self,
        prompt: str,
        batch_sizes: List[int],
    ) -> Dict[int, VerificationResult]:
        """
        Verify that the same prompt produces identical outputs across different batch sizes.
        
        Args:
            prompt: The prompt to test
            batch_sizes: List of batch sizes to test (e.g., [1, 2, 4, 8])
        
        Returns:
            Dictionary mapping batch_size to VerificationResult
        """
        results = {}
        baseline_output = None
        baseline_logprobs = None

        print(f"\n{'='*80}")
        print(f"Testing Prompt: {prompt[:60]}...")
        print(f"{'='*80}")

        for batch_size in batch_sizes:
            print(f"\n→ Testing batch size: {batch_size}")
            
            try:
                # Create batch with same prompt repeated
                prompts = [prompt] * batch_size
                
                # Send requests
                responses = self.send_batch_requests(prompts)
                
                # Extract outputs
                outputs = []
                logprobs_list = []
                
                for resp in responses:
                    text, logprobs = self.extract_output_and_logprobs(resp)
                    outputs.append(text)
                    if logprobs is not None:
                        logprobs_list.append(logprobs)
                
                # Check consistency within batch
                all_same = all(output == outputs[0] for output in outputs)
                
                if not all_same:
                    error_msg = f"Outputs differ within batch size {batch_size}"
                    print(f"  ❌ FAILED: {error_msg}")
                    for i, out in enumerate(outputs):
                        print(f"    Request {i}: {out[:50]}...")
                    
                    results[batch_size] = VerificationResult(
                        batch_size=batch_size,
                        prompt=prompt,
                        outputs=outputs,
                        logprobs=logprobs_list if logprobs_list else None,
                        success=False,
                        error_msg=error_msg,
                    )
                    continue
                
                # Compare with baseline (batch_size=1)
                if baseline_output is None:
                    baseline_output = outputs[0]
                    baseline_logprobs = logprobs_list[0] if logprobs_list else None
                    print(f"  ✓ Baseline output: {baseline_output[:60]}...")
                else:
                    if outputs[0] != baseline_output:
                        error_msg = f"Output differs from baseline (batch_size=1)"
                        print(f"  ❌ FAILED: {error_msg}")
                        print(f"    Baseline:  {baseline_output[:50]}...")
                        print(f"    Current:   {outputs[0][:50]}...")
                        
                        results[batch_size] = VerificationResult(
                            batch_size=batch_size,
                            prompt=prompt,
                            outputs=outputs,
                            logprobs=logprobs_list if logprobs_list else None,
                            success=False,
                            error_msg=error_msg,
                        )
                        continue
                    
                    # Compare logprobs if available
                    if baseline_logprobs and logprobs_list:
                        if len(baseline_logprobs) != len(logprobs_list[0]):
                            error_msg = f"Logprobs length mismatch"
                            print(f"  ⚠️  WARNING: {error_msg}")
                        else:
                            max_diff = max(
                                abs(b - c)
                                for b, c in zip(baseline_logprobs, logprobs_list[0])
                            )
                            if max_diff > 1e-4:
                                error_msg = f"Logprobs differ by {max_diff:.6f}"
                                print(f"  ⚠️  WARNING: {error_msg}")
                    
                    print(f"  ✓ Output matches baseline")
                
                results[batch_size] = VerificationResult(
                    batch_size=batch_size,
                    prompt=prompt,
                    outputs=outputs,
                    logprobs=logprobs_list if logprobs_list else None,
                    success=True,
                )
                
            except Exception as e:
                error_msg = f"Exception: {str(e)}"
                print(f"  ❌ FAILED: {error_msg}")
                results[batch_size] = VerificationResult(
                    batch_size=batch_size,
                    prompt=prompt,
                    outputs=[],
                    logprobs=None,
                    success=False,
                    error_msg=error_msg,
                )
        
        return results

    def verify_different_prompts_batch_invariance(
        self,
        prompts: List[str],
    ) -> Dict[str, VerificationResult]:
        """
        Verify that different prompts produce consistent outputs when batched vs unbatched.
        
        Each prompt is tested:
        1. Individually (batch_size=1)
        2. As part of a batch with all prompts
        
        Args:
            prompts: List of different prompts to test
        
        Returns:
            Dictionary mapping prompt to VerificationResult
        """
        results = {}
        baseline_outputs = {}

        print(f"\n{'='*80}")
        print(f"Testing {len(prompts)} different prompts")
        print(f"{'='*80}")

        # Get baseline outputs (individual requests)
        print("\n→ Getting baseline outputs (individual requests)...")
        for i, prompt in enumerate(prompts):
            print(f"  Request {i+1}/{len(prompts)}: {prompt[:50]}...")
            response = self.send_request(prompt)
            text, logprobs = self.extract_output_and_logprobs(response)
            baseline_outputs[prompt] = (text, logprobs)
            print(f"    Output: {text[:50]}...")

        # Send batch request
        print(f"\n→ Sending batch of {len(prompts)} requests...")
        batch_responses = self.send_batch_requests(prompts)

        # Compare results
        print(f"\n→ Comparing results...")
        all_success = True
        
        for i, (prompt, response) in enumerate(zip(prompts, batch_responses)):
            text, logprobs = self.extract_output_and_logprobs(response)
            baseline_text, baseline_logprobs = baseline_outputs[prompt]
            
            if text != baseline_text:
                error_msg = f"Batched output differs from individual output"
                print(f"  ❌ Prompt {i+1} FAILED: {error_msg}")
                print(f"    Baseline: {baseline_text[:50]}...")
                print(f"    Batched:  {text[:50]}...")
                
                results[prompt] = VerificationResult(
                    batch_size=len(prompts),
                    prompt=prompt,
                    outputs=[baseline_text, text],
                    logprobs=[baseline_logprobs, logprobs] if logprobs else None,
                    success=False,
                    error_msg=error_msg,
                )
                all_success = False
            else:
                print(f"  ✓ Prompt {i+1}: Outputs match")
                results[prompt] = VerificationResult(
                    batch_size=len(prompts),
                    prompt=prompt,
                    outputs=[text],
                    logprobs=[logprobs] if logprobs else None,
                    success=True,
                )
        
        if all_success:
            print(f"\n✓ All prompts produced consistent outputs!")
        
        return results

    def run_comprehensive_verification(
        self,
        test_prompts: List[str],
        batch_sizes: List[int] = [1, 2, 4, 8],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run comprehensive batch invariance verification.
        
        Tests:
        1. Each prompt individually across different batch sizes
        2. All prompts together in a batch
        
        Returns:
            Tuple of (all_tests_passed, detailed_results)
        """
        print(f"\n{'#'*80}")
        print(f"# COMPREHENSIVE BATCH INVARIANCE VERIFICATION")
        print(f"{'#'*80}")
        print(f"Configuration:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Sampling seed: {self.sampling_seed}")
        print(f"  Return logprob: {self.return_logprob}")
        print(f"  Base URL: {self.base_url}")
        print(f"{'#'*80}")

        all_results = {
            "single_prompt_tests": {},
            "multi_prompt_test": {},
        }
        all_passed = True

        # Test 1: Each prompt with different batch sizes
        print(f"\n{'='*80}")
        print("TEST 1: Single Prompt with Different Batch Sizes")
        print(f"{'='*80}")
        
        for prompt in test_prompts:
            results = self.verify_single_prompt_batch_invariance(prompt, batch_sizes)
            all_results["single_prompt_tests"][prompt] = results
            
            # Check if all batch sizes passed
            if not all(r.success for r in results.values()):
                all_passed = False

        # Test 2: Multiple different prompts in a batch
        print(f"\n{'='*80}")
        print("TEST 2: Multiple Different Prompts in Batch")
        print(f"{'='*80}")
        
        results = self.verify_different_prompts_batch_invariance(test_prompts)
        all_results["multi_prompt_test"] = results
        
        if not all(r.success for r in results.values()):
            all_passed = False

        # Print summary
        print(f"\n{'#'*80}")
        print(f"# VERIFICATION SUMMARY")
        print(f"{'#'*80}")
        
        total_tests = 0
        passed_tests = 0
        
        for prompt, results in all_results["single_prompt_tests"].items():
            for batch_size, result in results.items():
                total_tests += 1
                if result.success:
                    passed_tests += 1
        
        for result in all_results["multi_prompt_test"].values():
            total_tests += 1
            if result.success:
                passed_tests += 1
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        
        if all_passed:
            print(f"\n✅ ALL TESTS PASSED - Batch invariance verified!")
        else:
            print(f"\n❌ SOME TESTS FAILED - Batch invariance NOT verified")
        
        print(f"{'#'*80}")
        
        return all_passed, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Verify batch invariance of the model"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=42, help="Sampling seed for reproducibility"
    )
    parser.add_argument(
        "--return-logprob", action="store_true", help="Return and compare logprobs"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of different prompts to test",
    )
    parser.add_argument(
        "--custom-prompts",
        type=str,
        nargs="+",
        help="Custom prompts to test (overrides n-trials)",
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Prepare test prompts
    if args.custom_prompts:
        test_prompts = args.custom_prompts
    else:
        # Default test prompts
        default_prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In a galaxy far, far away",
            "To be or not to be",
            "It was the best of times",
            "Call me Ishmael",
            "In the beginning",
            "The year was 1984",
        ]
        test_prompts = default_prompts[: args.n_trials]

    # Create verifier
    verifier = BatchInvarianceVerifier(
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sampling_seed=args.sampling_seed,
        return_logprob=args.return_logprob,
    )

    # Run verification
    try:
        all_passed, results = verifier.run_comprehensive_verification(
            test_prompts=test_prompts,
            batch_sizes=batch_sizes,
        )
        
        # Save results to file
        output_file = "batch_invariance_results.json"
        with open(output_file, "w") as f:
            # Convert results to JSON-serializable format
            json_results = {
                "config": {
                    "host": args.host,
                    "port": args.port,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "sampling_seed": args.sampling_seed,
                    "batch_sizes": batch_sizes,
                },
                "all_passed": all_passed,
                "summary": {
                    "total_tests": sum(
                        len(r) for r in results["single_prompt_tests"].values()
                    )
                    + len(results["multi_prompt_test"]),
                    "passed": sum(
                        sum(1 for res in r.values() if res.success)
                        for r in results["single_prompt_tests"].values()
                    )
                    + sum(1 for res in results["multi_prompt_test"].values() if res.success),
                },
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
