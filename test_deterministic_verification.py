#!/usr/bin/env python3
"""
Test script to verify deterministic verification behavior.

This script tests that:
1. Non-deterministic requests (is_deterministic=False) run normally in non-det mode
2. Deterministic requests (is_deterministic=True):
   - First run decode in non-det mode
   - Then verify by running in det-mode
   - Update KV-cache in-place during det-mode
   - Print output_ids on mismatch
"""

import requests
import json
import time
from typing import Dict

SERVER_URL = "http://127.0.0.1:30000"

def send_request(prompt: str, is_deterministic: bool, max_tokens: int = 20, temperature: float = 0.0) -> Dict:
    """Send a completion request to the server."""
    url = f"{SERVER_URL}/v1/completions"
    
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "is_deterministic": is_deterministic,
        "stream": False
    }
    
    print(f"\n{'='*80}")
    print(f"Sending request:")
    print(f"  Prompt: {prompt[:50]}...")
    print(f"  is_deterministic: {is_deterministic}")
    print(f"  temperature: {temperature}")
    print(f"  max_tokens: {max_tokens}")
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        output = data['choices'][0]['text']
        print(f"\nResponse (took {elapsed:.2f}s):")
        print(f"  Output: {output[:100]}...")
        print(f"  Full output length: {len(output)}")
        return data
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def test_non_deterministic():
    """Test that non-deterministic requests work normally (no verification)."""
    print("\n" + "="*80)
    print("TEST 1: Non-Deterministic Requests (is_deterministic=False)")
    print("="*80)
    print("Expected behavior: Normal inference in non-det mode, no verification")
    
    # Send non-deterministic request with temperature > 0
    send_request(
        prompt="The capital of France is",
        is_deterministic=False,
        max_tokens=10,
        temperature=0.7
    )
    
    # Send another one
    send_request(
        prompt="Python is a programming language that",
        is_deterministic=False,
        max_tokens=10,
        temperature=0.8
    )

def test_deterministic():
    """Test that deterministic requests go through verification."""
    print("\n" + "="*80)
    print("TEST 2: Deterministic Requests (is_deterministic=True)")
    print("="*80)
    print("Expected behavior:")
    print("  1. Decode in non-det mode (token buffered, not committed)")
    print("  2. Verify by re-running in det-mode")
    print("  3. Update KV-cache in-place during verification")
    print("  4. Print output_ids on mismatch (check server logs)")
    
    # Send deterministic request with temperature=0 (greedy)
    send_request(
        prompt="The capital of France is",
        is_deterministic=True,
        max_tokens=15,
        temperature=0.0
    )
    
    # Send another one to test multiple tokens
    send_request(
        prompt="Machine learning is a field of",
        is_deterministic=True,
        max_tokens=15,
        temperature=0.0
    )
    
    # Test with a longer generation
    send_request(
        prompt="Deep neural networks are",
        is_deterministic=True,
        max_tokens=30,
        temperature=0.0
    )

def test_mixed():
    """Test mixed non-det and det requests."""
    print("\n" + "="*80)
    print("TEST 3: Mixed Non-Det and Det Requests")
    print("="*80)
    print("Expected behavior: Non-det runs normally, det goes through verification")
    
    # Interleave non-det and det requests
    send_request(
        prompt="Hello, how are",
        is_deterministic=False,
        max_tokens=10,
        temperature=0.5
    )
    
    send_request(
        prompt="Hello, how are",
        is_deterministic=True,
        max_tokens=10,
        temperature=0.0
    )
    
    send_request(
        prompt="The quick brown fox",
        is_deterministic=False,
        max_tokens=10,
        temperature=0.7
    )
    
    send_request(
        prompt="The quick brown fox",
        is_deterministic=True,
        max_tokens=10,
        temperature=0.0
    )

if __name__ == "__main__":
    print("="*80)
    print("Deterministic Verification Test Suite")
    print("="*80)
    print("\nNOTE: Check server logs for verification details and mismatch messages")
    print("      Server must be started with --enable-deterministic-inference")
    
    # Wait for server to be ready
    time.sleep(1)
    
    try:
        # Test 1: Non-deterministic requests
        test_non_deterministic()
        time.sleep(1)
        
        # Test 2: Deterministic requests  
        test_deterministic()
        time.sleep(1)
        
        # Test 3: Mixed requests
        test_mixed()
        
        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80)
        print("\nCheck server logs for:")
        print("  - 'Deterministic verification mismatch' warnings (if any)")
        print("  - Non-det vs det token comparisons")
        print("  - KV-cache updates")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nError during tests: {e}")
        import traceback
        traceback.print_exc()
