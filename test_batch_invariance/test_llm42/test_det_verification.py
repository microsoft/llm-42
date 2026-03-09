#!/usr/bin/env python3
"""
Test script for deterministic verification system.

This tests the following behavior:
1. Non-deterministic requests (is_deterministic=False) run normally
2. Deterministic requests (is_deterministic=True):
   - Run decode normally in non-det mode
   - When finished, verification is queued
   - During idle, verification re-runs in det-mode
   - Compares results and prints mismatches
"""

import openai
import time
import concurrent.futures
from typing import List

# Point to the local server
client = openai.Client(base_url="http://localhost:30000/v1", api_key="EMPTY")

def test_non_deterministic_request():
    """Test that non-deterministic requests run normally."""
    print("\n=== Testing Non-Deterministic Request ===")
    response = client.completions.create(
        model="default",
        prompt="Once upon a time",
        max_tokens=20,
        temperature=0.8,
        extra_body={"is_deterministic": False}
    )
    print(f"Non-det response: {response.choices[0].text}")
    print("✓ Non-deterministic request completed normally")

def test_deterministic_request():
    """Test that deterministic requests are verified."""
    print("\n=== Testing Deterministic Request ===")
    response = client.completions.create(
        model="default",
        prompt="Once upon a time",
        max_tokens=47,
        temperature=0.0,
        extra_body={"is_deterministic": True}
    )
    print(f"Det response: {response.choices[0].text}")
    print("✓ Deterministic request completed")
    print("⏳ Verification should happen during next idle period...")
    
    # Wait a bit for verification to happen
    time.sleep(2)
    print("✓ Check server logs for [VERIFICATION_*] messages")

def test_multiple_requests():
    """Test multiple requests with mixed deterministic/non-deterministic."""
    print("\n=== Testing Mixed Requests ===")
    
    # Send non-det request
    response1 = client.completions.create(
        model="default",
        prompt="The quick brown fox",
        max_tokens=15,
        temperature=0.8,
        extra_body={"is_deterministic": False}
    )
    print(f"1. Non-det: {response1.choices[0].text[:50]}")
    
    # Send det request
    response2 = client.completions.create(
        model="default",
        prompt="The quick brown fox",
        max_tokens=15,
        temperature=0.0,
        extra_body={"is_deterministic": True}
    )
    print(f"2. Det: {response2.choices[0].text[:50]}")
    
    # Send another det request
    response3 = client.completions.create(
        model="default",
        prompt="Hello world",
        max_tokens=15,
        temperature=0.0,
        extra_body={"is_deterministic": True}
    )
    print(f"3. Det: {response3.choices[0].text[:50]}")
    
    print("✓ All requests completed")
    print("⏳ Waiting for verifications...")
    time.sleep(3)
    print("✓ Check server logs for [VERIFICATION_*] messages")

def send_single_request(request_id: int, n: int, is_det: bool = True):
    """Send a single request and return the response."""
    print(f"→ Sending Request {request_id} ({'det' if is_det else 'non-det'})")
    print(f"Prompt: Request {request_id}: Once upon a time")
    try:
        prompt = f"Request {request_id}: Once upon a time"
        # if request_id == 2 or n == 1:
        #     prompt = f"Request 2: Once upon a time"
        response = client.completions.create(
            model="default",
            prompt=prompt,
            max_tokens=8,
            temperature=0.6,
            seed=42,
            extra_body={"is_deterministic": is_det}
        )
        result = response.choices[0].text[:50]
        print(f"✓ Request {request_id} ({'det' if is_det else 'non-det'}): {result}")
        return request_id, result
    except Exception as e:
        print(f"❌ Request {request_id} failed: {e}")
        return request_id, None

def test_concurrent_requests(n: int = 5, deterministic: bool = True):
    """Test n concurrent requests."""
    print(f"\n=== Testing {n} Concurrent {'Deterministic' if deterministic else 'Non-Deterministic'} Requests ===")
    
    start_time = time.time()
    
    # Send all requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(send_single_request, i, deterministic) for i in range(n)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ All {n} requests completed in {elapsed:.2f}s")
    

def test_mixed_concurrent_requests(n_det: int = 3, n_nondet: int = 2):
    """Test mixed deterministic and non-deterministic concurrent requests."""
    print(f"\n=== Testing {n_det} Det + {n_nondet} Non-Det Concurrent Requests ===")
    
    start_time = time.time()
    
    # Send all requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_det + n_nondet) as executor:
        futures = []
        # Send deterministic requests
        for i in range(n_det):
            futures.append(executor.submit(send_single_request, i, True))
        # Send non-deterministic requests
        for i in range(n_det, n_det + n_nondet):
            futures.append(executor.submit(send_single_request, i, False))
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ All {n_det + n_nondet} requests completed in {elapsed:.2f}s")
    print("⏳ Waiting for verifications to complete...")
    time.sleep(3)
    print("✓ Check server logs for [VERIFICATION_*] messages")

def main():
    print("=" * 60)
    print("Deterministic Verification Test")
    print("=" * 60)
    
    try:
        # Test different scenarios (uncomment as needed)
        
        # Single request tests
        # test_non_deterministic_request()
        # test_deterministic_request()
        # test_multiple_requests()
        
        # Concurrent request tests
        test_concurrent_requests(n=1, deterministic=True)  # 1 concurrent det request
        test_concurrent_requests(n=2, deterministic=True)  # 2 concurrent det requests
        test_concurrent_requests(n=4, deterministic=True)  # 4 concurrent det requests
        test_concurrent_requests(n=8, deterministic=True)  # 8 concurrent det requests
        test_concurrent_requests(n=16, deterministic=True)  # 16 concurrent det requests
        test_concurrent_requests(n=32, deterministic=True)  # 32 concurrent det requests
        # test_mixed_concurrent_requests(n_det=5, n_nondet=3)  # Mixed concurrent requests
        
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the server is running on port 30000")

if __name__ == "__main__":
    main()
