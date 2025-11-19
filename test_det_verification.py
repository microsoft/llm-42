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
        max_tokens=20,
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

def main():
    print("=" * 60)
    print("Deterministic Verification Test")
    print("=" * 60)
    
    try:
        # test_non_deterministic_request()
        test_deterministic_request()
        # test_multiple_requests()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the server is running on port 30000")

if __name__ == "__main__":
    main()
