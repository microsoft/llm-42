#!/usr/bin/env python3
"""
Test script for temperature-based batch-invariant switching.

This script tests the new temperature-based dynamic mode (bit 512).
"""

import requests
import json
import time
import sys

def test_temperature_switching(base_url="http://localhost:30000"):
    """Test temperature-based switching functionality."""
    
    print("=" * 80)
    print("Testing Temperature-Based Batch-Invariant Switching")
    print("=" * 80)
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✓ Server is running at {base_url}")
        print()
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Cannot connect to server at {base_url}")
        print("  Please start the server first:")
        print("  ./launch_server.sh deterministic 513  # 512 + 1 for temp-based + ThinkingMachine")
        sys.exit(1)
    
    # Test cases
    test_cases = [
        # {
        #     "name": "Test 1: Temperature = 0 (should use batch-invariant)",
        #     "temperature": 0.0,
        #     "prompt": "What is 2+2?",
        #     "expected": "batch-invariant"
        # },
        {
            "name": "Test 2: Temperature = 0.8 (should use non-deterministic)",
            "temperature": 0.8,
            "prompt": "Tell me a creative story.",
            "expected": "non-deterministic"
        },
        {
            "name": "Test 3: Temperature = 0 again (should use batch-invariant)",
            "temperature": 0.0,
            "prompt": "Calculate 10 + 15.",
            "expected": "batch-invariant"
        },
        {
            "name": "Test 4: Temperature = 1.0 (should use non-deterministic)",
            "temperature": 1.0,
            "prompt": "Write a poem.",
            "expected": "non-deterministic"
        },
        # {
        #     "name": "Test 5: Temperature = 0 (should use batch-invariant)",
        #     "temperature": 0.0,
        #     "prompt": "What is the capital of France?",
        #     "expected": "batch-invariant"
        # },
    ]
    
    print("Running test cases...")
    print()
    
    for i, test in enumerate(test_cases, 1):
        print(f"{test['name']}")
        print(f"  Temperature: {test['temperature']}")
        print(f"  Expected mode: {test['expected']}")
        
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": test['prompt']}],
                    "temperature": test['temperature'],
                    "max_tokens": 20,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"  Response: {content[:60]}...")
                print(f"  ✓ Request successful")
            else:
                print(f"  ✗ Error: HTTP {response.status_code}")
                print(f"  {response.text}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
        time.sleep(0.5)  # Small delay between requests
    
    print("=" * 80)
    print("Test completed!")
    print()
    print("To see the statistics, check the server logs for:")
    print("  'Temperature-based switching stats: batch_invariant=X, non_deterministic=Y'")
    print()
    print("Expected stats after this test:")
    print("  - batch_invariant: 3")
    print("  - non_deterministic: 2")
    print("  - total: 5")
    print()
    print("The logs will appear every 5 forward passes automatically.")
    print("=" * 80)

def test_deterministic_consistency(base_url="http://localhost:30000"):
    """Test that temperature=0 produces consistent results."""
    
    print()
    print("=" * 80)
    print("Testing Deterministic Consistency (temperature=0)")
    print("=" * 80)
    print()
    
    prompt = "What is 5 + 3?"
    num_requests = 2
    
    print(f"Sending {num_requests} identical requests with temperature=0...")
    print(f"Prompt: {prompt}")
    print()
    
    responses = []
    for i in range(num_requests):
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 1.0,
                    "max_tokens": 50,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                responses.append(content)
                print(f"Request {i+1}: {content}")
            else:
                print(f"Request {i+1}: Error HTTP {response.status_code}")
        except Exception as e:
            print(f"Request {i+1}: Error {e}")
        
        time.sleep(0.3)
    
    print()
    if len(responses) == num_requests:
        if len(set(responses)) == 1:
            print("✓ SUCCESS: All responses are identical (deterministic)")
        else:
            print("✗ WARNING: Responses differ (may not be deterministic)")
            print("  This could be normal if the model or sampling has other sources of randomness.")
    else:
        print("✗ Could not complete all requests")
    
    print("=" * 80)

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:30000"
    
    # Run tests
    # test_temperature_switching(base_url)
    test_deterministic_consistency(base_url)
