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
    
    # Configuration: which positions should have temperature=0
    # You can modify this list to control which requests use temperature=0
    # For example: [0, 10, 20, 30, 40] means requests at positions 0, 10, 20, 30, 40 will use temp=0
    temp_zero_positions = [0, 10, 20, 30, 40]  # Modify this to control temperature=0 positions
    
    num_requests = 50
    interval = 0.1  # seconds between requests
    
    print(f"Configuration:")
    print(f"  Total requests: {num_requests}")
    print(f"  Interval: {interval} seconds")
    print(f"  Temperature=0 positions: {temp_zero_positions}")
    print(f"  Temperature>0 positions: All other positions")
    print()
    
    # Generate test cases
    test_cases = []
    prompts_temp_zero = [
        "What is 2+2?",
        "Calculate 10 + 15.",
        "What is the capital of France?",
        "What is 7 * 8?",
        "What is the square root of 16?",
    ]
    
    prompts_temp_high = [
        "Tell me a creative story.",
        "Write a poem.",
        "Describe a sunset.",
        "Invent a new recipe.",
        "Create a character.",
    ]
    
    for i in range(num_requests):
        if i in temp_zero_positions:
            temperature = 0.0
            prompt = prompts_temp_zero[i % len(prompts_temp_zero)]
            expected = "batch-invariant"
        else:
            temperature = 0.8
            prompt = prompts_temp_high[i % len(prompts_temp_high)]
            expected = "non-deterministic"
        
        test_cases.append({
            "name": f"Request {i+1}/{num_requests}: Temperature = {temperature}",
            "temperature": temperature,
            "prompt": prompt,
            "expected": expected
        })
    
    print("Running test cases...")
    print()
    
    start_time = time.time()
    successful_requests = 0
    batch_invariant_count = 0
    non_deterministic_count = 0
    
    for i, test in enumerate(test_cases):
        print(f"{test['name']}")
        print(f"  Temperature: {test['temperature']}")
        print(f"  Expected mode: {test['expected']}")
        
        request_start = time.time()
        
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
                successful_requests += 1
                
                if test['expected'] == "batch-invariant":
                    batch_invariant_count += 1
                else:
                    non_deterministic_count += 1
            else:
                print(f"  ✗ Error: HTTP {response.status_code}")
                print(f"  {response.text}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
        
        # Wait for the interval, accounting for request time
        elapsed = time.time() - request_start
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    total_time = time.time() - start_time
    
    print("=" * 80)
    print("Test completed!")
    print()
    print("Summary:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful requests: {successful_requests}")
    print(f"  Failed requests: {num_requests - successful_requests}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per request: {total_time/num_requests:.3f} seconds")
    print()
    print("Expected distribution:")
    print(f"  - batch_invariant (temp=0): {batch_invariant_count}")
    print(f"  - non_deterministic (temp>0): {non_deterministic_count}")
    print()
    print("To see the actual statistics, check the server logs for:")
    print("  'Temperature-based switching stats: batch_invariant=X, non_deterministic=Y'")
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
                    "temperature": 0.0,
                    "is_deterministic": False,
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
    #test_temperature_switching(base_url)
    test_deterministic_consistency(base_url)
