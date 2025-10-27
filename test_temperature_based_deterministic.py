#!/usr/bin/env python3
"""
Test script to demonstrate temperature-based dynamic batch-invariant mode.

This script demonstrates how the system now behaves:
1. When temperature == 0, the system uses batch-invariant (deterministic) mode
2. When temperature > 0, the system uses non-deterministic mode
3. In a batch, if ALL requests are non-deterministic (temp > 0), use default implementation
4. Otherwise (at least one request has temp == 0), use batch-invariant implementation

Usage:
    # Enable dynamic temperature-based mode (bit 512 = 512)
    # Combined with mode 1 (existing kernel): 512 + 1 = 513
    python -m sglang.launch_server \
        --model-path <model_path> \
        --enable-deterministic-inference 513

    # Or with mode 2 (CUDA kernel): 512 + 2 = 514
    python -m sglang.launch_server \
        --model-path <model_path> \
        --enable-deterministic-inference 514
"""

import requests
import json


def test_temperature_based_batching():
    """Test that temperature determines deterministic vs non-deterministic execution."""
    
    base_url = "http://localhost:30000"
    
    print("=" * 80)
    print("Testing Temperature-Based Dynamic Batch-Invariant Mode")
    print("=" * 80)
    
    # Test 1: Single request with temperature = 0 (deterministic)
    print("\nTest 1: Single request with temperature = 0 (should use batch-invariant)")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0.0,
            "max_tokens": 10,
        }
    )
    print(f"Response: {response.json()}")
    
    # Test 2: Single request with temperature > 0 (non-deterministic)
    print("\nTest 2: Single request with temperature > 0 (should use default implementation)")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0.8,
            "max_tokens": 10,
        }
    )
    print(f"Response: {response.json()}")
    
    # Test 3: Batch with mixed temperatures (should use batch-invariant)
    print("\nTest 3: Batch with mixed temperatures (should use batch-invariant)")
    # Note: This requires the batch API endpoint
    # For demonstration purposes only - actual implementation may vary
    
    # Test 4: Batch with all temperature > 0 (should use default)
    print("\nTest 4: Batch with all temperature > 0 (should use default implementation)")
    # Note: This requires the batch API endpoint
    
    print("\n" + "=" * 80)
    print("How it works:")
    print("=" * 80)
    print("1. Set --enable-deterministic-inference to 513 (or 514, 516, etc.)")
    print("   - Bit 512 (value 512) enables dynamic temperature-based mode")
    print("   - Remaining bits specify the deterministic mode (1, 2, 4, etc.)")
    print("   - Example: 513 = 512 (dynamic) + 1 (mode 1)")
    print("")
    print("2. When a batch is processed:")
    print("   - If ANY request has temperature == 0, use batch-invariant mode")
    print("   - If ALL requests have temperature > 0, use default (non-deterministic) mode")
    print("")
    print("3. Benefits:")
    print("   - Deterministic results when needed (temp == 0)")
    print("   - Non-deterministic sampling when desired (temp > 0)")
    print("   - Optimal performance: batch-invariant only when necessary")
    print("=" * 80)


if __name__ == "__main__":
    test_temperature_based_batching()
