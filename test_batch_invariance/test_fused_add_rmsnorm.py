"""
Test fusedAddRMS positional independence.

Tests that fused_add_rmsnorm produces identical outputs for fixed residual and 
hidden_states inputs placed at any position i within a batch of size n, 
regardless of random values in other positions.
"""

import torch
import sgl_kernel


def test_fused_add_rmsnorm_positional_independence():
    device = "cuda"
    dtype = torch.bfloat16
    hidden_size = 4096
    eps = 1e-6
    
    # Fixed seed for reproducible probe tensors
    torch.manual_seed(42)
    
    # Create fixed probe tensors (values in [-1, 1])
    probe_hidden = (torch.rand(1, hidden_size, dtype=dtype, device=device) * 2 - 1)
    probe_residual = (torch.rand(1, hidden_size, dtype=dtype, device=device) * 2 - 1)
    
    # Create fixed RMS weight (values in [-1, 1])
    weight = (torch.rand(hidden_size, dtype=dtype, device=device) * 2 - 1)
    
    print(f"Probe hidden (first 10): {probe_hidden[0][:10]}")
    print(f"Probe residual (first 10): {probe_residual[0][:10]}")
    print()
    
    # Test for n from 1 to 16384
    max_n = 32768
    
    for n in range(1, max_n + 1):
        all_hidden_outputs = []
        all_residual_outputs = []
        
        # For each n, run n iterations with probe at position i
        for i in range(n):
            # Different seed for random values in each run
            torch.manual_seed(1000 * n + i)
            
            # Create random tensors (values in [-1, 1])
            hidden_states = (torch.rand(n, hidden_size, dtype=dtype, device=device) * 2 - 1)
            residual = (torch.rand(n, hidden_size, dtype=dtype, device=device) * 2 - 1)
            
            # Place fixed probe at position i
            hidden_states[i] = probe_hidden[0].clone()
            residual[i] = probe_residual[0].clone()
            
            # Run kernel (in-place)
            sgl_kernel.fused_add_rmsnorm(hidden_states, residual, weight, eps)
            torch.cuda.synchronize()
            
            # Extract output at position i
            all_hidden_outputs.append(hidden_states[i].clone())
            all_residual_outputs.append(residual[i].clone())
        
        # Check all outputs are identical (compare with first output)
        passed = True
        failed_pos = -1
        
        if n > 0:
            first_hidden = all_hidden_outputs[0]
            first_residual = all_residual_outputs[0]
            
            for i in range(1, n):
                if not torch.equal(all_hidden_outputs[i], first_hidden):
                    passed = False
                    failed_pos = i
                    break
                if not torch.equal(all_residual_outputs[i], first_residual):
                    passed = False
                    failed_pos = i
                    break
        
        if passed:
            if n <= max_n:
                print(f"n={n}: PASSED ({n} iterations)")
                print(f"  First hidden (first 10): {first_hidden[:10]}")
                print(f"  First residual (first 10): {first_residual[:10]}")
        else:
            print(f"n={n}: FAILED at position {failed_pos}")
            print(f"  First hidden (first 10): {first_hidden[:10]}")
            print(f"  Got hidden (first 10):   {all_hidden_outputs[failed_pos][:10]}")
            print(f"  First residual (first 10): {first_residual[:10]}")
            print(f"  Got residual (first 10):   {all_residual_outputs[failed_pos][:10]}")
            
            # Compute diff
            hidden_diff = (all_hidden_outputs[failed_pos] - first_hidden).abs()
            residual_diff = (all_residual_outputs[failed_pos] - first_residual).abs()
            print(f"  Max hidden diff: {hidden_diff.max().item()}")
            print(f"  Max residual diff: {residual_diff.max().item()}")
            print(f"  Non-zero hidden diffs: {(hidden_diff > 0).sum().item()}")
            print(f"  Non-zero residual diffs: {(residual_diff > 0).sum().item()}")
            return False
    
    print(f"\nAll tests passed for n=1 to {max_n}!")
    return True


if __name__ == "__main__":
    test_fused_add_rmsnorm_positional_independence()
