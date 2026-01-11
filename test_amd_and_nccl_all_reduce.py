"""
Test to confirm non-determinism of default NCCL all-reduce with batch size invariance.

This test uses the default torch.distributed.all_reduce (NCCL) which can be
NON-DETERMINISTIC due to tree-based reduction algorithms that don't guarantee
fixed accumulation order for bfloat16/float16.

This test compares:
1. Default all-reduce (same batch size) - should be DETERMINISTIC
2. Default all-reduce (different batch size) - typically NON-DETERMINISTIC for bfloat16
3. Default all-reduce (position invariance) - fixed batch size, permuted positions

Usage:
    python test_ar.py --test 1      # Run only test 1
    python test_ar.py --test 2      # Run only test 2
    python test_ar.py --test 3      # Run only test 3
    python test_ar.py --test 1 2 3  # Run tests 1, 2, and 3
    python test_ar.py               # Run all tests
"""

import argparse
import multiprocessing as mp
import socket

import torch
import torch.distributed as dist


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test1_same_batch_size(rank, device, num_trials, BS, hidden_dim):
    """TEST 1: Default all-reduce (same batch size) - should be DETERMINISTIC"""
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 1: Default NCCL all_reduce (same batch size)")
        print(f"{'='*70}")
    dist.barrier()

    for batch_size in range(1, BS + 1):
        if rank == 0:
            print(f"  Batch size: {batch_size}")
        dist.barrier()

        base_input = torch.empty((batch_size, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)

        dist.barrier()

        results_allreduce_only = []
        for trial in range(num_trials):
            # Clone the same input
            inp = base_input.clone()

            # Use default NCCL all-reduce
            dist.all_reduce(inp)
            torch.cuda.synchronize()

            # Only compare output corresponding to first request
            out_first_req = inp[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_allreduce_only.append((checksum, first_vals))

        # Check determinism
        if rank == 0:
            ref_sum, ref_vals = results_allreduce_only[0]
            all_match = True
            for i, (s, vals) in enumerate(results_allreduce_only[1:], 1):
                if not torch.equal(ref_vals, vals) or ref_sum != s:
                    all_match = False
                    print(f"    Trial {i+1} DIFFERS! ref_sum={ref_sum:.6f}, got={s:.6f}")

            if all_match:
                print(f"    ✓ DEFAULT ALL_REDUCE (fixed BS={batch_size}): DETERMINISTIC (as expected)")
            else:
                print(f"    ✗ DEFAULT ALL_REDUCE (fixed BS={batch_size}): NON-DETERMINISTIC (unexpected!)")

    dist.barrier()


def test2_different_batch_size(rank, device, num_trials, BS, hidden_dim):
    """TEST 2: Default all-reduce (different batch size) - typically NON-DETERMINISTIC"""
    base_input = torch.empty((1, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    base_input_rand = torch.empty((1, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)

    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 2: Default NCCL all_reduce (different batch size)")
        print("Batches: [a], [a,x], [a,x,x], ...")
        print(f"{'='*70}")
    dist.barrier()

    results_allreduce_only = {trial: [] for trial in range(num_trials)}
    for trial in range(num_trials):
        for bs in range(1, BS + 1):
            # Construct batch: (batch_size, hidden_dim)
            # First element is base_input, rest are base_input_rand
            batch = torch.stack([base_input] + [base_input_rand] * (bs - 1), dim=0)
            # Shape: (bs, hidden_dim)

            # Flatten for all-reduce: (bs * hidden_dim,)
            batch_flat = batch.view(-1)

            # Use default NCCL all-reduce
            dist.all_reduce(batch_flat)
            torch.cuda.synchronize()

            # Reshape back to (bs, hidden_dim)
            batch_out = batch_flat.view(bs, hidden_dim)

            # Only compare output corresponding to first request
            out_first_req = batch_out[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_allreduce_only[trial].append((bs, checksum, first_vals))

    # Check determinism
    if rank == 0:
        for trial in range(num_trials):
            results = results_allreduce_only[trial]

            _, ref_sum, ref_vals = results[0]
            all_match = True
            for _, s, vals in results[1:]:
                if abs(ref_sum - s) > 1e-3 or not torch.allclose(
                    ref_vals, vals, rtol=1e-3
                ):
                    all_match = False

        if all_match:
            print("  ✓ DEFAULT ALL_REDUCE (variant BS): DETERMINISTIC")
        else:
            print("  ✗ DEFAULT ALL_REDUCE (variant BS): NON-DETERMINISTIC")

    dist.barrier()


def test3_position_invariance(rank, device, num_trials, hidden_dim, fixed_batch_sizes=None):
    """TEST 3: Default all-reduce (position invariance) - fixed batch size, permuted positions"""
    if fixed_batch_sizes is None:
        fixed_batch_sizes = [8, 128, 256, 512]

    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 3: Default NCCL all_reduce (position invariance)")
        print("Fixed batch size, element 'a' at different positions")
        print("Batches: [a,x,x,...], [x,a,x,...], [x,x,a,...], ...")
        print(f"{'='*70}")
    dist.barrier()
    results_position_invariance = []
    for fixed_bs in fixed_batch_sizes:
        if rank == 0:
            print(f"\n  Testing batch size: {fixed_bs}")
        dist.barrier()

        # Create base inputs - different per rank
        torch.manual_seed(142 + rank)
        target_input = torch.empty((1, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        filler_input = torch.empty((fixed_bs, hidden_dim), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        dist.barrier()
        for pos in range(fixed_bs):
            # Construct batch: target element at position 'pos', rest are filler
            batch = filler_input.clone()
            batch[pos] = target_input.clone()
            # Shape: (fixed_bs, hidden_dim)
            torch.cuda.synchronize()

            # Use default NCCL all-reduce
            dist.all_reduce(batch)
            torch.cuda.synchronize()

            # Extract output corresponding to target element (at position 'pos')
            out_target = batch[pos].clone()
            checksum = out_target.sum().item()
            results_position_invariance.append((pos, checksum, out_target.clone()))

        # Check position invariance
        if rank == 0:
            all_trials_match = True
            _, ref_sum, ref_vals = results_position_invariance[0]
            for pos, s, vals in results_position_invariance[1:]:
                if ref_sum != s or not torch.equal(ref_vals, vals):
                    all_trials_match = False
                    print(f"    Position {pos} DIFFERS! ref_sum={ref_sum:.6f}, got={s:.6f}")

            if all_trials_match:
                print(f"    ✓ DEFAULT ALL_REDUCE (position invariance, BS={fixed_bs}): POSITION-INVARIANT")
            else:
                print(f"    ✗ DEFAULT ALL_REDUCE (position invariance, BS={fixed_bs}): NOT POSITION-INVARIANT")

        dist.barrier()


def worker(world_size, rank, port, tests_to_run):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    num_trials = 200

    # Matrix sizes similar to real model layers
    # Format: (batch_size, hidden_dim) - typical tensor shape for all-reduce
    BS = 32768  # max batch_size (1..BS)
    hidden_dim = 4096  # hidden dimension / intermediate dimension

    # Different seed per rank - each GPU has DIFFERENT input
    torch.manual_seed(42 + rank)

    # Run selected tests
    if 1 in tests_to_run:
        test1_same_batch_size(rank, device, num_trials, BS, hidden_dim)

    if 2 in tests_to_run:
        test2_different_batch_size(rank, device, num_trials, BS, hidden_dim)

    if 3 in tests_to_run:
        test3_position_invariance(rank, device, num_trials, hidden_dim, fixed_batch_sizes=[8, 128, 256, 512, 8192, 32768])

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="NCCL All-Reduce Determinism Test")
    parser.add_argument(
        "--test", "-t",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which tests to run (1, 2, 3, or combination). Default: all tests"
    )
    args = parser.parse_args()

    tests_to_run = set(args.test)

    world_size = 4
    available_gpus = torch.cuda.device_count()

    print("=" * 70)
    print("Default NCCL All-Reduce Determinism Test")
    print("=" * 70)
    print(f"Available GPUs: {available_gpus}")
    print(f"Using world_size: {world_size}")
    print(f"Running tests: {sorted(tests_to_run)}")

    if available_gpus < world_size:
        print(
            f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead"
        )
        world_size = available_gpus

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this test")
        return
    world_size = 2
    mp.set_start_method("spawn", force=True)
    port = get_open_port()

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(world_size, rank, port, tests_to_run))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()