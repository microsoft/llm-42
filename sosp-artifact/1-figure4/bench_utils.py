"""Shared utilities for matmul microbenchmarks."""

import os
import re
import sys

# Maps known GPU families to short names.
# Checked against torch.cuda.get_device_name() strings.
_GPU_SHORT_NAMES = [
    (r"\bB200\b", "b200"),
    (r"\bB100\b", "b100"),
    (r"\bH200\b", "h200"),
    (r"\bH100\b", "h100"),
    (r"\bH20\b", "h20"),
    (r"\bA100\b", "a100"),
    (r"\bA10G\b", "a10g"),
    (r"\bA10\b", "a10"),
    (r"\bL40S\b", "l40s"),
    (r"\bL40\b", "l40"),
    (r"\bL4\b", "l4"),
    (r"\bRTX\s*6000\s*Ada\b", "rtx6000ada"),
    (r"\bRTX\s*4090\b", "rtx4090"),
    (r"\bRTX\s*3090\b", "rtx3090"),
    (r"\bV100\b", "v100"),
    (r"\bT4\b", "t4"),
]


def get_short_gpu_name() -> str:
    """Return a short GPU name like 'a100', 'h100', 'b200' from CUDA device 0."""
    import torch

    full_name = torch.cuda.get_device_name(0)
    for pattern, short in _GPU_SHORT_NAMES:
        if re.search(pattern, full_name, re.IGNORECASE):
            return short
    # Fallback: lowercase, keep only alphanumerics, collapse underscores
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', full_name.lower())).strip('_')


def setup_run_dir(model_name: str) -> str:
    """Create and return the run output directory: runs/<gpu>/<model>/."""
    gpu = get_short_gpu_name()
    run_dir = os.path.join("runs", gpu, model_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class Tee:
    """Duplicate stdout to a file while preserving console output."""

    def __init__(self, filepath, mode='w'):
        self.file = open(filepath, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def tee_stdout(run_dir: str, filename: str = "results.txt"):
    """Redirect stdout to both console and a results file in *run_dir*."""
    sys.stdout = Tee(os.path.join(run_dir, filename))
