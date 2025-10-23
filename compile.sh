#!/bin/bash

set -e

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y libnuma-dev libibverbs-dev

# Create conda environment
conda create -n sglang python=3.12 -y | true
source $(conda info --base)/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)" && conda activate sglang

# Set CUDA environment variables
# Check if CUDA 12.8 is available, otherwise fall back to 13.0
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
    echo "Using CUDA 12.8 (matches PyTorch)"
elif [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME=/usr/local/cuda-12.4
    echo "Using CUDA 12.4 (compatible with PyTorch)"
else
    export CUDA_HOME=/usr/local/cuda-13.0
    echo "Using CUDA 13.0 (newer than PyTorch CUDA 12.8)"
    echo "Warning: This may cause compatibility issues with pre-built PyTorch"
fi

export CUDACXX=$CUDA_HOME/bin/nvcc
export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Allow CUDA version mismatch for flash-attn build
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export FORCE_CUDA="1"

# Verify CUDA is found
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDACXX: $CUDACXX"
echo "nvcc version:"
$CUDACXX --version

# Install build dependencies
pip install --upgrade pip setuptools wheel ninja cmake scikit-build-core pybind11

# Install PyTorch with CUDA 12.8 (compatible with CUDA 13.0 for most operations)
# Note: CUDA 13.0 is forward compatible with CUDA 12.x libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Note: flash-attn is NOT installed separately
# It is compiled as part of sgl-kernel build (see sgl-kernel/CMakeLists.txt)
echo "Skipping separate flash-attn installation (included in sgl-kernel)"
# Build and install sgl-kernel from source
echo "Building sgl-kernel..."
cd sgl-kernel
# Limit parallel jobs to avoid OOM during flash attention compilation
# Compile only for compute capability 80 (A100) and 90 (H100)
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="8.0;9.0" CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$CUDACXX -DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install -e . --no-build-isolation -vv
cd ..

# Build and install sgl-router from source
echo "Building sgl-router..."
cd sgl-router
pip install -e . --no-build-isolation -vv
cd ..

# Build and install main sglang package from source
echo "Building sglang..."
cd python
pip install -e ".[all]" --no-build-isolation -vv
cd ..