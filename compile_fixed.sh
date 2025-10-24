#!/bin/bash

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================"
echo "SGLang Compilation Setup Script"
echo "================================"
echo ""
echo "Note: This script requires root/sudo privileges for system package installation."
echo ""

# Check if running as root for apt-get commands
if [ "$EUID" -ne 0 ]; then 
    echo "Warning: Not running as root. apt-get commands will use sudo."
    SUDO="sudo"
else
    SUDO=""
fi

# Install Miniconda
echo "Installing Miniconda..."
mkdir -p ~/miniconda3
if [ ! -f ~/miniconda3/miniconda.sh ]; then
    echo "Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
fi
if [ ! -d ~/miniconda3/bin ]; then
    echo "Installing Miniconda..."
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    ~/miniconda3/condabin/conda init bash
fi

export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# Initialize conda for this shell session
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Create tokenweave environment
echo "Checking for tokenweave environment..."
if ! ~/miniconda3/bin/conda env list | grep -q "^tokenweave " 2>/dev/null; then
    echo "Creating tokenweave environment..."
    ~/miniconda3/bin/conda create -n tokenweave python=3.12 -y
else
    echo "tokenweave environment already exists"
fi

# Activate tokenweave environment
echo "Activating tokenweave environment..."
conda activate tokenweave

# Install PyTorch 2.8.0 with CUDA 12.8 support
echo "Installing PyTorch 2.8.0 with CUDA 12.8 support..."
pip install --no-cache-dir torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Deactivate conda temporarily for system package installation
echo "Deactivating conda for system package installation..."
conda deactivate || true

# Install system dependencies
echo "Installing basic system dependencies..."
$SUDO apt-get update
$SUDO apt-get install -y libnuma-dev libibverbs-dev pkg-config libssl-dev protobuf-compiler

echo "Installing Python 3.12 and related packages..."
$SUDO apt-get update
$SUDO apt-get install -y wget software-properties-common

# Check if deadsnakes PPA is already added
if ! grep -q "deadsnakes/ppa" /etc/apt/sources.list.d/*.list 2>/dev/null; then
    echo "Adding deadsnakes PPA..."
    # Temporarily unset conda paths to use system Python
    PATH_BACKUP="$PATH"
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    $SUDO add-apt-repository ppa:deadsnakes/ppa -y
    export PATH="$PATH_BACKUP"
else
    echo "deadsnakes PPA already added, skipping..."
fi

$SUDO apt-get update
$SUDO apt-get install -y python3.12-full python3.12-dev python3.10-venv

echo "Setting Python 3.12 as default..."
$SUDO update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
$SUDO update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
$SUDO update-alternatives --set python3 /usr/bin/python3.12

echo "Installing comprehensive system dependencies..."
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common netcat-openbsd kmod unzip openssh-server \
    curl wget lsof zsh ccache tmux htop git-lfs tree \
    build-essential cmake \
    libopenmpi-dev libnuma1 libnuma-dev \
    libibverbs-dev libibverbs1 libibumad3 \
    librdmacm1 libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
    ibverbs-providers infiniband-diags perftest \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libunwind-dev \
    libboost-all-dev libssl-dev \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev libcurl4-openssl-dev \
    libczmq4 libczmq-dev \
    libfabric-dev \
    patchelf \
    devscripts debhelper fakeroot dkms check libsubunit0 libsubunit-dev

# Try to install nvidia-dkms-550 (may fail if not available)
$SUDO apt-get install -y nvidia-dkms-550 || echo "Warning: nvidia-dkms-550 not available, skipping..."

$SUDO ln -sf /usr/bin/python3.12 /usr/bin/python
$SUDO rm -rf /var/lib/apt/lists/*
$SUDO apt-get clean

# Reactivate conda environment
echo "Reactivating conda environment..."
conda activate tokenweave

# GDRCopy installation
echo "Installing GDRCopy..."
if [ ! -d /tmp/gdrcopy ]; then
    mkdir -p /tmp/gdrcopy
    cd /tmp
    git clone https://github.com/NVIDIA/gdrcopy.git -b v2.4.4
    cd gdrcopy/packages
    CUDA=/usr/local/cuda ./build-deb-packages.sh
    $SUDO dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb || echo "Warning: GDRCopy installation failed, continuing..."
    cd /
    rm -rf /tmp/gdrcopy
else
    echo "GDRCopy already downloaded, skipping..."
fi

# Fix IBGDA symlink
echo "Creating IBGDA symlink..."
$SUDO ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so || echo "Warning: Could not create libmlx5.so symlink"

# Upgrade pip and install basic Python tools
echo "Upgrading pip and installing basic Python tools..."
python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel html5lib six

# Install nvidia-nccl
echo "Installing NVIDIA NCCL..."
python3 -m pip install --no-cache-dir nvidia-nccl-cu12==2.27.6 --force-reinstall --no-deps

# Install flashinfer and download cubin files
echo "Installing flashinfer..."
python3 -m pip install --no-cache-dir flashinfer

# Install Python development tools
echo "Installing Python development tools..."
python3 -m pip install --no-cache-dir \
    datamodel_code_generator \
    mooncake-transfer-engine==0.3.6.post1 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core \
    nixl \
    py-spy

# Install additional development tools and utilities
echo "Installing additional development tools..."
$SUDO apt-get update
$SUDO apt-get install -y \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    wget \
    curl \
    locales \
    lsof \
    git \
    git-lfs \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    unzip \
    pkg-config \
    libssl-dev \
    bear \
    ccache \
    less \
    rdma-core infiniband-diags openssh-server perftest \
    ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1

$SUDO rm -rf /var/lib/apt/lists/*
$SUDO apt-get clean

# Install build dependencies via pip
echo "Installing build dependencies..."
pip install --upgrade pip setuptools wheel ninja cmake scikit-build-core pybind11

# Build and install sgl-kernel from source
echo "Building sgl-kernel..."
if [ -d "$SCRIPT_DIR/sgl-kernel" ]; then
    cd "$SCRIPT_DIR/sgl-kernel"
    # Limit parallel jobs to avoid OOM during flash attention compilation
    # Compile only for compute capability 80 (A100) and 90 (H100)
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"
    export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    pip install -e . --no-build-isolation -vv
    
    # Manually copy/link the built .so files to the Python package directory
    echo "Linking compiled libraries to Python package..."
    if [ -d "build/sm90" ]; then
        mkdir -p python/sgl_kernel/sm90
        ln -sf "$SCRIPT_DIR/sgl-kernel/build/sm90/"*.so python/sgl_kernel/sm90/ 2>/dev/null || \
        cp -v build/sm90/*.so python/sgl_kernel/sm90/
    fi
    if [ -f "build/flash_ops.abi3.so" ]; then
        ln -sf "$SCRIPT_DIR/sgl-kernel/build/flash_ops.abi3.so" python/sgl_kernel/ 2>/dev/null || \
        cp -v build/flash_ops.abi3.so python/sgl_kernel/
    fi
    if [ -f "build/spatial_ops.abi3.so" ]; then
        ln -sf "$SCRIPT_DIR/sgl-kernel/build/spatial_ops.abi3.so" python/sgl_kernel/ 2>/dev/null || \
        cp -v build/spatial_ops.abi3.so python/sgl_kernel/
    fi
    
    cd "$SCRIPT_DIR"
    
    # Verify installation
    echo "Verifying sgl-kernel installation..."
    python3 -c "import sgl_kernel; print('sgl-kernel version:', sgl_kernel.__version__)"
else
    echo "Warning: sgl-kernel directory not found at $SCRIPT_DIR/sgl-kernel"
    echo "Skipping sgl-kernel installation."
fi

# Install remaining Python packages from python directory if it exists
if [ -d "$SCRIPT_DIR/python" ]; then
    echo "Installing Python package from $SCRIPT_DIR/python..."
    cd "$SCRIPT_DIR"
    python3 -m pip install --no-cache-dir -e "python[all]"
else
    echo "No python directory found, skipping..."
fi

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "To use this environment in the future, run:"
echo "  conda activate tokenweave"
echo ""
