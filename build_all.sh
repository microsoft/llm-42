#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}SGLang Build Script${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Check if CUDA is available
check_cuda() {
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        print_status "CUDA $CUDA_VERSION detected"
        export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        print_status "CUDA_HOME: $CUDA_HOME"
    else
        print_warning "CUDA not found. Building without CUDA support."
    fi
}

# Function to build sgl-kernel
build_sgl_kernel() {
    print_status "Building sgl-kernel..."
    cd "$SCRIPT_DIR/sgl-kernel"
    
    # Show PyTorch and CUDA info
    print_status "Checking PyTorch and CUDA versions..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" || true
    
    # Initialize submodules
    print_status "Initializing submodules..."
    git submodule update --init --recursive
    
    # Clean previous builds if requested
    if [ "$CLEAN_BUILD" = true ]; then
        print_status "Cleaning previous build artifacts..."
        rm -rf build dist *.egg-info
    fi
    
    # Install dependencies first
    print_status "Installing build dependencies..."
    pip install scikit-build-core uv
    pip install uvloop==0.21.0
    
    # Build wheel (similar to make build)
    print_status "Building sgl-kernel wheel..."
    rm -rf dist/* || true
    CMAKE_POLICY_VERSION_MINIMUM=3.5 MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32 \
        uv build --wheel -Cbuild-dir=build . --verbose --color=always --no-build-isolation
    
    # Install the built wheel
    print_status "Installing sgl-kernel wheel..."
    pip3 install dist/*.whl --force-reinstall --no-deps
    
    # Verify installation
    print_status "Verifying sgl-kernel installation..."
    if python -c "import sgl_kernel; print(f'sgl-kernel version: {sgl_kernel.__version__}')"; then
        print_status "Basic import successful!"
        
        # Try to import the actual ops that were failing
        print_status "Testing Flash Attention ops..."
        if python -c "from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache; print('Flash Attention ops loaded successfully!')" 2>&1; then
            print_status "sgl-kernel installed and verified successfully!"
        else
            print_warning "sgl-kernel imported but Flash Attention ops may have issues."
            print_warning "This might cause runtime errors. Check PyTorch/CUDA version compatibility."
        fi
    else
        print_error "Failed to import sgl-kernel!"
        print_error "This usually means the C++ extensions failed to compile."
        print_error "Try running with --clean to do a fresh rebuild."
        return 1
    fi
    echo ""
}

# Function to install sglang Python dependencies (including PyTorch)
# This must run BEFORE building sgl-kernel to ensure the kernel is compiled
# against the same PyTorch version that will be used at runtime.
install_sglang_deps() {
    print_status "Installing sglang Python dependencies (including PyTorch)..."
    cd "$SCRIPT_DIR"
    pip install -e "python[all]" --no-build-isolation 2>&1 | tail -5
    print_status "Verifying PyTorch version after dependency install..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" || true
    echo ""
}

# Function to build sglang
build_sglang() {
    print_status "Building sglang..."
    cd "$SCRIPT_DIR"
    
    # Install python dependencies
    print_status "Installing sglang in editable mode..."
    pip install -e "python[all]"
    
    # Verify installation
    print_status "Verifying sglang installation..."
    if python -c "import sglang; print(f'sglang version: {sglang.__version__}')"; then
        print_status "sglang installed successfully!"
    else
        print_error "Failed to import sglang!"
        return 1
    fi
    echo ""
}

# Main build process
main() {
    echo -e "${GREEN}Starting build process...${NC}"
    echo ""
    
    # Check CUDA
    check_cuda
    echo ""
    
    # Parse arguments
    BUILD_KERNEL=true
    BUILD_SGLANG=true
    CLEAN_BUILD=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --kernel-only)
                BUILD_SGLANG=false
                shift
                ;;
            --sglang-only)
                BUILD_KERNEL=false
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --kernel-only    Build only sgl-kernel"
                echo "  --sglang-only    Build only sglang"
                echo "  --clean          Clean build artifacts before building"
                echo "  -h, --help       Show this help message"
                echo ""
                echo "Default: Build both sgl-kernel and sglang"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Install sglang deps first to pin PyTorch version BEFORE compiling sgl-kernel.
    # This prevents ABI mismatches where sgl-kernel is compiled against one PyTorch
    # version but a different version is installed later by `pip install sglang[all]`.
    if [ "$BUILD_KERNEL" = true ] && [ "$BUILD_SGLANG" = true ]; then
        install_sglang_deps
    fi

    # Build components
    if [ "$BUILD_KERNEL" = true ]; then
        build_sgl_kernel
    fi
    
    if [ "$BUILD_SGLANG" = true ]; then
        build_sglang
    fi
    
    # Summary
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    
    if [ "$BUILD_KERNEL" = true ]; then
        echo "✓ sgl-kernel built"
    fi
    if [ "$BUILD_SGLANG" = true ]; then
        echo "✓ sglang installed"
    fi
    echo ""
    echo "You can now use SGLang:"
    echo "  python -m sglang.launch_server --model-path <model-path>"
    echo ""
}

# Run main function
main "$@"
