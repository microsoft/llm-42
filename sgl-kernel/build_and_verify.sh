#!/bin/bash
set -ex
cd "$(dirname "${BASH_SOURCE[0]}")"
rm -rf build dist python/sgl_kernel/sm90/*.so
TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
mkdir -p build && cd build
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake .. -DCMAKE_PREFIX_PATH="$TORCH_PATH"
make common_ops_sm90_build -j$(nproc)
cp sm90/common_ops*.so sm90/common_ops.abi3.so
mkdir -p ../python/sgl_kernel/sm90 && ln -sf $(pwd)/sm90/common_ops.abi3.so ../python/sgl_kernel/sm90/
cd .. && python3 -c "import sgl_kernel; print('✓ sgl-kernel version:', sgl_kernel.__version__)"
