## CUDA Installation Issue & Fix for NanoOWL on Jetson Orin

### Current Problem:
- PyTorch 2.9.1+cpu (CPU-only) is installed
- CUDA 12.6 is available but PyTorch can't use it
- Need NVIDIA's Jetson-specific PyTorch build

### Solution Options:

#### Option 1: Reinstall NVIDIA PyTorch (Recommended for Jetson)
```bash
# Uninstall current PyTorch
pip3 uninstall torch torchvision functorch -y

# Install NVIDIA's PyTorch for Jetson (JP 6.x)
# Download from: https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Or use the version that was previously installed (check /opt/ or backup locations)
