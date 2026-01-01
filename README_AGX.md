# NanoOWL-FT for NVIDIA Jetson AGX Nano 64GB - Optimized Implementation

## Project Summary

This project implements **real-time object detection** on NVIDIA Jetson AGX Nano 64GB using OWL-ViT (Open World Localization - Vision Transformer) with TensorRT FP16 optimization. The system achieves high-quality 720p video streaming with minimal latency using a USB camera.

### Key Achievements

✅ **TensorRT FP16 Optimization**: 7.81x speedup (2.76 → 21.56 FPS)  
✅ **Inference Performance**: 40ms per frame, 21.56 FPS capability  
✅ **Video Quality**: Full 720p (1280x720) HD streaming at 95% JPEG quality  
✅ **Stable Performance**: USB camera eliminates WiFi latency issues  
✅ **Real-time Detection**: Person, face, and hand detection with bounding boxes  
✅ **Web Interface**: Browser-based viewing with dynamic prompt configuration  

### Performance Metrics

- **TensorRT Engine Size**: 174.98 MiB (FP16 optimized)
- **Inference Time**: ~40ms per frame
- **Streaming FPS**: 14-15 FPS (camera-limited, not inference-limited)
- **Resolution**: 1280x720 (720p HD)
- **JPEG Quality**: 95% (default high quality)
- **Latency**: Minimal (<100ms end-to-end)

---

## Hardware Configuration

### NVIDIA Jetson AGX Nano 64GB
- **JetPack Version**: R36.4.7 (JP 6.2.1)
- **CUDA Version**: 12.6
- **Compute Capability**: 8.7
- **Unified Memory**: 61.37 GB
- **Architecture**: ARM64

### Camera
- **Model**: Microsoft LifeCam HD-3000
- **Connection**: USB 2.0
- **USB ID**: 045e:0810
- **Maximum Resolution**: 1280x720 (720p)
- **Frame Rate**: 30 FPS
- **Typical Streaming Performance**: 14-15 FPS (MJPEG over Flask)

### Network
- **Primary**: Ethernet at 192.168.2.239
- **Server Port**: 5000 (Flask development server)

---

## Software Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.10.12 | Runtime environment |
| **PyTorch** | 2.8.0 | Deep learning framework |
| **TensorRT** | 10.3.0 | Inference optimization |
| **CUDA** | 12.6 | GPU acceleration |
| **OpenCV** | 4.6.0.66 | Video capture and processing |
| **Flask** | 3.1.2 | Web server |
| **Pillow** | 12.0.0 | Image processing |
| **NumPy** | 1.26.4 | Numerical operations |
| **SciPy** | 1.8.0 | Scientific computing |
| **Matplotlib** | 3.5.1 | Visualization |
| **torchvision** | 0.23.0 | Computer vision utilities |
| **torch2trt** | 0.5.0 | PyTorch to TensorRT conversion |

### Additional Packages (Experimental)
| Package | Version | Notes |
|---------|---------|-------|
| **aiohttp** | 3.13.2 | Async HTTP (for WebRTC attempts) |
| **aiortc** | 1.14.0 | WebRTC implementation (experimental) |
| **av** | 16.0.1 | Media processing |

### Critical Configuration

**CUDA Allocator Fix for AGX Nano:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```
This environment variable is **required** for stable CUDA memory allocation on AGX Nano.

---

## Installation Guide

### Prerequisites
1. NVIDIA Jetson AGX Nano 64GB with JetPack 6.2.1 (R36.4.7) installed
2. USB camera (Microsoft LifeCam HD-3000 or compatible)
3. Internet connection for package installation

### Step 1: Clone Repository
```bash
cd ~/Downloads
git clone https://github.com/ulise75/nanoowlFT-AGX.git
cd nanoowlFT-AGX
```

### Step 2: Install System Dependencies
```bash
# Update package manager
sudo apt update

# Install system packages
sudo apt install -y python3-pip python3-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y v4l-utils  # Video4Linux utilities
```

### Step 3: Install Python Packages

**Core packages (required):**
```bash
pip3 install Flask==3.1.2
pip3 install pillow==12.0.0
pip3 install matplotlib==3.5.1
pip3 install opencv-python==4.6.0.66
```

**Note**: PyTorch, TensorRT, and torchvision should already be installed with JetPack. Verify with:
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"
```

### Step 4: Load USB Camera Driver
```bash
# Load UVC video driver
sudo modprobe uvcvideo

# Verify camera detected
ls -la /dev/video*
v4l2-ctl --list-devices
```

### Step 5: Set CUDA Environment Variable
Add to `~/.bashrc`:
```bash
echo 'export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync' >> ~/.bashrc
source ~/.bashrc
```

---

## Running the System

### Option 1: TensorRT Object Detection Server (Recommended)

**Start the server:**
```bash
cd ~/Downloads/nanoowlFT-AGX/examples
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python3 camera_web_server_trt.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --engine ../data/owl_image_encoder_patch32.engine
```

**Access the interface:**
- Open browser: `http://192.168.2.239:5000` (replace with your Jetson IP)
- Default detection prompts: `a person`, `a face`, `a hand`
- Adjust threshold and prompts via web interface

**Command-line options:**
```bash
--camera 0              # Camera device ID (0 for /dev/video0)
--width 1280            # Video width (720p)
--height 720            # Video height (720p)
--engine <path>         # Path to TensorRT engine file
--prompt "text"         # Detection prompts (comma-separated)
--threshold 0.15        # Detection confidence threshold (0.0-1.0)
--host 0.0.0.0         # Server host
--port 5000            # Server port
```

### Option 2: Simple Camera Stream (No Inference)

For testing camera quality without object detection:
```bash
cd ~/Downloads/nanoowlFT-AGX
python3 simple_camera_stream.py
```
Access at: `http://192.168.2.239:5000`

### Option 3: Background Service

Run as background service:
```bash
cd ~/Downloads/nanoowlFT-AGX/examples
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
nohup python3 camera_web_server_trt.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --engine ../data/owl_image_encoder_patch32.engine \
  > /tmp/trt_server.log 2>&1 &

# Check logs
tail -f /tmp/trt_server.log

# Stop server
pkill -f camera_web_server_trt
```

---

## Building TensorRT Engine (Optional)

If you need to rebuild the TensorRT engine:

```bash
cd ~/Downloads/nanoowlFT-AGX/examples
python3 build_trt_engine.py \
  --model google/owlvit-base-patch32 \
  --output ../data/owl_image_encoder_patch32_agx.engine \
  --fp16
```

**Note**: FP16 precision provides 7.81x speedup with minimal accuracy loss.

---

## Troubleshooting

### Issue: Camera not detected
```bash
# Check camera connection
lsusb | grep -i camera

# Reload driver
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo

# Check permissions
sudo usermod -aG video $USER
# Log out and back in
```

### Issue: CUDA out of memory
```bash
# Ensure CUDA allocator is set
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# Check memory usage
sudo tegrastats
```

### Issue: Low FPS performance
- **Camera bottleneck**: Microsoft LifeCam HD-3000 limited to 14-15 FPS over USB 2.0
- **Solution**: Use a higher-performance USB 3.0 camera for better FPS
- **Inference is NOT the bottleneck**: TensorRT runs at 40ms (25 FPS capable)

### Issue: Poor video quality
- Check JPEG quality setting (default is 95%)
- Ensure no downscaling in code (should process full 720p)
- Verify camera resolution: `v4l2-ctl --device=/dev/video0 --list-formats-ext`

### Issue: Server won't start
```bash
# Check if port is in use
sudo netstat -tulpn | grep :5000

# Kill existing processes
pkill -f camera_web_server
pkill -f simple_camera
```

---

## File Structure

```
nanoowlFT-AGX/
├── README_AGX.md                 # This file
├── data/
│   ├── owl_image_encoder_patch32.engine        # TensorRT FP16 engine (symlink)
│   └── owl_image_encoder_patch32_agx.engine   # Actual engine file
├── examples/
│   ├── camera_web_server_trt.py  # Main server (TensorRT optimized)
│   └── build_trt_engine.py       # Engine builder
├── simple_camera_stream.py       # Test camera without inference
├── MSCamera-Test.py              # Reference test script
├── test_camera_fps.py            # USB camera FPS test
└── test_rtsp_camera.py           # RTSP camera test (for WiFi cameras)
```

---

## Known Limitations

1. **Camera FPS**: Microsoft LifeCam HD-3000 limited to 14-15 FPS over USB 2.0/Flask
   - TensorRT inference capable of 21.56 FPS - camera is the bottleneck
   - Consider USB 3.0 camera for higher FPS

2. **WiFi Camera Issues**: RTSP cameras over WiFi experienced 2-3 second latency
   - Root cause: WiFi signal instability (-48 dBm, ping spikes up to 318ms)
   - Solution: Use USB cameras for stable, low-latency streaming

3. **Flask Server**: Development server - not recommended for production
   - Use production WSGI server (gunicorn, uWSGI) for deployment

4. **NumPy Warning**: SciPy 1.8.0 expects NumPy <1.25.0 but 1.26.4 is installed
   - Non-critical warning, does not affect functionality

---

## Future Improvements

- [ ] Implement hardware-accelerated GStreamer pipeline for higher FPS
- [ ] Add support for multiple camera sources
- [ ] Implement video recording functionality
- [ ] Add database logging for detection events
- [ ] Optimize for INT8 precision for even faster inference
- [ ] Deploy with production WSGI server
- [ ] Add REST API for programmatic access
- [ ] Implement motion detection to trigger inference only when needed

---

## Performance Comparison

| Configuration | Resolution | FPS | Inference Time | Notes |
|--------------|------------|-----|----------------|-------|
| **Baseline PyTorch** | 640x480 | 2.76 | ~360ms | Original model |
| **TensorRT FP16** | 640x480 | 21.56 | 40ms | 7.81x speedup |
| **Final Implementation** | 1280x720 | 14-15 | 40ms | Camera-limited |

---

## References

- NanoOWL Original: https://github.com/NVIDIA-AI-IOT/nanoowl
- OWL-ViT Model: https://huggingface.co/google/owlvit-base-patch32
- TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/
- Jetson AGX Nano: https://developer.nvidia.com/embedded/jetson-agx-orin

---

## Contributors

- Ulises (ulise75) - AGX Nano optimization and implementation

## License

Follow the original NanoOWL license terms.

---

## Support

For issues specific to AGX Nano implementation, please open an issue on:
https://github.com/ulise75/nanoowlFT-AGX/issues

---

**Last Updated**: January 1, 2026
