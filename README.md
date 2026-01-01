<h1 align="center">NanoOWL FT (Full-Featured)</h1>

<p align="center"><a href="#features"/>✨ Features</a> - <a href="#usage"/>👍 Usage</a> - <a href="#web-interface"/>🌐 Web Interface</a> - <a href="#performance"/>⏱️ Performance</a> - <a href="#setup">🛠️ Setup</a> - <a href="#examples">🤸 Examples</a> <br> - <a href="#acknowledgement">👏 Acknowledgment</a> - <a href="#see-also">🔗 See also</a></p>

NanoOWL FT is an enhanced fork of [NVIDIA's NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) that optimizes [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) to run 🔥 ***real-time*** 🔥 on [NVIDIA Jetson Orin Platforms](https://store.nvidia.com/en-us/jetson/store) with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt). This fork adds USB camera support, an interactive web interface, and remote access capabilities for production deployments.

<p align="center">
<img src="assets/jetson_person_2x.gif" height="50%" width="50%"/></p>

> Interested in detecting object masks as well?  Try combining NanoOWL with
> [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) for zero-shot open-vocabulary 
> instance segmentation.

<a id="features"></a>
## ✨ New Features in NanoOWL FT

This fork adds several production-ready features for real-world deployments:

### 🎥 USB Camera Support
- Real-time inference from USB cameras
- Multiple camera device support
- Configurable resolution (default 640x480)
- GPU-optimized frame processing (~7.5 FPS on Orin Nano)

### 🌐 Interactive Web Interface
- **Remote browser access** - View detections from any device on your network
- **Live prompt editing** - Change detection targets without restarting
- **Quick presets** - One-click common detection scenarios
- **Real-time metrics** - FPS, inference time, and detection counts
- **MJPEG streaming** - Low-latency video feed

### ⚡ Performance Optimizations
- GPU memory management for stable long-running inference
- cuDNN benchmark mode support
- TensorRT FP16 optimization ready
- Headless mode for server deployments

### 🛠️ Production Tools
- **Desktop management** - Enable/disable graphical interface to save memory (~500MB-1GB)
- **Systemd service** templates for auto-start on boot
- **Comprehensive documentation** for deployment scenarios

<a id="usage"></a>
## 👍 Usage

You can use NanoOWL in Python like this

```python3
from nanoowl.owl_predictor import OwlPredictor

predictor = OwlPredictor(
    "google/owlvit-base-patch32",
    image_encoder_engine="data/owlvit-base-patch32-image-encoder.engine"
)

image = PIL.Image.open("assets/owl_glove_small.jpg")

output = predictor.predict(image=image, text=["an owl", "a glove"], threshold=0.1)

print(output)
```

<a id="web-interface"></a>
## 🌐 Web Interface Usage (New!)

The easiest way to use NanoOWL is through the interactive web interface:

```bash
cd examples
python3 camera_web_server.py \
    --device cuda \
    --prompt "a person,a bottle,a phone" \
    --port 5000
```

Then open your browser to `http://<jetson-ip>:5000` to:
- 👀 View live camera feed with detections
- ✏️ Change detection prompts in real-time
- ⚙️ Adjust confidence threshold
- 📊 Monitor FPS and performance

**Quick Examples:**
```bash
# Detect people and objects
python3 camera_web_server.py --device cuda --prompt "a person,a bottle"

# Detect vehicles
python3 camera_web_server.py --device cuda --prompt "a car,a truck,a bicycle"

# With TensorRT engine (2-3x faster!)
python3 camera_web_server_trt.py \
    --device cuda \
    --engine ../data/owl_image_encoder_patch32.engine \
    --prompt "a person,a face"
```

See [examples/CAMERA_INFERENCE.md](examples/CAMERA_INFERENCE.md) and [examples/REMOTE_ACCESS.md](examples/REMOTE_ACCESS.md) for detailed instructions.

## 👍 Python API Usage

You can also use NanoOWL in Python like this

```python3
from nanoowl.owl_predictor import OwlPredictor

predictor = OwlPredictor(
    "google/owlvit-base-patch32",
    image_encoder_engine="data/owlvit-base-patch32-image-encoder.engine"
)

image = PIL.Image.open("assets/owl_glove_small.jpg")

output = predictor.predict(image=image, text=["an owl", "a glove"], threshold=0.1)

print(output)
```

Or better yet, to use OWL-ViT in conjunction with CLIP to detect and classify anything,
at any level, check out the tree predictor example below!

> See [Setup](#setup) for instructions on how to build the image encoder engine.

<a id="performance"></a>
## ⏱️ Performance

NanoOWL runs real-time on Jetson Orin Nano.

### Original TensorRT Performance
<table style="border-top: solid 1px; border-left: solid 1px; border-right: solid 1px; border-bottom: solid 1px">
    <thead>
        <tr>
            <th rowspan=1 style="text-align: center; border-right: solid 1px">Model †</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">Image Size</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">Patch Size</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">⏱️ Jetson Orin Nano (FPS)</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">⏱️ Jetson AGX Orin (FPS)</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">🎯 Accuracy (mAP)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: center; border-right: solid 1px">OWL-ViT (ViT-B/32)</td>
            <td style="text-align: center; border-right: solid 1px">768</td>
            <td style="text-align: center; border-right: solid 1px">32</td>
            <td style="text-align: center; border-right: solid 1px">TBD</td>
            <td style="text-align: center; border-right: solid 1px">95</td>
            <td style="text-align: center; border-right: solid 1px">28</td>
        </tr>
        <tr>
            <td style="text-align: center; border-right: solid 1px">OWL-ViT (ViT-B/16)</td>
            <td style="text-align: center; border-right: solid 1px">768</td>
            <td style="text-align: center; border-right: solid 1px">16</td>
            <td style="text-align: center; border-right: solid 1px">TBD</td>
            <td style="text-align: center; border-right: solid 1px">25</td>
            <td style="text-align: center; border-right: solid 1px">31.7</td>
        </tr>
    </tbody>
</table>

### Real-Time Camera Performance (Orin Nano, 640x480)
| Mode | Backend | FPS | Inference Time | Notes |
|------|---------|-----|----------------|-------|
| **GPU** | PyTorch CUDA | **~7.5** | ~134ms | Default, stable |
| **GPU** | TensorRT FP16 | **15-20*** | ~50-70ms* | Requires engine build |
| CPU | PyTorch | ~0.6 | ~1660ms | Not recommended |

\* *TensorRT performance estimated - requires building engine on machine with 16GB+ GPU memory*

<a id="setup"></a>
## 🛠️ Setup

### Prerequisites
- NVIDIA Jetson Orin (Nano, NX, or AGX)
- JetPack 6.0+ (tested on 6.2.1)
- CUDA 12.0+
- Python 3.8+
- USB Camera (for camera examples)

### Tested Package Versions

This project has been tested with the following package versions on Jetson Orin Nano:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.12 | System default |
| PyTorch | 2.8.0 | Jetson-specific build with CUDA |
| torchvision | 0.23.0 | Compatible with PyTorch 2.8.0 |
| TensorRT | 10.3.0 | Included with JetPack 6.2.1 |
| CUDA | 12.6 | Included with JetPack |
| transformers | 4.57.3 | Hugging Face library |
| numpy | 1.26.4 | **Must be <2.0** for PyTorch compatibility |
| Pillow | 12.0.0 | Image processing |
| opencv-python | 4.6.0 | Camera capture and visualization |
| Flask | 3.1.2 | Web server framework |
| CLIP | latest | Installed from GitHub |
| torch2trt | 0.5.0 | TensorRT converter |

> ⚠️ **Important:** Use `numpy<2` as NumPy 2.x is not compatible with PyTorch 2.8.0 on Jetson.

### Installation Steps

1. **Install PyTorch for Jetson**

   Follow [NVIDIA's instructions](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) or use:
   ```bash
   # For JetPack 6.x
   pip3 install torch torchvision
   ```

2. **Install torch2trt**
   
   ```bash
   git clone https://github.com/NVIDIA-AI-IOT/torch2trt
   cd torch2trt
   python3 setup.py install --user
   cd ..
   ```

3. **Install dependencies**

   ```bash
   pip3 install transformers pillow opencv-python flask
   pip3 install "numpy<2"  # Compatibility requirement
   pip3 install git+https://github.com/openai/CLIP.git
   ```

4. **Install NanoOWL FT**

   ```bash
   git clone https://github.com/ulise75/nanoowlFT.git
   cd nanoowlFT
   python3 setup.py develop --user
   ```

5. **Build TensorRT engine (Optional - for 2-3x speedup)**

   > ⚠️ **Note:** Building the TensorRT engine on Jetson Orin Nano may fail due to limited GPU memory (8GB). 
   > For best results, build on a machine with 16GB+ GPU memory and transfer the `.engine` file.

   On a machine with sufficient GPU memory:
   ```bash
   mkdir -p data
   python3 -m nanoowl.build_image_encoder_engine \
       data/owl_image_encoder_patch32.engine \
       --fp16_mode 1
   ```
   
   Then transfer to Jetson:
   ```bash
   scp data/owl_image_encoder_patch32.engine user@jetson-ip:~/nanoowlFT/data/
   ```

6. **Test the installation**

   ```bash
   cd examples
   
   # Test GPU inference (no TensorRT)
   python3 owl_predict_gpu.py
   
   # Test web server
   python3 camera_web_server.py --device cuda --prompt "a person"
   # Open browser to http://<jetson-ip>:5000
   ```

### Memory Optimization (Optional)

Free up ~500MB-1GB by disabling the graphical desktop:
```bash
./scripts/manage_desktop.sh disable
sudo reboot
```

Re-enable anytime with:
```bash
./scripts/manage_desktop.sh enable
sudo reboot
```  

<a id="examples"></a>
## 🤸 Examples

### Example 1 - Web Interface with Live Camera (NEW! ⭐)

The interactive web interface is the easiest way to use NanoOWL:

```bash
cd examples
python3 camera_web_server.py --device cuda --prompt "a person,a bottle"
```

Open browser to `http://<jetson-ip>:5000` for:
- Live camera feed with detections
- Real-time prompt editing
- Performance metrics (FPS, inference time)
- Quick preset buttons

**For TensorRT-accelerated version (2-3x faster):**
```bash
python3 camera_web_server_trt.py \
    --device cuda \
    --engine ../data/owl_image_encoder_patch32.engine \
    --prompt "a person,a face"
```

See [examples/CAMERA_INFERENCE.md](examples/CAMERA_INFERENCE.md) for detailed usage.

### Example 2 - Headless Camera Inference (NEW!)

For server deployments without display:

```bash
cd examples
python3 camera_inference_headless.py --device cuda --max_frames 100
```

Saves detected frames to `data/camera_output/`.

### Example 3 - Basic Prediction

<img src="assets/owl_predict_out.jpg" height="256px"/>

This example demonstrates how to use the TensorRT optimized OWL-ViT model to
detect objects by providing text descriptions of the object labels.

To run the example, first navigate to the examples folder

```bash
cd examples
```

**GPU-accelerated version:**
```bash
python3 owl_predict_gpu.py
```

**With TensorRT engine:**
```bash
python3 owl_predict.py \
    --prompt="[an owl, a glove]" \
    --threshold=0.1 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

By default the output will be saved to ``data/owl_predict_out.jpg``. 

You can also use this example to profile inference.  Simply set the flag ``--profile``.

### Example 4 - Tree Prediction

<img src="assets/tree_predict_out.jpg" height="256px"/>

This example demonstrates how to use the tree predictor class to detect and
classify objects at any level.

To run the example, first navigate to the examples folder

```bash
cd examples
```

To detect all owls, and the detect all wings and eyes in each detect owl region
of interest, type

```bash
python3 tree_predict.py \
    --prompt="[an owl [a wing, an eye]]" \
    --threshold=0.15 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

By default the output will be saved to ``data/tree_predict_out.jpg``.

To classify the image as indoors or outdoors, type

```bash
python3 tree_predict.py \
    --prompt="(indoors, outdoors)" \
    --threshold=0.15 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

To classify the image as indoors or outdoors, and if it's outdoors then detect
all owls, type

```bash
python3 tree_predict.py \
    --prompt="(indoors, outdoors [an owl])" \
    --threshold=0.15 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```


### Example 5 - Tree Prediction (Live Camera)

<img src="assets/jetson_person_2x.gif" height="50%" width="50%"/>

This example demonstrates the tree predictor running on a live camera feed with
live-edited text prompts.  To run the example

1. Ensure you have a camera device connected

2. Launch the demo
    ```bash
    cd examples/tree_demo
    python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
    ```
3. Second, open your browser to ``http://<ip address>:7860``
4. Type whatever prompt you like to see what works!  Here are some examples
    - Example: [a face [a nose, an eye, a mouth]]
    - Example: [a face (interested, yawning / bored)]
    - Example: (indoors, outdoors)



<a id="acknowledgement"></a>
## 👏 Acknowledgement

- Thanks to [NVIDIA AI-IOT](https://github.com/NVIDIA-AI-IOT) for the original [NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) implementation
- Thanks to the authors of [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) for the great open-vocabulary detection work
- This fork adds production-ready features including web interface, camera support, and remote access capabilities

<a id="see-also"></a>
## 🔗 See also

- [Original NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) - The upstream project this fork is based on
- [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) - A real-time Segment Anything (SAM) model variant for NVIDIA Jetson Orin platforms.
- [Jetson Introduction to Knowledge Distillation Tutorial](https://github.com/NVIDIA-AI-IOT/jetson-intro-to-distillation) - For an introduction to knowledge distillation as a model optimization technique.
- [Jetson Generative AI Playground](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/) - For instructions and tips for using a variety of LLMs and transformers on Jetson.
- [Jetson Containers](https://github.com/dusty-nv/jetson-containers) - For a variety of easily deployable and modular Jetson Containers

## 📝 Additional Documentation

- [examples/CAMERA_INFERENCE.md](examples/CAMERA_INFERENCE.md) - USB camera setup and usage
- [examples/REMOTE_ACCESS.md](examples/REMOTE_ACCESS.md) - Remote access configuration and systemd service setup
- [CUDA_FIX.md](CUDA_FIX.md) - Troubleshooting CUDA and PyTorch issues on Jetson
