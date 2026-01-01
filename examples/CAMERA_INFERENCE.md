# NanoOWL Camera Inference

Real-time object detection using USB camera with NanoOWL on GPU.

## Available Scripts

### 1. Basic Object Detection (`camera_inference.py`)
Real-time detection of multiple objects with customizable prompts.

**Usage:**
```bash
cd /home/sombras4/Downloads/nanoowlOrin/examples

# Basic usage (default: detect person, face, hand)
python3 camera_inference.py

# Custom detection prompts
python3 camera_inference.py --prompt "a bottle,a cup,a phone,a laptop"

# With FPS counter
python3 camera_inference.py --show-fps

# Different camera device
python3 camera_inference.py --camera 1

# Lower threshold for more detections
python3 camera_inference.py --threshold 0.1

# Full example
python3 camera_inference.py --camera 0 --width 640 --height 480 \
    --prompt "a person,a dog,a cat,a car" --threshold 0.15 --show-fps
```

### 2. Tree Detection (`camera_tree_inference.py`)
Hierarchical detection and classification in real-time.

**Usage:**
```bash
cd /home/sombras4/Downloads/nanoowlOrin/examples

# Basic usage (detect person, classify as man/woman)
python3 camera_tree_inference.py

# Custom tree prompt
python3 camera_tree_inference.py --prompt "[a vehicle](a car, a truck, a motorcycle)"

# Detect objects and classify environment
python3 camera_tree_inference.py --prompt "[an object](indoors, outdoors)"

# Full example
python3 camera_tree_inference.py --camera 0 --width 640 --height 480 \
    --prompt "[a face](happy, sad, neutral)" --threshold 0.2 --show-fps
```

## Options

Both scripts support the following arguments:

- `--camera ID` - Camera device ID (default: 0)
- `--width WIDTH` - Camera width in pixels (default: 640)
- `--height HEIGHT` - Camera height in pixels (default: 480)
- `--threshold VALUE` - Detection confidence threshold 0-1 (default: 0.15)
- `--device cuda|cpu` - Device to run on (default: cuda)
- `--show-fps` - Display FPS counter
- `--prompt TEXT` - Detection prompts (comma-separated for basic, tree format for tree)

## Controls

While running:
- Press **'q'** to quit
- Press **'s'** to save the current frame to `data/` directory

## Detection Prompt Examples

### Basic Detection (`camera_inference.py`):
```bash
# Common objects
--prompt "a person,a face,a hand,a bottle,a cup,a phone"

# Animals
--prompt "a dog,a cat,a bird,a horse"

# Vehicles
--prompt "a car,a bus,a truck,a bicycle,a motorcycle"

# Indoor objects
--prompt "a chair,a table,a laptop,a keyboard,a mouse,a monitor"
```

### Tree Detection (`camera_tree_inference.py`):
```bash
# Hierarchical detection then classification
--prompt "[a person](a man, a woman, a child)"

# Detect object, classify by category
--prompt "[an object](a furniture, an electronic, a vehicle)"

# Detect and classify environment
--prompt "[a scene](indoors, outdoors)"

# Multiple levels
--prompt "[a person][a face](happy, sad, neutral)"
```

## Performance

On Jetson Orin with GPU:
- Basic detection: ~100-200ms per frame
- Tree detection: ~150-250ms per frame
- Resolution: 640x480 (adjustable)

## Troubleshooting

**Camera not found:**
```bash
# List available cameras
ls -la /dev/video*

# Try different camera ID
python3 camera_inference.py --camera 1
```

**Permission denied:**
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

**Low FPS:**
- Lower camera resolution: `--width 320 --height 240`
- Increase detection threshold: `--threshold 0.2`
- Use simpler prompts with fewer classes

**Out of memory:**
- Run on CPU: `--device cpu`
- Lower resolution
- Reduce number of detection classes
