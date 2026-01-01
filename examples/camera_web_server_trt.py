#!/usr/bin/env python3
# Web-based Camera Inference with NanoOWL + TensorRT optimization

import cv2
import numpy as np
import PIL.Image
import time
import argparse
import os
import gc
import torch
import threading
from flask import Flask, Response, render_template_string, request, jsonify
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output

app = Flask(__name__)

class ThreadedCamera:
    """Thread-based camera reader to eliminate buffering lag"""
    def __init__(self, src):
        self.capture = src
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.read_count = 0
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            # Grab multiple frames quickly to flush buffer
            for _ in range(2):  # Grab 2 frames to stay current
                self.capture.grab()
            
            # Retrieve the last grabbed frame
            ret, frame = self.capture.retrieve()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.read_count += 1
            time.sleep(0.001)  # Minimal delay
                    
    def read(self):
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        self.stopped = True

# Global variables for video stream
camera = None
threaded_camera = None
predictor = None
text_encodings = None
detection_text = []
args = None

# Stats tracking
stats = {
    'fps': 0.0,
    'inference_time_ms': 0.0,
    'detections': 0,
    'detected_labels': [],
    'engine_type': 'TensorRT'
}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>NanoOWL Live Detection (TensorRT)</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: #fff;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #76b900;
        }
        .container {
            max-width: 1200px;
            width: 100%;
        }
        .video-container {
            background-color: #000;
            padding: 10px;
            border-radius: 8px;
            margin: 20px 0;
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .info {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .status {
            color: #76b900;
            font-weight: bold;
        }
        .trt-badge {
            background-color: #76b900;
            color: #000;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            display: inline-block;
        }
        .control-panel {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .input-group {
            margin: 10px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #76b900;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 10px;
            background-color: #1a1a1a;
            border: 2px solid #444;
            border-radius: 4px;
            color: #fff;
            font-size: 14px;
            box-sizing: border-box;
        }
        input[type="text"]:focus, input[type="number"]:focus {
            outline: none;
            border-color: #76b900;
        }
        button {
            background-color: #76b900;
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            width: 100%;
        }
        button:hover {
            background-color: #5a9100;
        }
        button:active {
            background-color: #446e00;
        }
        .message {
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
        }
        .message.success {
            background-color: #2a4a2a;
            color: #76b900;
            display: block;
        }
        .message.error {
            background-color: #4a2a2a;
            color: #ff6b6b;
            display: block;
        }
        .hint {
            color: #888;
            font-size: 12px;
            margin-top: 5px;
        }
        .quick-presets {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }
        .preset-btn {
            background-color: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            width: auto;
            margin: 0;
        }
        .preset-btn:hover {
            background-color: #444;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 NanoOWL Live Camera Detection <span class="trt-badge">TensorRT</span></h1>
        
        <div class="control-panel">
            <h2>🎯 Detection Settings</h2>
            <div class="input-group">
                <label for="prompts">What to detect (comma-separated):</label>
                <input type="text" id="prompts" value="{{ prompts }}" placeholder="a person, a bottle, a car">
                <div class="hint">Examples: a person, a bottle, a car, a dog, a phone, a laptop</div>
                <div class="quick-presets">
                    <button class="preset-btn" onclick="setPrompts('a person, a face')">Person & Face</button>
                    <button class="preset-btn" onclick="setPrompts('a bottle, a cup, a phone')">Objects</button>
                    <button class="preset-btn" onclick="setPrompts('a car, a truck, a bicycle')">Vehicles</button>
                    <button class="preset-btn" onclick="setPrompts('a dog, a cat, a bird')">Animals</button>
                </div>
            </div>
            <div class="input-group">
                <label for="threshold">Detection threshold (0.0-1.0):</label>
                <input type="number" id="threshold" value="{{ threshold }}" min="0" max="1" step="0.05">
                <div class="hint">Lower = more detections but less accurate. Higher = fewer but more confident.</div>
            </div>
            <button onclick="updateSettings()">Apply Settings</button>
            <div id="message" class="message"></div>
        </div>
        
        <div class="info">
            <p><span class="status">Acceleration:</span> TensorRT FP16 🚀</p>
            <p><span class="status">Device:</span> {{ device }}</p>
            <p><span class="status">Current prompts:</span> <span id="current-prompts">{{ prompts }}</span></p>
            <p><span class="status">Current threshold:</span> <span id="current-threshold">{{ threshold }}</span></p>
            <p><span class="status">Resolution:</span> {{ width }}x{{ height }}</p>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
        </div>
        
        <div class="info">
            <p>💡 TensorRT-optimized inference for maximum performance!</p>
        </div>
    </div>
    
    <script>
        function setPrompts(value) {
            document.getElementById('prompts').value = value;
        }
        
        function updateSettings() {
            const prompts = document.getElementById('prompts').value;
            const threshold = parseFloat(document.getElementById('threshold').value);
            const messageDiv = document.getElementById('message');
            
            if (!prompts.trim()) {
                messageDiv.className = 'message error';
                messageDiv.textContent = '❌ Please enter at least one detection prompt';
                return;
            }
            
            if (threshold < 0 || threshold > 1) {
                messageDiv.className = 'message error';
                messageDiv.textContent = '❌ Threshold must be between 0.0 and 1.0';
                return;
            }
            
            messageDiv.className = 'message';
            messageDiv.textContent = '⏳ Updating settings...';
            messageDiv.style.display = 'block';
            
            fetch('/update_settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompts: prompts,
                    threshold: threshold
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    messageDiv.className = 'message success';
                    messageDiv.textContent = '✅ ' + data.message;
                    document.getElementById('current-prompts').textContent = data.prompts;
                    document.getElementById('current-threshold').textContent = data.threshold;
                } else {
                    messageDiv.className = 'message error';
                    messageDiv.textContent = '❌ ' + data.message;
                }
            })
            .catch(error => {
                messageDiv.className = 'message error';
                messageDiv.textContent = '❌ Failed to update settings: ' + error;
            });
        }
    </script>
</body>
</html>
'''

def initialize_camera():
    """Initialize camera with settings optimized for Jetson"""
    global camera, threaded_camera, args
    
    # Check if RTSP stream
    if args.camera.startswith('rtsp://'):
        print(f"📡 Connecting to RTSP stream: {args.camera}")
        
        # Try GStreamer first (lowest latency for RTSP on Jetson)
        gst_rtsp_pipeline = (
            f"rtspsrc location={args.camera} latency=0 ! "
            f"rtph264depay ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"videoscale ! video/x-raw,width={args.width},height={args.height} ! "
            f"appsink drop=1 max-buffers=1"
        )
        
        try:
            camera = cv2.VideoCapture(gst_rtsp_pipeline, cv2.CAP_GSTREAMER)
            if camera.isOpened():
                print("✅ Using GStreamer RTSP (hardware accelerated, low latency)")
                threaded_camera = ThreadedCamera(camera).start()
                print("✅ Started threaded frame reader (eliminates buffering lag)")
                return True
        except Exception as e:
            print(f"⚠️  GStreamer RTSP failed: {e}")
        
        # Fallback to FFMPEG with low latency settings
        print("⚠️  Trying FFMPEG with low latency settings...")
        camera = cv2.VideoCapture(args.camera, cv2.CAP_FFMPEG)
        if camera.isOpened():
            # Set low latency options
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("✅ Using FFMPEG RTSP (fallback)")
            threaded_camera = ThreadedCamera(camera).start()
            print("✅ Started threaded frame reader (eliminates buffering lag)")
            return True
        return False
    
    # Convert camera to int for local cameras
    try:
        camera_id = int(args.camera)
    except ValueError:
        print(f"❌ Invalid camera ID: {args.camera}")
        return False
    
    # Try GStreamer pipeline first (optimized for Jetson)
    gst_pipeline = (
        f"v4l2src device=/dev/video{camera_id} ! "
        f"video/x-raw,width={args.width},height={args.height},framerate=30/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )
    
    try:
        camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if camera.isOpened():
            print("✅ Using GStreamer backend (optimized)")
            return True
    except:
        print("⚠️  GStreamer backend failed, trying V4L2...")
    
    # Fallback to V4L2 - MATCH MSCamera-Test.py settings exactly
    camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if camera.isOpened():
        # EXACT same settings as MSCamera-Test.py
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FPS, 30)
        # NO BUFFERSIZE setting - match MSCamera-Test.py
        
        # Verify actual resolution
        actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Using V4L2 backend - Actual resolution: {actual_w}x{actual_h}")
        return True
    
    return False

def generate_frames():
    """Generator function that yields video frames with detections"""
    global camera, threaded_camera, predictor, text_encodings, detection_text, args, stats
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0.0
    last_inference_time = 0
    skip_frames = 0
    
    # Use threaded camera if available, otherwise use direct camera
    cam_source = threaded_camera if threaded_camera else camera
    
    while True:
        success, frame = cam_source.read()
        if not success:
            break
        
        frame_count += 1
        
        # Skip frames if inference is taking too long (adaptive frame skipping)
        if last_inference_time > 0.05 and frame_count % 2 == 0:  # If inference > 50ms, skip every other frame
            skip_frames += 1
            continue
        
        # NO RESIZE - keep original quality like MSCamera-Test.py
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = PIL.Image.fromarray(frame_rgb)
        
        # Run inference with TensorRT
        inference_start = time.time()
        try:
            with torch.no_grad():
                output = predictor.predict(
                    image=image_pil,
                    text=detection_text,
                    text_encodings=text_encodings,
                    threshold=args.threshold,
                    pad_square=False
                )
            inference_time = time.time() - inference_start
            last_inference_time = inference_time
            
            # Less aggressive cache clearing with TensorRT
            if args.device == 'cuda' and frame_count % 100 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Inference error: {e}")
            continue
        
        # Draw detections
        output_image = draw_owl_output(image_pil, output, text=detection_text, draw_text=True)
        output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Update stats
        stats['fps'] = round(fps_display, 2)
        stats['inference_time_ms'] = round(inference_time * 1000, 2)
        stats['detections'] = len(output.labels)
        stats['detected_labels'] = [detection_text[label] for label in output.labels]
        
        # Add info overlay (simplified for speed)
        info_text = f"FPS: {fps_display:.1f} | Inference: {inference_time*1000:.0f}ms | Det: {len(output.labels)}"
        cv2.putText(output_frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(output.labels) > 0:
            det_labels = ', '.join([detection_text[label] for label in output.labels[:3]])  # Show max 3
            if len(output.labels) > 3:
                det_labels += f" +{len(output.labels)-3}"
            cv2.putText(output_frame, det_labels, (10, output_frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Encode frame as JPEG (default quality 95 for best image)
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        
        frame_count += 1
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template_string(
        HTML_TEMPLATE,
        device=args.device.upper(),
        prompts=', '.join(detection_text),
        threshold=args.threshold,
        width=args.width,
        height=args.height
    )

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Return current performance statistics"""
    global stats
    return jsonify(stats)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update detection settings on the fly"""
    global predictor, text_encodings, detection_text, args
    
    try:
        data = request.get_json()
        new_prompts = data.get('prompts', '')
        new_threshold = data.get('threshold', args.threshold)
        
        # Validate inputs
        if not new_prompts.strip():
            return jsonify({
                'status': 'error',
                'message': 'Prompts cannot be empty'
            }), 400
        
        # Parse new prompts
        new_detection_text = [t.strip() for t in new_prompts.split(',') if t.strip()]
        
        if len(new_detection_text) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No valid prompts provided'
            }), 400
        
        # Update text encodings
        print(f"\nUpdating detection prompts to: {new_detection_text}")
        with torch.no_grad():
            new_text_encodings = predictor.encode_text(new_detection_text)
        
        # Update global variables
        detection_text = new_detection_text
        text_encodings = new_text_encodings
        args.threshold = float(new_threshold)
        
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"Updated threshold to: {args.threshold}")
        
        return jsonify({
            'status': 'success',
            'message': 'Settings updated successfully!',
            'prompts': ', '.join(detection_text),
            'threshold': args.threshold
        })
        
    except Exception as e:
        print(f"Error updating settings: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def main():
    global predictor, text_encodings, detection_text, args
    
    parser = argparse.ArgumentParser(description='Web-based camera inference with NanoOWL + TensorRT')
    parser.add_argument('--camera', type=str, default='0', help='Camera device ID or RTSP URL (e.g., rtsp://192.168.1.254/live)')
    parser.add_argument('--width', type=int, default=320, help='Camera width')
    parser.add_argument('--height', type=int, default=240, help='Camera height')
    parser.add_argument('--prompt', type=str, default='a person,a face,a hand',
                        help='Comma-separated detection prompts')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind web server to (0.0.0.0 for all interfaces)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for web server')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to TensorRT engine file')
    args = parser.parse_args()
    
    # Check if engine file exists
    if not os.path.exists(args.engine):
        print(f"Error: TensorRT engine file not found: {args.engine}")
        print("\nPlease build the engine first or provide the correct path.")
        print("See instructions for building on another machine.")
        return
    
    # Set GPU memory optimization
    if args.device == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        torch.cuda.empty_cache()
        gc.collect()
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    # Parse detection prompts
    detection_text = [t.strip() for t in args.prompt.split(',')]
    print(f"Detection prompts: {detection_text}")
    
    # Initialize predictor with TensorRT engine
    print(f"Initializing OWL-ViT predictor with TensorRT engine on {args.device.upper()}...")
    try:
        predictor = OwlPredictor(
            "google/owlvit-base-patch32",
            device=args.device,
            image_encoder_engine=args.engine
        )
        text_encodings = predictor.encode_text(detection_text)
        
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        
        print("✅ TensorRT engine loaded successfully!")
            
    except RuntimeError as e:
        print(f"Error initializing model: {e}")
        return
    
    # Initialize camera
    print(f"Opening camera {args.camera}...")
    if not initialize_camera():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print(f"\n{'='*60}")
    print(f"🌐 Web Server Starting (TensorRT Accelerated)")
    print(f"{'='*60}")
    print(f"  Device: {args.device.upper()}")
    print(f"  TensorRT Engine: {args.engine}")
    print(f"  Camera: {args.camera} ({args.width}x{args.height})")
    print(f"  Detection: {', '.join(detection_text)}")
    print(f"  Threshold: {args.threshold}")
    print(f"\n  🔗 Open in browser:")
    print(f"     Local:   http://localhost:{args.port}")
    print(f"     Network: http://192.168.2.88:{args.port}")
    print(f"\n  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if camera:
            camera.release()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print("Server stopped.")

if __name__ == '__main__':
    main()
