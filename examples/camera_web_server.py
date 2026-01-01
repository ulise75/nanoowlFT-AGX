#!/usr/bin/env python3
# Web-based Camera Inference with NanoOWL - Stream to remote browser

import cv2
import numpy as np
import PIL.Image
import time
import argparse
import os
import gc
import torch
from flask import Flask, Response, render_template_string, request, jsonify
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output

app = Flask(__name__)

# Global variables for video stream
camera = None
predictor = None
text_encodings = None
detection_text = []
args = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>NanoOWL Live Detection</title>
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
        <h1>🎥 NanoOWL Live Camera Detection</h1>
        
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
            <p><span class="status">Device:</span> {{ device }}</p>
            <p><span class="status">Current prompts:</span> <span id="current-prompts">{{ prompts }}</span></p>
            <p><span class="status">Current threshold:</span> <span id="current-threshold">{{ threshold }}</span></p>
            <p><span class="status">Resolution:</span> {{ width }}x{{ height }}</p>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
        </div>
        
        <div class="info">
            <p>💡 Tip: Update detection settings without restarting the server!</p>
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
    """Initialize camera with settings"""
    global camera, args
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    return camera.isOpened()

def generate_frames():
    """Generator function that yields video frames with detections"""
    global camera, predictor, text_encodings, detection_text, args
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0.0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = PIL.Image.fromarray(frame_rgb)
        
        # Run inference
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
            
            # Clear GPU cache periodically
            if args.device == 'cuda' and frame_count % 10 == 0:
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
        
        # Add info overlay
        info_text = f"Detections: {len(output.labels)} | FPS: {fps_display:.1f} | Inference: {inference_time*1000:.1f}ms"
        cv2.putText(output_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(output.labels) > 0:
            det_info = f"Found: {', '.join([detection_text[label] for label in output.labels])}"
            cv2.putText(output_frame, det_info, (10, output_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    
    parser = argparse.ArgumentParser(description='Web-based camera inference with NanoOWL')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
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
    args = parser.parse_args()
    
    # Set GPU memory optimization
    if args.device == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        torch.cuda.empty_cache()
        gc.collect()
    
    # Parse detection prompts
    detection_text = [t.strip() for t in args.prompt.split(',')]
    print(f"Detection prompts: {detection_text}")
    
    # Initialize predictor
    print(f"Initializing OWL-ViT predictor on {args.device.upper()}...")
    try:
        predictor = OwlPredictor(
            "google/owlvit-base-patch32",
            device=args.device
        )
        text_encodings = predictor.encode_text(detection_text)
        
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        print(f"Error initializing model: {e}")
        return
    
    # Initialize camera
    print(f"Opening camera {args.camera}...")
    if not initialize_camera():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print(f"\n{'='*60}")
    print(f"🌐 Web Server Starting")
    print(f"{'='*60}")
    print(f"  Device: {args.device.upper()}")
    print(f"  Camera: {args.camera} ({args.width}x{args.height})")
    print(f"  Detection: {', '.join(detection_text)}")
    print(f"  Threshold: {args.threshold}")
    print(f"\n  🔗 Open in browser:")
    print(f"     Local:   http://localhost:{args.port}")
    print(f"     Network: http://<jetson-ip>:{args.port}")
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
