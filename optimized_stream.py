#!/usr/bin/env python3
"""
Optimized MJPEG streaming with minimal latency - tested and reliable approach.
Focus on speed over image quality.
"""

import cv2
import time
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# RTSP configuration
RTSP_URL = "rtsp://192.168.1.254/live"
camera = None

def init_camera():
    """Initialize camera with aggressive low-latency settings"""
    global camera
    print(f"📡 Connecting to: {RTSP_URL}")
    
    camera = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not camera.isOpened():
        print("❌ Failed to open camera")
        return False
    
    print("✅ Camera opened")
    return True

def generate_frames():
    """Ultra-fast frame generation with aggressive optimizations"""
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    
    while True:
        # Aggressive buffer flush - grab and discard 4 frames
        for _ in range(4):
            if not camera.grab():
                break
        
        # Retrieve only the latest
        ret, frame = camera.retrieve()
        
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        fps_counter += 1
        
        # Downscale for speed (lower res = faster encoding)
        frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LINEAR)
        
        # Calculate FPS
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        
        # Minimal overlay
        cv2.putText(frame, f"FPS: {fps_display:.0f}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Fast JPEG encoding - quality 30 for maximum speed
        ret, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 30,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0  # Disable optimization for speed
        ])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Optimized Low Latency Stream</title>
    <style>
        body { 
            background: #000; 
            color: #0f0;
            text-align: center;
            font-family: monospace;
            padding: 20px;
            margin: 0;
        }
        h1 { color: #0f0; margin: 10px; }
        img { 
            width: 90%;
            max-width: 800px;
            border: 2px solid #0f0;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }
        .info {
            font-size: 14px;
            margin: 10px;
            color: #ff0;
        }
    </style>
</head>
<body>
    <h1>⚡ OPTIMIZED LOW LATENCY STREAM</h1>
    <div class="info">480x360 | 4x buffer flush | JPEG Q30 | No optimization delay</div>
    <img src="{{ url_for('video_feed') }}" alt="Video Stream">
    <div class="info">Resolution lowered for minimum latency</div>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if init_camera():
        print("\n" + "="*70)
        print("⚡ Optimized Low-Latency Stream Server")
        print("="*70)
        print(f"   URL: http://192.168.2.239:5000")
        print(f"   Optimizations:")
        print(f"     • 480x360 resolution (faster than 640x480)")
        print(f"     • JPEG quality 30 (vs 70-85)")
        print(f"     • 4x buffer flush")
        print(f"     • No JPEG optimization (faster encode)")
        print("="*70 + "\n")
        
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True)
        finally:
            if camera:
                camera.release()
    else:
        print("Failed to initialize camera")
