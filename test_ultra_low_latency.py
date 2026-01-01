#!/usr/bin/env python3
"""
Ultra-low latency RTSP camera test with aggressive optimization.
"""

import cv2
import time
import threading
import os
from flask import Flask, Response

app = Flask(__name__)

class UltraLowLatencyCamera:
    """Aggressive frame grabbing for minimal latency"""
    def __init__(self, src):
        self.capture = src
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.frame_time = 0
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            # Aggressively flush buffer - grab and discard 5 frames
            for _ in range(5):
                self.capture.grab()
            
            # Retrieve only the latest frame
            ret, frame = self.capture.retrieve()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_time = time.time()
            # No sleep - grab as fast as possible
                    
    def read(self):
        with self.lock:
            if self.frame is not None:
                latency = time.time() - self.frame_time if self.frame_time > 0 else 0
                return True, self.frame.copy(), latency
            return False, None, 0

# Test different RTSP configurations
print("\n" + "="*70)
print("Testing RTSP Configurations for Minimal Latency")
print("="*70)

rtsp_url = "rtsp://192.168.1.254/live"

# Configuration 1: FFMPEG with aggressive low-latency flags
print("\n📡 Testing FFMPEG with ultra-low latency settings...")
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32'

camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if camera.isOpened():
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("✅ Camera opened with FFMPEG")
else:
    print("❌ Failed to open camera")
    exit(1)

print("✅ Starting ultra-low latency frame grabber...")
threaded_camera = UltraLowLatencyCamera(camera).start()

# Wait for stabilization
time.sleep(3)

def generate_frames():
    """Generate frames with latency measurement"""
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    latency_sum = 0
    latency_count = 0
    
    while True:
        success, frame, grab_latency = threaded_camera.read()
        if not success or frame is None:
            time.sleep(0.001)
            continue
        
        frame_count += 1
        fps_counter += 1
        latency_sum += grab_latency
        latency_count += 1
        
        # Resize to 640x480
        if frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 480))
        
        # Calculate FPS
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        
        # Calculate average latency
        avg_latency = (latency_sum / latency_count * 1000) if latency_count > 0 else 0
        if latency_count >= 30:  # Reset every 30 frames
            latency_sum = 0
            latency_count = 0
        
        # Minimal overlay
        cv2.putText(frame, f"FPS: {fps_display:.1f} | Grab Latency: {avg_latency:.0f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "ULTRA-LOW LATENCY MODE (5x buffer flush)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Fast encode with moderate quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultra-Low Latency Test</title>
        <style>
            body { background: #000; color: #0f0; text-align: center; font-family: monospace; }
            img { max-width: 90%; border: 3px solid #0f0; margin: 20px; }
            .info { font-size: 18px; margin: 10px; }
        </style>
    </head>
    <body>
        <h1>🚀 ULTRA-LOW LATENCY RTSP TEST</h1>
        <div class="info">640x480 | 5x Buffer Flush | No Inference</div>
        <img src="/video_feed">
        <div class="info">Check FPS and grab latency values above</div>
        <div class="info">Goal: &lt;100ms total latency, 25-30 FPS</div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Ultra-Low Latency Camera Test Server")
    print("="*70)
    print("   URL: http://192.168.2.239:5000")
    print("   Optimizations:")
    print("     • 5x buffer flush (aggressive)")
    print("     • UDP RTSP transport")
    print("     • No buffer flags")
    print("     • Zero analysis/probe delay")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
