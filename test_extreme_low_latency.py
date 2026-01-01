#!/usr/bin/env python3
"""
Extreme low-latency test - try GStreamer with hardware acceleration and minimal buffering.
"""

import cv2
import time
import threading
from flask import Flask, Response

app = Flask(__name__)

class MinimalLatencyCamera:
    """Minimal latency frame grabber - grab latest only"""
    def __init__(self, src):
        self.capture = src
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            # Just read continuously - OpenCV will handle buffer
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    
    def read(self):
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None

# Try GStreamer pipeline optimized for low latency
rtsp_url = "rtsp://192.168.1.254/live"

print("\n" + "="*70)
print("Extreme Low-Latency Test - GStreamer + Minimal Buffering")
print("="*70)

# GStreamer pipeline with minimal latency settings
gst_pipeline = (
    f"rtspsrc location={rtsp_url} latency=0 buffer-mode=0 ! "
    f"application/x-rtp,media=video ! rtph264depay ! h264parse ! "
    f"avdec_h264 ! videoconvert ! videoscale ! "
    f"video/x-raw,width=640,height=480,format=BGR ! "
    f"appsink drop=1 max-buffers=1 sync=0"
)

print(f"\n📡 Trying GStreamer with zero-latency settings...")
camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not camera.isOpened():
    print("⚠️  GStreamer failed, trying FFMPEG with TCP...")
    # Fallback to FFMPEG with TCP (sometimes more stable)
    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("✅ Camera opened")
threaded_camera = MinimalLatencyCamera(camera).start()
print("✅ Threaded reader started (continuous read mode)")

# Wait for first frames
time.sleep(2)

def generate_frames():
    """Generate frames with timestamp for latency check"""
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    
    while True:
        success, frame = threaded_camera.read()
        if not success or frame is None:
            time.sleep(0.001)
            continue
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        
        # Add timestamp to verify latency
        current_time = time.strftime("%H:%M:%S", time.localtime())
        current_ms = int((time.time() % 1) * 1000)
        
        cv2.putText(frame, f"FPS: {fps_display:.1f} | Time: {current_time}.{current_ms:03d}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Compare this time with your device clock", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Medium quality for balance
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Extreme Low Latency Test</title>
        <style>
            body { background: #000; color: #0f0; text-align: center; font-family: monospace; padding: 20px; }
            img { max-width: 95%; border: 3px solid #0f0; }
            .clock { font-size: 36px; color: #ff0; margin: 20px; font-weight: bold; }
            .info { font-size: 16px; margin: 10px; }
        </style>
        <script>
            function updateClock() {
                const now = new Date();
                const timeStr = now.getHours().toString().padStart(2,'0') + ':' + 
                               now.getMinutes().toString().padStart(2,'0') + ':' + 
                               now.getSeconds().toString().padStart(2,'0') + '.' +
                               now.getMilliseconds().toString().padStart(3,'0');
                document.getElementById('clock').innerText = 'Browser Time: ' + timeStr;
            }
            setInterval(updateClock, 50);
        </script>
    </head>
    <body onload="updateClock()">
        <h1>⚡ EXTREME LOW LATENCY TEST</h1>
        <div class="clock" id="clock"></div>
        <div class="info">Compare timestamp in video with browser time above</div>
        <img src="/video_feed">
        <div class="info">Latency = Browser Time - Video Time</div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("⚡ Extreme Low-Latency Camera Test")
    print("="*70)
    print("   URL: http://192.168.2.239:5000")
    print("   Method: GStreamer with zero-latency settings")
    print("   Buffer: 1 frame max, drop mode, no sync")
    print("   Compare timestamps to measure actual latency")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
