#!/usr/bin/env python3
"""
Test RTSP camera feed only (no inference) to measure pure streaming latency.
"""

import cv2
import time
import threading
from flask import Flask, Response

app = Flask(__name__)

class ThreadedCamera:
    """Thread-based camera reader"""
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
            # Grab multiple frames to flush buffer
            for _ in range(2):
                self.capture.grab()
            
            ret, frame = self.capture.retrieve()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.001)
                    
    def read(self):
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None

# Initialize camera
rtsp_url = "rtsp://192.168.1.254/live"
print(f"Connecting to: {rtsp_url}")

camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("✅ Camera opened")
threaded_camera = ThreadedCamera(camera).start()
print("✅ Threaded reader started")

# Wait for first frame
time.sleep(2)

def generate_frames():
    """Generate raw frames with only FPS counter"""
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    
    while True:
        success, frame = threaded_camera.read()
        if not success or frame is None:
            continue
        
        frame_count += 1
        fps_counter += 1
        
        # Resize to 640x480 if needed
        if frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 480))
        
        # Calculate FPS
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        
        # Add minimal overlay
        cv2.putText(frame, f"FPS: {fps_display:.1f} | Raw Feed (No Inference)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode with high quality to see actual camera quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Raw Camera Feed Test</title></head>
    <body style="background: #000; color: #fff; text-align: center; font-family: Arial;">
        <h1>Raw RTSP Camera Feed (640x480) - No Inference</h1>
        <img src="/video_feed" style="max-width: 90%; border: 2px solid #0f0;">
        <p>Testing pure camera streaming latency</p>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Starting Raw Camera Feed Test Server")
    print("="*60)
    print("   URL: http://192.168.2.239:5000")
    print("   Resolution: 640x480")
    print("   NO INFERENCE - Pure camera feed test")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
