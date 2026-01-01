#!/usr/bin/env python3
"""
Stable low-latency test - single-threaded to avoid crashes.
"""

import cv2
import time
from flask import Flask, Response

app = Flask(__name__)

# Initialize camera
rtsp_url = "rtsp://192.168.1.254/live"
print(f"\n📡 Connecting to {rtsp_url}...")

camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("✅ Camera opened (single-threaded mode)")

def generate_frames():
    """Generate frames - single threaded, flush old frames"""
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    
    while True:
        # Flush buffer by grabbing multiple times
        for _ in range(3):
            camera.grab()
        
        # Get the latest frame
        ret, frame = camera.retrieve()
        
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        fps_counter += 1
        
        # Resize if needed
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))
        
        # Calculate FPS
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        
        # Add timestamp
        current_time = time.strftime("%H:%M:%S", time.localtime())
        current_ms = int((time.time() % 1) * 1000)
        
        cv2.putText(frame, f"FPS: {fps_display:.1f} | {current_time}.{current_ms:03d}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode
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
        <title>Stable Low Latency Test</title>
        <style>
            body { background: #000; color: #0f0; text-align: center; font-family: monospace; padding: 20px; }
            img { max-width: 90%; border: 3px solid #0f0; }
            .clock { font-size: 32px; color: #ff0; margin: 20px; font-weight: bold; }
        </style>
        <script>
            function updateClock() {
                const now = new Date();
                const timeStr = now.toLocaleTimeString('en-US', { hour12: false }) + '.' +
                               now.getMilliseconds().toString().padStart(3,'0');
                document.getElementById('clock').innerText = 'Browser: ' + timeStr;
            }
            setInterval(updateClock, 50);
        </script>
    </head>
    <body onload="updateClock()">
        <h1>📹 STABLE LOW LATENCY TEST</h1>
        <div class="clock" id="clock"></div>
        <div style="margin: 10px;">Compare video timestamp with browser time</div>
        <img src="/video_feed">
        <div style="margin: 20px; font-size: 14px;">
            Single-threaded | 3x buffer flush | 640x480
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*60)
    print("🌐 Server: http://192.168.2.239:5000")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        camera.release()
