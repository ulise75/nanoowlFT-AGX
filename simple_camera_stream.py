#!/usr/bin/env python3
"""
Simple camera streaming - matching MSCamera-Test.py settings exactly
"""

import cv2
from flask import Flask, Response

app = Flask(__name__)

# Camera setup - EXACT same as MSCamera-Test.py
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)

def generate_frames():
    """Generator function - EXACT same as MSCamera-Test.py"""
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format (default quality 95%)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Use multipart/x-mixed-replace to stream individual frames
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Camera - 720p Test</title>
        <style>
            body { margin: 0; padding: 20px; background: #1a1a1a; color: white; 
                   font-family: Arial; text-align: center; }
            h1 { color: #4CAF50; }
            img { max-width: 90%; border: 3px solid #4CAF50; margin: 20px auto; display: block; }
        </style>
    </head>
    <body>
        <h1>📷 Simple Camera Stream - 720p (No Inference)</h1>
        <p>Matching MSCamera-Test.py settings exactly</p>
        <img src="/video_feed" alt="Camera Feed">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route - EXACT same as MSCamera-Test.py"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Simple Camera Stream (720p - No Inference)")
    print("="*60)
    print(f"  Matching MSCamera-Test.py settings")
    print(f"  Resolution: 1280x720")
    print(f"  JPEG Quality: 95% (default)")
    print(f"\n  🔗 Open in browser:")
    print(f"     http://192.168.2.239:5000")
    print(f"\n  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
