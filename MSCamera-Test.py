import cv2
from flask import Flask, Response

app = Flask(__name__)

# Initialize the LifeCam HD-3000
# 0 is typically the default integrated webcam; use 1 or 2 if you have multiple cameras
camera = cv2.VideoCapture(0)

# Force 720p HD resolution for LifeCam HD-3000
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)

def generate_frames():
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Use multipart/x-mixed-replace to stream individual frames
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    # Return the response generated along with the specific media type
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run locally on your PC
    app.run(host='0.0.0.0', port=5000)