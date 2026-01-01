#!/usr/bin/env python3
"""
WebRTC low-latency video streaming from RTSP camera.
"""

import asyncio
import cv2
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import time

class RTSPVideoTrack(VideoStreamTrack):
    """Video track from RTSP camera"""
    
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_count = 0
        
    async def recv(self):
        """Receive the next video frame"""
        if self.cap is None:
            print(f"📡 Opening RTSP: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        # Flush buffer - grab multiple frames
        for _ in range(2):
            self.cap.grab()
        
        ret, frame = self.cap.retrieve()
        
        if not ret or frame is None:
            # Reconnect if failed
            self.cap.release()
            self.cap = None
            await asyncio.sleep(0.1)
            return await self.recv()
        
        # Resize if needed
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))
        
        self.frame_count += 1
        
        # Add FPS counter
        if self.frame_count % 30 == 0:
            cv2.putText(frame, f"Frame: {self.frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create video frame
        video_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame.pts = self.frame_count
        video_frame.time_base = '1/30'
        
        return video_frame

# Global state
pcs = set()
rtsp_url = "rtsp://192.168.1.254/live"

async def offer(request):
    """Handle WebRTC offer"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
    
    # Add video track BEFORE setting remote description
    video_track = RTSPVideoTrack(rtsp_url)
    pc.addTrack(video_track)
    print(f"✅ Video track added")
    
    # Set remote description first
    await pc.setRemoteDescription(offer)
    print(f"✅ Remote description set")
    
    # Create and set local description
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print(f"✅ Local description set")
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

async def index(request):
    """Serve the HTML page"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>WebRTC Low Latency Stream</title>
    <style>
        body {
            background: #000;
            color: #0f0;
            font-family: monospace;
            text-align: center;
            padding: 20px;
        }
        video {
            width: 90%;
            max-width: 800px;
            border: 3px solid #0f0;
            margin: 20px auto;
        }
        button {
            background: #0f0;
            color: #000;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            cursor: pointer;
            font-family: monospace;
            font-weight: bold;
        }
        button:hover {
            background: #0c0;
        }
        .status {
            margin: 20px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <h1>🚀 WebRTC Ultra-Low Latency</h1>
    <div class="status" id="status">Ready to connect...</div>
    <button id="start" onclick="start()">START STREAM</button>
    <br><br>
    <video id="video" autoplay playsinline muted></video>
    
    <script>
        const video = document.getElementById('video');
        const status = document.getElementById('status');
        let pc = null;
        
        async function start() {
            status.innerText = 'Connecting...';
            
            pc = new RTCPeerConnection({
                iceServers: []
            });
            
            pc.ontrack = (event) => {
                video.srcObject = event.streams[0];
                status.innerText = '✅ Connected - Ultra Low Latency';
            };
            
            pc.oniceconnectionstatechange = () => {
                status.innerText = 'ICE: ' + pc.iceConnectionState;
            };
            
            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            // Send offer to server
            const response = await fetch('/offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                })
            });
            
            const answer = await response.json();
            await pc.setRemoteDescription(answer);
        }
        
        // Auto-start
        setTimeout(start, 500);
    </script>
</body>
</html>
    """
    return web.Response(text=html, content_type="text/html")

async def on_shutdown(app):
    """Close peer connections on shutdown"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 WebRTC Ultra-Low Latency Server")
    print("="*70)
    print(f"   RTSP Source: {rtsp_url}")
    print(f"   WebRTC Server: http://192.168.2.239:8080")
    print(f"   Expected Latency: <500ms (much better than MJPEG)")
    print("="*70 + "\n")
    
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    
    web.run_app(app, host="0.0.0.0", port=8080)
