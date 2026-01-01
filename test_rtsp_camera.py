#!/usr/bin/env python3
"""Test RTSP camera FPS from Vantrue E360"""

import cv2
import time

def test_rtsp_camera(rtsp_url, duration=10):
    print("="*70)
    print("RTSP Camera FPS Test - Vantrue E360")
    print("="*70)
    print(f"Stream: {rtsp_url}\n")
    
    # Try OpenCV with RTSP
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Failed to open RTSP stream with OpenCV")
        return
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_prop = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📷 Camera Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Reported FPS: {fps_prop}")
    print(f"\n⏱️  Testing for {duration} seconds...\n")
    
    # Warm up
    for _ in range(5):
        cap.read()
    
    frame_count = 0
    start_time = time.time()
    last_report = start_time
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            current = time.time()
            elapsed = current - start_time
            
            if current - last_report >= 1.0:
                fps = frame_count / elapsed
                print(f"   Frame {frame_count:4d} | {elapsed:5.1f}s | FPS: {fps:5.2f}")
                last_report = current
            
            if elapsed >= duration:
                break
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        cap.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print("\n" + "="*70)
        print("📊 Results")
        print("="*70)
        print(f"   Frames: {frame_count}")
        print(f"   Duration: {total_time:.2f}s")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Frame time: {1000/avg_fps:.2f}ms")
        
        if avg_fps >= 25:
            print(f"\n   ✅ Excellent - {avg_fps:.1f} FPS!")
        elif avg_fps >= 15:
            print(f"\n   ✅ Good - {avg_fps:.1f} FPS")
        else:
            print(f"\n   ⚠️  Low - {avg_fps:.1f} FPS")
        print("="*70)

if __name__ == "__main__":
    test_rtsp_camera("rtsp://192.168.1.254/live", duration=10)
