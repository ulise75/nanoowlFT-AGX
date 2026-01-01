#!/usr/bin/env python3
"""
Test camera FPS with optimized OpenCV settings for USB cameras.
"""

import cv2
import time

def test_optimized_camera(camera_id=0, duration=10):
    print("="*70)
    print("Optimized Camera FPS Test (USB Camera)")
    print("="*70)
    
    # Open with V4L2 backend explicitly
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return
    
    # Optimize settings for USB camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))  # YUYV format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    # Get actual settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_setting = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"\n📷 Camera: Microsoft LifeCam HD-3000")
    print(f"   Resolution: {width}x{height}")
    print(f"   Format: {fourcc_str}")
    print(f"   Target FPS: {fps_setting}")
    print(f"\n⏱️  Testing for {duration} seconds...\n")
    
    frame_count = 0
    start_time = time.time()
    last_report = start_time
    
    # Warm up - discard first few frames
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
            print(f"\n   ✅ Camera OK - {avg_fps:.1f} FPS is sufficient")
        elif avg_fps >= 15:
            print(f"\n   ⚠️  Camera moderate - {avg_fps:.1f} FPS (acceptable but not ideal)")
        else:
            print(f"\n   ❌ Camera slow - {avg_fps:.1f} FPS (this IS the bottleneck)")
            print("\n   Recommendations:")
            print("   • Microsoft LifeCam HD-3000 may have USB bandwidth limitations")
            print("   • Try disconnecting other USB devices")
            print("   • Consider using a better camera (e.g., Logitech C920/C922)")
        print("="*70)

if __name__ == "__main__":
    test_optimized_camera(0, 10)
