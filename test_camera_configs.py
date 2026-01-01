#!/usr/bin/env python3
"""
Test camera with different configurations to find optimal settings.
"""

import cv2
import time

def test_camera_with_settings(camera_id=0, width=640, height=480, fps=30, duration=5):
    print(f"\n{'='*70}")
    print(f"Testing: {width}x{height} @ {fps} FPS")
    print(f"{'='*70}")
    
    cap = cv2.VideoCapture(camera_id)
    
    # Try to set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera")
        return None
    
    # Get actual properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"   Actual: {actual_width}x{actual_height} @ {actual_fps} FPS")
    print(f"   Format: {fourcc_str}")
    print(f"   Testing for {duration}s...")
    
    frame_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
    
    cap.release()
    
    total_time = time.time() - start_time
    measured_fps = frame_count / total_time
    
    print(f"   📊 Result: {measured_fps:.2f} FPS ({frame_count} frames in {total_time:.2f}s)")
    
    return measured_fps

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Camera Configuration Optimizer")
    print("="*70)
    
    configs = [
        # (width, height, fps)
        (640, 480, 30),
        (640, 480, 60),
        (320, 240, 30),
        (320, 240, 60),
        (1280, 720, 30),
    ]
    
    results = []
    for width, height, fps in configs:
        measured = test_camera_with_settings(0, width, height, fps, duration=5)
        if measured:
            results.append((width, height, fps, measured))
    
    print("\n" + "="*70)
    print("📊 Summary - Best Configurations")
    print("="*70)
    results.sort(key=lambda x: x[3], reverse=True)
    for i, (w, h, target_fps, actual_fps) in enumerate(results[:3], 1):
        print(f"   {i}. {w}x{h} → {actual_fps:.2f} FPS")
    print("="*70)
