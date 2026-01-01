#!/usr/bin/env python3
"""
Test camera maximum FPS without any inference processing.
This helps identify if the camera is the bottleneck.
"""

import cv2
import time

def test_camera_fps(camera_id=0, duration=10):
    print("=" * 70)
    print("Camera FPS Test (No Inference)")
    print("=" * 70)
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_setting = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📷 Camera Information:")
    print(f"   Camera ID: {camera_id}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Configured FPS: {fps_setting}")
    print(f"\n⏱️  Testing for {duration} seconds...\n")
    
    frame_count = 0
    start_time = time.time()
    last_report_time = start_time
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Report every second
            if current_time - last_report_time >= 1.0:
                fps = frame_count / elapsed
                print(f"   Frame {frame_count:4d} | Elapsed: {elapsed:.1f}s | Current FPS: {fps:.2f}")
                last_report_time = current_time
            
            # Stop after duration
            if elapsed >= duration:
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    
    finally:
        cap.release()
        
        # Final results
        total_time = time.time() - start_time
        average_fps = frame_count / total_time
        
        print("\n" + "=" * 70)
        print("📊 Camera Test Results")
        print("=" * 70)
        print(f"   Total frames captured: {frame_count}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average FPS: {average_fps:.2f}")
        print(f"   Frame time: {1000/average_fps:.2f} ms")
        print("\n   Analysis:")
        if average_fps >= 25:
            print(f"   ✅ Camera performing well (>{25} FPS)")
            print("   → Camera is NOT the bottleneck")
        elif average_fps >= 15:
            print(f"   ⚠️  Camera FPS moderate ({average_fps:.1f} FPS)")
            print("   → Camera may be limiting performance")
        else:
            print(f"   ❌ Camera FPS low ({average_fps:.1f} FPS)")
            print("   → Camera IS the bottleneck")
        print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test camera maximum FPS")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    args = parser.parse_args()
    
    test_camera_fps(args.camera, args.duration)
