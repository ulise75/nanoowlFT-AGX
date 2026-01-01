#!/usr/bin/env python3
# Real-time USB Camera Inference with NanoOWL on GPU

import cv2
import numpy as np
import PIL.Image
import time
import argparse
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output

def main():
    parser = argparse.ArgumentParser(description='Real-time object detection with USB camera')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--prompt', type=str, default='a person,a face,a hand', 
                        help='Comma-separated detection prompts (default: "a person,a face,a hand")')
    parser.add_argument('--threshold', type=float, default=0.15, 
                        help='Detection confidence threshold (default: 0.15)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--show-fps', action='store_true', help='Show FPS counter')
    args = parser.parse_args()

    # Parse detection prompts
    text = [t.strip() for t in args.prompt.split(',')]
    print(f"Detection prompts: {text}")

    # Initialize predictor on GPU
    print(f"Initializing OWL-ViT predictor on {args.device.upper()}...")
    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        device=args.device
    )

    # Encode text once (can be reused for all frames)
    print("Encoding detection prompts...")
    text_encodings = predictor.encode_text(text)
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("\nStarting inference...")
    print("Press 'q' to quit, 's' to save current frame\n")
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0.0
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = PIL.Image.fromarray(frame_rgb)
            
            # Run inference
            inference_start = time.time()
            output = predictor.predict(
                image=image_pil,
                text=text,
                text_encodings=text_encodings,
                threshold=args.threshold,
                pad_square=False
            )
            inference_time = time.time() - inference_start
            
            # Draw detections on PIL image
            output_image = draw_owl_output(image_pil, output, text=text, draw_text=True)
            
            # Convert back to OpenCV format for display
            output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            # Add info overlay
            info_text = f"Detections: {len(output.labels)} | Inference: {inference_time*1000:.1f}ms"
            cv2.putText(output_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate and display FPS
            if args.show_fps:
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps_display = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                cv2.putText(output_frame, f"FPS: {fps_display:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show detection details
            if len(output.labels) > 0:
                det_info = f"Found: {', '.join([text[label] for label in output.labels])}"
                cv2.putText(output_frame, det_info, (10, output_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('NanoOWL Camera Inference', output_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames - Last inference: {inference_time*1000:.1f}ms - Detections: {len(output.labels)}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"data/camera_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"Saved frame to: {filename}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames processed: {frame_count}")
        print("Camera closed.")

if __name__ == "__main__":
    main()
