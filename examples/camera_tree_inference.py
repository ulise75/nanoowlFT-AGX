#!/usr/bin/env python3
# Real-time USB Camera Inference with Tree Detection (NanoOWL) on GPU

import cv2
import numpy as np
import PIL.Image
import time
import argparse
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree import Tree
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.clip_predictor import ClipPredictor

def main():
    parser = argparse.ArgumentParser(description='Real-time tree detection with USB camera')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--prompt', type=str, default='[a person](a man, a woman)', 
                        help='Tree detection prompt (default: "[a person](a man, a woman)")')
    parser.add_argument('--threshold', type=float, default=0.15, 
                        help='Detection confidence threshold (default: 0.15)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--show-fps', action='store_true', help='Show FPS counter')
    args = parser.parse_args()

    # Parse tree prompt
    print(f"Tree prompt: {args.prompt}")
    tree = Tree.from_prompt(args.prompt)
    print(f"Tree has {len(tree.nodes)} nodes")

    # Initialize predictors on GPU
    print(f"Initializing OWL and CLIP predictors on {args.device.upper()}...")
    owl_predictor = OwlPredictor(
        model_name="google/owlvit-base-patch32",
        device=args.device
    )
    clip_predictor = ClipPredictor(
        model_name="ViT-B/32",
        device=args.device
    )
    
    tree_predictor = TreePredictor(
        owl_predictor=owl_predictor,
        clip_predictor=clip_predictor,
        device=args.device
    )
    
    # Pre-encode text for the tree
    print("Encoding tree prompts...")
    clip_text_encodings = tree_predictor.encode_clip_text(tree)
    owl_text_encodings = tree_predictor.encode_owl_text(tree)
    
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
    print("\nStarting tree inference...")
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
            
            # Run tree inference
            inference_start = time.time()
            output = tree_predictor.predict(
                image=image_pil,
                tree=tree,
                threshold=args.threshold,
                clip_text_encodings=clip_text_encodings,
                owl_text_encodings=owl_text_encodings
            )
            inference_time = time.time() - inference_start
            
            # Draw tree detections on PIL image
            output_image = draw_tree_output(image_pil, output, tree)
            
            # Convert back to OpenCV format for display
            output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            # Add info overlay
            info_text = f"Detections: {len(output.detections)} | Inference: {inference_time*1000:.1f}ms"
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
            
            # Display frame
            cv2.imshow('NanoOWL Tree Camera Inference', output_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames - Last inference: {inference_time*1000:.1f}ms - Detections: {len(output.detections)}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"data/camera_tree_capture_{int(time.time())}.jpg"
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
