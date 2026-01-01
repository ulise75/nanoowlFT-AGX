#!/usr/bin/env python3
# Headless USB Camera Inference with NanoOWL (saves frames instead of displaying)

import cv2
import numpy as np
import PIL.Image
import time
import argparse
import os
import gc
import torch
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output

def main():
    parser = argparse.ArgumentParser(description='Headless object detection with USB camera')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--prompt', type=str, default='a person,a face,a hand', 
                        help='Comma-separated detection prompts')
    parser.add_argument('--threshold', type=float, default=0.15, 
                        help='Detection confidence threshold (default: 0.15)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Maximum frames to process (default: 100)')
    parser.add_argument('--save-interval', type=int, default=30,
                        help='Save every Nth frame (default: 30)')
    parser.add_argument('--output-dir', type=str, default='../data/camera_output',
                        help='Output directory for saved frames')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.7,
                        help='Fraction of GPU memory to use (default: 0.7)')
    args = parser.parse_args()
    
    # Set PyTorch memory allocator settings for better GPU stability
    if args.device == 'cuda':
        # Optimize memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        # Clear any existing cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU Memory before initialization: {torch.cuda.memory_allocated()/1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse detection prompts
    text = [t.strip() for t in args.prompt.split(',')]
    print(f"Detection prompts: {text}")

    # Initialize predictor
    print(f"Initializing OWL-ViT predictor on {args.device.upper()}...")
    try:
        predictor = OwlPredictor(
            "google/owlvit-base-patch32",
            device=args.device
        )
        
        # Encode text once
        print("Encoding detection prompts...")
        text_encodings = predictor.encode_text(text)
        
        # If using GPU, clear cache after initialization
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            
    except RuntimeError as e:
        print(f"\nError initializing model on {args.device.upper()}: {e}")
        print("Try reducing --gpu-memory-fraction or running on CPU with --device cpu")
        return
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"\nStarting inference (headless mode)...")
    print(f"Will save every {args.save_interval} frames to: {args.output_dir}")
    print(f"Processing up to {args.max_frames} frames\n")
    
    frame_count = 0
    total_inference_time = 0
    saved_count = 0
    
    try:
        while frame_count < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = PIL.Image.fromarray(frame_rgb)
            
            # Run inference with explicit memory management
            inference_start = time.time()
            try:
                with torch.no_grad():  # Disable gradient computation for inference
                    output = predictor.predict(
                        image=image_pil,
                        text=text,
                        text_encodings=text_encodings,
                        threshold=args.threshold,
                        pad_square=False
                    )
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                
                # Aggressive GPU memory management
                if args.device == 'cuda':
                    # Clear cache after every frame to prevent buildup
                    torch.cuda.empty_cache()
                    # Synchronize to ensure operations complete
                    torch.cuda.synchronize()
                    
            except RuntimeError as e:
                print(f"\nError during inference on frame {frame_count}: {e}")
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                print("Attempting to continue...")
                time.sleep(0.1)  # Brief pause
                continue
            
            frame_count += 1
            
            # Print progress
            detections = len(output.labels)
            det_labels = [text[label] for label in output.labels] if detections > 0 else []
            print(f"Frame {frame_count}/{args.max_frames} | "
                  f"Inference: {inference_time*1000:.1f}ms | "
                  f"Detections: {detections} {det_labels}")
            
            # Save frame at intervals
            if frame_count % args.save_interval == 0:
                # Draw detections
                output_image = draw_owl_output(image_pil, output, text=text, draw_text=True)
                output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
                
                # Add info overlay
                info_text = f"Frame {frame_count} | Detections: {detections} | {inference_time*1000:.1f}ms"
                cv2.putText(output_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save frame
                filename = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(filename, output_frame)
                saved_count += 1
                print(f"  → Saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        
        # Clear GPU memory
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames saved: {saved_count}")
        print(f"  Average inference time: {(total_inference_time/frame_count)*1000:.1f}ms")
        print(f"  Average FPS: {frame_count/total_inference_time:.2f}")
        print(f"  Output directory: {os.path.abspath(args.output_dir)}")
        print(f"{'='*60}")
        print("Camera closed.")

if __name__ == "__main__":
    main()
