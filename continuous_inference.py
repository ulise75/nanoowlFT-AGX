#!/usr/bin/env python3
"""
Continuous inference demo with TensorRT engine
Runs for a specified duration and reports performance metrics
"""

import argparse
import PIL.Image
import time
import torch
import os
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description='Run continuous inference with TensorRT engine')
    parser.add_argument("--image", type=str, default="assets/owl_glove_small.jpg",
                        help="Test image to use")
    parser.add_argument("--prompt", type=str, default="an owl,a glove",
                        help="Detection prompts (comma-separated)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Detection threshold")
    parser.add_argument("--engine", type=str, default="data/owl_image_encoder_patch32.engine",
                        help="Path to TensorRT engine")
    parser.add_argument("--duration", type=int, default=600,
                        help="Duration to run in seconds (default: 600 = 10 minutes)")
    parser.add_argument("--output_dir", type=str, default="data/continuous_outputs",
                        help="Directory to save output images")
    parser.add_argument("--save_interval", type=int, default=30,
                        help="Save output image every N seconds")
    args = parser.parse_args()

    # Parse prompts
    text = [t.strip() for t in args.prompt.split(',')]
    print(f"Detection labels: {text}")
    print(f"Threshold: {args.threshold}")
    print(f"Duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    print(f"Using TensorRT engine: {args.engine}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check engine exists
    if not os.path.exists(args.engine):
        print(f"ERROR: TensorRT engine not found at {args.engine}")
        return

    # Load image
    print("Loading test image...")
    image = PIL.Image.open(args.image)
    print(f"Image size: {image.size}\n")

    # Initialize predictor
    print("Initializing TensorRT predictor...")
    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        image_encoder_engine=args.engine
    )

    # Encode text once
    print("Encoding detection prompts...")
    text_encodings = predictor.encode_text(text)
    print("Ready to start inference!\n")

    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = predictor.predict(
            image=image,
            text=text,
            text_encodings=text_encodings,
            threshold=args.threshold,
            pad_square=False
        )
    torch.cuda.current_stream().synchronize()
    print("Warmup complete!\n")

    # Start continuous inference
    print("="*80)
    print(f"STARTING CONTINUOUS INFERENCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    start_time = time.time()
    end_time = start_time + args.duration
    iteration = 0
    total_inference_time = 0
    last_save_time = start_time
    last_report_time = start_time

    try:
        while time.time() < end_time:
            # Run inference
            torch.cuda.current_stream().synchronize()
            t0 = time.perf_counter()
            
            output = predictor.predict(
                image=image,
                text=text,
                text_encodings=text_encodings,
                threshold=args.threshold,
                pad_square=False
            )
            
            torch.cuda.current_stream().synchronize()
            t1 = time.perf_counter()
            
            inference_time = t1 - t0
            total_inference_time += inference_time
            iteration += 1
            
            current_time = time.time()
            
            # Save image periodically
            if current_time - last_save_time >= args.save_interval:
                output_image = draw_owl_output(image, output, text=text, draw_text=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(args.output_dir, f"detection_{timestamp}.jpg")
                output_image.save(output_path)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved detection to: {output_path}")
                last_save_time = current_time
            
            # Report stats every 10 seconds
            if current_time - last_report_time >= 10:
                elapsed = current_time - start_time
                avg_fps = iteration / total_inference_time if total_inference_time > 0 else 0
                avg_latency = (total_inference_time / iteration * 1000) if iteration > 0 else 0
                remaining = end_time - current_time
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update:")
                print(f"  Elapsed: {elapsed:.1f}s / {args.duration}s ({elapsed/args.duration*100:.1f}%)")
                print(f"  Remaining: {remaining:.1f}s")
                print(f"  Iterations: {iteration}")
                print(f"  Avg FPS: {avg_fps:.2f}")
                print(f"  Avg Latency: {avg_latency:.2f} ms")
                print(f"  Detections: {len(output.labels)}")
                
                last_report_time = current_time

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    
    # Final report
    total_elapsed = time.time() - start_time
    avg_fps = iteration / total_inference_time if total_inference_time > 0 else 0
    avg_latency = (total_inference_time / iteration * 1000) if iteration > 0 else 0
    
    print("\n" + "="*80)
    print(f"FINAL REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Total Runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Total Iterations: {iteration}")
    print(f"Total Inference Time: {total_inference_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {iteration/total_elapsed:.2f} inferences/second (wall time)")
    print(f"Images saved to: {args.output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
