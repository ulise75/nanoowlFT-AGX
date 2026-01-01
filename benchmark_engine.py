#!/usr/bin/env python3
"""
Benchmark script to compare OWL-ViT inference performance:
- Without TensorRT engine (PyTorch only)
- With TensorRT FP16 engine
"""

import argparse
import PIL.Image
import time
import torch
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output
import os

def benchmark_model(predictor, image, text, text_encodings, threshold, num_runs=30, warmup_runs=5):
    """Run inference benchmark"""
    
    # Warmup runs
    print(f"  Running {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        output = predictor.predict(
            image=image, 
            text=text, 
            text_encodings=text_encodings,
            threshold=threshold,
            pad_square=False
        )
    
    # Benchmark runs
    print(f"  Running {num_runs} benchmark iterations...")
    torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter_ns()
    
    for i in range(num_runs):
        output = predictor.predict(
            image=image, 
            text=text, 
            text_encodings=text_encodings,
            threshold=threshold,
            pad_square=False
        )
    
    torch.cuda.current_stream().synchronize()
    t1 = time.perf_counter_ns()
    
    dt = (t1 - t0) / 1e9
    fps = num_runs / dt
    avg_latency_ms = (dt / num_runs) * 1000
    
    return fps, avg_latency_ms, output

def main():
    parser = argparse.ArgumentParser(description='Benchmark OWL-ViT with and without TensorRT')
    parser.add_argument("--image", type=str, default="assets/owl_glove_small.jpg", 
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="[an owl, a glove]",
                        help="Detection prompt")
    parser.add_argument("--threshold", type=str, default="0.1,0.1",
                        help="Detection thresholds")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32",
                        help="Model name")
    parser.add_argument("--engine", type=str, default="data/owl_image_encoder_patch32.engine",
                        help="Path to TensorRT engine")
    parser.add_argument("--num_runs", type=int, default=30,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup_runs", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--output_pytorch", type=str, default="data/benchmark_pytorch_out.jpg",
                        help="Output image for PyTorch inference")
    parser.add_argument("--output_trt", type=str, default="data/benchmark_trt_out.jpg",
                        help="Output image for TensorRT inference")
    args = parser.parse_args()

    # Parse prompt and thresholds
    prompt = args.prompt.strip("][()")
    text = [t.strip() for t in prompt.split(',')]
    print(f"Detection labels: {text}")

    thresholds = args.threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]
    print(f"Thresholds: {thresholds}\n")

    # Load image
    image = PIL.Image.open(args.image)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}\n")

    print("="*80)
    print("BENCHMARK 1: PyTorch (No TensorRT)")
    print("="*80)
    
    # Initialize predictor without TensorRT
    print("Initializing PyTorch predictor...")
    predictor_pytorch = OwlPredictor(
        args.model,
        image_encoder_engine=None  # No TensorRT engine
    )
    
    print("Encoding text...")
    text_encodings = predictor_pytorch.encode_text(text)
    
    # Benchmark PyTorch
    fps_pytorch, latency_pytorch, output_pytorch = benchmark_model(
        predictor_pytorch, image, text, text_encodings, 
        thresholds, args.num_runs, args.warmup_runs
    )
    
    print(f"\n  Results:")
    print(f"    FPS: {fps_pytorch:.2f}")
    print(f"    Average Latency: {latency_pytorch:.2f} ms")
    print(f"    Detections: {len(output_pytorch.labels)}")
    
    # Save output
    image_out_pytorch = draw_owl_output(image, output_pytorch, text=text, draw_text=True)
    os.makedirs(os.path.dirname(args.output_pytorch) or '.', exist_ok=True)
    image_out_pytorch.save(args.output_pytorch)
    print(f"    Output saved to: {args.output_pytorch}")
    
    print("\n" + "="*80)
    print("BENCHMARK 2: TensorRT FP16 Engine")
    print("="*80)
    
    # Check if engine exists
    if not os.path.exists(args.engine):
        print(f"ERROR: TensorRT engine not found at {args.engine}")
        print("Please build the engine first using:")
        print(f"  python3 -m nanoowl.build_image_encoder_engine {args.engine} --fp16_mode 1")
        return
    
    # Initialize predictor with TensorRT
    print(f"Initializing TensorRT predictor with engine: {args.engine}")
    predictor_trt = OwlPredictor(
        args.model,
        image_encoder_engine=args.engine
    )
    
    print("Encoding text...")
    text_encodings = predictor_trt.encode_text(text)
    
    # Benchmark TensorRT
    fps_trt, latency_trt, output_trt = benchmark_model(
        predictor_trt, image, text, text_encodings,
        thresholds, args.num_runs, args.warmup_runs
    )
    
    print(f"\n  Results:")
    print(f"    FPS: {fps_trt:.2f}")
    print(f"    Average Latency: {latency_trt:.2f} ms")
    print(f"    Detections: {len(output_trt.labels)}")
    
    # Save output
    image_out_trt = draw_owl_output(image, output_trt, text=text, draw_text=True)
    image_out_trt.save(args.output_trt)
    print(f"    Output saved to: {args.output_trt}")
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    speedup = fps_trt / fps_pytorch
    latency_improvement = (latency_pytorch - latency_trt) / latency_pytorch * 100
    
    print(f"\nPyTorch (No TensorRT):")
    print(f"  FPS:          {fps_pytorch:.2f}")
    print(f"  Latency:      {latency_pytorch:.2f} ms")
    
    print(f"\nTensorRT FP16:")
    print(f"  FPS:          {fps_trt:.2f}")
    print(f"  Latency:      {latency_trt:.2f} ms")
    
    print(f"\nImprovement:")
    print(f"  Speedup:      {speedup:.2f}x")
    print(f"  Latency:      {latency_improvement:.1f}% faster")
    
    print("\n" + "="*80)
    print("ACCURACY CHECK")
    print("="*80)
    print(f"PyTorch detections: {len(output_pytorch.labels)}")
    print(f"TensorRT detections: {len(output_trt.labels)}")
    
    if len(output_pytorch.labels) == len(output_trt.labels):
        print("✓ Same number of detections")
    else:
        print("⚠ Different number of detections (may be due to FP16 precision)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
