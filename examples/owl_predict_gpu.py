#!/usr/bin/env python3
# GPU-based OWL-ViT prediction example (no TensorRT engine required)

import PIL.Image
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output

# Initialize predictor on GPU (no TensorRT engine)
print("Initializing OWL-ViT predictor on GPU...")
predictor = OwlPredictor(
    "google/owlvit-base-patch32",
    device="cuda"  # Use GPU
)

# Load image
print("Loading image...")
image = PIL.Image.open("../assets/owl_glove_small.jpg")

# Define what to detect
text = ["an owl", "a glove"]
threshold = 0.1

print(f"Detecting: {text} with threshold {threshold}")

# Encode text
print("Encoding text...")
text_encodings = predictor.encode_text(text)

# Run prediction on GPU
print("Running prediction on GPU...")
output = predictor.predict(
    image=image,
    text=text,
    text_encodings=text_encodings,
    threshold=threshold,
    pad_square=False
)

# Print results
print(f"\nDetection Results:")
print(f"  Found {len(output.labels)} objects:")
for i, (label, score) in enumerate(zip(output.labels, output.scores)):
    print(f"    {i+1}. {text[label]} (confidence: {score:.3f})")

# Draw and save visualization
print("\nSaving visualization...")
output_image = draw_owl_output(image, output, text=text, draw_text=True)
output_image.save("../data/owl_predict_gpu_out.jpg")
print("✓ Saved to: data/owl_predict_gpu_out.jpg")
print("\n✓ GPU-accelerated detection complete!")
