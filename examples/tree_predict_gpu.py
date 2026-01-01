#!/usr/bin/env python3
# CPU-based Tree Prediction example
# Demonstrates hierarchical detection and classification

import PIL.Image
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree import Tree
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.clip_predictor import ClipPredictor

# Initialize predictors on GPU
print("Initializing OWL and CLIP predictors on GPU...")
owl_predictor = OwlPredictor(
    model_name="google/owlvit-base-patch32",
    device="cuda"
)
clip_predictor = ClipPredictor(
    model_name="ViT-B/32",
    device="cuda"
)

# Initialize tree predictor
tree_predictor = TreePredictor(
    owl_predictor=owl_predictor,
    clip_predictor=clip_predictor,
    device="cuda"
)

# Load image
print("Loading image...")
image = PIL.Image.open("../assets/owl_glove_small.jpg")

# Define hierarchical detection tree
# First detect object, then classify if it's a bird or a clothing item
tree_prompt = "[an object](a bird, a clothing item)"
print(f"\nTree prompt: {tree_prompt}")
print("This will detect objects and then classify them")

# Parse tree from prompt
tree = Tree.from_prompt(tree_prompt)
print(f"Tree has {len(tree.nodes)} nodes")

# Run tree prediction
print("\nRunning tree prediction...")
output = tree_predictor.predict(
    image=image,
    tree=tree,
    threshold=0.15
)

# Print results
print(f"\nTree Detection Results:")
print(f"  Found {len(output.detections)} detections:")
for i, detection in enumerate(output.detections):
    print(f"    {i+1}. Detection ID: {detection.id}, Parent ID: {detection.parent_id}")
    print(f"       Box: {detection.box}")
    print(f"       Labels: {detection.labels}, Scores: {detection.scores}")

# Draw and save visualization
print("\nSaving visualization...")
output_image = draw_tree_output(image, output, tree)
output_image.save("../data/tree_predict_gpu_out.jpg")
print("✓ Saved to: data/tree_predict_gpu_out.jpg")
print("\n✓ GPU-accelerated tree detection complete!")
