#!/usr/bin/env python3
"""
Simple test to verify NanoOWL setup is working correctly on CPU
"""

import sys
import torch
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree import Tree

def test_imports():
    print("✓ All imports successful")

def test_owl_predictor_cpu():
    print("\n[1/3] Testing OwlPredictor on CPU...")
    try:
        predictor = OwlPredictor(
            "google/owlvit-base-patch32",
            device="cpu"
        )
        print("✓ OwlPredictor initialized on CPU")
        
        # Test encoding text
        text = ["an owl", "a glove"]
        text_encodings = predictor.encode_text(text)
        print(f"✓ Text encoding successful: {text_encodings.text_embeds.shape}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_tree_parsing():
    print("\n[2/3] Testing Tree parsing...")
    try:
        # Test various tree prompts
        tree1 = Tree.from_prompt("[a face]")
        print(f"✓ Simple detection tree parsed: {len(tree1.nodes)} nodes")
        
        tree2 = Tree.from_prompt("[a face](a dog, a cat)")
        print(f"✓ Hierarchical tree parsed: {len(tree2.nodes)} nodes")
        
        tree3 = Tree.from_prompt("(indoors, outdoors)")
        print(f"✓ Classification tree parsed: {len(tree3.nodes)} nodes")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_clip():
    print("\n[3/3] Testing CLIP on CPU...")
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        print("✓ CLIP model loaded on CPU")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("NanoOWL Setup Test")
    print("=" * 60)
    
    try:
        test_imports()
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)
    
    results = []
    results.append(test_owl_predictor_cpu())
    results.append(test_tree_parsing())
    results.append(test_clip())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)
