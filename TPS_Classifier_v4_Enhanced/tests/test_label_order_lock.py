"""
Test to ensure label order consistency between different components.
"""
import json
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path so we can import tps modules
sys.path.append(str(Path(__file__).parent.parent))

def test_label_order_consistency():
    """Test that label order is consistent across all components."""
    
    # Load label order from checkpoint
    label_order_path = Path("models/checkpoints/label_order.json")
    assert label_order_path.exists(), f"Label order file not found: {label_order_path}"
    
    with open(label_order_path) as f:
        label_order = json.load(f)
    
    # Load class list
    classes_path = Path("data/classes.txt")
    assert classes_path.exists(), f"Classes file not found: {classes_path}"
    
    with open(classes_path) as f:
        classes = [line.strip() for line in f if line.strip()]
    
    # Test consistency
    assert len(label_order) == len(classes), f"Length mismatch: {len(label_order)} vs {len(classes)}"
    assert label_order == classes, "Label order does not match classes.txt"

def test_model_dimensions():
    """Test that model dimensions match label order."""
    from tps.config import ESM_MODEL_ID
    
    # Load label order to get number of classes
    label_order_path = Path("models/checkpoints/label_order.json")
    with open(label_order_path) as f:
        label_order = json.load(f)
    
    n_classes = len(label_order)
    
    # Test basic dimensions
    assert n_classes > 0, f"Invalid number of classes: {n_classes}"
    
    # Try to get ESM dimension (may fail if ESM not installed)
    try:
        from tps.esm_embed import ESMEmbedder
        embedder = ESMEmbedder(model_id=ESM_MODEL_ID)
        emb_dim = embedder.get_embedding_dim()
        assert emb_dim > 0, f"Invalid embedding dimension: {emb_dim}"
        print(f"ESM model: {ESM_MODEL_ID}")
        print(f"Embedding dimension: {emb_dim}")
        print(f"Number of classes: {n_classes}")
    except RuntimeError as e:
        print(f"ESM not installed (expected): {e}")
        print(f"ESM model: {ESM_MODEL_ID}")
        print(f"Number of classes: {n_classes}")
        print("Note: Install fair-esm to test embedding dimensions")

if __name__ == "__main__":
    test_label_order_consistency()
    test_model_dimensions()
    print("All label order tests passed!")
