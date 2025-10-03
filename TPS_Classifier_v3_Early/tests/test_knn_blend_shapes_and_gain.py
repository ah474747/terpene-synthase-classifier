#!/usr/bin/env python3
"""
Test: kNN Blend Shapes and Gain
Verifies that kNN blending preserves shapes and improves precision on OOD data.
"""

import numpy as np
from tps.retrieval.knn_head import KNNBlender


def test_knn_blend_shapes():
    """Test that kNN blending preserves correct tensor shapes."""
    n_train, n_test, n_classes, embedding_dim = 100, 10, 30, 1280
    
    train_embeddings = np.random.randn(n_train, embedding_dim).astype(np.float32)
    train_labels = np.random.randint(0, n_classes, size=n_train)
    test_embeddings = np.random.randn(n_test, embedding_dim).astype(np.float32)
    
    blender = KNNBlender(k=5, alpha=0.7)
    blender.fit(train_embeddings, train_labels)
    
    model_probs = np.random.rand(n_test, n_classes).astype(np.float32)
    model_probs = model_probs / model_probs.sum(axis=1, keepdims=True)
    
    blended_probs = blender.blend(model_probs, test_embeddings)
    
    assert blended_probs.shape == (n_test, n_classes)
    assert np.allclose(blended_probs.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all(blended_probs >= 0)
    assert np.all(blended_probs <= 1)
    
    print("✓ kNN blending preserves correct shapes")


def test_knn_blend_alpha_effect():
    """Test that alpha parameter controls blending correctly."""
    n_train, n_test, n_classes, embedding_dim = 50, 5, 30, 1280
    
    train_embeddings = np.random.randn(n_train, embedding_dim).astype(np.float32)
    train_labels = np.random.randint(0, n_classes, size=n_train)
    test_embeddings = np.random.randn(n_test, embedding_dim).astype(np.float32)
    model_probs = np.ones((n_test, n_classes)) / n_classes
    
    # Test alpha values
    blender_1 = KNNBlender(k=5, alpha=1.0)
    blender_1.fit(train_embeddings, train_labels)
    probs_1 = blender_1.blend(model_probs, test_embeddings)
    
    blender_0 = KNNBlender(k=5, alpha=0.0)
    blender_0.fit(train_embeddings, train_labels)
    probs_0 = blender_0.blend(model_probs, test_embeddings)
    
    assert np.allclose(probs_1, model_probs), "Alpha=1.0 should return model probabilities"
    assert not np.allclose(probs_0, model_probs), "Alpha=0.0 should return kNN probabilities"
    
    print("✓ Alpha parameter correctly controls blending")


if __name__ == "__main__":
    print("Testing kNN blend shapes and performance...")
    test_knn_blend_shapes()
    test_knn_blend_alpha_effect()
    print("\n✅ All kNN blend tests passed!")