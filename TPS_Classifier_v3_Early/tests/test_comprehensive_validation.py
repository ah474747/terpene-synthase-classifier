#!/usr/bin/env python3
"""
Comprehensive TPS Classifier Validation Test
Tests all stabilization components in one file
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tps.config import ESM_MODEL_ID, ESM2_DIM, ENG_DIM, N_CLASSES
from tps.features.engineered import generate_engineered_features
from tps.features.structure import generate_gcn_features
from tps.retrieval.knn_head import KNNBlender
from tps.hierarchy.head import HierarchyHead
from tps.eval.calibration import CalibratedPredictor
from tps.eval.identity_split import IdentitySplitter
from tps.utils import set_seed


def test_deterministic_features():
    """Test that features are deterministic"""
    print("Testing deterministic features...")
    
    sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN"
    
    # Test engineered features determinism
    set_seed(42)
    features1 = generate_engineered_features(sequence)
    
    set_seed(123)
    features2 = generate_engineered_features(sequence)
    
    set_seed(42)
    features3 = generate_engineered_features(sequence)
    
    assert np.allclose(features1, features2), "Engineered features should be deterministic"
    assert np.allclose(features1, features3), "Engineered features should be identical with same seed"
    
    # Test structural features determinism
    set_seed(42)
    graph1, has_struct1 = generate_gcn_features(sequence, "TEST123")
    
    set_seed(123)
    graph2, has_struct2 = generate_gcn_features(sequence, "TEST123")
    
    assert has_struct1 == has_struct2, "Structure detection should be deterministic"
    assert np.allclose(graph1.node_features, graph2.node_features), "Fallback features should be identical"
    
    print("✓ Deterministic features test passed")


def test_knn_blending():
    """Test kNN blending functionality"""
    print("Testing kNN blending...")
    
    n_train, n_test, n_classes, embedding_dim = 100, 10, 30, 1280
    
    train_embeddings = np.random.randn(n_train, embedding_dim).astype(np.float32)
    train_labels = np.random.randint(0, n_classes, size=n_train)
    test_embeddings = np.random.randn(n_test, embedding_dim).astype(np.float32)
    
    blender = KNNBlender(k=5, alpha=0.7)
    blender.fit(train_embeddings, train_labels)
    
    model_probs = np.random.rand(n_test, n_classes).astype(np.float32)
    model_probs = model_probs / model_probs.sum(axis=1, keepdims=True)
    
    blended_probs = blender.blend(model_probs, test_embeddings)
    
    assert blended_probs.shape == (n_test, n_classes), "Blended probabilities should have correct shape"
    assert np.allclose(blended_probs.sum(axis=1), 1.0, rtol=1e-5), "Probabilities should sum to 1"
    assert np.all(blended_probs >= 0), "All probabilities should be non-negative"
    assert np.all(blended_probs <= 1), "All probabilities should be <= 1"
    
    print("✓ kNN blending test passed")


def test_hierarchy_masking():
    """Test hierarchy masking functionality"""
    print("Testing hierarchy masking...")
    
    hierarchy_head = HierarchyHead(n_classes=30, latent_dim=512)
    latent_features = torch.randn(2, 512)
    type_logits = hierarchy_head.type_head(latent_features)
    type_probs = torch.softmax(type_logits, dim=-1)
    fine_logits = torch.randn(2, 30)
    masked_logits = hierarchy_head.apply_type_mask(fine_logits, type_probs)
    
    assert masked_logits.shape == fine_logits.shape, "Masked logits should have same shape"
    assert not torch.allclose(masked_logits, fine_logits), "Masking should modify logits"
    
    print("✓ Hierarchy masking test passed")


def test_calibration():
    """Test calibration functionality"""
    print("Testing calibration...")
    
    n_samples, n_classes = 100, 30
    raw_logits = np.random.randn(n_samples, n_classes)
    true_labels = np.random.randint(0, n_classes, size=n_samples)
    
    calibrator = CalibratedPredictor()
    calibrator.fit(raw_logits, true_labels)
    
    calibrated_probs = calibrator.predict_proba(raw_logits)
    
    assert calibrated_probs.shape == (n_samples, n_classes), "Calibrated probabilities should have correct shape"
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0, rtol=1e-5), "Probabilities should sum to 1"
    assert np.all(calibrated_probs >= 0), "All probabilities should be non-negative"
    assert np.all(calibrated_probs <= 1), "All probabilities should be <= 1"
    
    print("✓ Calibration test passed")


def test_identity_splitting():
    """Test identity splitting functionality"""
    print("Testing identity splitting...")
    
    sequences = {
        'seq1': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN',
        'seq2': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN',  # Identical
        'seq3': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',  # Very different
    }
    
    splitter = IdentitySplitter(identity_threshold=0.4)
    clusters = splitter.cluster_sequences(sequences)
    
    assert len(clusters) >= 1, "Should have at least one cluster"
    
    # Check that identical sequences are clustered together
    cluster_ids = {}
    for cluster_id, seq_list in clusters.items():
        for seq_id in seq_list:
            cluster_ids[seq_id] = cluster_id
    
    assert cluster_ids['seq1'] == cluster_ids['seq2'], "Identical sequences should be in same cluster"
    
    print("✓ Identity splitting test passed")


def test_config_consistency():
    """Test configuration consistency"""
    print("Testing configuration consistency...")
    
    # Check that ESM model ID is defined
    assert ESM_MODEL_ID is not None, "ESM_MODEL_ID should be defined"
    assert isinstance(ESM_MODEL_ID, str), "ESM_MODEL_ID should be a string"
    assert "esm" in ESM_MODEL_ID.lower(), "Should be an ESM model"
    
    # Check dimensions
    assert ESM2_DIM > 0, "ESM2_DIM should be positive"
    assert ENG_DIM > 0, "ENG_DIM should be positive"
    assert N_CLASSES > 0, "N_CLASSES should be positive"
    assert ESM2_DIM >= 512, "ESM2_DIM should be >= 512"
    
    print("✓ Configuration consistency test passed")


def main():
    """Run all validation tests"""
    print("🧪 Running Comprehensive TPS Classifier Validation")
    print("=" * 60)
    
    try:
        test_config_consistency()
        test_deterministic_features()
        test_knn_blending()
        test_hierarchy_masking()
        test_calibration()
        test_identity_splitting()
        
        print("\n" + "=" * 60)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ TPS Classifier stabilization is working correctly")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


