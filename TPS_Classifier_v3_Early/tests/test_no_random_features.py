#!/usr/bin/env python3
"""
Test: No Random Features
Verifies that engineered and structural features are deterministic (no randomness).
"""

import numpy as np
import torch
from tps.features.engineered import generate_engineered_features
from tps.features.structure import generate_gcn_features
from tps.utils import set_seed


def test_deterministic_engineered_features():
    """Test that engineered features are identical across runs with same seed."""
    sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN"
    
    # Run 1
    set_seed(42)
    features1 = generate_engineered_features(sequence)
    
    # Run 2 (different seed)
    set_seed(123)
    features2 = generate_engineered_features(sequence)
    
    # Run 3 (back to original seed)
    set_seed(42)
    features3 = generate_engineered_features(sequence)
    
    # Features should be identical regardless of seed
    assert np.allclose(features1, features2), "Engineered features should be deterministic"
    assert np.allclose(features1, features3), "Engineered features should be identical with same seed"
    print("✓ Engineered features are deterministic")


def test_deterministic_structural_features():
    """Test that structural features are deterministic when structure is missing."""
    sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN"
    uniprot_id = "TEST123"  # Non-existent ID to force fallback
    
    # Run 1
    set_seed(42)
    graph1, has_struct1 = generate_gcn_features(sequence, uniprot_id)
    
    # Run 2 (different seed)
    set_seed(123)
    graph2, has_struct2 = generate_gcn_features(sequence, uniprot_id)
    
    # Run 3 (back to original seed)
    set_seed(42)
    graph3, has_struct3 = generate_gcn_features(sequence, uniprot_id)
    
    # Structure availability should be deterministic
    assert has_struct1 == has_struct2 == has_struct3, "Structure detection should be deterministic"
    
    # When structure is missing, features should be identical (all zeros)
    if not has_struct1:
        assert np.allclose(graph1.node_features, graph2.node_features), "Fallback features should be identical"
        assert np.allclose(graph1.node_features, graph3.node_features), "Fallback features should be identical"
        assert np.allclose(graph1.edge_index, graph2.edge_index), "Fallback edges should be identical"
        assert np.allclose(graph1.edge_index, graph3.edge_index), "Fallback edges should be identical"
    
    print("✓ Structural features are deterministic")


def test_no_random_placeholders():
    """Test that no random placeholders exist in feature generation."""
    sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN"
    
    # Check engineered features contain no random values
    features = generate_engineered_features(sequence)
    assert not np.any(np.isnan(features)), "Engineered features should not contain NaN"
    assert not np.any(np.isinf(features)), "Engineered features should not contain Inf"
    
    # Check that features are not all zeros (indicating proper computation)
    assert np.sum(np.abs(features)) > 0, "Engineered features should be non-zero"
    
    print("✓ No random placeholders in engineered features")


def test_structural_fallback_consistency():
    """Test that structural fallback is consistent and deterministic."""
    sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQN"
    uniprot_id = "NONEXISTENT123"
    
    # Multiple runs should give identical fallback
    graphs = []
    has_structs = []
    
    for i in range(5):
        set_seed(i * 100)
        graph, has_struct = generate_gcn_features(sequence, uniprot_id)
        graphs.append(graph)
        has_structs.append(has_struct)
    
    # All should have same structure availability
    assert all(h == has_structs[0] for h in has_structs), "Structure availability should be consistent"
    
    # If no structure, all graphs should be identical
    if not has_structs[0]:
        for i in range(1, len(graphs)):
            assert np.allclose(graphs[0].node_features, graphs[i].node_features), f"Graph {i} should match first"
            assert np.allclose(graphs[0].edge_index, graphs[i].edge_index), f"Graph {i} edges should match first"
    
    print("✓ Structural fallback is consistent")


if __name__ == "__main__":
    print("Testing deterministic feature generation...")
    test_deterministic_engineered_features()
    test_deterministic_structural_features()
    test_no_random_placeholders()
    test_structural_fallback_consistency()
    print("\n✅ All deterministic feature tests passed!")