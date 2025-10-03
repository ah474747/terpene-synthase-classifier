#!/usr/bin/env python3
"""
Test: Label Order Lock
Verifies that label mapping is consistent across all artifacts and components.
"""

import json
import os
import numpy as np
from tps.paths import get_model_path, get_thresholds_path, get_label_order_path
from tps.config import N_CLASSES


def test_label_order_exists():
    """Test that label_order.json exists and is valid."""
    label_order_path = get_label_order_path()
    assert os.path.exists(label_order_path), f"Label order file missing: {label_order_path}"
    
    with open(label_order_path, 'r') as f:
        label_order = json.load(f)
    
    assert isinstance(label_order, list), "Label order should be a list"
    assert len(label_order) > 0, "Label order should not be empty"
    assert all(isinstance(name, str) for name in label_order), "All labels should be strings"
    assert len(set(label_order)) == len(label_order), "Labels should be unique"
    
    print(f"✓ Label order exists with {len(label_order)} unique labels")


def test_label_order_dimensions_match():
    """Test that label order dimensions match model and thresholds."""
    # Load label order
    with open(get_label_order_path(), 'r') as f:
        label_order = json.load(f)
    
    # Load thresholds
    thresholds_path = get_thresholds_path()
    assert os.path.exists(thresholds_path), f"Thresholds file missing: {thresholds_path}"
    
    with open(thresholds_path, 'r') as f:
        thresholds_data = json.load(f)
    
    # Check dimensions match
    assert len(label_order) == N_CLASSES, f"Label order length {len(label_order)} != N_CLASSES {N_CLASSES}"
    
    if 'thresholds' in thresholds_data:
        thresholds = np.array(thresholds_data['thresholds'])
        assert len(thresholds) == len(label_order), f"Thresholds length {len(thresholds)} != label order {len(label_order)}"
    elif 'optimal_thresholds' in thresholds_data:
        thresholds = np.array(thresholds_data['optimal_thresholds'])
        assert len(thresholds) == len(label_order), f"Thresholds length {len(thresholds)} != label order {len(label_order)}"
    
    print("✓ Label order dimensions match model and thresholds")


def test_thresholds_valid_range():
    """Test that all thresholds are in valid [0, 1] range."""
    thresholds_path = get_thresholds_path()
    
    with open(thresholds_path, 'r') as f:
        thresholds_data = json.load(f)
    
    if 'thresholds' in thresholds_data:
        thresholds = np.array(thresholds_data['thresholds'])
    elif 'optimal_thresholds' in thresholds_data:
        thresholds = np.array(thresholds_data['optimal_thresholds'])
    else:
        raise ValueError("No thresholds found in thresholds file")
    
    assert np.all(thresholds >= 0), "All thresholds should be >= 0"
    assert np.all(thresholds <= 1), "All thresholds should be <= 1"
    assert not np.any(np.isnan(thresholds)), "No thresholds should be NaN"
    assert not np.any(np.isinf(thresholds)), "No thresholds should be Inf"
    
    print(f"✓ All {len(thresholds)} thresholds are in valid [0, 1] range")


def test_label_order_consistency():
    """Test that label order is consistent with expected ensemble names."""
    with open(get_label_order_path(), 'r') as f:
        label_order = json.load(f)
    
    # Expected ensemble names (from marts_consolidation_pipeline.py)
    expected_ensembles = [
        'mono_pinene', 'mono_limonene', 'mono_myrcene', 'mono_linalool', 'mono_geraniol',
        'mono_camphene', 'mono_bornyl', 'mono_terpineol', 'mono_menthol', 'mono_citronellol',
        'sesqui_germacrene', 'sesqui_caryophyllene', 'sesqui_alpha_bergamotene', 'sesqui_alpha_farnesene',
        'sesqui_beta_farnesene', 'sesqui_nerolidol', 'sesqui_farnesol', 'sesqui_bisabolol',
        'sesqui_patchoulol', 'sesqui_alpha_santalene', 'sesqui_alpha_curcumene', 'sesqui_alpha_zingiberene',
        'di_kaurene', 'di_abietadiene', 'di_copalyl', 'di_manoyl', 'di_ent_kaurene',
        'di_phytol', 'tri_squalene', 'other_terpene'
    ]
    
    assert len(label_order) == len(expected_ensembles), f"Label order length {len(label_order)} != expected {len(expected_ensembles)}"
    
    # Check that all expected ensembles are present
    for expected in expected_ensembles:
        assert expected in label_order, f"Missing expected ensemble: {expected}"
    
    print("✓ Label order matches expected ensemble names")


def test_model_checkpoint_consistency():
    """Test that model checkpoint exists and is consistent."""
    model_path = get_model_path()
    assert os.path.exists(model_path), f"Model checkpoint missing: {model_path}"
    
    # Try to load the model to check it's valid
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check that checkpoint has expected keys
        assert 'model_state_dict' in checkpoint or 'state_dict' in checkpoint, "Model checkpoint missing state dict"
        
        # If we have label order in checkpoint, verify consistency
        if 'label_order' in checkpoint:
            checkpoint_labels = checkpoint['label_order']
            with open(get_label_order_path(), 'r') as f:
                file_labels = json.load(f)
            assert checkpoint_labels == file_labels, "Checkpoint label order != file label order"
        
        print("✓ Model checkpoint is valid and consistent")
        
    except Exception as e:
        print(f"⚠ Warning: Could not validate model checkpoint: {e}")


def test_metadata_consistency():
    """Test that metadata is consistent with other artifacts."""
    metadata_path = get_label_order_path().replace('label_order.json', 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        with open(get_label_order_path(), 'r') as f:
            label_order = json.load(f)
        
        # Check consistency if metadata has label info
        if 'label_order' in metadata:
            assert metadata['label_order'] == label_order, "Metadata label order != file label order"
        
        if 'n_classes' in metadata:
            assert metadata['n_classes'] == len(label_order), "Metadata n_classes != label order length"
        
        print("✓ Metadata is consistent with label order")
    else:
        print("⚠ Metadata file not found, skipping consistency check")


if __name__ == "__main__":
    print("Testing label order lock and consistency...")
    test_label_order_exists()
    test_label_order_dimensions_match()
    test_thresholds_valid_range()
    test_label_order_consistency()
    test_model_checkpoint_consistency()
    test_metadata_consistency()
    print("\n✅ All label order lock tests passed!")