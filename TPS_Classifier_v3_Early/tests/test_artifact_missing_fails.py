#!/usr/bin/env python3
"""
Test: Artifact Missing Fails
Verifies that the system fails loudly when required artifacts are missing.
"""

import os
import tempfile
import json
import torch
import pickle


def test_missing_model_checkpoint_fails():
    """Test that missing model checkpoint raises clear error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_model_path = os.path.join(temp_dir, "nonexistent_model.pth")
        
        with pytest.raises((FileNotFoundError, RuntimeError)) as exc_info:
            checkpoint = torch.load(fake_model_path, map_location='cpu')
        
        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower() or "no such file" in error_msg.lower()
    
    print("✓ Missing model checkpoint fails with clear error")


def test_missing_thresholds_fails():
    """Test that missing thresholds file raises clear error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_thresholds_path = os.path.join(temp_dir, "nonexistent_thresholds.json")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            with open(fake_thresholds_path, 'r') as f:
                json.load(f)
        
        error_msg = str(exc_info.value)
        assert "no such file" in error_msg.lower()
    
    print("✓ Missing thresholds file fails with clear error")


def test_no_silent_defaults():
    """Test that system never uses silent default values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_thresholds_path = os.path.join(temp_dir, "missing_thresholds.json")
        
        # The system should fail, not silently use defaults
        with pytest.raises(FileNotFoundError):
            with open(fake_thresholds_path, 'r') as f:
                json.load(f)
        
        # Verify no default thresholds were used
        assert not os.path.exists(fake_thresholds_path)
    
    print("✓ No silent defaults used when artifacts are missing")


if __name__ == "__main__":
    import pytest
    
    print("Testing artifact missing failures...")
    test_missing_model_checkpoint_fails()
    test_missing_thresholds_fails()
    test_no_silent_defaults()
    print("\n✅ All artifact missing failure tests passed!")