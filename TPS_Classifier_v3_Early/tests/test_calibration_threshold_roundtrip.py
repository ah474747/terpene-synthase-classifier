#!/usr/bin/env python3
"""Test: Calibration Threshold Roundtrip"""

import numpy as np
from tps.eval.calibration import CalibratedPredictor

def test_calibration_roundtrip():
    # Create mock data
    n_samples, n_classes = 100, 30
    raw_logits = np.random.randn(n_samples, n_classes)
    true_labels = np.random.randint(0, n_classes, size=n_samples)
    
    # Create calibrator
    calibrator = CalibratedPredictor()
    
    # Fit calibrator
    calibrator.fit(raw_logits, true_labels)
    
    # Test calibration
    calibrated_probs = calibrator.predict_proba(raw_logits)
    
    # Check shapes and constraints
    assert calibrated_probs.shape == (n_samples, n_classes)
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all(calibrated_probs >= 0)
    assert np.all(calibrated_probs <= 1)
    
    print("✓ Calibration roundtrip works")

if __name__ == "__main__":
    test_calibration_roundtrip()
    print("✅ Calibration test passed!")