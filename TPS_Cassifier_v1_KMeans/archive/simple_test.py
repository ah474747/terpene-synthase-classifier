#!/usr/bin/env python3
"""
Simple test script that doesn't require loading the full protein language model
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terpene_classifier import TerpeneClassifier


def test_basic_functionality():
    """Test basic functionality without model loading"""
    print("=== Basic Functionality Test ===")
    
    # Test classifier initialization
    print("Testing classifier initialization...")
    classifier = TerpeneClassifier()
    print(f"✓ Classifier initialized with model: {classifier.model_name}")
    
    # Test sequence loading
    print("\nTesting sequence loading...")
    test_fasta = "test_sequences.fasta"
    with open(test_fasta, 'w') as f:
        f.write(">test_seq_1\n")
        f.write("MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL\n")
        f.write(">test_seq_2\n")
        f.write("MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL\n")
    
    df = classifier.load_sequences(test_fasta)
    print(f"✓ Loaded {len(df)} sequences")
    
    # Test MARTS-DB parsing
    print("\nTesting MARTS-DB parsing...")
    test_marts = "test_marts.fasta"
    with open(test_marts, 'w') as f:
        f.write(">germacrene_synthase_1\n")
        f.write("MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL\n")
        f.write(">limonene_synthase_1\n")
        f.write("MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL\n")
    
    marts_df = classifier.parse_marts_db(test_marts)
    print(f"✓ Parsed MARTS-DB with {len(marts_df)} sequences")
    print(f"  - Germacrene sequences: {marts_df['is_germacrene'].sum()}")
    
    # Test synthetic model training
    print("\nTesting synthetic model training...")
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    
    # Create synthetic data
    n_samples = 20
    embedding_dim = 100
    X = np.random.randn(n_samples, embedding_dim)
    y = np.random.randint(0, 2, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    classifier.scaler.fit(X_train)
    X_train_scaled = classifier.scaler.transform(X_train)
    X_test_scaled = classifier.scaler.transform(X_test)
    
    # Train model
    model = xgb.XGBClassifier(random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)
    
    # Test predictions
    predictions = model.predict(X_test_scaled)
    print(f"✓ Model training and prediction successful")
    print(f"  - Test accuracy: {np.mean(predictions == y_test):.3f}")
    
    # Clean up test files
    os.remove(test_fasta)
    os.remove(test_marts)
    
    print("\n✓ All basic functionality tests passed!")
    return True


def test_configuration():
    """Test configuration loading"""
    print("\n=== Configuration Test ===")
    
    try:
        from config import config
        print(f"✓ Configuration loaded successfully")
        print(f"  - Default model: {config.model.DEFAULT_MODEL}")
        print(f"  - Available models: {len(config.model.AVAILABLE_MODELS)}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Germacrene Synthase Classifier - Simple Test Suite")
    print("=" * 55)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Configuration", test_configuration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 55)
    print("TEST SUMMARY")
    print("=" * 55)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic functionality is working! Ready for data upload and training.")
    else:
        print("⚠ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

