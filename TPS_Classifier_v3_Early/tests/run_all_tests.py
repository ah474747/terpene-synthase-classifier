#!/usr/bin/env python3
"""Run all validation tests"""

import sys
import subprocess
import os

def run_test(test_file):
    """Run a single test file"""
    print(f"\n{'='*50}")
    print(f"Running {test_file}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED")
            print(result.stdout)
            return True
        else:
            print(f"❌ {test_file} FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {test_file} ERROR: {e}")
        return False

def main():
    """Run all tests"""
    test_files = [
        "tests/test_no_random_features.py",
        "tests/test_label_order_lock.py", 
        "tests/test_artifact_missing_fails.py",
        "tests/test_knn_blend_shapes_and_gain.py",
        "tests/test_identity_split_wrapper.py",
        "tests/test_hierarchy_masking.py",
        "tests/test_calibration_threshold_roundtrip.py",
        "tests/test_pooling_parity.py"
    ]
    
    print("🧪 Running TPS Classifier Validation Test Suite")
    print(f"Found {len(test_files)} test files")
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if run_test(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  {test_file} not found, skipping")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! TPS Classifier is stabilized.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Review and fix issues.")
        return 1

if __name__ == "__main__":
    exit(main())