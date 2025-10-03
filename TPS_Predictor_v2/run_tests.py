"""
Test runner for Terpene Synthase Product Predictor v2
"""

import sys
import os
import pytest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all tests"""
    print("Running tests for Terpene Synthase Product Predictor v2...")
    print("=" * 60)
    
    # Test directory
    test_dir = Path(__file__).parent
    
    # Run tests with verbose output
    result = pytest.main([
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
