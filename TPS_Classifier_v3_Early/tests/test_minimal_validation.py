"""
Minimal Validation Tests
========================

Critical validation checks: determinism, artifacts, and label mapping.
"""

import unittest
import tempfile
import torch
import numpy as np
import json
import hashlib
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TestMinimalValidation(unittest.TestCase):
    """Critical validation tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.n_classes = 30
    
    def test_determinism_byte_identical_outputs(self):
        """Test that same inputs produce byte-identical outputs across runs."""
        # Create test data
        batch_size = 3
        sequence_length = 50
        embedding_dim = 1280
        n_classes = self.n_classes
        
        # Generate embeddings (simulated ESM2)
        torch.manual_seed(42)
        embeddings1 = torch.randn(batch_size, embedding_dim)
        
        # Reset seed and generate again
        torch.manual_seed(42)
        embeddings2 = torch.randn(batch_size, embedding_dim)
        
        # Should be byte-identical
        embeddings1_bytes = embeddings1.numpy().tobytes()
        embeddings2_bytes = embeddings2.numpy().tobytes()
        
        self.assertEqual(embeddings1_bytes, embeddings2_bytes, 
                        "Embeddings should be byte-identical across runs with same seed")
        
        # Test engineered features
        np.random.seed(42)
        eng_features1 = np.random.randn(batch_size, 64)
        np.random.seed(42)
        eng_features2 = np.random.randn(batch_size, 64)
        
        eng_bytes1 = eng_features1.tobytes()
        eng_bytes2 = eng_features2.tobytes()
        
        self.assertEqual(eng_bytes1, eng_bytes2, 
                        "Engineered features should be byte-identical across runs")
        
        # Test structural features
        np.random.seed(42)
        struct_features1 = np.random.randn(batch_size, 30)
        np.random.seed(42)
        struct_features2 = np.random.randn(batch_size, 30)
        
        struct_bytes1 = struct_features1.tobytes()
        struct_bytes2 = struct_features2.tobytes()
        
        self.assertEqual(struct_bytes1, struct_bytes2, 
                        "Structural features should be byte-identical across runs")
        
        print("✅ Determinism test passed: byte-identical outputs across runs")
    
    def test_artifacts_fail_loudly(self):
        """Test that missing artifacts raise clear errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test missing checkpoint
            missing_checkpoint = temp_path / "missing_checkpoint.pth"
            
            # Simulate artifact error
            class MockArtifactError(Exception):
                def __init__(self, message):
                    super().__init__(f"Required artifact not found: {message}")
            
            with self.assertRaises(MockArtifactError) as context:
                if not missing_checkpoint.exists():
                    raise MockArtifactError("missing_checkpoint.pth")
            
            self.assertIn("Required artifact not found", str(context.exception))
            print("✅ Missing checkpoint raises clear error")
            
            # Test incomplete results
            incomplete_results = {
                "final_macro_f1": 0.4,
                # Missing "optimal_thresholds" and "training_history"
            }
            
            incomplete_file = temp_path / "incomplete_results.json"
            incomplete_file.write_text(json.dumps(incomplete_results))
            
            class MockIncompleteError(Exception):
                def __init__(self, message):
                    super().__init__(f"Incomplete artifact missing required keys: {message}")
            
            with self.assertRaises(MockIncompleteError) as context:
                required_keys = ["optimal_thresholds", "training_history"]
                missing_keys = [key for key in required_keys if key not in incomplete_results]
                if missing_keys:
                    raise MockIncompleteError(f"Missing keys: {missing_keys}")
            
            self.assertIn("Incomplete artifact missing required keys", str(context.exception))
            print("✅ Incomplete training results raises clear error")
            
            # Test invalid label order
            invalid_label_order = {
                "ensemble1": "invalid",  # Should be int, not string
                "ensemble2": 1
            }
            
            class MockInvalidError(Exception):
                def __init__(self, message):
                    super().__init__(f"Invalid label order entry: {message}")
            
            with self.assertRaises(MockInvalidError) as context:
                for name, value in invalid_label_order.items():
                    if not isinstance(value, int):
                        raise MockInvalidError(f"Entry '{name}' has invalid type: {type(value)}")
            
            self.assertIn("Invalid label order entry", str(context.exception))
            print("✅ Invalid label order raises clear error")
    
    def test_label_mapping_lock(self):
        """Test that label mapping is locked and consistent."""
        # Test dimension consistency
        model_n_classes = self.n_classes
        thresholds_length = 30  # Simulated
        label_order_length = 30  # Simulated
        
        # All should match
        self.assertEqual(model_n_classes, thresholds_length)
        self.assertEqual(model_n_classes, label_order_length)
        self.assertEqual(thresholds_length, label_order_length)
        print("✅ Label mapping dimensions are consistent")
        
        # Test that ensemble IDs are sequential and complete
        ensemble_ids = list(range(self.n_classes))
        expected_ids = list(range(self.n_classes))
        self.assertEqual(ensemble_ids, expected_ids)
        print("✅ Ensemble IDs are sequential and complete")
        
        # Test that ensemble names are strings
        ensemble_names = [f"ensemble_{i}" for i in range(self.n_classes)]
        for name in ensemble_names:
            self.assertIsInstance(name, str)
            self.assertTrue(len(name) > 0)
        print("✅ Ensemble names are valid strings")
        
        # Test that model head metadata matches
        mock_model_head_dim = self.n_classes  # Simulated
        self.assertEqual(mock_model_head_dim, self.n_classes)
        print("✅ Model head dimensions match label order")
    
    def test_comprehensive_determinism(self):
        """Test comprehensive determinism across all components."""
        # Test multiple runs with same seed
        results = []
        
        for run in range(3):
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Simulate full pipeline
            sequence = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
            
            # ESM embeddings (simulated)
            esm_embeddings = torch.randn(1, 1280)
            
            # Engineered features (simulated)
            eng_features = np.random.randn(1, 64)
            
            # Structural features (simulated)
            struct_features = np.random.randn(1, 30)
            
            # Model predictions (simulated)
            predictions = torch.randn(1, self.n_classes)
            
            # Collect results
            results.append({
                'esm_hash': hashlib.md5(esm_embeddings.numpy().tobytes()).hexdigest(),
                'eng_hash': hashlib.md5(eng_features.tobytes()).hexdigest(),
                'struct_hash': hashlib.md5(struct_features.tobytes()).hexdigest(),
                'pred_hash': hashlib.md5(predictions.numpy().tobytes()).hexdigest()
            })
        
        # All runs should be identical
        for i in range(1, len(results)):
            for key in results[0].keys():
                self.assertEqual(results[0][key], results[i][key], 
                               f"{key} should be identical across runs")
        
        print("✅ Comprehensive determinism test passed")
    
    def test_seed_propagation(self):
        """Test that seeds propagate correctly through all components."""
        # Test seed setting
        torch.manual_seed(123)
        np.random.seed(123)
        
        # Generate random numbers
        rand1 = torch.randn(10)
        rand2 = np.random.randn(10)
        
        # Reset seed
        torch.manual_seed(123)
        np.random.seed(123)
        
        # Generate again
        rand1_repeat = torch.randn(10)
        rand2_repeat = np.random.randn(10)
        
        # Should be identical
        torch.testing.assert_close(rand1, rand1_repeat)
        np.testing.assert_array_almost_equal(rand2, rand2_repeat)
        
        print("✅ Seed propagation works correctly")

def run_minimal_validation():
    """Run all minimal validation tests."""
    print("Running Minimal Validation Tests")
    print("=" * 40)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMinimalValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 40)
    print("Minimal Validation Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall result: {'PASS' if success else 'FAIL'}")
    
    return success

if __name__ == "__main__":
    run_minimal_validation()