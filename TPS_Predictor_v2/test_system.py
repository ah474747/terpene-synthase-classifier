"""
Simple Test Script for Terpene Synthase Product Predictor v2

This script tests the basic functionality of our new system.
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_molecular_encoding():
    """Test molecular fingerprint encoding"""
    logger.info("Testing molecular fingerprint encoding...")
    
    try:
        from models.molecular_encoder import TerpeneProductEncoder
        
        # Initialize encoder
        encoder = TerpeneProductEncoder()
        
        # Test encoding
        products = ["limonene", "pinene", "myrcene"]
        encoded_products = encoder.encode_dataset(products)
        
        if encoded_products:
            logger.info(f"✅ Molecular encoding successful: {len(encoded_products)} products encoded")
            return True
        else:
            logger.error("❌ Molecular encoding failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Molecular encoding error: {e}")
        return False

def test_protein_encoding():
    """Test protein sequence encoding"""
    logger.info("Testing protein sequence encoding...")
    
    try:
        from models.saprot_encoder import SaProtEncoder
        
        # Initialize encoder
        encoder = SaProtEncoder()
        
        # Test encoding
        sequences = [
            "MSTEQFVLPDLLESCPLKDATNPYYKEAAAESRAWINGYDIFTDRKRAEFIQGQNELLCSHVYWYAGREQLRTTCDFVNLLFVVDEVSDEQNGKGARETGQVFFKAMKYPDWDDGSILAKVTKEFMARFTRLAGPRNTKRFIDLCESYTACVGEEAELRERSELLDLASYIPLRRQNSAVLLCFALVEYILGIDLADEVYEDEMFMKAYWAACDQVCWANDIYSYDMEQSKGLAGNNIVSILMNENGTNLQETADYIGERCGEFVSDYISAKSQISPSLGPEALQFIDFVGYWMIGNIEWCFETPRYFGSRHLEIKETRVVHLRPKEVPEGLSSEDCIESDDE"
        ]
        
        embeddings = encoder.encode_sequences(sequences)
        
        if embeddings:
            logger.info(f"✅ Protein encoding successful: {len(embeddings)} sequences encoded")
            return True
        else:
            logger.error("❌ Protein encoding failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Protein encoding error: {e}")
        return False

def test_attention_model():
    """Test attention-based classifier"""
    logger.info("Testing attention-based classifier...")
    
    try:
        from models.attention_classifier import TerpenePredictor, ModelConfig
        
        # Create model config
        config = ModelConfig(
            protein_embedding_dim=1280,
            molecular_fingerprint_dim=2223,
            num_classes=5
        )
        
        # Initialize model
        model = TerpenePredictor(config)
        
        # Test forward pass
        import torch
        protein_embeddings = torch.randn(2, 1280)
        molecular_fingerprints = torch.randn(2, 2223)
        
        logits, attention_weights = model(protein_embeddings, molecular_fingerprints)
        
        if logits.shape == (2, 5):
            logger.info("✅ Attention model successful: Forward pass works")
            return True
        else:
            logger.error("❌ Attention model failed: Wrong output shape")
            return False
            
    except Exception as e:
        logger.error(f"❌ Attention model error: {e}")
        return False

def test_training_pipeline():
    """Test training pipeline"""
    logger.info("Testing training pipeline...")
    
    try:
        from training.training_pipeline import TrainingPipeline, TrainingConfig
        
        # Create sample data
        num_samples = 100
        protein_embeddings = np.random.randn(num_samples, 1280)
        molecular_fingerprints = np.random.randn(num_samples, 2223)
        labels = np.random.choice(["limonene", "pinene", "myrcene", "linalool", "germacrene_a"], num_samples)
        
        # Create training config
        config = TrainingConfig(
            protein_embedding_dim=1280,
            molecular_fingerprint_dim=2223,
            num_classes=5,
            min_samples_per_class=5,
            max_samples_per_class=50,
            num_epochs=2,  # Just 2 epochs for testing
            early_stopping_patience=1
        )
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        # Test data loading
        protein_embeddings, molecular_fingerprints, labels = pipeline.load_data(
            protein_embeddings, molecular_fingerprints, labels
        )
        
        logger.info("✅ Training pipeline successful: Data loading works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline error: {e}")
        return False

def main():
    """Main test function"""
    
    print("Terpene Synthase Product Predictor v2 - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Molecular Encoding", test_molecular_encoding),
        ("Protein Encoding", test_protein_encoding),
        ("Attention Model", test_attention_model),
        ("Training Pipeline", test_training_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
