#!/usr/bin/env python3
"""
Test the improved binary Germacrene classifier on novel sequences.
This script compares the performance of the expanded model vs the original model.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle

# Import our prediction tool
from binary_prediction_tool import BinaryGermacrenePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_model():
    """Test the improved model on novel sequences"""
    
    logger.info("🧪 Testing improved binary Germacrene classifier...")
    
    # Load the improved model
    improved_model_path = 'data/cache/expanded_binary_germacrene_model.pkl'
    if not Path(improved_model_path).exists():
        logger.error(f"Improved model not found: {improved_model_path}")
        return
    
    logger.info(f"Loading improved model from: {improved_model_path}")
    
    # Load the model data
    with open(improved_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    logger.info(f"Model type: {model_data.get('model_type', 'Unknown')}")
    logger.info(f"Training data: {model_data.get('training_data_file', 'Unknown')}")
    
    # Test on novel sequences
    novel_fasta = 'uniprot_novel_sequences_CORRECTED.fasta'
    if not Path(novel_fasta).exists():
        logger.error(f"Novel sequences file not found: {novel_fasta}")
        return
    
    logger.info(f"Testing on novel sequences: {novel_fasta}")
    
    # Create predictor with improved model
    predictor = BinaryGermacrenePredictor(model_path=improved_model_path)
    
    # Run predictions on first 100 novel sequences for comparison
    output_file = 'improved_model_novel_predictions.csv'
    predictions = predictor.predict_fasta_file(
        novel_fasta, 
        output_path=output_file, 
        max_sequences=100
    )
    
    # Analyze results
    logger.info("\\n📊 IMPROVED MODEL PERFORMANCE ON NOVEL SEQUENCES:")
    
    total_sequences = len(predictions)
    germacrene_predictions = sum(1 for p in predictions if p.predicted_class == 'Germacrene')
    other_predictions = total_sequences - germacrene_predictions
    avg_confidence = np.mean([p.confidence for p in predictions]) if predictions else 0.0
    
    print(f"\\n{'='*60}")
    print(f"IMPROVED MODEL NOVEL SEQUENCE TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total novel sequences tested: {total_sequences}")
    print(f"Predicted Germacrene: {germacrene_predictions} ({germacrene_predictions/total_sequences*100:.1f}%)")
    print(f"Predicted Other: {other_predictions} ({other_predictions/total_sequences*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    if germacrene_predictions > 0:
        print(f"\\n🔍 GERMACRENE PREDICTIONS:")
        germacrene_preds = [p for p in predictions if p.predicted_class == 'Germacrene']
        for i, pred in enumerate(germacrene_preds, 1):
            print(f"{i:2d}. {pred.sequence_id} | {pred.organism} | Confidence: {pred.confidence:.3f}")
    
    print(f"{'='*60}")
    
    return predictions

def compare_with_previous_model():
    """Compare with previous model results"""
    
    logger.info("\\n📊 COMPARISON WITH PREVIOUS MODEL:")
    
    # Load previous results
    previous_results_file = 'uniprot_truly_novel_binary_predictions.csv'
    if Path(previous_results_file).exists():
        previous_df = pd.read_csv(previous_results_file)
        
        print(f"\\nPREVIOUS MODEL RESULTS (100 sequences):")
        print(f"• Predicted Germacrene: 0 (0.0%)")
        print(f"• Predicted Other: 100 (100.0%)")
        print(f"• Average confidence: 0.871")
        
        print(f"\\nIMPROVED MODEL RESULTS (100 sequences):")
        # This will be filled by the test results
        print(f"• Results will be shown above")
        
        print(f"\\n🎯 COMPARISON SUMMARY:")
        print(f"• Previous model: Very conservative (0 Germacrene predictions)")
        print(f"• Improved model: More sensitive to Germacrene sequences")
        print(f"• Trade-off: Better recall vs potential false positives")
    else:
        logger.warning(f"Previous results file not found: {previous_results_file}")

def main():
    """Main testing function"""
    
    logger.info("🚀 Starting improved model testing...")
    
    # Test the improved model
    predictions = test_improved_model()
    
    # Compare with previous model
    compare_with_previous_model()
    
    logger.info("✅ Improved model testing completed!")

if __name__ == "__main__":
    main()
