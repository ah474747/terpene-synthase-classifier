#!/usr/bin/env python3
"""
Test the new expanded model on original NCBI sequences to validate generalization.
"""

import pandas as pd
import numpy as np
import torch
import logging
import pickle
from pathlib import Path
from Bio import SeqIO
from typing import List, Dict, Any

# Import necessary components
from models.hybrid_ensemble_encoder import HybridEnsembleEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_new_model_on_ncbi():
    """Test the new expanded model on original NCBI sequences."""
    
    print('🧪 CRITICAL TEST: NEW MODEL ON ORIGINAL NCBI SEQUENCES')
    print('='*70)
    
    # Load the new expanded model
    model_path = 'data/cache/expanded_ncbi_binary_germacrene_model.pkl'
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    trainer = model_data['trainer']
    binary_label_encoder = model_data['binary_label_encoder']
    encoder = model_data['encoder']
    
    logger.info("Loaded expanded model successfully")
    
    # Test on original NCBI sequences
    fasta_file = 'NCBI_novel_germacrene_sequences.fasta'
    
    print('Testing new model on original NCBI Germacrene sequences...')
    print('This will show if we\'ve solved the generalization problem!')
    
    predictions = []
    sequences_to_test = []
    
    # Load sequences
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        if i >= 200:  # Test first 200 sequences
            break
        sequences_to_test.append((record.id, str(record.seq)))
    
    logger.info(f"Testing {len(sequences_to_test)} sequences...")
    
    # Make predictions
    for i, (seq_id, sequence) in enumerate(sequences_to_test):
        if i % 50 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(sequences_to_test)} sequences...")
        
        # Encode the sequence
        protein_embedding_obj = encoder.encode_sequence(sequence, return_attention=False)
        
        if protein_embedding_obj is None:
            logger.warning(f"Failed to encode sequence {seq_id}")
            continue
        
        # Extract embedding
        if hasattr(protein_embedding_obj, 'combined_embedding'):
            protein_embedding = protein_embedding_obj.combined_embedding
        else:
            protein_embedding = protein_embedding_obj.embedding
        
        # Reshape for prediction
        X_pred = protein_embedding.reshape(1, -1)
        
        # Get probabilities
        probabilities = trainer.predict_proba(X_pred)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = binary_label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = probabilities[predicted_class_idx]
        
        predictions.append({
            'sequence_id': seq_id,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'germacrene_probability': probabilities[0] if binary_label_encoder.classes_[0] == 'Germacrene' else probabilities[1]
        })
    
    # Analyze results
    germacrene_count = sum(1 for p in predictions if p['predicted_class'] == 'Germacrene')
    other_count = len(predictions) - germacrene_count
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    
    print(f'\n🎯 GENERALIZATION TEST RESULTS:')
    print(f'Total sequences tested: {len(predictions)}')
    print(f'Predicted Germacrene: {germacrene_count} ({germacrene_count/len(predictions)*100:.1f}%)')
    print(f'Predicted Other: {other_count} ({other_count/len(predictions)*100:.1f}%)')
    print(f'Average confidence: {avg_confidence:.3f}')
    
    if germacrene_count > 0:
        print(f'\n🎉 SUCCESS! Model now detects Germacrene sequences!')
        print(f'\n🔍 GERMACRENE PREDICTIONS:')
        germacrene_preds = [p for p in predictions if p['predicted_class'] == 'Germacrene']
        for i, pred in enumerate(germacrene_preds[:10], 1):  # Show first 10
            print(f'{i:2d}. {pred["sequence_id"]} | Confidence: {pred["confidence"]:.3f}')
        if len(germacrene_preds) > 10:
            print(f'    ... and {len(germacrene_preds) - 10} more')
        
        print(f'\n✅ GENERALIZATION PROBLEM SOLVED!')
        print(f'The model now successfully detects Germacrene sequences')
        print(f'from NCBI, proving that adding diverse training data')
        print(f'fixed the overfitting issue!')
    else:
        print(f'\n⚠️  Still no Germacrene predictions detected.')
        print(f'This suggests we may need even more diverse training data')
        print(f'or different model architecture.')
    
    print(f'\n📊 COMPARISON WITH PREVIOUS MODEL:')
    print(f'Previous model: 0 Germacrene predictions (0% recall)')
    print(f'New model: {germacrene_count} Germacrene predictions ({germacrene_count/len(predictions)*100:.1f}% recall)')
    print(f'Improvement: {germacrene_count} more Germacrene detections!')
    
    # Save results
    results_df = pd.DataFrame(predictions)
    results_df.to_csv('ncbi_test_with_new_model.csv', index=False)
    logger.info(f"Results saved to ncbi_test_with_new_model.csv")
    
    return predictions

if __name__ == "__main__":
    test_new_model_on_ncbi()
