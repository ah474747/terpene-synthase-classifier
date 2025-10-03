#!/usr/bin/env python3
"""
Binary Germacrene Prediction Tool

This tool uses the trained binary Germacrene classifier to predict whether
terpene synthase sequences produce Germacrene or other terpenes.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
from dataclasses import dataclass
import argparse
from Bio import SeqIO
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BinaryPredictionResult:
    """Container for binary prediction results"""
    sequence_id: str
    sequence: str
    predicted_class: str  # "Germacrene" or "Other"
    confidence: float
    germacrene_probability: float
    other_probability: float
    organism: Optional[str] = None

class BinaryGermacrenePredictor:
    """Binary Germacrene prediction interface"""
    
    def __init__(self, model_path: str = "data/cache/binary_germacrene_model.pkl"):
        self.model_path = Path(model_path)
        
        # Load trained binary model
        self._load_model()
        
        logger.info("Binary Germacrene Predictor initialized successfully")
    
    def _load_model(self):
        """Load the trained binary model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Binary model not found at {self.model_path}")
        
        logger.info(f"Loading binary model from {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.trainer = model_data['trainer']
        self.binary_label_encoder = model_data['binary_label_encoder']
        self.encoder = model_data['encoder']
        self.results = model_data['results']
        self.germacrene_variants = model_data['germacrene_variants']
        
        logger.info("Binary model loaded successfully")
        logger.info(f"Model performance: {self.results['test_accuracy']:.3f} accuracy")
        logger.info(f"Germacrene variants: {len(self.germacrene_variants)}")
    
    def predict_sequence(self, sequence: str, sequence_id: str = "unknown") -> BinaryPredictionResult:
        """Predict if a sequence produces Germacrene"""
        logger.info(f"Predicting sequence {sequence_id} (length: {len(sequence)})")
        
        # Encode sequence
        embedding = self.encoder.encode_sequence(sequence)
        if not embedding:
            logger.error(f"Failed to encode sequence {sequence_id}")
            return None
        
        # Get prediction
        if hasattr(embedding, 'combined_embedding'):
            protein_tensor = embedding.combined_embedding
        else:
            protein_tensor = embedding.embedding
        
        # Predict using Random Forest
        prediction = self.trainer.predict([protein_tensor])[0]
        probabilities = self.trainer.predict_proba([protein_tensor])[0]
        
        # Convert back to labels
        predicted_class = self.binary_label_encoder.inverse_transform([prediction])[0]
        class_name = 'Germacrene' if predicted_class == 1 else 'Other'
        
        result = BinaryPredictionResult(
            sequence_id=sequence_id,
            sequence=sequence,
            predicted_class=class_name,
            confidence=probabilities[prediction],
            germacrene_probability=probabilities[1],
            other_probability=probabilities[0]
        )
        
        logger.info(f"Prediction: {class_name} (confidence: {result.confidence:.3f})")
        return result
    
    def predict_fasta_file(self, fasta_path: str, output_path: str = None, 
                          max_sequences: int = None) -> List[BinaryPredictionResult]:
        """Predict sequences from a FASTA file"""
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        logger.info(f"Processing FASTA file: {fasta_path}")
        
        # Parse FASTA file
        sequences = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequence_id = record.id
            sequence = str(record.seq)
            
            # Skip if sequence is too short
            if len(sequence) < 50:
                logger.warning(f"Skipping short sequence {sequence_id} (length: {len(sequence)})")
                continue
            
            sequences.append((sequence_id, sequence))
            
            # Limit number of sequences if specified
            if max_sequences and len(sequences) >= max_sequences:
                logger.info(f"Reached maximum sequence limit: {max_sequences}")
                break
        
        logger.info(f"Found {len(sequences)} valid sequences to process")
        
        # Predict each sequence
        results = []
        for i, (sequence_id, sequence) in enumerate(sequences):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(sequences)} sequences...")
            
            result = self.predict_sequence(sequence, sequence_id)
            if result:
                results.append(result)
        
        logger.info(f"Successfully predicted {len(results)} sequences")
        
        # Save results if output path specified
        if output_path:
            self.save_predictions(results, output_path)
        
        return results
    
    def save_predictions(self, results: List[BinaryPredictionResult], output_path: str):
        """Save predictions to CSV file"""
        output_path = Path(output_path)
        
        logger.info(f"Saving predictions to {output_path}")
        
        # Prepare data for CSV
        data = []
        for result in results:
            data.append({
                'sequence_id': result.sequence_id,
                'sequence_length': len(result.sequence),
                'predicted_class': result.predicted_class,
                'confidence': result.confidence,
                'germacrene_probability': result.germacrene_probability,
                'other_probability': result.other_probability,
                'organism': result.organism or 'unknown'
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        # Print summary
        germacrene_count = sum(1 for r in results if r.predicted_class == 'Germacrene')
        other_count = len(results) - germacrene_count
        
        logger.info(f"Prediction Summary:")
        logger.info(f"  Total sequences: {len(results)}")
        logger.info(f"  Predicted Germacrene: {germacrene_count} ({germacrene_count/len(results)*100:.1f}%)")
        logger.info(f"  Predicted Other: {other_count} ({other_count/len(results)*100:.1f}%)")
        logger.info(f"  Average confidence: {np.mean([r.confidence for r in results]):.3f}")
        
        return output_path

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Binary Germacrene Prediction Tool')
    parser.add_argument('--fasta', required=True, help='Input FASTA file path')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--max-sequences', type=int, help='Maximum number of sequences to process')
    parser.add_argument('--model', default='data/cache/binary_germacrene_model.pkl', 
                       help='Path to binary model file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BinaryGermacrenePredictor(model_path=args.model)
    
    # Generate output path if not provided
    if not args.output:
        fasta_path = Path(args.fasta)
        args.output = f"{fasta_path.stem}_binary_predictions.csv"
    
    # Run predictions
    results = predictor.predict_fasta_file(
        fasta_path=args.fasta,
        output_path=args.output,
        max_sequences=args.max_sequences
    )
    
    logger.info(f"Binary prediction completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()
