#!/usr/bin/env python3
"""
Predict Germacrene Synthase Activity using Trained Model
======================================================

This script uses the trained model to make predictions on new sequences.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_embedding_generator import RobustEmbeddingGenerator


class TrainedGermacrenePredictor:
    """
    Predictor using the trained Germacrene synthase classifier
    """
    
    def __init__(self, model_path: str = "models/germacrene_classifier_robust.pkl"):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.embedding_generator = None
        
        # Load the trained model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and embedding generator"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading trained model from {self.model_path}...")
        
        # Load model data
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data['model']
        
        print(f"✓ Model loaded successfully")
        print(f"  - Embedding dimension: {self.model_data['embedding_dim']}")
        print(f"  - Training sequences: {self.model_data['training_sequences']}")
        print(f"  - Positive samples: {self.model_data['positive_samples']}")
        print(f"  - Negative samples: {self.model_data['negative_samples']}")
        
        # Initialize embedding generator
        self.embedding_generator = RobustEmbeddingGenerator()
    
    def predict_single_sequence(self, sequence: str) -> dict:
        """
        Predict Germacrene synthase activity for a single sequence
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Generate embedding
            embedding_df = self.embedding_generator.generate_embeddings([sequence])
            
            # Extract embedding
            embedding = np.array([np.array(emb) for emb in embedding_df['embedding']])[0]
            
            # Make prediction
            confidence = self.model.predict_proba(embedding.reshape(1, -1))[0, 1]
            
            # Determine prediction
            prediction = "Germacrene Synthase" if confidence > 0.5 else "Other"
            
            return {
                'sequence': sequence,
                'confidence': float(confidence),
                'prediction': prediction,
                'is_germacrene': confidence > 0.5,
                'sequence_length': len(sequence)
            }
            
        except Exception as e:
            return {
                'sequence': sequence,
                'confidence': None,
                'prediction': 'Error',
                'is_germacrene': None,
                'sequence_length': len(sequence),
                'error': str(e)
            }
    
    def predict_multiple_sequences(self, sequences: list) -> pd.DataFrame:
        """
        Predict Germacrene synthase activity for multiple sequences
        
        Args:
            sequences: List of amino acid sequences
            
        Returns:
            DataFrame with prediction results
        """
        print(f"Predicting Germacrene synthase activity for {len(sequences)} sequences...")
        
        results = []
        
        for i, sequence in enumerate(sequences):
            if i % 10 == 0:
                print(f"Processing sequence {i+1}/{len(sequences)}")
            
            result = self.predict_single_sequence(sequence)
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Summary statistics
        successful_predictions = df[df['confidence'].notna()]
        if len(successful_predictions) > 0:
            germacrene_count = successful_predictions['is_germacrene'].sum()
            avg_confidence = successful_predictions['confidence'].mean()
            
            print(f"\nPrediction Summary:")
            print(f"  - Total sequences: {len(sequences)}")
            print(f"  - Successful predictions: {len(successful_predictions)}")
            print(f"  - Predicted Germacrene synthases: {germacrene_count}")
            print(f"  - Average confidence: {avg_confidence:.3f}")
        
        return df


def test_predictions():
    """Test the predictor with sample sequences"""
    print("=" * 60)
    print("TESTING GERMACRENE SYNTHASE PREDICTOR")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = TrainedGermacrenePredictor()
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {e}")
        return False
    
    # Test sequences (mix of known and unknown)
    test_sequences = [
        # Example sequences from MARTS-DB
        "MSVSLSFAASATFGFRGGLGGFSRPAAAIKQWRCLPRIQCHSAEQSQSPLRRSGNYQPSIWTHDRIQSLTLSHTADEDDHGERIKLLKCQTNKLMEEKKGEVGEQLQLIDHLQQLGVAYHFKDEIKDTLRGFYASFEDISLQFKDNLHASALLFRLLRENGFSVSEDIFKKFKDDQKGQFEDRLQSQAEGLLSLYEASYLEKDGEELLHEAREFTTKHLKNLLEEEGSLKPGLIREQVAYALELPLNRRFQRLHTKWFIGAWQRDPTMDPALLLLAKLDFNALQNMYKRELNEVSRWWTDLGLPQKLPFFRDRLTENYLWAVVFAFEPDSWAFREMDTKTNCFITMIDDVYDVYGTLDELELFTDIMERWDVNAIDKLPEYMKICFLAVFNTVNDAGYEVMRDKGVNIIPYLKRAWAELCKMYMREARWYHTGYTPTLDEYLDGAWISISGALILSTAYCMGKDLTKEDLDKFSTYPSIVQPSCMLLRLHDDFGTSTEELARGDVQKAVQCCMHERKVPEAVAREHIKQVMEAKWRVLNGNRVAASSFEEYFQNVAINLPRAAQFFYGKGDGYANADGETQKQVMSLLIEPVQ",
        
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELWKEFVNQHLCGSHLVEALYLVCGERGFFYTPKA",
        
        # Shorter test sequence
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
    ]
    
    print(f"\nTesting {len(test_sequences)} sequences...")
    
    # Make predictions
    try:
        results_df = predictor.predict_multiple_sequences(test_sequences)
        
        print(f"\nDetailed Results:")
        print("-" * 60)
        
        for i, row in results_df.iterrows():
            print(f"\nSequence {i+1}:")
            print(f"  Length: {row['sequence_length']} amino acids")
            
            if row['confidence'] is not None:
                print(f"  Prediction: {row['prediction']}")
                print(f"  Confidence: {row['confidence']:.3f}")
                
                # Confidence interpretation
                if row['confidence'] > 0.8:
                    print(f"  Interpretation: High confidence prediction")
                elif row['confidence'] > 0.6:
                    print(f"  Interpretation: Moderate confidence prediction")
                elif row['confidence'] > 0.4:
                    print(f"  Interpretation: Low confidence prediction")
                else:
                    print(f"  Interpretation: Very low confidence prediction")
            else:
                print(f"  Error: {row.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction testing failed: {e}")
        return False


def predict_from_fasta(fasta_file: str, output_file: str = None):
    """
    Predict Germacrene synthase activity for sequences in a FASTA file
    
    Args:
        fasta_file: Path to FASTA file
        output_file: Path to output CSV file (optional)
    """
    print(f"Predicting Germacrene synthase activity for sequences in {fasta_file}")
    
    # Initialize predictor
    try:
        predictor = TrainedGermacrenePredictor()
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {e}")
        return False
    
    # Load sequences from FASTA file
    from Bio import SeqIO
    
    sequences = []
    sequence_ids = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        sequence_ids.append(record.id)
    
    print(f"Loaded {len(sequences)} sequences from FASTA file")
    
    # Make predictions
    try:
        results_df = predictor.predict_multiple_sequences(sequences)
        
        # Add sequence IDs
        results_df['sequence_id'] = sequence_ids
        
        # Reorder columns
        columns = ['sequence_id', 'sequence', 'prediction', 'confidence', 'is_germacrene', 'sequence_length']
        if 'error' in results_df.columns:
            columns.append('error')
        
        results_df = results_df[columns]
        
        # Save results
        if output_file is None:
            output_file = fasta_file.replace('.fasta', '_predictions.csv')
        
        results_df.to_csv(output_file, index=False)
        print(f"✓ Predictions saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict Germacrene synthase activity")
    parser.add_argument('--test', action='store_true', help='Run test predictions')
    parser.add_argument('--fasta', type=str, help='FASTA file to predict')
    parser.add_argument('--output', type=str, help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.test:
        success = test_predictions()
        sys.exit(0 if success else 1)
    
    elif args.fasta:
        success = predict_from_fasta(args.fasta, args.output)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
