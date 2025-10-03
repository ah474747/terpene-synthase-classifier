"""
FASTA Sequence Analyzer for Germacrene Synthase Prediction
Processes FASTA files and outputs CSV with predictions for Germacrene A, D, or Other.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our predictor
from germacrene_predictor import GermacreneSynthasePredictor


class FastaAnalyzer:
    """
    Analyzes FASTA files for germacrene synthase prediction.
    """
    
    def __init__(self, model_path: str = "germacrene_predictor.joblib"):
        """Initialize the analyzer with a trained model."""
        self.model_path = model_path
        self.predictor = GermacreneSynthasePredictor()
        self.is_loaded = False
        
        print(f"FASTA Sequence Analyzer")
        print(f"Model path: {model_path}")
        
    def load_model(self) -> bool:
        """Load the trained model."""
        try:
            if not os.path.exists(self.model_path):
                print(f"Error: Model file not found at {self.model_path}")
                print("Please train the model first by running: python3 germacrene_predictor.py")
                return False
            
            print(f"Loading trained model from {self.model_path}...")
            success = self.predictor.load_model(self.model_path)
            
            if success:
                self.is_loaded = True
                print("✅ Model loaded successfully!")
                return True
            else:
                print("❌ Failed to load model")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def read_fasta_file(self, fasta_path: str) -> List[Dict[str, str]]:
        """
        Read FASTA file and return list of sequences with headers.
        """
        print(f"Reading FASTA file: {fasta_path}")
        
        if not os.path.exists(fasta_path):
            print(f"Error: FASTA file not found at {fasta_path}")
            return []
        
        sequences = []
        
        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                # Get sequence as string
                sequence = str(record.seq)
                
                # Skip if sequence is too short (less than 50 amino acids)
                if len(sequence) < 50:
                    print(f"⚠️  Skipping {record.id}: sequence too short ({len(sequence)} amino acids)")
                    continue
                
                # Skip if sequence contains invalid characters
                valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                if not all(aa in valid_aa for aa in sequence):
                    print(f"⚠️  Skipping {record.id}: contains invalid amino acids")
                    continue
                
                sequences.append({
                    'header': record.id,
                    'description': record.description,
                    'sequence': sequence,
                    'length': len(sequence)
                })
            
            print(f"✅ Successfully read {len(sequences)} sequences from FASTA file")
            
            if len(sequences) == 0:
                print("⚠️  No valid sequences found in FASTA file")
            
            return sequences
            
        except Exception as e:
            print(f"❌ Error reading FASTA file: {e}")
            return []
    
    def analyze_sequences(self, sequences: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Analyze sequences using the trained model.
        """
        if not self.is_loaded:
            print("❌ Model not loaded. Please load model first.")
            return []
        
        if len(sequences) == 0:
            print("❌ No sequences to analyze")
            return []
        
        print(f"Analyzing {len(sequences)} sequences...")
        
        # Extract sequences for prediction
        seq_strings = [seq['sequence'] for seq in sequences]
        
        # Get predictions
        predictions = self.predictor.predict_germacrene(seq_strings)
        
        if not predictions:
            print("❌ Failed to get predictions")
            return []
        
        # Combine input data with predictions
        results = []
        for i, (seq_data, pred_data) in enumerate(zip(sequences, predictions['predictions'])):
            results.append({
                'fasta_name': seq_data['header'],
                'fasta_description': seq_data['description'],
                'sequence': seq_data['sequence'],
                'sequence_length': seq_data['length'],
                'predicted_type': pred_data['predicted_type'],
                'prediction': pred_data['prediction'],
                'confidence': pred_data['confidence'],
                'prob_germacrene_a': pred_data['probabilities']['germacrene_a'],
                'prob_germacrene_d': pred_data['probabilities']['germacrene_d'],
                'prob_other': pred_data['probabilities']['other']
            })
        
        print(f"✅ Analysis complete for {len(results)} sequences")
        return results
    
    def save_results_csv(self, results: List[Dict[str, any]], output_path: str) -> bool:
        """
        Save results to CSV file.
        """
        if len(results) == 0:
            print("❌ No results to save")
            return False
        
        try:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Select columns for basic CSV (as requested)
            basic_columns = ['fasta_name', 'sequence', 'prediction']
            basic_df = df[basic_columns]
            
            # Save basic CSV
            basic_df.to_csv(output_path, index=False)
            print(f"✅ Basic results saved to: {output_path}")
            
            # Also save detailed CSV with all information
            detailed_path = output_path.replace('.csv', '_detailed.csv')
            df.to_csv(detailed_path, index=False)
            print(f"✅ Detailed results saved to: {detailed_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def print_summary(self, results: List[Dict[str, any]]):
        """
        Print summary of analysis results.
        """
        if len(results) == 0:
            print("No results to summarize")
            return
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Count predictions
        germacrene_a_count = sum(1 for r in results if r['predicted_type'] == 'germacrene_a')
        germacrene_d_count = sum(1 for r in results if r['predicted_type'] == 'germacrene_d')
        other_count = sum(1 for r in results if r['predicted_type'] == 'other')
        
        print(f"Total sequences analyzed: {len(results)}")
        print(f"Germacrene A synthases: {germacrene_a_count}")
        print(f"Germacrene D synthases: {germacrene_d_count}")
        print(f"Other synthases: {other_count}")
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # High confidence predictions
        high_conf = [r for r in results if r['confidence'] > 0.9]
        print(f"High confidence predictions (>90%): {len(high_conf)}")
        
        print(f"\nDetailed Results:")
        print(f"{'Name':<30} {'Prediction':<20} {'Confidence':<10}")
        print(f"{'-'*60}")
        
        for result in results:
            name = result['fasta_name'][:29]  # Truncate long names
            prediction = result['prediction']
            confidence = f"{result['confidence']:.3f}"
            print(f"{name:<30} {prediction:<20} {confidence:<10}")
    
    def analyze_fasta_file(self, fasta_path: str, output_path: str = None) -> bool:
        """
        Complete analysis pipeline: read FASTA, analyze, save results.
        """
        print(f"\n{'='*60}")
        print(f"FASTA ANALYSIS PIPELINE")
        print(f"{'='*60}")
        
        # Load model if not already loaded
        if not self.is_loaded:
            if not self.load_model():
                return False
        
        # Read FASTA file
        sequences = self.read_fasta_file(fasta_path)
        if len(sequences) == 0:
            return False
        
        # Analyze sequences
        results = self.analyze_sequences(sequences)
        if len(results) == 0:
            return False
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(fasta_path))[0]
            output_path = f"{base_name}_germacrene_predictions.csv"
        
        # Save results
        if not self.save_results_csv(results, output_path):
            return False
        
        # Print summary
        self.print_summary(results)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Input: {fasta_path}")
        print(f"✅ Output: {output_path}")
        print(f"✅ Sequences analyzed: {len(results)}")
        
        return True


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze FASTA sequences for germacrene synthase prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 fasta_analyzer.py sequences.fasta
  python3 fasta_analyzer.py sequences.fasta -o results.csv
  python3 fasta_analyzer.py sequences.fasta -m custom_model.joblib
        """
    )
    
    parser.add_argument(
        'fasta_file',
        help='Path to input FASTA file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to output CSV file (default: auto-generated)'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='germacrene_predictor.joblib',
        help='Path to trained model file (default: germacrene_predictor.joblib)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FastaAnalyzer(model_path=args.model)
    
    # Run analysis
    success = analyzer.analyze_fasta_file(args.fasta_file, args.output)
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
