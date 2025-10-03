#!/usr/bin/env python3
"""
Germacrene Synthase Prediction Tool
==================================

A simple command-line tool for predicting Germacrene synthase activity
from protein sequences using the trained classifier.

Usage:
    python predict.py sequence.fasta
    python predict.py -s "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL..."
    python predict.py --interactive
"""

import argparse
import sys
import os
from pathlib import Path
from Bio import SeqIO
from terpene_classifier import TerpeneClassifier


def predict_from_fasta(filepath: str, model_path: str = "models/germacrene_classifier.pkl"):
    """
    Predict Germacrene synthase activity for sequences in a FASTA file
    
    Args:
        filepath: Path to FASTA file
        model_path: Path to trained model
    """
    # Load classifier
    classifier = TerpeneClassifier()
    classifier.load_model(model_path)
    
    print(f"Predicting Germacrene synthase activity for sequences in {filepath}")
    print("=" * 60)
    
    # Process sequences
    for i, record in enumerate(SeqIO.parse(filepath, "fasta")):
        sequence = str(record.seq)
        confidence = classifier.predict_germacrene(sequence)
        
        prediction = "Germacrene Synthase" if confidence > 0.5 else "Other"
        
        print(f"Sequence {i+1}: {record.id}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Prediction: {prediction}")
        print(f"Sequence length: {len(sequence)} amino acids")
        print("-" * 40)


def predict_from_string(sequence: str, model_path: str = "models/germacrene_classifier.pkl"):
    """
    Predict Germacrene synthase activity for a single sequence
    
    Args:
        sequence: Amino acid sequence
        model_path: Path to trained model
    """
    # Load classifier
    classifier = TerpeneClassifier()
    classifier.load_model(model_path)
    
    confidence = classifier.predict_germacrene(sequence)
    prediction = "Germacrene Synthase" if confidence > 0.5 else "Other"
    
    print("Germacrene Synthase Prediction")
    print("=" * 30)
    print(f"Confidence: {confidence:.3f}")
    print(f"Prediction: {prediction}")
    print(f"Sequence length: {len(sequence)} amino acids")
    
    if confidence > 0.8:
        print("High confidence prediction!")
    elif confidence > 0.6:
        print("Moderate confidence prediction.")
    else:
        print("Low confidence prediction.")


def interactive_mode(model_path: str = "models/germacrene_classifier.pkl"):
    """
    Interactive prediction mode
    
    Args:
        model_path: Path to trained model
    """
    # Load classifier
    classifier = TerpeneClassifier()
    classifier.load_model(model_path)
    
    print("Interactive Germacrene Synthase Predictor")
    print("=" * 40)
    print("Enter protein sequences (one per line) or 'quit' to exit")
    print("Example: MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL")
    print()
    
    while True:
        try:
            sequence = input("Enter sequence: ").strip()
            
            if sequence.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not sequence:
                continue
            
            # Validate sequence (basic check)
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_aa for aa in sequence.upper()):
                print("Warning: Sequence contains invalid amino acid characters")
                continue
            
            confidence = classifier.predict_germacrene(sequence.upper())
            prediction = "Germacrene Synthase" if confidence > 0.5 else "Other"
            
            print(f"Confidence: {confidence:.3f} | Prediction: {prediction}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Predict Germacrene synthase activity from protein sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py sequences.fasta
  python predict.py -s "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
  python predict.py --interactive
        """
    )
    
    parser.add_argument(
        'fasta_file',
        nargs='?',
        help='FASTA file containing protein sequences'
    )
    
    parser.add_argument(
        '-s', '--sequence',
        help='Single protein sequence to predict'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='models/germacrene_classifier.pkl',
        help='Path to trained model file (default: models/germacrene_classifier.pkl)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please train the model first by running: python terpene_classifier.py")
        sys.exit(1)
    
    try:
        if args.interactive:
            interactive_mode(args.model)
        elif args.sequence:
            predict_from_string(args.sequence, args.model)
        elif args.fasta_file:
            if not os.path.exists(args.fasta_file):
                print(f"Error: FASTA file not found at {args.fasta_file}")
                sys.exit(1)
            predict_from_fasta(args.fasta_file, args.model)
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

