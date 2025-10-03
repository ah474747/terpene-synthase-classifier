"""
Terpene Synthase Prediction Tool

This module provides a user-friendly interface for predicting terpene synthase products
from protein sequences using the trained SaProt-based model.
"""

import pandas as pd
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pickle
from dataclasses import dataclass
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Container for prediction results"""
    sequence_id: str
    sequence: str
    predicted_product: str
    confidence: float
    all_probabilities: Dict[str, float]
    attention_weights: Optional[np.ndarray] = None
    organism: Optional[str] = None

class TerpenePredictor:
    """Main prediction interface for terpene synthase products"""
    
    def __init__(self, model_path: str = "data/cache/best_model.pth", 
                 pipeline_path: str = "data/cache/terpene_predictor_pipeline.pkl"):
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path)
        
        # Load trained pipeline
        self._load_pipeline()
        
        logger.info("Terpene Predictor initialized successfully")
    
    def _load_pipeline(self):
        """Load the trained pipeline and model"""
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline data not found at {self.pipeline_path}")
        
        with open(self.pipeline_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.trainer = pipeline_data['trainer']
        
        # Always reinitialize encoder to avoid pickle issues
        # Check if pipeline has Hybrid, ProtT5, or SaProt encoder
        if 'hybrid_encoder' in pipeline_data:
            from models.hybrid_ensemble_encoder import HybridEnsembleEncoder
            self.encoder = HybridEnsembleEncoder()
            logger.info("Using Hybrid Ensemble encoder")
        elif 'prott5_encoder' in pipeline_data:
            from models.prott5_encoder import ProtT5Encoder
            self.encoder = ProtT5Encoder()
            logger.info("Using ProtT5 encoder")
        elif 'saprot_encoder' in pipeline_data:
            from models.saprot_encoder import SaProtEncoder
            self.encoder = SaProtEncoder()
            logger.info("Using SaProt encoder (legacy)")
        else:
            raise ValueError("No encoder found in pipeline data")
        
        logger.info("Pipeline loaded successfully")
    
    def predict_single_sequence(self, sequence: str, sequence_id: str = "unknown", 
                               organism: str = None, return_attention: bool = False) -> PredictionResult:
        """Predict terpene product for a single protein sequence"""
        
        # Validate sequence
        if not self._validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
        
        # Encode sequence with protein encoder
        protein_embedding = self.encoder.encode_sequence(sequence, return_attention=return_attention)
        if protein_embedding is None:
            raise ValueError("Failed to encode protein sequence")
        
        # Create molecular fingerprint (use default for unknown products)
        default_fingerprint = np.zeros(2048)  # Default molecular fingerprint
        molecular_fingerprint = default_fingerprint
        
        # Prepare inputs
        if hasattr(protein_embedding, 'combined_embedding'):
            # Hybrid ensemble embedding
            protein_tensor = torch.FloatTensor(protein_embedding.combined_embedding).unsqueeze(0).to(self.trainer.device)
        else:
            # Standard embedding
            protein_tensor = torch.FloatTensor(protein_embedding.embedding).unsqueeze(0).to(self.trainer.device)
        molecular_tensor = torch.FloatTensor(molecular_fingerprint).unsqueeze(0).to(self.trainer.device)
        
        # Make prediction
        self.trainer.model.eval()
        with torch.no_grad():
            logits, attention_weights = self.trainer.model(protein_tensor, molecular_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        # Get predicted product
        predicted_product = self.trainer.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(np.max(probabilities))
        
        # Get all probabilities
        all_probabilities = {}
        for i, class_name in enumerate(self.trainer.label_encoder.classes_):
            all_probabilities[class_name] = float(probabilities[i])
        
        # Sort by probability
        all_probabilities = dict(sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True))
        
        return PredictionResult(
            sequence_id=sequence_id,
            sequence=sequence,
            predicted_product=predicted_product,
            confidence=confidence,
            all_probabilities=all_probabilities,
            attention_weights=attention_weights.cpu().numpy() if attention_weights is not None else None,
            organism=organism
        )
    
    def predict_multiple_sequences(self, sequences: List[str], sequence_ids: List[str] = None,
                                  organisms: List[str] = None, return_attention: bool = False) -> List[PredictionResult]:
        """Predict terpene products for multiple protein sequences"""
        
        if sequence_ids is None:
            sequence_ids = [f"seq_{i+1}" for i in range(len(sequences))]
        
        if organisms is None:
            organisms = [None] * len(sequences)
        
        if len(sequences) != len(sequence_ids) or len(sequences) != len(organisms):
            raise ValueError("All input lists must have the same length")
        
        results = []
        for i, (seq, seq_id, org) in enumerate(zip(sequences, sequence_ids, organisms)):
            try:
                result = self.predict_single_sequence(seq, seq_id, org, return_attention)
                results.append(result)
                logger.info(f"Predicted {seq_id}: {result.predicted_product} (confidence: {result.confidence:.3f})")
            except Exception as e:
                logger.error(f"Failed to predict {seq_id}: {e}")
                # Create error result
                results.append(PredictionResult(
                    sequence_id=seq_id,
                    sequence=seq,
                    predicted_product="ERROR",
                    confidence=0.0,
                    all_probabilities={},
                    organism=org
                ))
        
        return results
    
    def predict_from_fasta(self, fasta_file: str, return_attention: bool = False) -> List[PredictionResult]:
        """Predict terpene products from a FASTA file"""
        
        sequences, sequence_ids = self._parse_fasta(fasta_file)
        organisms = [None] * len(sequences)  # No organism info in FASTA
        
        return self.predict_multiple_sequences(sequences, sequence_ids, organisms, return_attention)
    
    def predict_from_csv(self, csv_file: str, sequence_col: str = "sequence", 
                        id_col: str = "sequence_id", organism_col: str = None,
                        return_attention: bool = False) -> List[PredictionResult]:
        """Predict terpene products from a CSV file"""
        
        df = pd.read_csv(csv_file)
        
        sequences = df[sequence_col].tolist()
        sequence_ids = df[id_col].tolist() if id_col in df.columns else [f"seq_{i+1}" for i in range(len(sequences))]
        organisms = df[organism_col].tolist() if organism_col and organism_col in df.columns else [None] * len(sequences)
        
        return self.predict_multiple_sequences(sequences, sequence_ids, organisms, return_attention)
    
    def save_predictions(self, results: List[PredictionResult], output_file: str):
        """Save prediction results to CSV file"""
        
        data = []
        for result in results:
            # Get top 5 predictions
            top_predictions = list(result.all_probabilities.items())[:5]
            
            row = {
                'sequence_id': result.sequence_id,
                'sequence': result.sequence,
                'predicted_product': result.predicted_product,
                'confidence': result.confidence,
                'organism': result.organism or 'unknown'
            }
            
            # Add top 5 predictions
            for i, (product, prob) in enumerate(top_predictions):
                row[f'prediction_{i+1}'] = product
                row[f'probability_{i+1}'] = prob
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
    
    def _validate_sequence(self, sequence: str) -> bool:
        """Validate protein sequence"""
        if not sequence or not isinstance(sequence, str):
            return False
        
        # Check for valid amino acids
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        sequence_upper = sequence.upper()
        
        if not all(aa in valid_aa for aa in sequence_upper):
            return False
        
        # Check length
        if len(sequence) < 50 or len(sequence) > 2000:
            return False
        
        return True
    
    def _parse_fasta(self, fasta_file: str) -> Tuple[List[str], List[str]]:
        """Parse FASTA file and return sequences and IDs"""
        
        sequences = []
        sequence_ids = []
        current_seq = ""
        current_id = ""
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_seq:
                        sequences.append(current_seq)
                        sequence_ids.append(current_id)
                    
                    # Start new sequence
                    current_id = line[1:].split()[0]  # Get ID (first word after >)
                    current_seq = ""
                else:
                    current_seq += line
        
        # Don't forget the last sequence
        if current_seq:
            sequences.append(current_seq)
            sequence_ids.append(current_id)
        
        return sequences, sequence_ids

def main():
    """Command-line interface for terpene prediction"""
    
    parser = argparse.ArgumentParser(description="Predict terpene synthase products from protein sequences")
    parser.add_argument("--input", "-i", required=True, help="Input file (FASTA or CSV)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    parser.add_argument("--sequence-col", default="sequence", help="Sequence column name for CSV")
    parser.add_argument("--id-col", default="sequence_id", help="ID column name for CSV")
    parser.add_argument("--organism-col", help="Organism column name for CSV")
    parser.add_argument("--attention", action="store_true", help="Return attention weights")
    parser.add_argument("--model-path", default="data/cache/best_model.pth", help="Path to trained model")
    parser.add_argument("--pipeline-path", default="data/cache/terpene_predictor_pipeline.pkl", help="Path to pipeline data")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TerpenePredictor(args.model_path, args.pipeline_path)
    
    # Determine input format and predict
    input_path = Path(args.input)
    
    if input_path.suffix.lower() == '.fasta':
        results = predictor.predict_from_fasta(args.input, args.attention)
    elif input_path.suffix.lower() == '.csv':
        results = predictor.predict_from_csv(
            args.input, 
            args.sequence_col, 
            args.id_col, 
            args.organism_col,
            args.attention
        )
    else:
        raise ValueError("Input file must be FASTA (.fasta) or CSV (.csv)")
    
    # Save results
    predictor.save_predictions(results, args.output)
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"Total sequences: {len(results)}")
    print(f"Successful predictions: {len([r for r in results if r.predicted_product != 'ERROR'])}")
    print(f"Failed predictions: {len([r for r in results if r.predicted_product == 'ERROR'])}")
    
    if results:
        avg_confidence = np.mean([r.confidence for r in results if r.predicted_product != 'ERROR'])
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Show top predictions
        print(f"\nTop 5 predictions:")
        product_counts = {}
        for result in results:
            if result.predicted_product != 'ERROR':
                product_counts[result.predicted_product] = product_counts.get(result.predicted_product, 0) + 1
        
        for product, count in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {product}: {count} sequences")

if __name__ == "__main__":
    main()
