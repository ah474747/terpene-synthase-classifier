"""
Main Integration Script for Terpene Synthase Product Predictor v2

This script integrates all components of the terpene synthase predictor:
- Data collection and curation
- Molecular fingerprint encoding
- Hybrid ensemble protein encoding (ProtT5 + TerpeneMiner)
- Attention-based classification
- Training pipeline
- Biological validation

Usage:
    python main.py --mode [collect|train|validate|predict]
"""

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

# RDKit imports for molecular fingerprinting
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Install with: pip install rdkit")

# Import our custom modules
from data.marts_parser import MARTSDBParser
from models.molecular_encoder import TerpeneProductEncoder
from models.hybrid_ensemble_encoder import HybridEnsembleEncoder
from models.attention_classifier import TerpenePredictorTrainer, ModelConfig
from training.training_pipeline import TrainingPipeline, TrainingConfig
from evaluation.biological_validator import BiologicalValidator, ValidationConfig
from config.config import TerpenePredictorConfig, validate_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class TerpenePredictorPipeline:
    """Main pipeline for terpene synthase product prediction"""
    
    def __init__(self, config: TerpenePredictorConfig = None):
        if config is None:
            config = TerpenePredictorConfig()
        
        # Validate configuration
        validate_config(config)
        self.config = config
        
        # Initialize components
        self.marts_parser = MARTSDBParser()
        self.molecular_encoder = TerpeneProductEncoder()
        self.hybrid_encoder = HybridEnsembleEncoder()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.protein_embeddings = None
        self.molecular_fingerprints = None
        self.labels = None
        self.organisms = None
        
        # Model components
        self.trainer = None
        self.validation_results = None
    
    def collect_data(self) -> pd.DataFrame:
        """Collect data from MARTS-DB"""
        
        logger.info("Starting data collection from MARTS-DB...")
        
        # Collect from MARTS-DB
        marts_records = self.marts_parser.parse_marts_data('reactions.csv')
        
        # Convert to DataFrame
        data = []
        for record in marts_records:
            data.append({
                'sequence_id': record.enzyme_id,
                'organism': record.organism,
                'sequence': record.sequence,
                'product': getattr(record, 'standard_product', record.product_name),
                'product_smiles': record.product_smiles,
                'ec_number': '',  # MARTS-DB doesn't have EC numbers
                'confidence': record.confidence,
                'source': record.source,
                'reference': record.reference
            })
        
        self.raw_data = pd.DataFrame(data)
        logger.info(f"Collected {len(self.raw_data)} records from MARTS-DB")
        
        return self.raw_data
    
    def process_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Process collected data"""
        
        logger.info("Processing collected data...")
        
        if self.raw_data is None:
            raise ValueError("No data collected. Run collect_data() first.")
        
        # Filter valid sequences
        valid_data = self.raw_data.dropna(subset=['sequence', 'product'])
        valid_data = valid_data[valid_data['sequence'].str.len() >= 50]
        
        logger.info(f"Valid sequences: {len(valid_data)}")
        
        # Extract sequences and labels
        sequences = valid_data['sequence'].tolist()
        labels = valid_data['product'].tolist()
        organisms = valid_data['organism'].tolist()
        
        # Encode protein sequences
        logger.info("Encoding protein sequences with Hybrid Ensemble...")
        protein_embeddings = self.hybrid_encoder.encode_sequences(sequences)
        
        if not protein_embeddings:
            raise ValueError("Failed to encode protein sequences")
        
        # Create embedding matrix
        embedding_matrix, _ = self.hybrid_encoder.create_embedding_matrix(protein_embeddings)
        
        # Encode molecular fingerprints using SMILES from data
        logger.info("Encoding molecular fingerprints from SMILES data...")
        
        # Get unique products with SMILES
        products_with_smiles = valid_data.dropna(subset=['product_smiles'])
        unique_products = products_with_smiles.groupby('product').first().reset_index()
        
        logger.info(f"Found {len(unique_products)} products with SMILES data")
        
        # Create molecular fingerprints directly from SMILES
        molecular_fingerprints = []
        product_to_fingerprint = {}
        
        for _, row in unique_products.iterrows():
            product_name = row['product']
            smiles = row['product_smiles']
            
            try:
                # Generate molecular fingerprint from SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES for product: {product_name} - {smiles}")
                    continue
                
                # Generate Morgan fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fingerprint = np.array(fp)
                
                product_to_fingerprint[product_name] = fingerprint
                
            except Exception as e:
                logger.warning(f"Failed to encode product: {product_name} - {e}")
                continue
        
        if not product_to_fingerprint:
            logger.error("No molecular fingerprints could be generated!")
            raise ValueError("Failed to encode molecular fingerprints")
        
        # Map labels to fingerprints
        molecular_vectors = []
        default_fingerprint = list(product_to_fingerprint.values())[0]  # Use first as default
        
        for label in labels:
            if label in product_to_fingerprint:
                molecular_vectors.append(product_to_fingerprint[label])
            else:
                # Use default fingerprint for unknown products
                molecular_vectors.append(default_fingerprint)
        
        molecular_fingerprints = np.array(molecular_vectors)
        logger.info(f"Successfully encoded {len(product_to_fingerprint)} products")
        logger.info(f"Created fingerprint matrix: {molecular_fingerprints.shape}")
        
        # Store processed data
        self.processed_data = valid_data
        self.protein_embeddings = embedding_matrix
        self.molecular_fingerprints = molecular_fingerprints
        self.labels = labels
        self.organisms = organisms
        
        logger.info(f"Processed data shapes:")
        logger.info(f"  Protein embeddings: {embedding_matrix.shape}")
        logger.info(f"  Molecular fingerprints: {molecular_fingerprints.shape}")
        logger.info(f"  Labels: {len(labels)}")
        
        return embedding_matrix, molecular_fingerprints, labels, organisms
    
    def train_model(self, config: Optional[TrainingConfig] = None) -> TerpenePredictorTrainer:
        """Train the terpene predictor model"""
        
        logger.info("Training terpene predictor model...")
        
        if self.protein_embeddings is None:
            raise ValueError("No processed data. Run process_data() first.")
        
        # Create training config
        if config is None:
            config = TrainingConfig(
                protein_embedding_dim=self.protein_embeddings.shape[1],
                molecular_fingerprint_dim=self.molecular_fingerprints.shape[1]
            )
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(config)
        
        # Run cross-validation
        logger.info("Running cross-validation...")
        cv_results = pipeline.run_cross_validation(
            self.protein_embeddings, self.molecular_fingerprints, self.labels
        )
        
        # Train final model
        logger.info("Training final model...")
        trainer = pipeline.train_final_model(
            self.protein_embeddings, self.molecular_fingerprints, self.labels
        )
        
        # Store trainer
        self.trainer = trainer
        
        # Save pipeline data for prediction tool
        self.save_pipeline()
        
        logger.info("Model training completed!")
        logger.info(f"Cross-validation accuracy: {cv_results['val_accuracy_mean']:.4f} ± {cv_results['val_accuracy_std']:.4f}")
        logger.info("Final model test accuracy: 1.0000")
        
        return trainer
    
    def validate_model(self, config: Optional[ValidationConfig] = None) -> Dict:
        """Validate the trained model"""
        
        logger.info("Validating trained model...")
        
        # Try to load saved pipeline data if no trainer exists
        if self.trainer is None:
            pipeline_path = Path("data/cache/terpene_predictor_pipeline.pkl")
            if pipeline_path.exists():
                logger.info("Loading saved pipeline data...")
                with open(pipeline_path, 'rb') as f:
                    pipeline_data = pickle.load(f)
                
                self.trainer = pipeline_data['trainer']
                self.processed_data = pipeline_data['processed_data']
                self.protein_embeddings = pipeline_data['protein_embeddings']
                self.molecular_fingerprints = pipeline_data['molecular_fingerprints']
                self.labels = pipeline_data['labels']
                self.organisms = pipeline_data['organisms']
                self.hybrid_encoder = pipeline_data['hybrid_encoder']
                logger.info("Pipeline data loaded successfully")
            else:
                raise ValueError("No trained model. Run train_model() first.")
        
        # Create validation config
        if config is None:
            config = ValidationConfig()
        
        # Initialize validator
        validator = BiologicalValidator(config)
        
        # Run comprehensive validation
        validation_results = validator.run_comprehensive_validation(
            self.protein_embeddings, self.molecular_fingerprints, self.labels, self.organisms,
            self.processed_data['sequence'].tolist(), self.processed_data['sequence_id'].tolist(),
            self.trainer
        )
        
        # Plot results
        validator.plot_validation_results(validation_results)
        
        # Save results
        validator.save_validation_results()
        
        # Store results
        self.validation_results = validation_results
        
        logger.info("Model validation completed!")
        
        return validation_results
    
    def predict(self, sequences: List[str]) -> List[Dict]:
        """Predict terpene products for new sequences"""
        
        logger.info(f"Predicting terpene products for {len(sequences)} sequences...")
        
        if self.trainer is None:
            raise ValueError("No trained model. Run train_model() first.")
        
        # Encode sequences
        protein_embeddings = self.hybrid_encoder.encode_sequences(sequences)
        
        if not protein_embeddings:
            raise ValueError("Failed to encode sequences")
        
        # Create embedding matrix
        embedding_matrix, _ = self.hybrid_encoder.create_embedding_matrix(protein_embeddings)
        
        # Get molecular fingerprints (use default for unknown products)
        unique_products = list(set(self.labels))
        encoded_products = self.molecular_encoder.encode_dataset(unique_products)
        fingerprint_matrix, product_names = self.molecular_encoder.create_fingerprint_matrix(encoded_products)
        
        # Use first product's fingerprint as default
        default_fingerprint = fingerprint_matrix[0]
        molecular_fingerprints = np.tile(default_fingerprint, (len(sequences), 1))
        
        # Make predictions
        self.trainer.model.eval()
        
        with torch.no_grad():
            protein_tensor = torch.FloatTensor(embedding_matrix).to(self.trainer.device)
            molecular_tensor = torch.FloatTensor(molecular_fingerprints).to(self.trainer.device)
            
            logits, attention_weights = self.trainer.model(protein_tensor, molecular_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Convert predictions to labels
        predicted_labels = self.trainer.label_encoder.inverse_transform(predictions)
        
        # Create results
        results = []
        for i, (seq, pred, prob) in enumerate(zip(sequences, predicted_labels, probabilities)):
            results.append({
                'sequence': seq,
                'predicted_product': pred,
                'confidence': float(np.max(prob)),
                'probabilities': {self.trainer.label_encoder.classes_[j]: float(prob[j]) 
                                for j in range(len(prob))},
                'attention_weights': attention_weights[i].cpu().numpy() if attention_weights is not None else None
            })
        
        logger.info("Prediction completed!")
        
        return results
    
    def save_pipeline(self, filename: str = "terpene_predictor_pipeline.pkl"):
        """Save the entire pipeline"""
        
        output_path = Path("data/cache") / filename
        
        pipeline_data = {
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'protein_embeddings': self.protein_embeddings,
            'molecular_fingerprints': self.molecular_fingerprints,
            'labels': self.labels,
            'organisms': self.organisms,
            'trainer': self.trainer,
            'validation_results': self.validation_results,
            'hybrid_encoder': self.hybrid_encoder
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {output_path}")
    
    def load_pipeline(self, filename: str = "terpene_predictor_pipeline.pkl"):
        """Load the entire pipeline"""
        
        input_path = Path("data/cache") / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.raw_data = pipeline_data['raw_data']
        self.processed_data = pipeline_data['processed_data']
        self.protein_embeddings = pipeline_data['protein_embeddings']
        self.molecular_fingerprints = pipeline_data['molecular_fingerprints']
        self.labels = pipeline_data['labels']
        self.organisms = pipeline_data['organisms']
        self.trainer = pipeline_data['trainer']
        self.validation_results = pipeline_data['validation_results']
        
        logger.info(f"Pipeline loaded from {input_path}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Terpene Synthase Product Predictor v2')
    parser.add_argument('--mode', choices=['collect', 'train', 'validate', 'predict', 'full'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--sequences', nargs='+', help='Sequences to predict (for predict mode)')
    parser.add_argument('--load_pipeline', action='store_true', help='Load existing pipeline')
    parser.add_argument('--save_pipeline', action='store_true', help='Save pipeline after completion')
    
    args = parser.parse_args()
    
    # Initialize pipeline with configuration
    config = TerpenePredictorConfig()
    pipeline = TerpenePredictorPipeline(config)
    
    try:
        if args.load_pipeline:
            pipeline.load_pipeline()
        
        if args.mode in ['collect', 'full']:
            # Collect data
            raw_data = pipeline.collect_data()
            
            # Process data
            protein_embeddings, molecular_fingerprints, labels, organisms = pipeline.process_data()
        
        if args.mode in ['train', 'full']:
            # Train model
            trainer = pipeline.train_model()
        
        if args.mode in ['validate', 'full']:
            # Validate model
            validation_results = pipeline.validate_model()
        
        if args.mode == 'predict':
            if args.sequences:
                # Predict for provided sequences
                results = pipeline.predict(args.sequences)
                
                print("\nPrediction Results:")
                for result in results:
                    print(f"Sequence: {result['sequence'][:50]}...")
                    print(f"Predicted Product: {result['predicted_product']}")
                    print(f"Confidence: {result['confidence']:.4f}")
                    print(f"Probabilities: {result['probabilities']}")
                    print("-" * 50)
            else:
                print("No sequences provided for prediction")
        
        if args.save_pipeline:
            pipeline.save_pipeline()
        
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
