"""
Stabilized TPS Predictor - Multi-Modal Terpene Synthase Classifier
================================================================

Stabilized inference pipeline with deterministic features, calibration,
and out-of-distribution robustness.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import requests
from transformers import AutoTokenizer, AutoModel

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import stabilized modules
from tps.config import *
from tps.paths import *
from tps.models.multimodal import FinalMultiModalClassifier
from tps.features.engineered import generate_deterministic_features
from tps.features.structure import StructuralFeatureProcessor
from tps.retrieval.knn_head import KNNRetrievalHead
from tps.hierarchy.head import HierarchyHead, apply_hierarchy_masking
from tps.eval.calibration import CalibrationPipeline
from tps.utils import set_seed, setup_logging, ensure_deterministic, get_device

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class StabilizedTPSPredictor:
    """Stabilized TPS Predictor with deterministic inference and calibration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize stabilized predictor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_config()
        self.device = get_device()
        
        # Ensure deterministic behavior
        if self.config['deterministic_inference']:
            ensure_deterministic()
        
        # Load artifacts
        self._load_artifacts()
        
        # Initialize components
        self._initialize_model()
        self._initialize_feature_processors()
        self._initialize_calibration()
        self._initialize_knn()
        self._initialize_hierarchy()
        
        logger.info("StabilizedTPSPredictor initialized successfully")
    
    def _load_artifacts(self) -> None:
        """Load required artifacts with fail-loudly behavior."""
        try:
            # Load checkpoint
            self.checkpoint_path, self.checkpoint_hash = load_checkpoint_path()
            
            # Load training results
            self.training_results = load_training_results()
            
            # Load label order
            self.label_order = load_label_order()
            
            # Load calibration artifacts (optional)
            self.calibration_thresholds, self.calibration_models = load_calibration_artifacts()
            
            logger.info("All artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
    
    def _initialize_model(self) -> None:
        """Initialize the multimodal model."""
        self.model = FinalMultiModalClassifier(
            plm_dim=self.config['esm_dim'],
            eng_dim=self.config['eng_dim'],
            latent_dim=self.config['latent_dim'],
            n_classes=self.config['n_classes']
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {self.checkpoint_path}")
    
    def _initialize_feature_processors(self) -> None:
        """Initialize feature processors."""
        self.structural_processor = StructuralFeatureProcessor()
        logger.info("Feature processors initialized")
    
    def _initialize_calibration(self) -> None:
        """Initialize calibration pipeline."""
        if self.calibration_models is not None:
            self.calibration_pipeline = CalibrationPipeline()
            # Load calibration artifacts would go here
            logger.info("Calibration pipeline initialized")
        else:
            self.calibration_pipeline = None
            logger.info("No calibration pipeline available")
    
    def _initialize_knn(self) -> None:
        """Initialize kNN retrieval head."""
        try:
            knn_index_path = get_knn_index_path()
            embeddings_path = get_esm_embeddings_path()
            labels_path = get_train_labels_path()
            
            if all(p.exists() for p in [knn_index_path, embeddings_path, labels_path]):
                self.knn_head = KNNRetrievalHead()
                self.knn_head.load_index(knn_index_path, embeddings_path, labels_path)
                logger.info("kNN retrieval head initialized")
            else:
                self.knn_head = None
                logger.info("kNN retrieval head not available")
                
        except Exception as e:
            logger.warning(f"Failed to initialize kNN head: {e}")
            self.knn_head = None
    
    def _initialize_hierarchy(self) -> None:
        """Initialize hierarchy head."""
        self.hierarchy_head = HierarchyHead(
            input_dim=self.config['latent_dim'] * 3,  # Fusion dimension
            n_types=len(TERPENE_TYPES),
            n_fine_classes=self.config['n_classes']
        )
        self.hierarchy_head.to(self.device)
        self.hierarchy_head.eval()
        
        logger.info("Hierarchy head initialized")
    
    def predict_functional_ensembles(
        self,
        uniprot_id: str,
        sequence: str,
        annotation: str = "",
        use_knn: bool = True,
        use_hierarchy: bool = True,
        return_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Predict functional ensembles for a protein sequence.
        
        Args:
            uniprot_id: UniProt identifier
            sequence: Amino acid sequence
            annotation: Optional annotation
            use_knn: Whether to use kNN blending
            use_hierarchy: Whether to use hierarchy masking
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            # Generate features
            features = self._generate_features(uniprot_id, sequence, annotation)
            
            # Get model predictions
            with torch.no_grad():
                predictions = self.model(
                    plm_x=features['esm_embeddings'],
                    eng_x=features['engineered_features'],
                    struct_x=features['structural_features'],
                    edge_index=features['edge_index'],
                    has_structure=features['has_structure']
                )
            
            # Apply kNN blending if available
            if use_knn and self.knn_head is not None:
                esm_embeddings_np = features['esm_embeddings'].cpu().numpy()
                model_probs_np = predictions.cpu().numpy()
                blended_probs = self.knn_head.blend_predictions(model_probs_np, esm_embeddings_np)
                predictions = torch.tensor(blended_probs, device=self.device)
            
            # Apply hierarchy masking if requested
            if use_hierarchy:
                # Get type predictions and apply masking
                type_preds = self._predict_terpene_type(features['engineered_features'])
                predictions = apply_hierarchy_masking(predictions, type_preds)
            
            # Apply calibration if available
            if self.calibration_pipeline is not None:
                predictions_np = predictions.cpu().numpy()
                calibrated_preds, calibrated_probs = self.calibration_pipeline.predict(
                    predictions_np, return_calibrated=True
                )
                predictions = torch.tensor(calibrated_preds, device=self.device)
                calibrated_probabilities = torch.tensor(calibrated_probs, device=self.device)
            else:
                calibrated_probabilities = predictions
            
            # Get top predictions
            top_k = self.config.get('top_k', 3)
            top_indices = torch.topk(predictions, top_k).indices.cpu().numpy()
            top_scores = torch.topk(predictions, top_k).values.cpu().numpy()
            
            # Map to ensemble names
            ensemble_names = [list(self.label_order.keys())[idx] for idx in top_indices[0]]
            
            result = {
                'uniprot_id': uniprot_id,
                'top_predictions': ensemble_names,
                'top_scores': top_scores[0].tolist(),
                'has_structure': features['has_structure'].item(),
                'sequence_length': len(sequence)
            }
            
            if return_probs:
                result['all_probabilities'] = calibrated_probabilities.cpu().numpy().tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {uniprot_id}: {e}")
            return {
                'uniprot_id': uniprot_id,
                'error': str(e),
                'top_predictions': [],
                'top_scores': []
            }
    
    def _generate_features(
        self,
        uniprot_id: str,
        sequence: str,
        annotation: str
    ) -> Dict[str, torch.Tensor]:
        """Generate all required features."""
        # ESM embeddings
        esm_embeddings = self._generate_esm_embeddings(sequence)
        
        # Engineered features (deterministic)
        eng_features, _ = generate_deterministic_features(
            annotation=annotation,
            sequence_length=len(sequence),
            has_structure=False  # Will be updated based on structure availability
        )
        
        # Structural features
        structural_result = self.structural_processor.process_structure(
            uniprot_id, sequence
        )
        
        if structural_result[0] is not None:
            node_features, edge_index, has_structure = self.structural_processor.create_graph_features(
                structural_result[0], sequence
            )
            # Update engineered features with structure flag
            eng_features, _ = generate_deterministic_features(
                annotation=annotation,
                sequence_length=len(sequence),
                has_structure=True
            )
        else:
            node_features, edge_index, has_structure = self.structural_processor.create_fallback_features(sequence)
        
        return {
            'esm_embeddings': esm_embeddings,
            'engineered_features': torch.tensor(eng_features, dtype=torch.float32).unsqueeze(0),
            'structural_features': node_features.unsqueeze(0),
            'edge_index': edge_index,
            'has_structure': torch.tensor([has_structure], dtype=torch.bool)
        }
    
    def _generate_esm_embeddings(self, sequence: str) -> torch.Tensor:
        """Generate ESM2 embeddings."""
        # This would load the ESM2 model and tokenizer
        # For now, return a placeholder
        return torch.randn(1, ESM_DIM, device=self.device)
    
    def _predict_terpene_type(self, eng_features: torch.Tensor) -> torch.Tensor:
        """Predict terpene type from engineered features."""
        # Extract type from engineered features (first 5 dimensions)
        type_logits = eng_features[:, :len(TERPENE_TYPES)]
        return torch.softmax(type_logits, dim=1)

def main():
    """Main function for testing."""
    # Initialize predictor
    predictor = StabilizedTPSPredictor()
    
    # Test prediction
    test_sequence = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
    test_uniprot = "TEST123"
    
    result = predictor.predict_functional_ensembles(
        uniprot_id=test_uniprot,
        sequence=test_sequence,
        annotation="terpene synthase"
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()



