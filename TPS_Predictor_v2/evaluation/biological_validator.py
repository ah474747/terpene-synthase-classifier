"""
Biological Validation Framework for Terpene Synthase Product Predictor

This module implements comprehensive biological validation including
hold-out testing on unseen organisms, literature validation, and
uncertainty quantification.

Based on research best practices for biological model validation.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests
import json
import warnings

# Import our custom modules
from models.saprot_encoder import SaProtEncoder
from models.molecular_encoder import TerpeneProductEncoder
from models.attention_classifier import TerpenePredictorTrainer, ModelConfig
from training.training_pipeline import TrainingPipeline, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

@dataclass
class ValidationConfig:
    """Configuration for biological validation"""
    # Hold-out testing
    holdout_organisms: List[str] = None
    holdout_fraction: float = 0.2
    
    # Literature validation
    literature_sources: List[str] = None
    
    # Uncertainty quantification
    uncertainty_threshold: float = 0.8
    num_monte_carlo_samples: int = 100
    
    # Performance thresholds
    min_accuracy_threshold: float = 0.7
    min_f1_threshold: float = 0.6
    
    # Device
    device: str = "auto"

@dataclass
class ValidationResult:
    """Container for validation results"""
    test_name: str
    accuracy: float
    f1_score: float
    predictions: List[str]
    true_labels: List[str]
    probabilities: np.ndarray
    confidence_scores: np.ndarray
    uncertainty_scores: np.ndarray
    classification_report: Dict
    confusion_matrix: np.ndarray
    metadata: Dict

class HoldOutValidator:
    """Validates model on hold-out organisms"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def create_holdout_split(self, 
                           protein_embeddings: np.ndarray,
                           molecular_fingerprints: np.ndarray,
                           labels: List[str],
                           organisms: List[str]) -> Tuple[np.ndarray, ...]:
        """Create hold-out split based on organisms"""
        
        logger.info("Creating hold-out split based on organisms...")
        
        # Get unique organisms
        unique_organisms = list(set(organisms))
        logger.info(f"Found {len(unique_organisms)} unique organisms")
        
        # Select hold-out organisms
        if self.config.holdout_organisms is None:
            num_holdout = max(1, int(len(unique_organisms) * self.config.holdout_fraction))
            np.random.seed(42)
            holdout_organisms = np.random.choice(unique_organisms, num_holdout, replace=False)
        else:
            holdout_organisms = self.config.holdout_organisms
        
        logger.info(f"Hold-out organisms: {holdout_organisms}")
        
        # Create hold-out indices
        holdout_indices = [i for i, org in enumerate(organisms) if org in holdout_organisms]
        train_indices = [i for i, org in enumerate(organisms) if org not in holdout_organisms]
        
        logger.info(f"Hold-out samples: {len(holdout_indices)}")
        logger.info(f"Training samples: {len(train_indices)}")
        
        # Split data
        X_protein_train = protein_embeddings[train_indices]
        X_protein_holdout = protein_embeddings[holdout_indices]
        X_mol_train = molecular_fingerprints[train_indices]
        X_mol_holdout = molecular_fingerprints[holdout_indices]
        y_train = [labels[i] for i in train_indices]
        y_holdout = [labels[i] for i in holdout_indices]
        org_train = [organisms[i] for i in train_indices]
        org_holdout = [organisms[i] for i in holdout_indices]
        
        return (X_protein_train, X_protein_holdout, X_mol_train, X_mol_holdout, 
                y_train, y_holdout, org_train, org_holdout)
    
    def validate_holdout(self,
                       protein_embeddings: np.ndarray,
                       molecular_fingerprints: np.ndarray,
                       labels: List[str],
                       organisms: List[str],
                       trainer: TerpenePredictorTrainer) -> ValidationResult:
        """Validate model on hold-out organisms"""
        
        logger.info("Validating on hold-out organisms...")
        
        # Create hold-out split
        (X_protein_train, X_protein_holdout, X_mol_train, X_mol_holdout,
         y_train, y_holdout, org_train, org_holdout) = self.create_holdout_split(
            protein_embeddings, molecular_fingerprints, labels, organisms
        )
        
        # Train model on training data
        training_config = TrainingConfig(
            protein_embedding_dim=protein_embeddings.shape[1],
            molecular_fingerprint_dim=molecular_fingerprints.shape[1]
        )
        
        pipeline = TrainingPipeline(training_config)
        trained_trainer = pipeline.train_final_model(
            X_protein_train, X_mol_train, y_train
        )
        
        # Handle unseen products in hold-out data
        y_holdout_encoded = []
        unseen_products = []
        
        for label in y_holdout:
            if label in trained_trainer.label_encoder.classes_:
                y_holdout_encoded.append(trained_trainer.label_encoder.transform([label])[0])
            else:
                # Map unseen products to "unknown" class or use a default
                unseen_products.append(label)
                # Use the most common class as default for unseen products
                default_class_idx = 0  # First class in the training set
                y_holdout_encoded.append(default_class_idx)
        
        if unseen_products:
            logger.info(f"Found {len(unseen_products)} unseen products in hold-out data: {set(unseen_products)}")
            logger.info("Mapping unseen products to default class for evaluation")
        
        # Evaluate on hold-out data
        X_protein_holdout_tensor = torch.FloatTensor(X_protein_holdout).to(trained_trainer.device)
        X_mol_holdout_tensor = torch.FloatTensor(X_mol_holdout).to(trained_trainer.device)
        
        holdout_results = trained_trainer.evaluate(
            X_protein_holdout_tensor, X_mol_holdout_tensor, 
            torch.LongTensor(y_holdout_encoded).to(trained_trainer.device)
        )
        
        # Calculate confidence and uncertainty
        confidence_scores = np.max(holdout_results['probabilities'], axis=1)
        uncertainty_scores = 1 - confidence_scores
        
        # Create validation result
        result = ValidationResult(
            test_name="Hold-out Organisms",
            accuracy=holdout_results['accuracy'],
            f1_score=holdout_results['classification_report']['macro avg']['f1-score'],
            predictions=holdout_results['predictions'],
            true_labels=holdout_results['true_labels'],
            probabilities=holdout_results['probabilities'],
            confidence_scores=confidence_scores,
            uncertainty_scores=uncertainty_scores,
            classification_report=holdout_results['classification_report'],
            confusion_matrix=holdout_results['confusion_matrix'],
            metadata={
                'holdout_organisms': list(set(org_holdout)),
                'num_holdout_samples': len(y_holdout),
                'num_training_samples': len(y_train),
                'unseen_products': list(set(unseen_products)) if unseen_products else [],
                'num_unseen_products': len(set(unseen_products)) if unseen_products else 0
            }
        )
        
        logger.info(f"Hold-out validation accuracy: {result.accuracy:.4f}")
        logger.info(f"Hold-out validation F1-score: {result.f1_score:.4f}")
        
        return result

class LiteratureValidator:
    """Validates predictions against literature"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.literature_db = self._load_literature_database()
    
    def _load_literature_database(self) -> Dict[str, Dict]:
        """Load literature database of verified terpene synthase products"""
        
        # This would load from a real literature database
        # For now, return sample data
        return {
            "Q9XJ32": {  # Limonene synthase from Citrus limon
                "organism": "Citrus limon",
                "product": "limonene",
                "reference": "PMID:12345678",
                "confidence": 0.95
            },
            "P0CJ43": {  # Pinene synthase from Pinus taeda
                "organism": "Pinus taeda",
                "product": "pinene",
                "reference": "PMID:87654321",
                "confidence": 0.92
            },
            "A0A075FBG7": {  # Sample entry
                "organism": "Mentha spicata",
                "product": "linalool",
                "reference": "PMID:11111111",
                "confidence": 0.88
            }
        }
    
    def validate_predictions(self, 
                           predictions: List[str],
                           sequences: List[str],
                           sequence_ids: List[str]) -> ValidationResult:
        """Validate predictions against literature"""
        
        logger.info("Validating predictions against literature...")
        
        validated_predictions = []
        validated_labels = []
        confidence_scores = []
        literature_sources = []
        
        for i, (pred, seq_id) in enumerate(zip(predictions, sequence_ids)):
            if seq_id in self.literature_db:
                lit_data = self.literature_db[seq_id]
                validated_predictions.append(pred)
                validated_labels.append(lit_data['product'])
                confidence_scores.append(lit_data['confidence'])
                literature_sources.append(lit_data['reference'])
        
        if not validated_predictions:
            logger.warning("No literature validation data found")
            return None
        
        # Calculate accuracy
        accuracy = accuracy_score(validated_labels, validated_predictions)
        
        # Calculate F1-score
        f1 = f1_score(validated_labels, validated_predictions, average='macro')
        
        # Create validation result
        result = ValidationResult(
            test_name="Literature Validation",
            accuracy=accuracy,
            f1_score=f1,
            predictions=validated_predictions,
            true_labels=validated_labels,
            probabilities=np.array([[0.5, 0.5]] * len(validated_predictions)),  # Placeholder
            confidence_scores=np.array(confidence_scores),
            uncertainty_scores=1 - np.array(confidence_scores),
            classification_report=classification_report(validated_labels, validated_predictions, output_dict=True),
            confusion_matrix=confusion_matrix(validated_labels, validated_predictions),
            metadata={
                'num_validated': len(validated_predictions),
                'literature_sources': literature_sources
            }
        )
        
        logger.info(f"Literature validation accuracy: {result.accuracy:.4f}")
        logger.info(f"Literature validation F1-score: {result.f1_score:.4f}")
        
        return result

class UncertaintyQuantifier:
    """Quantifies prediction uncertainty"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def monte_carlo_dropout(self, 
                           model: torch.nn.Module,
                           protein_embeddings: torch.Tensor,
                           molecular_fingerprints: torch.Tensor,
                           num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty using Monte Carlo dropout"""
        
        model.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits, _ = model(protein_embeddings, molecular_fingerprints)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)  # [num_samples, batch_size, num_classes]
        
        # Calculate mean and variance
        mean_predictions = np.mean(predictions, axis=0)
        variance_predictions = np.var(predictions, axis=0)
        
        # Calculate uncertainty (entropy of mean predictions)
        uncertainty = -np.sum(mean_predictions * np.log(mean_predictions + 1e-8), axis=1)
        
        return mean_predictions, uncertainty
    
    def quantify_uncertainty(self,
                            trainer: TerpenePredictorTrainer,
                            protein_embeddings: np.ndarray,
                            molecular_fingerprints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantify prediction uncertainty"""
        
        logger.info("Quantifying prediction uncertainty...")
        
        # Convert to tensors
        protein_tensor = torch.FloatTensor(protein_embeddings).to(trainer.device)
        molecular_tensor = torch.FloatTensor(molecular_fingerprints).to(trainer.device)
        
        # Get standard predictions
        trainer.model.eval()
        with torch.no_grad():
            logits, _ = trainer.model(protein_tensor, molecular_tensor)
            standard_probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Get Monte Carlo predictions
        mc_probs, uncertainty = self.monte_carlo_dropout(
            trainer.model, protein_tensor, molecular_tensor, self.config.num_monte_carlo_samples
        )
        
        # Calculate confidence scores
        confidence_scores = np.max(standard_probs, axis=1)
        
        return confidence_scores, uncertainty

class BiologicalValidator:
    """Main biological validation framework"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.holdout_validator = HoldOutValidator(config)
        self.literature_validator = LiteratureValidator(config)
        self.uncertainty_quantifier = UncertaintyQuantifier(config)
        
        # Results storage
        self.validation_results = {}
    
    def run_comprehensive_validation(self,
                                   protein_embeddings: np.ndarray,
                                   molecular_fingerprints: np.ndarray,
                                   labels: List[str],
                                   organisms: List[str],
                                   sequences: List[str],
                                   sequence_ids: List[str],
                                   trainer: TerpenePredictorTrainer) -> Dict[str, ValidationResult]:
        """Run comprehensive biological validation"""
        
        logger.info("Starting comprehensive biological validation...")
        
        results = {}
        
        # 1. Hold-out organism validation
        logger.info("1. Hold-out organism validation...")
        holdout_result = self.holdout_validator.validate_holdout(
            protein_embeddings, molecular_fingerprints, labels, organisms, trainer
        )
        results['holdout'] = holdout_result
        
        # 2. Literature validation
        logger.info("2. Literature validation...")
        # Get test predictions by evaluating the trainer on test data
        # For now, skip literature validation as it requires test data that's not easily accessible
        # TODO: Implement proper test data access for literature validation
        logger.info("Skipping literature validation - requires test data access")
        literature_result = None
        if literature_result:
            results['literature'] = literature_result
        
        # 3. Uncertainty quantification
        logger.info("3. Uncertainty quantification...")
        confidence_scores, uncertainty_scores = self.uncertainty_quantifier.quantify_uncertainty(
            trainer, protein_embeddings, molecular_fingerprints
        )
        
        # Create uncertainty validation result
        uncertainty_result = ValidationResult(
            test_name="Uncertainty Quantification",
            accuracy=0.0,  # Not applicable
            f1_score=0.0,  # Not applicable
            predictions=[],  # Not applicable
            true_labels=[],  # Not applicable
            probabilities=np.array([]),  # Not applicable
            confidence_scores=confidence_scores,
            uncertainty_scores=uncertainty_scores,
            classification_report={},
            confusion_matrix=np.array([]),
            metadata={
                'mean_confidence': np.mean(confidence_scores),
                'mean_uncertainty': np.mean(uncertainty_scores),
                'high_uncertainty_count': np.sum(uncertainty_scores > 0.5)
            }
        )
        results['uncertainty'] = uncertainty_result
        
        # Store results
        self.validation_results = results
        
        # Log summary
        self._log_validation_summary(results)
        
        return results
    
    def _log_validation_summary(self, results: Dict[str, ValidationResult]):
        """Log validation summary"""
        
        logger.info("Biological Validation Summary:")
        logger.info("=" * 50)
        
        for test_name, result in results.items():
            logger.info(f"{test_name}:")
            logger.info(f"  Accuracy: {result.accuracy:.4f}")
            logger.info(f"  F1-score: {result.f1_score:.4f}")
            if hasattr(result, 'metadata'):
                for key, value in result.metadata.items():
                    logger.info(f"  {key}: {value}")
            logger.info("")
    
    def plot_validation_results(self, results: Dict[str, ValidationResult]):
        """Plot validation results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Accuracy comparison
        test_names = []
        accuracies = []
        
        for test_name, result in results.items():
            if result.accuracy > 0:  # Skip uncertainty quantification
                test_names.append(test_name)
                accuracies.append(result.accuracy)
        
        axes[0, 0].bar(test_names, accuracies)
        axes[0, 0].set_title('Validation Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: F1-score comparison
        f1_scores = []
        for test_name, result in results.items():
            if result.f1_score > 0:  # Skip uncertainty quantification
                f1_scores.append(result.f1_score)
        
        axes[0, 1].bar(test_names, f1_scores)
        axes[0, 1].set_title('Validation F1-Score')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: Confidence distribution
        if 'uncertainty' in results:
            confidence_scores = results['uncertainty'].confidence_scores
            axes[1, 0].hist(confidence_scores, bins=20, alpha=0.7)
            axes[1, 0].set_title('Confidence Score Distribution')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Uncertainty distribution
        if 'uncertainty' in results:
            uncertainty_scores = results['uncertainty'].uncertainty_scores
            axes[1, 1].hist(uncertainty_scores, bins=20, alpha=0.7)
            axes[1, 1].set_title('Uncertainty Score Distribution')
            axes[1, 1].set_xlabel('Uncertainty Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data/cache/validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_validation_results(self, filename: str = "validation_results.pkl"):
        """Save validation results"""
        output_path = Path("data/cache") / filename
        
        # Convert to serializable format
        serializable_results = {}
        for test_name, result in self.validation_results.items():
            serializable_results[test_name] = {
                'test_name': result.test_name,
                'accuracy': result.accuracy,
                'f1_score': result.f1_score,
                'predictions': result.predictions,
                'true_labels': result.true_labels,
                'probabilities': result.probabilities,
                'confidence_scores': result.confidence_scores,
                'uncertainty_scores': result.uncertainty_scores,
                'classification_report': result.classification_report,
                'confusion_matrix': result.confusion_matrix,
                'metadata': result.metadata
            }
        
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_results, f)
        
        logger.info(f"Validation results saved to {output_path}")

def main():
    """Main function to demonstrate biological validation"""
    logger.info("Starting biological validation framework...")
    
    # Create sample data
    num_samples = 1000
    protein_embedding_dim = 1280
    molecular_fingerprint_dim = 2223
    
    # Generate random data for demonstration
    protein_embeddings = np.random.randn(num_samples, protein_embedding_dim)
    molecular_fingerprints = np.random.randn(num_samples, molecular_fingerprint_dim)
    
    # Create sample labels and organisms
    products = ['limonene', 'pinene', 'myrcene', 'linalool', 'germacrene_a']
    organisms = ['Citrus limon', 'Pinus taeda', 'Mentha spicata', 'Lavandula angustifolia', 'Artemisia annua']
    
    labels = np.random.choice(products, num_samples)
    org_labels = np.random.choice(organisms, num_samples)
    
    # Create sample sequences and IDs
    sequences = [f"MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL{i:03d}" for i in range(num_samples)]
    sequence_ids = [f"SEQ_{i:06d}" for i in range(num_samples)]
    
    # Create validation config
    config = ValidationConfig(
        holdout_organisms=['Citrus limon', 'Pinus taeda'],
        min_accuracy_threshold=0.7,
        min_f1_threshold=0.6
    )
    
    # Initialize validator
    validator = BiologicalValidator(config)
    
    # Create a mock trainer for demonstration
    model_config = ModelConfig(
        protein_embedding_dim=protein_embedding_dim,
        molecular_fingerprint_dim=molecular_fingerprint_dim,
        num_classes=len(products)
    )
    
    trainer = TerpenePredictorTrainer(model_config)
    
    # Run validation
    results = validator.run_comprehensive_validation(
        protein_embeddings, molecular_fingerprints, labels, org_labels,
        sequences, sequence_ids, trainer
    )
    
    # Plot results
    validator.plot_validation_results(results)
    
    # Save results
    validator.save_validation_results()
    
    # Print summary
    print(f"\nBiological Validation Summary:")
    for test_name, result in results.items():
        print(f"{test_name}: Accuracy={result.accuracy:.4f}, F1={result.f1_score:.4f}")

if __name__ == "__main__":
    main()
