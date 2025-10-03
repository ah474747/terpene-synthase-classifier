"""
Training Pipeline for Terpene Synthase Product Predictor

This module implements the complete training pipeline including data loading,
preprocessing, balanced sampling, cross-validation, and model evaluation.

Based on research best practices for enzyme product prediction.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Import our custom modules
from models.saprot_encoder import SaProtEncoder, ProteinEmbedding
from models.molecular_encoder import TerpeneProductEncoder, MolecularFingerprint
from models.attention_classifier import TerpenePredictorTrainer, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Data parameters
    min_samples_per_class: int = 10
    max_samples_per_class: int = 1000
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    
    # Model parameters
    protein_embedding_dim: int = 1280
    molecular_fingerprint_dim: int = 2223
    hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout_rate: float = 0.3
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Cross-validation
    cv_folds: int = 5
    
    # Device
    device: str = "auto"

class BalancedSampler:
    """Handles balanced sampling for imbalanced datasets"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def balance_dataset(self, 
                       protein_embeddings: np.ndarray,
                       molecular_fingerprints: np.ndarray,
                       labels: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Balance the dataset by sampling"""
        
        # Count samples per class
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"Original class distribution:\n{label_counts}")
        
        # Filter classes with sufficient samples
        valid_classes = label_counts[label_counts >= self.config.min_samples_per_class].index
        logger.info(f"Valid classes (≥{self.config.min_samples_per_class} samples): {len(valid_classes)}")
        
        # Filter data
        valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
        
        filtered_protein = protein_embeddings[valid_indices]
        filtered_molecular = molecular_fingerprints[valid_indices]
        filtered_labels = [labels[i] for i in valid_indices]
        
        # Balance classes
        balanced_indices = []
        
        for class_label in valid_classes:
            class_indices = [i for i, label in enumerate(filtered_labels) if label == class_label]
            
            # Sample up to max_samples_per_class
            if len(class_indices) > self.config.max_samples_per_class:
                np.random.seed(self.config.random_state)
                class_indices = np.random.choice(class_indices, self.config.max_samples_per_class, replace=False)
            
            balanced_indices.extend(class_indices)
        
        # Create balanced dataset
        balanced_protein = filtered_protein[balanced_indices]
        balanced_molecular = filtered_molecular[balanced_indices]
        balanced_labels = [filtered_labels[i] for i in balanced_indices]
        
        # Log final distribution
        final_counts = pd.Series(balanced_labels).value_counts()
        logger.info(f"Balanced class distribution:\n{final_counts}")
        
        return balanced_protein, balanced_molecular, balanced_labels

class DataPreprocessor:
    """Handles data preprocessing and feature scaling"""
    
    def __init__(self):
        self.protein_scaler = StandardScaler()
        self.molecular_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def fit_transform(self, 
                     protein_embeddings: np.ndarray,
                     molecular_fingerprints: np.ndarray,
                     labels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit scalers and transform data"""
        
        # Scale features
        protein_scaled = self.protein_scaler.fit_transform(protein_embeddings)
        molecular_scaled = self.molecular_scaler.fit_transform(molecular_fingerprints)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Scaled protein embeddings: {protein_scaled.shape}")
        logger.info(f"Scaled molecular fingerprints: {molecular_scaled.shape}")
        logger.info(f"Encoded labels: {labels_encoded.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return protein_scaled, molecular_scaled, labels_encoded
    
    def transform(self, 
                 protein_embeddings: np.ndarray,
                 molecular_fingerprints: np.ndarray,
                 labels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform data using fitted scalers"""
        
        protein_scaled = self.protein_scaler.transform(protein_embeddings)
        molecular_scaled = self.molecular_scaler.transform(molecular_fingerprints)
        labels_encoded = self.label_encoder.transform(labels)
        
        return protein_scaled, molecular_scaled, labels_encoded

class CrossValidator:
    """Handles cross-validation for model evaluation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def cross_validate(self,
                      protein_embeddings: np.ndarray,
                      molecular_fingerprints: np.ndarray,
                      labels: List[str],
                      preprocessor: DataPreprocessor) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        
        logger.info(f"Starting {self.config.cv_folds}-fold cross-validation...")
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        
        cv_results = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': []
        }
        
        fold = 0
        for train_idx, val_idx in skf.split(protein_embeddings, labels):
            fold += 1
            logger.info(f"Fold {fold}/{self.config.cv_folds}")
            
            # Split data
            X_protein_train, X_protein_val = protein_embeddings[train_idx], protein_embeddings[val_idx]
            X_mol_train, X_mol_val = molecular_fingerprints[train_idx], molecular_fingerprints[val_idx]
            y_train, y_val = [labels[i] for i in train_idx], [labels[i] for i in val_idx]
            
            # Preprocess
            X_protein_train_scaled, X_mol_train_scaled, y_train_encoded = preprocessor.fit_transform(
                X_protein_train, X_mol_train, y_train
            )
            X_protein_val_scaled, X_mol_val_scaled, y_val_encoded = preprocessor.transform(
                X_protein_val, X_mol_val, y_val
            )
            
            # Train model
            model_config = ModelConfig(
                protein_embedding_dim=self.config.protein_embedding_dim,
                molecular_fingerprint_dim=self.config.molecular_fingerprint_dim,
                num_classes=len(preprocessor.label_encoder.classes_),
                hidden_dim=self.config.hidden_dim,
                num_attention_heads=self.config.num_attention_heads,
                dropout_rate=self.config.dropout_rate,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                num_epochs=self.config.num_epochs,
                early_stopping_patience=self.config.early_stopping_patience
            )
            
            trainer = TerpenePredictorTrainer(model_config)
            
            # Fit the label encoder with original labels
            trainer.label_encoder.fit(y_train)
            
            # Convert to tensors
            X_protein_train_tensor = torch.FloatTensor(X_protein_train_scaled).to(trainer.device)
            X_protein_val_tensor = torch.FloatTensor(X_protein_val_scaled).to(trainer.device)
            X_mol_train_tensor = torch.FloatTensor(X_mol_train_scaled).to(trainer.device)
            X_mol_val_tensor = torch.FloatTensor(X_mol_val_scaled).to(trainer.device)
            y_train_tensor = torch.LongTensor(y_train_encoded).to(trainer.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(trainer.device)
            
            # Train
            trainer.train(X_protein_train_tensor, X_mol_train_tensor, y_train_tensor,
                         X_protein_val_tensor, X_mol_val_tensor, y_val_tensor)
            
            # Evaluate
            train_results = trainer.evaluate(X_protein_train_tensor, X_mol_train_tensor, y_train_tensor)
            val_results = trainer.evaluate(X_protein_val_tensor, X_mol_val_tensor, y_val_tensor)
            
            # Store results
            cv_results['train_accuracy'].append(train_results['accuracy'])
            cv_results['val_accuracy'].append(val_results['accuracy'])
            cv_results['train_f1'].append(f1_score(train_results['true_labels'], train_results['predictions'], average='macro'))
            cv_results['val_f1'].append(f1_score(val_results['true_labels'], val_results['predictions'], average='macro'))
            
            logger.info(f"Fold {fold} - Train Acc: {train_results['accuracy']:.4f}, Val Acc: {val_results['accuracy']:.4f}")
        
        # Calculate mean and std
        metrics_to_process = list(cv_results.keys())
        for metric in metrics_to_process:
            if metric.endswith('_mean') or metric.endswith('_std'):
                continue
            values = cv_results[metric]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        logger.info("Cross-validation completed!")
        return cv_results

class TrainingPipeline:
    """Main training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.sampler = BalancedSampler(config)
        self.preprocessor = DataPreprocessor()
        self.cross_validator = CrossValidator(config)
        
        # Results storage
        self.results = {}
    
    def load_data(self, 
                 protein_embeddings: np.ndarray,
                 molecular_fingerprints: np.ndarray,
                 labels: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and validate data"""
        
        logger.info("Loading data...")
        
        # Validate data shapes
        assert len(protein_embeddings) == len(molecular_fingerprints) == len(labels), "Data length mismatch"
        assert protein_embeddings.shape[1] == self.config.protein_embedding_dim, "Protein embedding dimension mismatch"
        assert molecular_fingerprints.shape[1] == self.config.molecular_fingerprint_dim, "Molecular fingerprint dimension mismatch"
        
        logger.info(f"Loaded {len(labels)} samples")
        logger.info(f"Protein embeddings shape: {protein_embeddings.shape}")
        logger.info(f"Molecular fingerprints shape: {molecular_fingerprints.shape}")
        
        return protein_embeddings, molecular_fingerprints, labels
    
    def preprocess_data(self,
                       protein_embeddings: np.ndarray,
                       molecular_fingerprints: np.ndarray,
                       labels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data"""
        
        logger.info("Preprocessing data...")
        
        # Balance dataset
        balanced_protein, balanced_molecular, balanced_labels = self.sampler.balance_dataset(
            protein_embeddings, molecular_fingerprints, labels
        )
        
        # Scale features and encode labels
        protein_scaled, molecular_scaled, labels_encoded = self.preprocessor.fit_transform(
            balanced_protein, balanced_molecular, balanced_labels
        )
        
        return protein_scaled, molecular_scaled, labels_encoded
    
    def train_final_model(self,
                         protein_embeddings: np.ndarray,
                         molecular_fingerprints: np.ndarray,
                         labels: List[str]) -> TerpenePredictorTrainer:
        """Train final model on full dataset"""
        
        logger.info("Training final model...")
        
        # Preprocess data
        protein_scaled, molecular_scaled, labels_encoded = self.preprocess_data(
            protein_embeddings, molecular_fingerprints, labels
        )
        
        # Split into train/test
        X_protein_train, X_protein_test, X_mol_train, X_mol_test, y_train, y_test = train_test_split(
            protein_scaled, molecular_scaled, labels_encoded,
            test_size=self.config.test_size, random_state=self.config.random_state, stratify=labels_encoded
        )
        
        # Split train into train/val
        X_protein_train, X_protein_val, X_mol_train, X_mol_val, y_train, y_val = train_test_split(
            X_protein_train, X_mol_train, y_train,
            test_size=self.config.val_size, random_state=self.config.random_state, stratify=y_train
        )
        
        # Create model config
        model_config = ModelConfig(
            protein_embedding_dim=self.config.protein_embedding_dim,
            molecular_fingerprint_dim=self.config.molecular_fingerprint_dim,
            num_classes=len(self.preprocessor.label_encoder.classes_),
            hidden_dim=self.config.hidden_dim,
            num_attention_heads=self.config.num_attention_heads,
            dropout_rate=self.config.dropout_rate,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            early_stopping_patience=self.config.early_stopping_patience
        )
        
        # Initialize trainer
        trainer = TerpenePredictorTrainer(model_config)
        
        # Fit the label encoder with original labels
        trainer.label_encoder.fit(self.preprocessor.label_encoder.classes_)
        
        # Convert to tensors
        X_protein_train_tensor = torch.FloatTensor(X_protein_train).to(trainer.device)
        X_protein_val_tensor = torch.FloatTensor(X_protein_val).to(trainer.device)
        X_protein_test_tensor = torch.FloatTensor(X_protein_test).to(trainer.device)
        X_mol_train_tensor = torch.FloatTensor(X_mol_train).to(trainer.device)
        X_mol_val_tensor = torch.FloatTensor(X_mol_val).to(trainer.device)
        X_mol_test_tensor = torch.FloatTensor(X_mol_test).to(trainer.device)
        y_train_tensor = torch.LongTensor(y_train).to(trainer.device)
        y_val_tensor = torch.LongTensor(y_val).to(trainer.device)
        y_test_tensor = torch.LongTensor(y_test).to(trainer.device)
        
        # Train model
        trainer.train(X_protein_train_tensor, X_mol_train_tensor, y_train_tensor,
                     X_protein_val_tensor, X_mol_val_tensor, y_val_tensor)
        
        # Evaluate on test set
        test_results = trainer.evaluate(X_protein_test_tensor, X_mol_test_tensor, y_test_tensor)
        
        # Store results
        self.results['final_model'] = {
            'test_accuracy': test_results['accuracy'],
            'test_predictions': test_results['predictions'],
            'test_probabilities': test_results['probabilities'],
            'test_true_labels': test_results['true_labels'],
            'classification_report': test_results['classification_report'],
            'confusion_matrix': test_results['confusion_matrix']
        }
        
        logger.info(f"Final model test accuracy: {test_results['accuracy']:.4f}")
        
        return trainer
    
    def run_cross_validation(self,
                           protein_embeddings: np.ndarray,
                           molecular_fingerprints: np.ndarray,
                           labels: List[str]) -> Dict[str, List[float]]:
        """Run cross-validation"""
        
        logger.info("Running cross-validation...")
        
        # Balance dataset
        balanced_protein, balanced_molecular, balanced_labels = self.sampler.balance_dataset(
            protein_embeddings, molecular_fingerprints, labels
        )
        
        # Run cross-validation
        cv_results = self.cross_validator.cross_validate(
            balanced_protein, balanced_molecular, balanced_labels, self.preprocessor
        )
        
        # Store results
        self.results['cross_validation'] = cv_results
        
        # Log results
        logger.info("Cross-validation results:")
        logger.info(f"Train Accuracy: {cv_results['train_accuracy_mean']:.4f} ± {cv_results['train_accuracy_std']:.4f}")
        logger.info(f"Val Accuracy: {cv_results['val_accuracy_mean']:.4f} ± {cv_results['val_accuracy_std']:.4f}")
        logger.info(f"Train F1: {cv_results['train_f1_mean']:.4f} ± {cv_results['train_f1_std']:.4f}")
        logger.info(f"Val F1: {cv_results['val_f1_mean']:.4f} ± {cv_results['val_f1_std']:.4f}")
        
        return cv_results
    
    def save_results(self, filename: str = "training_results.pkl"):
        """Save training results"""
        output_path = Path("data/cache") / filename
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Results saved to {output_path}")
    
    def plot_training_history(self, trainer: TerpenePredictorTrainer):
        """Plot training history"""
        history = trainer.train_history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history['loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Train Accuracy')
        ax2.plot(history['val_accuracy'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('data/cache/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to demonstrate the training pipeline"""
    logger.info("Starting terpene synthase training pipeline...")
    
    # Create sample data
    num_samples = 2000
    protein_embedding_dim = 1280
    molecular_fingerprint_dim = 2223
    
    # Generate random data for demonstration
    protein_embeddings = np.random.randn(num_samples, protein_embedding_dim)
    molecular_fingerprints = np.random.randn(num_samples, molecular_fingerprint_dim)
    
    # Create imbalanced labels
    products = ['limonene', 'pinene', 'myrcene', 'linalool', 'germacrene_a', 
                'germacrene_d', 'caryophyllene', 'humulene', 'farnesene', 'bisabolene']
    
    # Create imbalanced distribution
    label_probs = [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.04, 0.03, 0.02, 0.01]
    labels = np.random.choice(products, num_samples, p=label_probs)
    
    # Create training config
    config = TrainingConfig(
        min_samples_per_class=50,
        max_samples_per_class=200,
        protein_embedding_dim=protein_embedding_dim,
        molecular_fingerprint_dim=molecular_fingerprint_dim
    )
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    # Load data
    protein_embeddings, molecular_fingerprints, labels = pipeline.load_data(
        protein_embeddings, molecular_fingerprints, labels
    )
    
    # Run cross-validation
    cv_results = pipeline.run_cross_validation(
        protein_embeddings, molecular_fingerprints, labels
    )
    
    # Train final model
    final_trainer = pipeline.train_final_model(
        protein_embeddings, molecular_fingerprints, labels
    )
    
    # Plot training history
    pipeline.plot_training_history(final_trainer)
    
    # Save results
    pipeline.save_results()
    
    # Print final summary
    print(f"\nTraining Pipeline Summary:")
    print(f"Cross-validation accuracy: {cv_results['val_accuracy_mean']:.4f} ± {cv_results['val_accuracy_std']:.4f}")
    print(f"Final model test accuracy: {pipeline.results['final_model']['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
