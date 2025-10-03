#!/usr/bin/env python3
"""
Retrain binary Germacrene classifier with expanded dataset including NCBI sequences.
This should dramatically improve generalization to NCBI sequences.
"""

import pandas as pd
import numpy as np
import torch
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary components from the existing pipeline
from models.hybrid_ensemble_encoder import HybridEnsembleEncoder
from data.marts_parser import MARTSDBParser, MARTSRecord

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExpandedBinaryGermacreneClassifierTrainer:
    def __init__(self, expanded_data_file: str = 'reactions_with_ncbi_germacrene.csv', 
                 model_output_path: str = 'data/cache/expanded_ncbi_binary_germacrene_model.pkl',
                 test_size: float = 0.2, random_state: int = 42):
        self.expanded_data_file = Path(expanded_data_file)
        self.model_output_path = Path(model_output_path)
        self.test_size = test_size
        self.random_state = random_state
        self.device = self._get_device()
        
        self.encoder = None  # Initialized in encode_sequences
        self.binary_label_encoder = LabelEncoder()
        self.trainer = None  # RandomForestClassifier
        self.results = None
        
        logger.info(f"Initializing expanded binary Germacrene classifier with NCBI data")
        logger.info(f"Using expanded dataset: {expanded_data_file}")
    
    def _get_device(self):
        """Determines and returns the appropriate device (MPS, CUDA, or CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def load_expanded_data(self) -> List[MARTSRecord]:
        """Load the expanded MARTS-DB dataset with NCBI sequences"""
        logger.info("Loading expanded MARTS-DB data with NCBI Germacrene sequences...")
        
        # Use MARTSDBParser to process the expanded data
        parser = MARTSDBParser()
        processed_data = parser.parse_marts_data(self.expanded_data_file)
        
        logger.info(f"Loaded {len(processed_data)} sequences from expanded dataset")
        
        # Check Germacrene count
        germacrene_count = sum(1 for record in processed_data if 'germacrene' in record.product_name.lower())
        logger.info(f"Germacrene sequences in expanded dataset: {germacrene_count}")
        logger.info(f"Germacrene percentage: {germacrene_count/len(processed_data)*100:.1f}%")
        
        return processed_data
    
    def prepare_binary_data(self, processed_data: List[MARTSRecord]) -> List[MARTSRecord]:
        """Prepare binary classification data (Germacrene vs Other)"""
        logger.info("Preparing binary classification data...")
        
        # Create binary labels
        def is_germacrene(product_name):
            if product_name is None:
                return False
            return 'germacrene' in str(product_name).lower()
        
        # Count classes
        germacrene_count = sum(1 for record in processed_data if is_germacrene(record.product_name))
        other_count = len(processed_data) - germacrene_count
        
        logger.info(f"Binary classification data prepared:")
        logger.info(f"  Germacrene sequences: {germacrene_count}")
        logger.info(f"  Other sequences: {other_count}")
        logger.info(f"  Germacrene percentage: {germacrene_count/len(processed_data)*100:.1f}%")
        logger.info(f"  Class ratio: {germacrene_count}:{other_count} ({germacrene_count/other_count:.1f}:1)")
        
        return processed_data

    def encode_sequences(self, processed_data: List[MARTSRecord]) -> np.ndarray:
        """Encode protein sequences using hybrid ensemble"""
        logger.info("Encoding protein sequences with hybrid ensemble...")
        
        # Initialize encoder
        self.encoder = HybridEnsembleEncoder(device=self.device)
        
        # Encode sequences
        sequences = [record.sequence for record in processed_data]
        embeddings = self.encoder.encode_sequences(sequences)
        
        # Create embedding matrix
        embedding_matrix, _ = self.encoder.create_embedding_matrix(embeddings)
        
        logger.info(f"Encoded {len(embeddings)} sequences")
        logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
        
        return embedding_matrix
    
    def train_model(self, X: np.ndarray, y: List[str]) -> Dict[str, Any]:
        """Train the binary classification model."""
        logger.info("Training expanded binary Germacrene classifier...")

        # Encode labels
        y_encoded = self.binary_label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        logger.info(f"Training set: {len(X_train)} sequences")
        logger.info(f"Test set: {len(X_test)} sequences")

        # Initialize and train RandomForestClassifier
        self.trainer = RandomForestClassifier(
            n_estimators=200,  # Increased for larger dataset
            max_depth=15,      # Increased for more complex patterns
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Handle class imbalance
            random_state=self.random_state,
            n_jobs=-1
        )
        
        logger.info("Training Random Forest classifier with balanced class weights...")
        self.trainer.fit(X_train, y_train_encoded)
        logger.info("Model training complete.")

        # Evaluate model on test set
        y_pred_encoded = self.trainer.predict(X_test)
        y_pred_proba = self.trainer.predict_proba(X_test)
        
        test_accuracy = self.trainer.score(X_test, y_test_encoded)
        logger.info(f"Test accuracy: {test_accuracy:.3f}")

        # Cross-validation
        logger.info("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = []
        for train_idx, val_idx in cv.split(X, y_encoded):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y_encoded[train_idx], y_encoded[val_idx]
            
            cv_trainer = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            cv_trainer.fit(X_cv_train, y_cv_train)
            cv_scores.append(cv_trainer.score(X_cv_val, y_cv_val))
        
        cv_accuracy_mean = np.mean(cv_scores)
        cv_accuracy_std = np.std(cv_scores)
        logger.info(f"Cross-validation accuracy: {cv_accuracy_mean:.3f} ± {cv_accuracy_std:.3f}")

        report = classification_report(y_test_encoded, y_pred_encoded, target_names=self.binary_label_encoder.classes_)
        logger.info(f"\nClassification Report:\n{report}")

        # Calculate metrics for Germacrene specifically
        germacrene_idx = np.where(self.binary_label_encoder.classes_ == 'Germacrene')[0][0]
        
        precision_germacrene = precision_score(y_test_encoded, y_pred_encoded, pos_label=germacrene_idx, average='binary')
        recall_germacrene = recall_score(y_test_encoded, y_pred_encoded, pos_label=germacrene_idx, average='binary')
        f1_germacrene = f1_score(y_test_encoded, y_pred_encoded, pos_label=germacrene_idx, average='binary')
        
        logger.info(f"Germacrene Precision: {precision_germacrene:.3f}")
        logger.info(f"Germacrene Recall: {recall_germacrene:.3f}")
        logger.info(f"Germacrene F1-score: {f1_germacrene:.3f}")

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.binary_label_encoder.classes_, 
                    yticklabels=self.binary_label_encoder.classes_)
        plt.title('Confusion Matrix (Expanded Dataset with NCBI)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Ensure output directory exists
        output_dir = Path("binary_classification_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'confusion_matrix_expanded_ncbi.png')
        logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix_expanded_ncbi.png'}")
        plt.close()

        # Store results
        self.results = {
            'test_accuracy': test_accuracy,
            'cv_accuracy_mean': cv_accuracy_mean,
            'cv_accuracy_std': cv_accuracy_std,
            'classification_report': report,
            'precision_germacrene': precision_germacrene,
            'recall_germacrene': recall_germacrene,
            'f1_germacrene': f1_germacrene,
            'confusion_matrix': cm,
            'test_true_labels': self.binary_label_encoder.inverse_transform(y_test_encoded),
            'test_predictions': self.binary_label_encoder.inverse_transform(y_pred_encoded),
            'test_probabilities': y_pred_proba
        }
        
        logger.info("Model training and evaluation with expanded NCBI data complete.")
        return self.results

    def save_model(self):
        """Save the trained model and associated components."""
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_output_path, 'wb') as f:
            pickle.dump({
                'trainer': self.trainer,
                'binary_label_encoder': self.binary_label_encoder,
                'encoder': self.encoder,
                'results': self.results,
                'data_path': self.expanded_data_file
            }, f)
        logger.info(f"Trained expanded binary model saved to {self.model_output_path}")

def main():
    expanded_file = 'reactions_with_ncbi_germacrene.csv'
    model_path = 'data/cache/expanded_ncbi_binary_germacrene_model.pkl'
    classifier = ExpandedBinaryGermacreneClassifierTrainer(expanded_data_file=expanded_file, model_output_path=model_path)
    
    # Load and prepare data
    processed_data = classifier.load_expanded_data()
    classifier.prepare_binary_data(processed_data)
    
    # Encode sequences
    X = classifier.encode_sequences(processed_data)
    
    # Create binary labels
    def is_germacrene(product_name):
        if product_name is None:
            return False
        return 'germacrene' in str(product_name).lower()
    
    y = ['Germacrene' if is_germacrene(record.product_name) else 'Other' for record in processed_data]
    
    # Train model
    results = classifier.train_model(X, y)
    
    # Save model
    classifier.save_model()

    # Print summary
    print(f"\n{'='*70}")
    print(f"EXPANDED BINARY GERMACRENE CLASSIFIER TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Training data: {expanded_file}")
    print(f"Total sequences: {len(processed_data)}")
    germacrene_count = sum(1 for record in processed_data if 'germacrene' in record.product_name.lower())
    print(f"Germacrene sequences: {germacrene_count}")
    print(f"Test accuracy: {results['test_accuracy']:.3f}")
    print(f"CV accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}")
    print(f"Germacrene recall: {results['recall_germacrene']:.3f}")
    print(f"Germacrene precision: {results['precision_germacrene']:.3f}")
    print(f"Germacrene F1-score: {results['f1_germacrene']:.3f}")
    print(f"Model saved to: {model_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
