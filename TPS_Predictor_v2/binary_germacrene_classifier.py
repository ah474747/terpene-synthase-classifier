#!/usr/bin/env python3
"""
Binary Germacrene Classification Pipeline

This script implements a binary classifier to distinguish between:
- Germacrene-producing terpene synthases (A, D, B, C variants)
- All other terpene synthases

This approach simplifies the multi-class problem and should yield higher accuracy.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import our existing components
from models.hybrid_ensemble_encoder import HybridEnsembleEncoder
from training.training_pipeline import TerpenePredictorTrainer, TrainingConfig, ModelConfig
from data.marts_parser import MARTSDBParser
from config.config import TerpenePredictorConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryGermacreneClassifier:
    """Binary classifier for Germacrene vs Other terpene synthases"""
    
    def __init__(self, config: TerpenePredictorConfig):
        self.config = config
        self.encoder = HybridEnsembleEncoder()
        self.trainer = None
        self.binary_label_encoder = None
        self.results = {}
        
        # Define Germacrene variants (matching MARTS-DB format)
        self.germacrene_variants = [
            'germacrene_a', 'germacrene_d', 'germacrene_b', 'germacrene_c',
            'germacrene a', 'germacrene d', 'germacrene b', 'germacrene c',
            'germacrene-a', 'germacrene-d', 'germacrene-b', 'germacrene-c',
            'bicyclogermacrene', 'helminthogermacrene'
        ]
        
        logger.info("Binary Germacrene Classifier initialized")
        logger.info(f"Germacrene variants: {self.germacrene_variants}")
    
    def load_training_data(self) -> Tuple[List[str], List[str]]:
        """Load and prepare training data for binary classification"""
        logger.info("Loading training data for binary classification...")
        
        # Load MARTS-DB data
        marts_parser = MARTSDBParser()
        records = marts_parser.parse_marts_data('reactions.csv')
        
        sequences = []
        binary_labels = []
        
        for record in records:
            if record.sequence and record.product_name:
                sequences.append(record.sequence)
                
                # Convert to binary: Germacrene (1) vs Other (0)
                product_lower = record.product_name.lower().strip()
                is_germacrene = any(variant in product_lower for variant in self.germacrene_variants)
                binary_labels.append(1 if is_germacrene else 0)
        
        logger.info(f"Loaded {len(sequences)} sequences")
        logger.info(f"Germacrene sequences: {sum(binary_labels)}")
        logger.info(f"Other sequences: {len(sequences) - sum(binary_labels)}")
        logger.info(f"Class balance: {sum(binary_labels)/len(sequences)*100:.1f}% Germacrene")
        
        return sequences, binary_labels
    
    def create_binary_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Create embeddings for binary classification"""
        logger.info("Creating binary classification embeddings...")
        
        # Encode sequences
        embeddings = self.encoder.encode_sequences(sequences)
        embedding_matrix, _ = self.encoder.create_embedding_matrix(embeddings)
        
        logger.info(f"Created embedding matrix: {embedding_matrix.shape}")
        return embedding_matrix
    
    def train_binary_model(self, X: np.ndarray, y: List[int]) -> Dict[str, Any]:
        """Train binary classification model"""
        logger.info("Training binary Germacrene classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} sequences")
        logger.info(f"Test set: {len(X_test)} sequences")
        logger.info(f"Training Germacrene: {sum(y_train)}")
        logger.info(f"Test Germacrene: {sum(y_test)}")
        
        # Create binary label encoder
        from sklearn.preprocessing import LabelEncoder
        self.binary_label_encoder = LabelEncoder()
        y_train_encoded = self.binary_label_encoder.fit_transform(y_train)
        y_test_encoded = self.binary_label_encoder.transform(y_test)
        
        # Use a simple MLP classifier for binary classification
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.model_selection import cross_val_score
        
        # Use Random Forest with class weighting to handle imbalance
        from sklearn.ensemble import RandomForestClassifier
        
        self.trainer = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        logger.info("Training Random Forest classifier with balanced class weights...")
        self.trainer.fit(X_train, y_train_encoded)
        
        # Evaluate on test set
        test_predictions = self.trainer.predict(X_test)
        test_accuracy = accuracy_score(y_test_encoded, test_predictions)
        test_precision = precision_score(y_test_encoded, test_predictions, average='binary')
        test_recall = recall_score(y_test_encoded, test_predictions, average='binary')
        test_f1 = f1_score(y_test_encoded, test_predictions, average='binary')
        
        # Cross-validation
        cv_scores = cross_val_score(self.trainer, X_train, y_train_encoded, cv=5, scoring='accuracy')
        
        results = {
            'train_accuracy': self.trainer.score(X_train, y_train_encoded),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test_encoded, test_predictions),
            'class_names': self.binary_label_encoder.classes_
        }
        
        logger.info(f"Binary classification results:")
        logger.info(f"  Training accuracy: {results['train_accuracy']:.4f}")
        logger.info(f"  Test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"  Test precision: {results['test_precision']:.4f}")
        logger.info(f"  Test recall: {results['test_recall']:.4f}")
        logger.info(f"  Test F1-score: {results['test_f1']:.4f}")
        logger.info(f"  CV accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        return results
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str = "binary_classification_results"):
        """Create visualizations for binary classification results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['Other', 'Germacrene'],
                    yticklabels=['Other', 'Germacrene'])
        plt.title('Binary Germacrene Classification - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()
        
        # Performance Metrics
        metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
        values = [results['test_accuracy'], results['test_precision'], 
                 results['test_recall'], results['test_f1']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Binary Germacrene Classification - Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_metrics.png")
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")
    
    def save_binary_model(self, output_path: str = "data/cache/binary_germacrene_model.pkl"):
        """Save the trained binary model"""
        model_data = {
            'trainer': self.trainer,
            'binary_label_encoder': self.binary_label_encoder,
            'encoder': self.encoder,
            'results': self.results,
            'germacrene_variants': self.germacrene_variants
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Binary model saved to {output_path}")
    
    def predict_sequence(self, sequence: str) -> Dict[str, Any]:
        """Predict if a sequence produces Germacrene"""
        if not self.trainer:
            raise ValueError("Model not trained. Run train_binary_model() first.")
        
        # Encode sequence
        embedding = self.encoder.encode_sequence(sequence)
        if not embedding:
            return {'error': 'Failed to encode sequence'}
        
        # Get prediction
        if hasattr(embedding, 'combined_embedding'):
            protein_tensor = embedding.combined_embedding
        else:
            protein_tensor = embedding.embedding
        
        # Predict using MLP classifier
        prediction = self.trainer.predict([protein_tensor])[0]
        probabilities = self.trainer.predict_proba([protein_tensor])[0]
        
        # Convert back to labels
        predicted_class = self.binary_label_encoder.inverse_transform([prediction])[0]
        class_name = 'Germacrene' if predicted_class == 1 else 'Other'
        
        return {
            'sequence': sequence,
            'predicted_class': class_name,
            'confidence': probabilities[prediction],
            'probabilities': {
                'Other': probabilities[0],
                'Germacrene': probabilities[1]
            }
        }

def main():
    """Main function to run binary Germacrene classification"""
    logger.info("Starting Binary Germacrene Classification Pipeline")
    
    # Initialize
    config = TerpenePredictorConfig()
    classifier = BinaryGermacreneClassifier(config)
    
    # Load and prepare data
    sequences, binary_labels = classifier.load_training_data()
    
    # Create embeddings
    embedding_matrix = classifier.create_binary_embeddings(sequences)
    
    # Train model
    results = classifier.train_binary_model(embedding_matrix, binary_labels)
    classifier.results = results
    
    # Create visualizations
    classifier.visualize_results(results)
    
    # Save model
    classifier.save_binary_model()
    
    # Test prediction
    logger.info("\n" + "="*60)
    logger.info("TESTING BINARY PREDICTION")
    logger.info("="*60)
    
    # Test with a few sequences
    test_sequences = sequences[:5]
    for i, seq in enumerate(test_sequences):
        result = classifier.predict_sequence(seq)
        true_label = 'Germacrene' if binary_labels[i] == 1 else 'Other'
        
        logger.info(f"\nSequence {i+1}:")
        logger.info(f"  True label: {true_label}")
        logger.info(f"  Predicted: {result['predicted_class']}")
        logger.info(f"  Confidence: {result['confidence']:.3f}")
        logger.info(f"  Probabilities: Other={result['probabilities']['Other']:.3f}, Germacrene={result['probabilities']['Germacrene']:.3f}")
    
    logger.info("\n" + "="*60)
    logger.info("BINARY CLASSIFICATION COMPLETED SUCCESSFULLY!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
