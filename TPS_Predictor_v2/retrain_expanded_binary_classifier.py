#!/usr/bin/env python3
"""
Retrain binary Germacrene classifier with expanded training data.
This script uses the expanded MARTS-DB dataset with 13 additional Germacrene sequences.
"""

import pandas as pd
import numpy as np
import torch
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from pathlib import Path

# Import our custom modules
from data.marts_parser import MARTSDBParser
from models.hybrid_ensemble_encoder import HybridEnsembleEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedBinaryGermacreneClassifier:
    """Binary classifier for Germacrene prediction with expanded training data"""
    
    def __init__(self, expanded_data_file='reactions_expanded.csv'):
        self.expanded_data_file = expanded_data_file
        self.trainer = None
        self.binary_label_encoder = None
        self.encoder = None
        self.results = None
        
        logger.info(f"Initializing expanded binary Germacrene classifier")
        logger.info(f"Using expanded dataset: {expanded_data_file}")
    
    def load_expanded_data(self):
        """Load the expanded MARTS-DB dataset"""
        logger.info("Loading expanded MARTS-DB data...")
        
        # Use MARTSDBParser to process the expanded data
        parser = MARTSDBParser()
        processed_data = parser.parse_marts_data(self.expanded_data_file)
        
        logger.info(f"Loaded {len(processed_data)} sequences from expanded dataset")
        
        # Check Germacrene count
        germacrene_count = sum(1 for record in processed_data if 'germacrene' in record.product_name.lower())
        logger.info(f"Germacrene sequences in expanded dataset: {germacrene_count}")
        
        return processed_data
    
    def prepare_binary_data(self, processed_data):
        """Prepare binary classification data (Germacrene vs Other)"""
        logger.info("Preparing binary classification data...")
        
        # Create binary labels
        def is_germacrene(product):
            if product is None:
                return False
            return 'germacrene' in str(product).lower()
        
        # Count classes
        germacrene_count = sum(1 for record in processed_data if is_germacrene(record.product_name))
        other_count = len(processed_data) - germacrene_count
        
        logger.info(f"Binary classification data prepared:")
        logger.info(f"  Germacrene sequences: {germacrene_count}")
        logger.info(f"  Other sequences: {other_count}")
        logger.info(f"  Germacrene percentage: {germacrene_count/len(processed_data)*100:.1f}%")
        
        return processed_data
    
    def encode_sequences(self, processed_data):
        """Encode protein sequences using hybrid ensemble"""
        logger.info("Encoding protein sequences with hybrid ensemble...")
        
        # Initialize encoder
        self.encoder = HybridEnsembleEncoder()
        
        # Encode sequences
        sequences = [record.sequence for record in processed_data]
        embeddings = self.encoder.encode_sequences(sequences)
        
        # Create embedding matrix
        embedding_matrix, _ = self.encoder.create_embedding_matrix(embeddings)
        
        logger.info(f"Encoded {len(embeddings)} sequences")
        logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
        
        return embedding_matrix
    
    def train_model(self, X, y):
        """Train the binary classifier"""
        logger.info("Training binary Germacrene classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} sequences")
        logger.info(f"Test set: {len(X_test)} sequences")
        
        # Encode labels
        self.binary_label_encoder = LabelEncoder()
        y_train_encoded = self.binary_label_encoder.fit_transform(y_train)
        y_test_encoded = self.binary_label_encoder.transform(y_test)
        
        # Train Random Forest with balanced class weights
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
        y_pred = self.trainer.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.3f}")
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(self.trainer, X_train, y_train_encoded, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Detailed classification report
        logger.info("Classification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=self.binary_label_encoder.classes_))
        
        # Store results
        self.results = {
            'test_accuracy': accuracy,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'classification_report': classification_report(y_test_encoded, y_pred, target_names=self.binary_label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
        }
        
        return self.results
    
    def save_model(self, model_path='data/cache/expanded_binary_germacrene_model.pkl'):
        """Save the trained model and components"""
        logger.info(f"Saving expanded binary model to: {model_path}")
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'trainer': self.trainer,
            'binary_label_encoder': self.binary_label_encoder,
            'encoder': self.encoder,
            'results': self.results,
            'germacrene_variants': ['Germacrene A', 'Germacrene B', 'Germacrene C', 'Germacrene D', 'Germacrene'],
            'model_type': 'expanded_binary_germacrene_classifier',
            'training_data_file': self.expanded_data_file
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Expanded binary model saved successfully!")
        return model_path

def main():
    """Main training function"""
    
    logger.info("🚀 Starting expanded binary Germacrene classifier training...")
    
    # Check if expanded data exists
    expanded_file = 'reactions_expanded.csv'
    if not Path(expanded_file).exists():
        logger.error(f"Expanded data file not found: {expanded_file}")
        logger.error("Please run expand_training_data.py first")
        return
    
    # Initialize classifier
    classifier = ExpandedBinaryGermacreneClassifier(expanded_file)
    
    # Load and prepare data
    processed_data = classifier.load_expanded_data()
    processed_data = classifier.prepare_binary_data(processed_data)
    
    # Encode sequences
    X = classifier.encode_sequences(processed_data)
    
    # Create binary labels
    def is_germacrene(product):
        if product is None:
            return False
        return 'germacrene' in str(product).lower()
    
    y = ['Germacrene' if is_germacrene(record.product_name) else 'Other' for record in processed_data]
    
    # Train model
    results = classifier.train_model(X, y)
    
    # Save model
    model_path = classifier.save_model()
    
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
    print(f"Model saved to: {model_path}")
    print(f"{'='*70}")
    
    logger.info("✅ Expanded binary Germacrene classifier training completed!")

if __name__ == "__main__":
    main()
