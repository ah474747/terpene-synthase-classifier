#!/usr/bin/env python3
"""
Phase 2: Semi-Supervised Learning Pipeline
==========================================

This script implements the semi-supervised learning pipeline for the germacrene classifier:
1. Load the trained model from Phase 1
2. Load the large-scale terpene synthase dataset
3. Generate embeddings for unlabeled sequences
4. Apply pseudo-labeling with confidence thresholds
5. Retrain the model on combined labeled + pseudo-labeled data
6. Evaluate performance improvements

Author: AI Assistant
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
import joblib
import warnings

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_embedding_generator import RobustEmbeddingGenerator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class SemiSupervisedGermacreneClassifier:
    """
    Semi-supervised learning pipeline for germacrene classification
    """
    
    def __init__(self, 
                 model_path: str = "models/germacrene_classifier_full_retrained.pkl",
                 confidence_threshold: float = 0.95,
                 max_pseudo_labels: int = 10000):
        """
        Initialize the semi-supervised classifier
        
        Args:
            model_path: Path to the trained model from Phase 1
            confidence_threshold: Confidence threshold for pseudo-labeling
            max_pseudo_labels: Maximum number of pseudo-labels to use
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_pseudo_labels = max_pseudo_labels
        
        # Load the trained model
        self.load_trained_model()
        
        # Initialize embedding generator
        self.embedding_generator = RobustEmbeddingGenerator(
            model_name="esm2_t33_650M_UR50D",
            max_length=512
        )
        
        # Data storage
        self.labeled_data = None
        self.unlabeled_data = None
        self.pseudo_labeled_data = None
        self.combined_data = None
        
        # Results storage
        self.results = {}
        
    def load_trained_model(self):
        """Load the trained model from Phase 1"""
        print(f"Loading trained model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = joblib.load(f)
        
        self.trained_model = model_data['model']
        self.scaler = StandardScaler()  # We'll fit this on the new data
        
        print(f"✓ Model loaded successfully")
        print(f"  - Training sequences: {model_data['training_sequences']}")
        print(f"  - Embedding dimension: {model_data['embedding_dimension']}")
        print(f"  - Class balance: {model_data['class_balance']:.3f}")
    
    def load_phase2_data(self, data_path: str = "data/phase2/processed_terpene_sequences.csv"):
        """
        Load the Phase 2 terpene synthase dataset
        
        Args:
            data_path: Path to the processed Phase 2 data
        """
        print(f"\n=== Loading Phase 2 Data ===")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Phase 2 data not found: {data_path}")
        
        # Load the data
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} sequences from Phase 2")
        
        # Separate labeled and unlabeled data
        # For now, we'll treat all Phase 2 data as unlabeled
        # In practice, you might have some labeled sequences here too
        self.unlabeled_data = df.copy()
        
        print(f"  - Unlabeled sequences: {len(self.unlabeled_data)}")
        print(f"  - Germacrene sequences (from description): {self.unlabeled_data['is_germacrene'].sum()}")
        print(f"  - Average sequence length: {self.unlabeled_data['length'].mean():.1f}")
        
        # Load original labeled data for comparison
        self.load_original_labeled_data()
    
    def load_original_labeled_data(self):
        """Load the original labeled data from Phase 1"""
        print(f"\n=== Loading Original Labeled Data ===")
        
        # Load the original MARTS-DB data
        df = pd.read_csv("data/marts_db_enhanced.csv")
        df = df.dropna(subset=['Aminoacid_sequence', 'is_germacrene_family'])
        df = df[df['Aminoacid_sequence'].str.len() > 10]
        
        self.labeled_data = df.copy()
        print(f"✓ Loaded {len(self.labeled_data)} original labeled sequences")
        print(f"  - Germacrene sequences: {self.labeled_data['is_germacrene_family'].sum()}")
        print(f"  - Class balance: {self.labeled_data['is_germacrene_family'].mean():.3f}")
    
    def generate_embeddings_for_unlabeled(self, chunk_size: int = 50):
        """
        Generate embeddings for unlabeled sequences
        
        Args:
            chunk_size: Number of sequences to process at once
        """
        print(f"\n=== Generating Embeddings for Unlabeled Data ===")
        print(f"Processing {len(self.unlabeled_data)} sequences in chunks of {chunk_size}")
        
        sequences = self.unlabeled_data['sequence'].tolist()
        all_embeddings = []
        failed_indices = []
        
        # Process in chunks
        num_chunks = (len(sequences) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(sequences))
            chunk_sequences = sequences[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_sequences)} sequences)...")
            
            try:
                # Generate embeddings for this chunk
                chunk_embeddings_df = self.embedding_generator.generate_embeddings(chunk_sequences)
                
                if chunk_embeddings_df is not None and len(chunk_embeddings_df) > 0:
                    # Extract embeddings from DataFrame
                    chunk_embeddings = chunk_embeddings_df['embedding'].tolist()
                    all_embeddings.extend(chunk_embeddings)
                    print(f"  ✓ Generated {len(chunk_embeddings)} embeddings")
                else:
                    print(f"  ✗ Failed to generate embeddings for chunk {chunk_idx + 1}")
                    failed_indices.extend(range(start_idx, end_idx))
                    
            except Exception as e:
                print(f"  ✗ Error processing chunk {chunk_idx + 1}: {e}")
                failed_indices.extend(range(start_idx, end_idx))
        
        if failed_indices:
            print(f"⚠️  Failed to process {len(failed_indices)} sequences")
            # Remove failed sequences
            valid_indices = [i for i in range(len(sequences)) if i not in failed_indices]
            self.unlabeled_data = self.unlabeled_data.iloc[valid_indices].reset_index(drop=True)
            print(f"  Remaining sequences: {len(self.unlabeled_data)}")
        
        if not all_embeddings:
            raise ValueError("No embeddings were generated successfully!")
        
        # Store embeddings
        self.unlabeled_embeddings = np.array(all_embeddings)
        print(f"✓ Generated {len(all_embeddings)} embeddings")
        print(f"✓ Embedding shape: {self.unlabeled_embeddings.shape}")
    
    def apply_pseudo_labeling(self):
        """
        Apply pseudo-labeling to unlabeled data using the trained model
        """
        print(f"\n=== Applying Pseudo-Labeling ===")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Scale embeddings using the same scaler as the trained model
        # Note: We'll fit the scaler on the unlabeled data for now
        # In practice, you might want to use the scaler from the trained model
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(self.unlabeled_embeddings)
        
        # Make predictions
        print("Making predictions on unlabeled data...")
        predictions = self.trained_model.predict(scaled_embeddings)
        prediction_probas = self.trained_model.predict_proba(scaled_embeddings)
        
        # Get confidence scores (max probability)
        confidence_scores = np.max(prediction_probas, axis=1)
        
        # Filter by confidence threshold
        high_confidence_mask = confidence_scores >= self.confidence_threshold
        
        print(f"Total predictions: {len(predictions)}")
        print(f"High confidence predictions: {np.sum(high_confidence_mask)}")
        print(f"High confidence rate: {np.mean(high_confidence_mask):.3f}")
        
        # Create pseudo-labeled dataset
        pseudo_labeled_indices = np.where(high_confidence_mask)[0]
        
        if len(pseudo_labeled_indices) > self.max_pseudo_labels:
            # Select top confidence predictions
            top_indices = np.argsort(confidence_scores[pseudo_labeled_indices])[-self.max_pseudo_labels:]
            pseudo_labeled_indices = pseudo_labeled_indices[top_indices]
        
        # Create pseudo-labeled data
        self.pseudo_labeled_data = self.unlabeled_data.iloc[pseudo_labeled_indices].copy()
        self.pseudo_labeled_data['pseudo_label'] = predictions[pseudo_labeled_indices]
        self.pseudo_labeled_data['confidence'] = confidence_scores[pseudo_labeled_indices]
        self.pseudo_labeled_embeddings = self.unlabeled_embeddings[pseudo_labeled_indices]
        
        print(f"Selected {len(self.pseudo_labeled_data)} pseudo-labels")
        print(f"Pseudo-label distribution:")
        print(f"  - Germacrene: {np.sum(self.pseudo_labeled_data['pseudo_label'])}")
        print(f"  - Other: {len(self.pseudo_labeled_data) - np.sum(self.pseudo_labeled_data['pseudo_label'])}")
        print(f"  - Average confidence: {self.pseudo_labeled_data['confidence'].mean():.3f}")
        
        # Save pseudo-labeled data
        pseudo_data_path = "data/phase2/pseudo_labeled_sequences.csv"
        self.pseudo_labeled_data.to_csv(pseudo_data_path, index=False)
        print(f"Pseudo-labeled data saved to: {pseudo_data_path}")
    
    def combine_datasets(self):
        """
        Combine original labeled data with pseudo-labeled data
        """
        print(f"\n=== Combining Datasets ===")
        
        # Generate embeddings for original labeled data
        print("Generating embeddings for original labeled data...")
        labeled_sequences = self.labeled_data['Aminoacid_sequence'].tolist()
        
        # Process in chunks
        chunk_size = 50
        all_labeled_embeddings = []
        
        for i in range(0, len(labeled_sequences), chunk_size):
            chunk = labeled_sequences[i:i+chunk_size]
            chunk_embeddings_df = self.embedding_generator.generate_embeddings(chunk)
            
            if chunk_embeddings_df is not None:
                chunk_embeddings = chunk_embeddings_df['embedding'].tolist()
                all_labeled_embeddings.extend(chunk_embeddings)
        
        labeled_embeddings = np.array(all_labeled_embeddings)
        
        # Combine embeddings
        combined_embeddings = np.vstack([labeled_embeddings, self.pseudo_labeled_embeddings])
        
        # Combine labels
        original_labels = self.labeled_data['is_germacrene_family'].values
        pseudo_labels = self.pseudo_labeled_data['pseudo_label'].values
        combined_labels = np.concatenate([original_labels, pseudo_labels])
        
        # Create combined dataset
        self.combined_embeddings = combined_embeddings
        self.combined_labels = combined_labels
        
        print(f"Combined dataset:")
        print(f"  - Total sequences: {len(combined_labels)}")
        print(f"  - Original labeled: {len(original_labels)}")
        print(f"  - Pseudo-labeled: {len(pseudo_labels)}")
        print(f"  - Germacrene sequences: {np.sum(combined_labels)}")
        print(f"  - Class balance: {np.mean(combined_labels):.3f}")
    
    def retrain_model(self, cv_folds: int = 5):
        """
        Retrain the model on the combined dataset
        
        Args:
            cv_folds: Number of cross-validation folds
        """
        print(f"\n=== Retraining Model ===")
        
        # Scale features
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(self.combined_embeddings)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'auc_pr_scores': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(scaled_embeddings, self.combined_labels)):
            print(f"\n--- Fold {fold + 1}/{cv_folds} ---")
            
            # Split data
            X_train, X_test = scaled_embeddings[train_idx], scaled_embeddings[test_idx]
            y_train, y_test = self.combined_labels[train_idx], self.combined_labels[test_idx]
            
            # Calculate class weights
            pos_count = np.sum(y_train)
            neg_count = len(y_train) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            print(f"Training samples: {len(y_train)} (pos: {pos_count}, neg: {neg_count})")
            print(f"Test samples: {len(y_test)}")
            print(f"Scale pos weight: {scale_pos_weight:.2f}")
            
            # Train model
            model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc_pr = average_precision_score(y_test, y_pred_proba)
            
            print(f"F1-Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"AUC-PR: {auc_pr:.4f}")
            
            # Store results
            cv_results['f1_scores'].append(f1)
            cv_results['precision_scores'].append(precision)
            cv_results['recall_scores'].append(recall)
            cv_results['auc_pr_scores'].append(auc_pr)
        
        # Train final model on all data
        print(f"\n--- Training Final Model ---")
        final_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        final_model.fit(scaled_embeddings, self.combined_labels)
        
        # Store results
        self.retrained_model = final_model
        self.retrained_scaler = scaler
        self.cv_results = cv_results
        
        # Print summary
        print(f"\n=== Cross-Validation Results ===")
        print(f"F1-Score: {np.mean(cv_results['f1_scores']):.4f} ± {np.std(cv_results['f1_scores']):.4f}")
        print(f"Precision: {np.mean(cv_results['precision_scores']):.4f} ± {np.std(cv_results['precision_scores']):.4f}")
        print(f"Recall: {np.mean(cv_results['recall_scores']):.4f} ± {np.std(cv_results['recall_scores']):.4f}")
        print(f"AUC-PR: {np.mean(cv_results['auc_pr_scores']):.4f} ± {np.std(cv_results['auc_pr_scores']):.4f}")
    
    def save_retrained_model(self):
        """Save the retrained model"""
        print(f"\n=== Saving Retrained Model ===")
        
        # Create model data
        model_data = {
            'model': self.retrained_model,
            'scaler': self.retrained_scaler,
            'embedding_dimension': self.combined_embeddings.shape[1],
            'training_sequences': len(self.combined_labels),
            'original_labeled': len(self.labeled_data),
            'pseudo_labeled': len(self.pseudo_labeled_data),
            'positive_samples': int(np.sum(self.combined_labels)),
            'negative_samples': int(len(self.combined_labels) - np.sum(self.combined_labels)),
            'class_balance': float(np.mean(self.combined_labels)),
            'confidence_threshold': self.confidence_threshold,
            'cv_results': self.cv_results
        }
        
        # Save model
        model_path = "models/germacrene_classifier_semi_supervised.pkl"
        joblib.dump(model_data, model_path)
        print(f"✓ Retrained model saved to: {model_path}")
        
        # Save results
        results_path = "results/semi_supervised_results.json"
        results_data = {
            'original_labeled_sequences': len(self.labeled_data),
            'pseudo_labeled_sequences': len(self.pseudo_labeled_data),
            'total_training_sequences': len(self.combined_labels),
            'confidence_threshold': self.confidence_threshold,
            'cv_f1_mean': float(np.mean(self.cv_results['f1_scores'])),
            'cv_f1_std': float(np.std(self.cv_results['f1_scores'])),
            'cv_precision_mean': float(np.mean(self.cv_results['precision_scores'])),
            'cv_precision_std': float(np.std(self.cv_results['precision_scores'])),
            'cv_recall_mean': float(np.mean(self.cv_results['recall_scores'])),
            'cv_recall_std': float(np.std(self.cv_results['recall_scores'])),
            'cv_auc_pr_mean': float(np.mean(self.cv_results['auc_pr_scores'])),
            'cv_auc_pr_std': float(np.std(self.cv_results['auc_pr_scores'])),
            'training_completed': True,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"✓ Results saved to: {results_path}")
    
    def compare_performance(self):
        """Compare performance before and after semi-supervised learning"""
        print(f"\n=== Performance Comparison ===")
        
        # Load original model results
        original_results_path = "results/full_dataset_retrained_results.json"
        if os.path.exists(original_results_path):
            with open(original_results_path, 'r') as f:
                original_results = json.load(f)
            
            print(f"Original Model (Phase 1):")
            print(f"  - F1-Score: {original_results['cv_f1_mean']:.4f} ± {original_results['cv_f1_std']:.4f}")
            print(f"  - AUC-PR: {original_results['cv_auc_pr_mean']:.4f} ± {original_results['cv_auc_pr_std']:.4f}")
            print(f"  - Training sequences: {original_results['training_sequences']}")
        
        print(f"\nSemi-Supervised Model (Phase 2):")
        print(f"  - F1-Score: {np.mean(self.cv_results['f1_scores']):.4f} ± {np.std(self.cv_results['f1_scores']):.4f}")
        print(f"  - AUC-PR: {np.mean(self.cv_results['auc_pr_scores']):.4f} ± {np.std(self.cv_results['auc_pr_scores']):.4f}")
        print(f"  - Training sequences: {len(self.combined_labels)}")
        print(f"  - Pseudo-labels added: {len(self.pseudo_labeled_data)}")
        
        # Calculate improvement
        if os.path.exists(original_results_path):
            f1_improvement = np.mean(self.cv_results['f1_scores']) - original_results['cv_f1_mean']
            auc_improvement = np.mean(self.cv_results['auc_pr_scores']) - original_results['cv_auc_pr_mean']
            
            print(f"\nImprovement:")
            print(f"  - F1-Score: {f1_improvement:+.4f}")
            print(f"  - AUC-PR: {auc_improvement:+.4f}")


def main():
    """Main execution function"""
    print("Phase 2: Semi-Supervised Learning Pipeline")
    print("=" * 60)
    
    # Initialize semi-supervised classifier
    classifier = SemiSupervisedGermacreneClassifier(
        model_path="models/germacrene_classifier_full_retrained.pkl",
        confidence_threshold=0.95,
        max_pseudo_labels=10000
    )
    
    # Load Phase 2 data
    classifier.load_phase2_data()
    
    # Generate embeddings for unlabeled data
    classifier.generate_embeddings_for_unlabeled()
    
    # Apply pseudo-labeling
    classifier.apply_pseudo_labeling()
    
    # Combine datasets
    classifier.combine_datasets()
    
    # Retrain model
    classifier.retrain_model()
    
    # Save retrained model
    classifier.save_retrained_model()
    
    # Compare performance
    classifier.compare_performance()
    
    print(f"\n✓ Phase 2 semi-supervised learning completed successfully!")


if __name__ == "__main__":
    main()
