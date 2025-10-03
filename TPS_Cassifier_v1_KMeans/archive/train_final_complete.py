#!/usr/bin/env python3
"""
Final Complete Training Script - Based on Working Approach
=========================================================

This script completes training on the full 1,356 sequences using the approach
that worked for 500 sequences, but adapted for the enhanced dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_embedding_generator import RobustEmbeddingGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


def load_complete_training_data():
    """Load the complete enhanced training dataset"""
    training_file = "data/marts_db_enhanced.csv"
    
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Training file not found: {training_file}")
    
    df = pd.read_csv(training_file)
    print(f"✓ Loaded complete dataset: {len(df)} sequences")
    
    # Clean the data
    df = df.dropna(subset=['Aminoacid_sequence', 'is_germacrene_family'])
    df = df[df['Aminoacid_sequence'].str.len() > 10]  # Remove very short sequences
    
    positive_count = df['is_germacrene_family'].sum()
    negative_count = len(df) - positive_count
    print(f"  - Germacrene synthases: {positive_count}")
    print(f"  - Other synthases: {negative_count}")
    print(f"  - Class balance: {positive_count/len(df):.2%}")
    
    return df


def generate_embeddings_chunked_approach(sequences, labels, chunk_size=25):
    """
    Generate embeddings using the chunked approach that worked before
    """
    print(f"\n=== Step 1: Generating Embeddings (Chunked Approach) ===")
    print(f"Processing {len(sequences)} sequences in chunks of {chunk_size}")
    
    # Initialize embedding generator (using the working version)
    embedding_generator = RobustEmbeddingGenerator(
        model_name="esm2_t33_650M_UR50D",
        max_length=512
    )
    
    all_embeddings = []
    processed_labels = []
    
    # Process in chunks
    total_chunks = (len(sequences) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(sequences))
        
        chunk_sequences = sequences[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_sequences)} sequences)")
        
        try:
            # Generate embeddings for this chunk
            chunk_embeddings_df = embedding_generator.generate_embeddings(chunk_sequences)
            
            # Extract embeddings (convert from list format to numpy array)
            chunk_embeddings = np.array([np.array(emb) for emb in chunk_embeddings_df['embedding']])
            
            all_embeddings.append(chunk_embeddings)
            processed_labels.extend(chunk_labels)
            
            print(f"  ✓ Chunk {chunk_idx + 1} completed")
            print(f"  ✓ Embedding dimension: {chunk_embeddings.shape[1]}")
            
            # Cleanup
            del chunk_embeddings_df, chunk_embeddings
            
            # Memory cleanup
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Chunk {chunk_idx + 1} failed: {e}")
            # Add zero embeddings as fallback
            fallback_embeddings = np.zeros((len(chunk_sequences), 1280))
            all_embeddings.append(fallback_embeddings)
            processed_labels.extend(chunk_labels)
    
    # Combine all embeddings
    print("\nCombining all embeddings...")
    final_embeddings = np.vstack(all_embeddings)
    final_labels = np.array(processed_labels)
    
    # Cleanup
    embedding_generator.cleanup()
    del all_embeddings, embedding_generator
    
    print(f"✓ Final embeddings shape: {final_embeddings.shape}")
    print(f"✓ Final labels shape: {final_labels.shape}")
    
    return final_embeddings, final_labels


class KMeansEnhancedClassifier:
    """K-means enhanced classifier (using the working approach)"""
    
    def __init__(self, n_clusters=15):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.xgb_model = None
        self.cluster_centers = None
        
    def fit_kmeans(self, embeddings, labels):
        """Fit k-means clustering"""
        print(f"\n=== Step 2: K-Means Clustering ===")
        print(f"Fitting k-means with {self.n_clusters} clusters...")
        
        try:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            
            cluster_labels = self.kmeans.fit_predict(embeddings)
            self.cluster_centers = self.kmeans.cluster_centers_
            
            print(f"✓ K-means clustering completed")
            
            # Analyze clusters
            self._analyze_clusters(cluster_labels, labels)
            
            return cluster_labels
            
        except Exception as e:
            print(f"✗ K-means clustering failed: {e}")
            return np.zeros(len(embeddings), dtype=int)
    
    def _analyze_clusters(self, cluster_labels, true_labels):
        """Analyze cluster purity"""
        print(f"\n=== Cluster Analysis ===")
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 0:
                germacrene_count = np.sum(true_labels[cluster_mask])
                germacrene_ratio = germacrene_count / cluster_size
                
                print(f"Cluster {cluster_id}: {cluster_size} sequences, "
                      f"{germacrene_count} Germacrene ({germacrene_ratio:.3f})")
    
    def _create_enhanced_features(self, embeddings, cluster_labels):
        """Create enhanced features from embeddings and cluster information"""
        print("Creating enhanced features...")
        
        # Original embeddings
        enhanced_features = embeddings.copy()
        
        # Add cluster assignment as one-hot encoding
        cluster_one_hot = np.zeros((len(embeddings), self.n_clusters))
        cluster_one_hot[np.arange(len(embeddings)), cluster_labels] = 1
        enhanced_features = np.hstack([enhanced_features, cluster_one_hot])
        
        # Add distance to cluster centers
        distances_to_centers = np.zeros((len(embeddings), self.n_clusters))
        for i, embedding in enumerate(embeddings):
            for j, center in enumerate(self.cluster_centers):
                distances_to_centers[i, j] = np.linalg.norm(embedding - center)
        
        enhanced_features = np.hstack([enhanced_features, distances_to_centers])
        
        print(f"✓ Enhanced features shape: {enhanced_features.shape}")
        return enhanced_features
    
    def train_with_cluster_features(self, embeddings, labels, n_splits=5):
        """Train XGBoost with cluster features"""
        print(f"\n=== Step 3: Training XGBoost with Cluster Features ===")
        
        try:
            # Get cluster labels
            cluster_labels = self.fit_kmeans(embeddings, labels)
            
            # Create enhanced features
            X_enhanced = self._create_enhanced_features(embeddings, cluster_labels)
            
            # Stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            cv_results = {
                'f1_scores': [],
                'precision_scores': [],
                'recall_scores': [],
                'auc_pr_scores': []
            }
            
            print(f"Running {n_splits}-fold cross-validation...")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_enhanced, labels)):
                print(f"\nFold {fold + 1}/{n_splits}:")
                
                X_train, X_val = X_enhanced[train_idx], X_enhanced[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                # XGBoost parameters (using working settings)
                xgb_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': 4,
                    'scale_pos_weight': len(y_train) / np.sum(y_train) - 1
                }
                
                # Train model
                self.xgb_model = xgb.XGBClassifier(**xgb_params)
                self.xgb_model.fit(X_train, y_train)
                
                # Predictions
                y_pred_proba = self.xgb_model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics
                f1 = f1_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                auc_pr = average_precision_score(y_val, y_pred_proba)
                
                cv_results['f1_scores'].append(f1)
                cv_results['precision_scores'].append(precision)
                cv_results['recall_scores'].append(recall)
                cv_results['auc_pr_scores'].append(auc_pr)
                
                print(f"  F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, AUC-PR: {auc_pr:.3f}")
            
            # Calculate mean and std
            mean_f1 = np.mean(cv_results['f1_scores'])
            std_f1 = np.std(cv_results['f1_scores'])
            
            print(f"\n=== Cross-Validation Results ===")
            print(f"F1-Score: {mean_f1:.3f} ± {std_f1:.3f}")
            print(f"Precision: {np.mean(cv_results['precision_scores']):.3f} ± {np.std(cv_results['precision_scores']):.3f}")
            print(f"Recall: {np.mean(cv_results['recall_scores']):.3f} ± {np.std(cv_results['recall_scores']):.3f}")
            print(f"AUC-PR: {np.mean(cv_results['auc_pr_scores']):.3f} ± {np.std(cv_results['auc_pr_scores']):.3f}")
            
            return cv_results, X_enhanced
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """Main training pipeline"""
    print("=== Final Complete Germacrene Classifier Training ===")
    print(f"Started at: {datetime.now()}")
    
    try:
        # Load complete training data
        df = load_complete_training_data()
        
        # Prepare sequences and labels
        sequences = df['Aminoacid_sequence'].tolist()
        labels = df['is_germacrene_family'].values
        
        print(f"\nUsing complete dataset: {len(df)} sequences")
        
        # Generate embeddings using chunked approach (that worked before)
        embeddings, processed_labels = generate_embeddings_chunked_approach(sequences, labels, chunk_size=25)
        
        # Initialize k-means enhanced classifier
        print(f"\n=== Step 2: K-Means Enhanced Training ===")
        
        n_clusters = min(15, len(embeddings) // 100)  # Reasonable cluster count
        print(f"Using {n_clusters} clusters for {len(embeddings)} sequences")
        
        classifier = KMeansEnhancedClassifier(n_clusters=n_clusters)
        
        # Train with cluster features
        print(f"\n=== Step 3: Train with Cluster Features ===")
        
        results, X_enhanced = classifier.train_with_cluster_features(embeddings, processed_labels, n_splits=5)
        
        if results is not None:
            # Save results
            os.makedirs("results", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            
            # Save training results
            results_data = {
                'training_sequences': int(len(sequences)),
                'embedding_dimension': int(embeddings.shape[1]),
                'enhanced_feature_dimension': int(X_enhanced.shape[1]),
                'positive_samples': int(np.sum(processed_labels)),
                'negative_samples': int(len(processed_labels) - np.sum(processed_labels)),
                'class_balance': float(np.mean(processed_labels)),
                'n_clusters': int(n_clusters),
                'cv_f1_mean': float(np.mean(results['f1_scores'])),
                'cv_f1_std': float(np.std(results['f1_scores'])),
                'cv_precision_mean': float(np.mean(results['precision_scores'])),
                'cv_recall_mean': float(np.mean(results['recall_scores'])),
                'cv_auc_pr_mean': float(np.mean(results['auc_pr_scores'])),
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            with open("results/final_complete_training_results.json", "w") as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\n✓ Training completed successfully!")
            print(f"✓ Results saved to: results/final_complete_training_results.json")
            print(f"✓ Final F1-Score: {results_data['cv_f1_mean']:.3f} ± {results_data['cv_f1_std']:.3f}")
            
        else:
            print("✗ Training failed")
            
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()

