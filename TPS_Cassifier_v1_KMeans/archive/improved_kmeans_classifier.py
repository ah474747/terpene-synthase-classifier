#!/usr/bin/env python3
"""
Improved K-Means Germacrene Classifier
=====================================

This script implements an improved k-means enhanced classifier that addresses:
1. Severe class imbalance (6% positive samples)
2. Proper feature scaling and normalization
3. Advanced class imbalance handling (SMOTE, class weights)
4. Better clustering strategy
5. Comprehensive evaluation metrics

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
import pickle
import warnings

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class ImprovedKMeansClassifier:
    """
    Improved k-means enhanced classifier with proper class imbalance handling
    """
    
    def __init__(self, n_clusters=10, use_smote=True, use_class_weights=True):
        self.n_clusters = n_clusters
        self.use_smote = use_smote
        self.use_class_weights = use_class_weights
        self.kmeans = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.xgb_model = None
        self.cluster_centers = None
        self.feature_names = None
        
    def load_training_data(self):
        """Load and prepare training data"""
        print("=== Loading Training Data ===")
        
        # Load the enhanced dataset
        df = pd.read_csv("data/marts_db_enhanced.csv")
        df = df.dropna(subset=['Aminoacid_sequence', 'is_germacrene_family'])
        df = df[df['Aminoacid_sequence'].str.len() > 10]
        
        sequences = df['Aminoacid_sequence'].tolist()
        labels = df['is_germacrene_family'].values
        
        print(f"✓ Loaded {len(sequences)} sequences")
        print(f"  - Positive samples (Germacrene): {np.sum(labels)}")
        print(f"  - Negative samples (Other): {len(labels) - np.sum(labels)}")
        print(f"  - Class balance: {np.mean(labels):.3f}")
        
        return sequences, labels
    
    def generate_improved_embeddings(self, sequences):
        """Generate improved embeddings with better feature engineering"""
        print(f"\n=== Generating Improved Embeddings ===")
        
        # For now, use synthetic embeddings but with better features
        # In production, you'd use real ESM embeddings
        embeddings = []
        
        for seq in sequences:
            # Basic sequence features
            seq_len = len(seq)
            aa_counts = {}
            for aa in 'ACDEFGHIKLMNPQRSTVWY':
                aa_counts[aa] = seq.count(aa) / seq_len
            
            # Create feature vector
            features = [
                seq_len,
                aa_counts['A'], aa_counts['C'], aa_counts['D'], aa_counts['E'],
                aa_counts['F'], aa_counts['G'], aa_counts['H'], aa_counts['I'],
                aa_counts['K'], aa_counts['L'], aa_counts['M'], aa_counts['N'],
                aa_counts['P'], aa_counts['Q'], aa_counts['R'], aa_counts['S'],
                aa_counts['T'], aa_counts['V'], aa_counts['W'], aa_counts['Y']
            ]
            
            # Add some sequence complexity features
            features.extend([
                len(set(seq)) / seq_len,  # Amino acid diversity
                seq.count('G') / seq_len,  # Glycine content (flexibility)
                seq.count('P') / seq_len,  # Proline content (rigidity)
                (seq.count('C') + seq.count('M')) / seq_len,  # Cysteine + Methionine
                seq.count('W') / seq_len,  # Tryptophan content
                seq.count('Y') / seq_len,  # Tyrosine content
            ])
            
            # Pad to consistent dimension
            while len(features) < 50:
                features.append(0.0)
            
            embeddings.append(features[:50])
        
        embeddings = np.array(embeddings)
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def fit_kmeans(self, embeddings, labels):
        """Fit k-means clustering with improved strategy"""
        print(f"\n=== K-Means Clustering ===")
        print(f"Fitting k-means with {self.n_clusters} clusters...")
        
        try:
            # Scale embeddings for k-means
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            # Use improved k-means settings
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=20,  # More initializations for stability
                max_iter=500,  # More iterations
                algorithm='lloyd',  # Most stable algorithm
                init='k-means++'  # Better initialization
            )
            
            cluster_labels = self.kmeans.fit_predict(embeddings_scaled)
            self.cluster_centers = self.kmeans.cluster_centers_
            
            print(f"✓ K-means clustering completed")
            
            # Analyze clusters
            self._analyze_clusters(cluster_labels, labels)
            
            return cluster_labels
            
        except Exception as e:
            print(f"✗ K-means clustering failed: {e}")
            return np.zeros(len(embeddings), dtype=int)
    
    def _analyze_clusters(self, cluster_labels, true_labels):
        """Analyze cluster purity and composition"""
        print(f"\n=== Cluster Analysis ===")
        
        cluster_stats = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 0:
                germacrene_count = np.sum(true_labels[cluster_mask])
                germacrene_ratio = germacrene_count / cluster_size
                
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'germacrene_count': germacrene_count,
                    'germacrene_ratio': germacrene_ratio
                })
                
                print(f"Cluster {cluster_id}: {cluster_size} sequences, "
                      f"{germacrene_count} Germacrene ({germacrene_ratio:.3f})")
        
        # Find best clusters (highest germacrene ratio)
        cluster_stats.sort(key=lambda x: x['germacrene_ratio'], reverse=True)
        print(f"\nTop clusters by germacrene ratio:")
        for i, stats in enumerate(cluster_stats[:3]):
            print(f"  {i+1}. Cluster {stats['cluster_id']}: {stats['germacrene_ratio']:.3f}")
        
        return cluster_stats
    
    def _create_enhanced_features(self, embeddings, cluster_labels):
        """Create enhanced features from embeddings and cluster information"""
        print(f"\n=== Creating Enhanced Features ===")
        
        # Start with scaled original embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        enhanced_features = embeddings_scaled.copy()
        
        # Add cluster assignment as one-hot encoding
        cluster_one_hot = np.zeros((len(embeddings), self.n_clusters))
        cluster_one_hot[np.arange(len(embeddings)), cluster_labels] = 1
        enhanced_features = np.hstack([enhanced_features, cluster_one_hot])
        
        # Add distance to cluster centers
        distances_to_centers = np.zeros((len(embeddings), self.n_clusters))
        for i, embedding in enumerate(embeddings_scaled):
            for j, center in enumerate(self.cluster_centers):
                distances_to_centers[i, j] = np.linalg.norm(embedding - center)
        
        enhanced_features = np.hstack([enhanced_features, distances_to_centers])
        
        # Add cluster-based features
        cluster_features = np.zeros((len(embeddings), 5))
        for i, cluster_id in enumerate(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Distance to cluster center (normalized)
            cluster_features[i, 0] = distances_to_centers[i, cluster_id]
            
            # Cluster size (normalized)
            cluster_features[i, 1] = cluster_size / len(embeddings)
            
            # Average distance to other points in cluster
            if cluster_size > 1:
                other_points = embeddings_scaled[cluster_mask]
                distances = [np.linalg.norm(embeddings_scaled[i] - other_point) 
                           for other_point in other_points if not np.array_equal(embeddings_scaled[i], other_point)]
                cluster_features[i, 2] = np.mean(distances) if distances else 0
            else:
                cluster_features[i, 2] = 0
            
            # Distance to nearest cluster center
            other_centers = [self.cluster_centers[j] for j in range(self.n_clusters) if j != cluster_id]
            if other_centers:
                min_dist = min([np.linalg.norm(embeddings_scaled[i] - center) for center in other_centers])
                cluster_features[i, 3] = min_dist
            
            # Cluster density (inverse of average intra-cluster distance)
            if cluster_size > 1:
                intra_distances = []
                cluster_points = embeddings_scaled[cluster_mask]
                for j, point1 in enumerate(cluster_points):
                    for k, point2 in enumerate(cluster_points):
                        if j != k:
                            intra_distances.append(np.linalg.norm(point1 - point2))
                cluster_features[i, 4] = 1.0 / (np.mean(intra_distances) + 1e-8)
            else:
                cluster_features[i, 4] = 1.0
        
        enhanced_features = np.hstack([enhanced_features, cluster_features])
        
        print(f"✓ Enhanced features shape: {enhanced_features.shape}")
        
        # Create feature names for interpretability
        self.feature_names = (
            [f"seq_feature_{i}" for i in range(embeddings.shape[1])] +
            [f"cluster_{i}" for i in range(self.n_clusters)] +
            [f"dist_to_cluster_{i}" for i in range(self.n_clusters)] +
            ["dist_to_center", "cluster_size", "avg_intra_dist", "min_inter_dist", "cluster_density"]
        )
        
        return enhanced_features
    
    def train_classifier(self, X, y):
        """Train the classifier with proper class imbalance handling"""
        print(f"\n=== Training Classifier ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Training class balance: {np.mean(y_train):.3f}")
        
        # Handle class imbalance
        if self.use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train) - 1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(X_train_balanced)} samples, class balance: {np.mean(y_train_balanced):.3f}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Calculate class weights
        if self.use_class_weights:
            class_counts = np.bincount(y_train_balanced)
            class_weights = len(y_train_balanced) / (len(class_counts) * class_counts)
            class_weight_dict = {i: class_weights[i] for i in range(len(class_counts))}
            print(f"Class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
        
        # Train XGBoost with improved parameters
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=class_weight_dict[1] if class_weight_dict else 1,
            eval_metric='logloss'
        )
        
        # Train the model
        self.xgb_model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate on test set
        y_pred = self.xgb_model.predict(X_test)
        y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        print(f"\n=== Test Set Performance ===")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        
        # Feature importance
        if hasattr(self.xgb_model, 'feature_importances_'):
            feature_importance = self.xgb_model.feature_importances_
            top_features = np.argsort(feature_importance)[-10:][::-1]
            print(f"\nTop 10 Most Important Features:")
            for i, idx in enumerate(top_features):
                if idx < len(self.feature_names):
                    print(f"  {i+1}. {self.feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_pr': auc_pr,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def cross_validate(self, X, y, cv_folds=5):
        """Perform stratified k-fold cross-validation"""
        print(f"\n=== Cross-Validation ({cv_folds} folds) ===")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'auc_pr_scores': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Handle class imbalance for this fold
            if self.use_smote:
                smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train) - 1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            else:
                X_train_balanced, y_train_balanced = X_train, y_train
            
            # Calculate class weights for this fold
            if self.use_class_weights:
                class_counts = np.bincount(y_train_balanced)
                class_weights = len(y_train_balanced) / (len(class_counts) * class_counts)
                class_weight_dict = {i: class_weights[i] for i in range(len(class_counts))}
                scale_pos_weight = class_weight_dict[1] if class_weight_dict else 1
            else:
                scale_pos_weight = 1
            
            # Train model for this fold
            fold_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            
            fold_model.fit(X_train_balanced, y_train_balanced, verbose=False)
            
            # Predict on validation set
            y_pred = fold_model.predict(X_val)
            y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            auc_pr = average_precision_score(y_val, y_pred_proba)
            
            cv_scores['f1_scores'].append(f1)
            cv_scores['precision_scores'].append(precision)
            cv_scores['recall_scores'].append(recall)
            cv_scores['auc_pr_scores'].append(auc_pr)
            
            print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC-PR: {auc_pr:.4f}")
        
        # Calculate mean and std
        mean_scores = {
            'f1_mean': np.mean(cv_scores['f1_scores']),
            'f1_std': np.std(cv_scores['f1_scores']),
            'precision_mean': np.mean(cv_scores['precision_scores']),
            'precision_std': np.std(cv_scores['precision_scores']),
            'recall_mean': np.mean(cv_scores['recall_scores']),
            'recall_std': np.std(cv_scores['recall_scores']),
            'auc_pr_mean': np.mean(cv_scores['auc_pr_scores']),
            'auc_pr_std': np.std(cv_scores['auc_pr_scores'])
        }
        
        print(f"\n=== Cross-Validation Results ===")
        print(f"F1-Score: {mean_scores['f1_mean']:.4f} ± {mean_scores['f1_std']:.4f}")
        print(f"Precision: {mean_scores['precision_mean']:.4f} ± {mean_scores['precision_std']:.4f}")
        print(f"Recall: {mean_scores['recall_mean']:.4f} ± {mean_scores['recall_std']:.4f}")
        print(f"AUC-PR: {mean_scores['auc_pr_mean']:.4f} ± {mean_scores['auc_pr_std']:.4f}")
        
        return mean_scores, cv_scores
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'xgb_model': self.xgb_model,
            'cluster_centers': self.cluster_centers,
            'feature_names': self.feature_names,
            'n_clusters': self.n_clusters
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.xgb_model = model_data['xgb_model']
        self.cluster_centers = model_data['cluster_centers']
        self.feature_names = model_data['feature_names']
        self.n_clusters = model_data['n_clusters']
        
        print(f"✓ Model loaded from {filepath}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("IMPROVED K-MEANS GERMACRENE CLASSIFIER")
    print("=" * 60)
    
    # Initialize classifier
    classifier = ImprovedKMeansClassifier(
        n_clusters=12,
        use_smote=True,
        use_class_weights=True
    )
    
    # Load data
    sequences, labels = classifier.load_training_data()
    
    # Generate embeddings
    embeddings = classifier.generate_improved_embeddings(sequences)
    
    # Fit k-means
    cluster_labels = classifier.fit_kmeans(embeddings, labels)
    
    # Create enhanced features
    enhanced_features = classifier._create_enhanced_features(embeddings, cluster_labels)
    
    # Cross-validation
    cv_results, cv_scores = classifier.cross_validate(enhanced_features, labels, cv_folds=5)
    
    # Train final model
    final_results = classifier.train_classifier(enhanced_features, labels)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/improved_kmeans_classifier.pkl')
    
    # Save results
    results = {
        'training_sequences': len(sequences),
        'embedding_dimension': embeddings.shape[1],
        'enhanced_feature_dimension': enhanced_features.shape[1],
        'positive_samples': int(np.sum(labels)),
        'negative_samples': int(len(labels) - np.sum(labels)),
        'class_balance': float(np.mean(labels)),
        'n_clusters': classifier.n_clusters,
        'use_smote': classifier.use_smote,
        'use_class_weights': classifier.use_class_weights,
        **cv_results,
        'final_test_f1': final_results['f1_score'],
        'final_test_precision': final_results['precision'],
        'final_test_recall': final_results['recall'],
        'final_test_auc_pr': final_results['auc_pr'],
        'training_completed': True,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/improved_kmeans_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Results saved to results/improved_kmeans_results.json")
    print(f"✓ Model saved to models/improved_kmeans_classifier.pkl")
    
    return results


if __name__ == "__main__":
    results = main()
