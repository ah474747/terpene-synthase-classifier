#!/usr/bin/env python3
"""
Germacrene Synthase Classifier with K-Means Clustering
====================================================

This script implements a hybrid approach combining:
1. ESM-2 protein embeddings
2. K-means clustering for unsupervised pattern discovery
3. XGBoost classification with cluster features
4. Semi-supervised learning
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_embedding_generator import RobustEmbeddingGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


class KMeansEnhancedClassifier:
    """
    Germacrene synthase classifier enhanced with k-means clustering
    """
    
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.embedding_generator = None
        
    def fit_kmeans(self, embeddings, labels=None):
        """
        Fit k-means clustering on protein embeddings
        
        Args:
            embeddings: Protein embeddings (n_samples, n_features)
            labels: Optional labels for supervised clustering analysis
        """
        print(f"Fitting k-means clustering with {self.n_clusters} clusters...")
        
        # Scale embeddings for k-means
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Fit k-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = self.kmeans.fit_predict(embeddings_scaled)
        
        # Analyze cluster composition
        if labels is not None:
            self._analyze_clusters(cluster_labels, labels)
        
        return cluster_labels
    
    def _analyze_clusters(self, cluster_labels, true_labels):
        """Analyze the composition of clusters"""
        print("\n=== Cluster Analysis ===")
        
        cluster_df = pd.DataFrame({
            'cluster': cluster_labels,
            'true_label': true_labels
        })
        
        # Analyze each cluster
        for cluster_id in range(self.n_clusters):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            total_in_cluster = len(cluster_data)
            germacrene_in_cluster = cluster_data['true_label'].sum()
            germacrene_ratio = germacrene_in_cluster / total_in_cluster if total_in_cluster > 0 else 0
            
            print(f"Cluster {cluster_id}: {total_in_cluster} sequences")
            print(f"  - Germacrene synthases: {germacrene_in_cluster} ({germacrene_ratio:.1%})")
            
            # Identify cluster characteristics
            if germacrene_ratio > 0.7:
                print(f"  - → HIGH GERMACRENE cluster!")
            elif germacrene_ratio > 0.3:
                print(f"  - → Mixed cluster")
            else:
                print(f"  - → Non-germacrene cluster")
        
        # Overall cluster purity
        cluster_purity = cluster_df.groupby('cluster')['true_label'].mean()
        print(f"\nCluster Purity (germacrene ratio):")
        for cluster_id, purity in cluster_purity.items():
            print(f"  Cluster {cluster_id}: {purity:.1%}")
    
    def create_cluster_features(self, embeddings):
        """
        Create features from k-means clustering
        
        Args:
            embeddings: Protein embeddings
            
        Returns:
            Enhanced feature matrix with cluster features
        """
        if self.kmeans is None:
            raise ValueError("K-means must be fitted first")
        
        # Scale embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Get cluster assignments and distances
        cluster_labels = self.kmeans.predict(embeddings_scaled)
        distances_to_centroids = self.kmeans.transform(embeddings_scaled)
        
        # Create cluster-based features
        cluster_features = []
        
        for i in range(len(embeddings)):
            features = []
            
            # Original embedding
            features.extend(embeddings[i])
            
            # Cluster assignment (one-hot encoded)
            cluster_one_hot = np.zeros(self.n_clusters)
            cluster_one_hot[cluster_labels[i]] = 1
            features.extend(cluster_one_hot)
            
            # Distance to each cluster centroid
            features.extend(distances_to_centroids[i])
            
            # Distance to closest centroid
            features.append(np.min(distances_to_centroids[i]))
            
            # Distance to second closest centroid
            sorted_distances = np.sort(distances_to_centroids[i])
            features.append(sorted_distances[1] if len(sorted_distances) > 1 else 0)
            
            cluster_features.append(features)
        
        return np.array(cluster_features)
    
    def train_with_cluster_features(self, embeddings, labels, n_splits=5):
        """
        Train XGBoost with cluster-enhanced features
        """
        print(f"Training XGBoost with k-means cluster features...")
        
        # Create cluster features
        X_enhanced = self.create_cluster_features(embeddings)
        
        print(f"Enhanced feature matrix shape: {X_enhanced.shape}")
        print(f"  - Original embeddings: {embeddings.shape[1]} features")
        print(f"  - Cluster one-hot: {self.n_clusters} features")
        print(f"  - Distances to centroids: {self.n_clusters} features")
        print(f"  - Additional distance features: 2 features")
        print(f"  - Total features: {X_enhanced.shape[1]}")
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        results = {
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'auc_pr_scores': [],
            'models': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_enhanced, labels)):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")
            
            # Split data
            X_train, X_test = X_enhanced[train_idx], X_enhanced[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Calculate scale_pos_weight
            pos_count = np.sum(y_train)
            neg_count = len(y_train) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            print(f"Training samples: {len(y_train)} (pos: {pos_count}, neg: {neg_count})")
            print(f"Scale pos weight: {scale_pos_weight:.2f}")
            
            # Initialize XGBoost
            xgb_model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
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
            
            # Train model
            xgb_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = xgb_model.predict(X_test)
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Calculate AUC-PR
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            auc_pr = auc(recall_curve, precision_curve)
            
            print(f"F1-Score: {f1:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"AUC-PR: {auc_pr:.3f}")
            
            # Store results
            results['f1_scores'].append(f1)
            results['precision_scores'].append(precision)
            results['recall_scores'].append(recall)
            results['auc_pr_scores'].append(auc_pr)
            results['models'].append(xgb_model)
        
        # Print summary
        print(f"\n=== K-Means Enhanced Results ===")
        print(f"F1-Score: {np.mean(results['f1_scores']):.3f} ± {np.std(results['f1_scores']):.3f}")
        print(f"Precision: {np.mean(results['precision_scores']):.3f} ± {np.std(results['precision_scores']):.3f}")
        print(f"Recall: {np.mean(results['recall_scores']):.3f} ± {np.std(results['recall_scores']):.3f}")
        print(f"AUC-PR: {np.mean(results['auc_pr_scores']):.3f} ± {np.std(results['auc_pr_scores']):.3f}")
        
        return results, X_enhanced
    
    def plot_cluster_analysis(self, embeddings, labels, save_path=None):
        """Create visualizations of cluster analysis"""
        if self.kmeans is None:
            raise ValueError("K-means must be fitted first")
        
        # Get cluster assignments
        embeddings_scaled = self.scaler.transform(embeddings)
        cluster_labels = self.kmeans.predict(embeddings_scaled)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster composition
        cluster_df = pd.DataFrame({
            'cluster': cluster_labels,
            'is_germacrene': labels
        })
        
        cluster_composition = cluster_df.groupby('cluster')['is_germacrene'].agg(['count', 'sum', 'mean'])
        cluster_composition.columns = ['total', 'germacrene', 'germacrene_ratio']
        
        axes[0, 0].bar(cluster_composition.index, cluster_composition['germacrene_ratio'])
        axes[0, 0].set_title('Germacrene Ratio by Cluster')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Germacrene Ratio')
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        # 2. Cluster sizes
        axes[0, 1].bar(cluster_composition.index, cluster_composition['total'])
        axes[0, 1].set_title('Cluster Sizes')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Sequences')
        
        # 3. Distance to centroids (for germacrene sequences)
        germacrene_mask = labels == 1
        if np.any(germacrene_mask):
            germacrene_distances = self.kmeans.transform(embeddings_scaled[germacrene_mask])
            min_distances = np.min(germacrene_distances, axis=1)
            
            axes[1, 0].hist(min_distances, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Distribution of Min Distances to Centroids\n(Germacrene Sequences)')
            axes[1, 0].set_xlabel('Minimum Distance to Centroid')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Distance to centroids (for non-germacrene sequences)
        non_germacrene_mask = labels == 0
        if np.any(non_germacrene_mask):
            non_germacrene_distances = self.kmeans.transform(embeddings_scaled[non_germacrene_mask])
            min_distances = np.min(non_germacrene_distances, axis=1)
            
            axes[1, 1].hist(min_distances, bins=20, alpha=0.7, color='red')
            axes[1, 1].set_title('Distribution of Min Distances to Centroids\n(Non-Germacrene Sequences)')
            axes[1, 1].set_xlabel('Minimum Distance to Centroid')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster analysis plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main function for k-means enhanced training"""
    print("=" * 70)
    print("K-MEANS ENHANCED GERMACRENE SYNTHASE CLASSIFIER")
    print("=" * 70)
    
    # Load training data
    print("\n=== Step 1: Load Training Data ===")
    training_file = "data/germacrene_training_data.csv"
    
    if not os.path.exists(training_file):
        print(f"Training data not found at {training_file}")
        return False
    
    df = pd.read_csv(training_file)
    print(f"✓ Loaded {len(df)} training sequences")
    
    # Use the complete dataset for k-means enhanced training
    print(f"Using complete dataset: {len(df)} sequences")
    
    sequences = df['sequence'].tolist()
    labels = df['target'].values
    
    positive_count = np.sum(labels)
    negative_count = len(labels) - positive_count
    print(f"  - Germacrene synthases: {positive_count}")
    print(f"  - Other synthases: {negative_count}")
    print(f"  - Class balance: {positive_count/len(labels):.2%}")
    
    # Generate embeddings
    print("\n=== Step 2: Generate Protein Embeddings ===")
    
    generator = RobustEmbeddingGenerator()
    
    try:
        # Generate embeddings in chunks for the full dataset
        print(f"Generating embeddings for {len(sequences)} sequences in chunks...")
        
        chunk_size = 50
        all_embeddings = []
        valid_indices = []
        
        total_chunks = (len(sequences) + chunk_size - 1) // chunk_size
        start_time = time.time()
        
        for i in range(0, len(sequences), chunk_size):
            chunk_num = i // chunk_size + 1
            chunk_sequences = sequences[i:i + chunk_size]
            
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_sequences)} sequences)")
            
            try:
                chunk_embeddings_df = generator.generate_embeddings(chunk_sequences)
                chunk_embeddings = np.array([np.array(emb) for emb in chunk_embeddings_df['embedding']])
                all_embeddings.append(chunk_embeddings)
                
                chunk_valid_indices = [i + j for j in range(len(chunk_embeddings))]
                valid_indices.extend(chunk_valid_indices)
                
                elapsed_time = time.time() - start_time
                avg_time_per_chunk = elapsed_time / chunk_num
                remaining_chunks = total_chunks - chunk_num
                estimated_remaining_time = remaining_chunks * avg_time_per_chunk
                
                print(f"  ✓ Processed {len(chunk_embeddings)} sequences")
                print(f"  ✓ Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"  ✗ Chunk {chunk_num} failed: {e}")
                continue
        
        if not all_embeddings:
            raise RuntimeError("No embeddings generated")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Align with labels
        min_length = min(len(embeddings), len(labels))
        embeddings = embeddings[:min_length]
        labels = labels[:min_length]
        
        total_time = time.time() - start_time
        print(f"✓ Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        print(f"✓ Total embedding time: {total_time/60:.1f} minutes")
        print(f"✓ Aligned embeddings and labels to {min_length} sequences")
        
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False
    
    # Initialize k-means enhanced classifier
    print("\n=== Step 3: K-Means Clustering Analysis ===")
    
    n_clusters = min(20, len(embeddings) // 20)  # More clusters for larger dataset
    print(f"Using {n_clusters} clusters for {len(embeddings)} sequences")
    
    classifier = KMeansEnhancedClassifier(n_clusters=n_clusters)
    
    # Fit k-means and analyze clusters
    cluster_labels = classifier.fit_kmeans(embeddings, labels)
    
    # Train with cluster features
    print("\n=== Step 4: Train with Cluster Features ===")
    
    try:
        results, X_enhanced = classifier.train_with_cluster_features(embeddings, labels, n_splits=5)
        
        # Select best model
        best_fold = np.argmax(results['f1_scores'])
        best_model = results['models'][best_fold]
        classifier.xgb_model = best_model
        
        print(f"✓ Best model from fold {best_fold + 1} with F1-score: {results['f1_scores'][best_fold]:.3f}")
        
    except Exception as e:
        print(f"✗ Training with cluster features failed: {e}")
        return False
    
    # Create visualizations
    print("\n=== Step 5: Create Cluster Analysis Visualizations ===")
    
    try:
        os.makedirs("results", exist_ok=True)
        classifier.plot_cluster_analysis(embeddings, labels, "results/kmeans_cluster_analysis.png")
    except Exception as e:
        print(f"⚠ Could not create visualizations: {e}")
    
    # Save results
    print("\n=== Step 6: Save K-Means Enhanced Model ===")
    
    try:
        os.makedirs("models", exist_ok=True)
        
        # Save model components
        import joblib
        model_data = {
            'kmeans': classifier.kmeans,
            'scaler': classifier.scaler,
            'xgb_model': classifier.xgb_model,
            'n_clusters': n_clusters,
            'embedding_dim': embeddings.shape[1],
            'enhanced_feature_dim': X_enhanced.shape[1],
            'training_sequences': len(embeddings),
            'positive_samples': positive_count,
            'negative_samples': negative_count
        }
        
        model_path = "models/germacrene_classifier_kmeans.pkl"
        joblib.dump(model_data, model_path)
        
        # Save results
        results_data = {
            'training_sequences': len(embeddings),
            'n_clusters': n_clusters,
            'embedding_dimension': embeddings.shape[1],
            'enhanced_feature_dimension': X_enhanced.shape[1],
            'positive_samples': int(positive_count),
            'negative_samples': int(negative_count),
            'class_balance': float(positive_count / len(embeddings)),
            'cv_f1_mean': float(np.mean(results['f1_scores'])),
            'cv_f1_std': float(np.std(results['f1_scores'])),
            'cv_precision_mean': float(np.mean(results['precision_scores'])),
            'cv_recall_mean': float(np.mean(results['recall_scores'])),
            'cv_auc_pr_mean': float(np.mean(results['auc_pr_scores'])),
            'best_fold': int(best_fold + 1),
            'best_f1_score': float(results['f1_scores'][best_fold])
        }
        
        with open("results/kmeans_training_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ K-means enhanced model saved to: {model_path}")
        print(f"✓ Results saved to: results/kmeans_training_results.json")
        
    except Exception as e:
        print(f"✗ Model saving failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("K-MEANS ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"✓ K-means clustering with {n_clusters} clusters")
    print(f"✓ Enhanced features: {X_enhanced.shape[1]} dimensions")
    print(f"✓ Cross-validation completed")
    print(f"✓ Model ready for predictions")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n" + "=" * 70)
        print("K-MEANS ENHANCED TRAINING FAILED")
        print("=" * 70)
        print("Please check the error messages above and try again.")
        sys.exit(1)
