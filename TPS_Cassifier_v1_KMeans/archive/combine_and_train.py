#!/usr/bin/env python3
"""
Combine the saved ESM embedding chunks and train the final model
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import glob
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')


def combine_embedding_chunks():
    """Combine all the saved embedding chunks"""
    print("=== Combining ESM Embedding Chunks ===")
    
    # Find all chunk files
    chunk_files = sorted(glob.glob("temp_embeddings_chunk_*.npy"))
    print(f"Found {len(chunk_files)} embedding chunks")
    
    if len(chunk_files) == 0:
        raise FileNotFoundError("No embedding chunks found!")
    
    # Load and combine chunks
    all_embeddings = []
    for i, chunk_file in enumerate(chunk_files):
        print(f"Loading chunk {i+1}/{len(chunk_files)}: {chunk_file}")
        chunk_embeddings = np.load(chunk_file)
        all_embeddings.append(chunk_embeddings)
        print(f"  Shape: {chunk_embeddings.shape}")
    
    # Combine all embeddings
    print("\nCombining all embeddings...")
    final_embeddings = np.vstack(all_embeddings)
    
    print(f"✓ Final embeddings shape: {final_embeddings.shape}")
    
    # Load labels
    df = pd.read_csv("data/marts_db_enhanced.csv")
    df = df.dropna(subset=['Aminoacid_sequence', 'is_germacrene_family'])
    df = df[df['Aminoacid_sequence'].str.len() > 10]
    
    # Make sure we have the right number of labels
    labels = df['is_germacrene_family'].values
    if len(labels) != len(final_embeddings):
        print(f"Warning: Labels ({len(labels)}) don't match embeddings ({len(final_embeddings)})")
        # Truncate to match
        min_len = min(len(labels), len(final_embeddings))
        labels = labels[:min_len]
        final_embeddings = final_embeddings[:min_len]
        print(f"Truncated to {min_len} sequences")
    
    print(f"✓ Labels shape: {labels.shape}")
    print(f"  - Positive samples (Germacrene): {np.sum(labels)}")
    print(f"  - Negative samples (Other): {len(labels) - np.sum(labels)}")
    print(f"  - Class balance: {np.mean(labels):.3f}")
    
    # Save combined embeddings
    np.save("combined_esm_embeddings.npy", final_embeddings)
    np.save("combined_labels.npy", labels)
    
    print("✓ Combined embeddings saved to combined_esm_embeddings.npy")
    
    return final_embeddings, labels


def train_kmeans_enhanced_classifier(embeddings, labels):
    """Train the k-means enhanced classifier"""
    print("\n=== Training K-Means Enhanced Classifier ===")
    
    # Determine number of clusters
    n_clusters = min(15, len(embeddings) // 100)
    print(f"Using {n_clusters} clusters for {len(embeddings)} sequences")
    
    # K-means clustering
    print("Performing k-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    print("✓ K-means clustering completed")
    
    # Analyze clusters
    print("\n=== Cluster Analysis ===")
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size > 0:
            germacrene_count = np.sum(labels[cluster_mask])
            germacrene_ratio = germacrene_count / cluster_size
            
            print(f"Cluster {cluster_id}: {cluster_size} sequences, "
                  f"{germacrene_count} Germacrene ({germacrene_ratio:.3f})")
    
    # Create enhanced features
    print("\nCreating enhanced features...")
    
    # Original embeddings
    enhanced_features = embeddings.copy()
    
    # Add cluster assignment as one-hot encoding
    cluster_one_hot = np.zeros((len(embeddings), n_clusters))
    cluster_one_hot[np.arange(len(embeddings)), cluster_labels] = 1
    enhanced_features = np.hstack([enhanced_features, cluster_one_hot])
    
    # Add distance to cluster centers
    distances_to_centers = np.zeros((len(embeddings), n_clusters))
    for i, embedding in enumerate(embeddings):
        for j, center in enumerate(kmeans.cluster_centers_):
            distances_to_centers[i, j] = np.linalg.norm(embedding - center)
    
    enhanced_features = np.hstack([enhanced_features, distances_to_centers])
    
    print(f"✓ Enhanced features shape: {enhanced_features.shape}")
    
    # Cross-validation
    print(f"\n=== Cross-Validation ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'auc_pr_scores': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(enhanced_features, labels)):
        print(f"\nFold {fold + 1}/5:")
        
        X_train, X_val = enhanced_features[train_idx], enhanced_features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # XGBoost parameters
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
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
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
    
    # Calculate final metrics
    mean_f1 = np.mean(cv_results['f1_scores'])
    std_f1 = np.std(cv_results['f1_scores'])
    
    print(f"\n=== Final Results (Real ESM Embeddings) ===")
    print(f"F1-Score: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Precision: {np.mean(cv_results['precision_scores']):.3f} ± {np.std(cv_results['precision_scores']):.3f}")
    print(f"Recall: {np.mean(cv_results['recall_scores']):.3f} ± {np.std(cv_results['recall_scores']):.3f}")
    print(f"AUC-PR: {np.mean(cv_results['auc_pr_scores']):.3f} ± {np.std(cv_results['auc_pr_scores']):.3f}")
    
    # Train final model on all data
    print(f"\n=== Training Final Model ===")
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(enhanced_features, labels)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Save training results
    results_data = {
        'training_sequences': int(len(embeddings)),
        'embedding_dimension': int(embeddings.shape[1]),
        'enhanced_feature_dimension': int(enhanced_features.shape[1]),
        'positive_samples': int(np.sum(labels)),
        'negative_samples': int(len(labels) - np.sum(labels)),
        'class_balance': float(np.mean(labels)),
        'n_clusters': int(n_clusters),
        'embedding_type': 'real_esm2_embeddings',
        'cv_f1_mean': float(mean_f1),
        'cv_f1_std': float(std_f1),
        'cv_precision_mean': float(np.mean(cv_results['precision_scores'])),
        'cv_recall_mean': float(np.mean(cv_results['recall_scores'])),
        'cv_auc_pr_mean': float(np.mean(cv_results['auc_pr_scores'])),
        'training_completed': True,
        'timestamp': datetime.now().isoformat()
    }
    
    with open("results/final_real_esm_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save the trained model
    model_data = {
        'model': final_model,
        'kmeans': kmeans,
        'cluster_centers': kmeans.cluster_centers_,
        'n_clusters': n_clusters,
        'embedding_dim': embeddings.shape[1],
        'training_sequences': len(embeddings),
        'positive_samples': int(np.sum(labels)),
        'negative_samples': int(len(labels) - np.sum(labels)),
        'model_name': 'real_esm_embedding_classifier',
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, "models/germacrene_classifier_final_real_esm.pkl")
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Results saved to: results/final_real_esm_results.json")
    print(f"✓ Model saved to: models/germacrene_classifier_final_real_esm.pkl")
    print(f"✓ Final F1-Score: {mean_f1:.3f} ± {std_f1:.3f}")
    
    return results_data


def cleanup_temp_files():
    """Clean up temporary chunk files"""
    print("\n=== Cleaning Up ===")
    chunk_files = glob.glob("temp_embeddings_chunk_*.npy")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    print(f"✓ Removed {len(chunk_files)} temporary files")


def main():
    """Main pipeline"""
    print("=== Final Training with Real ESM Embeddings ===")
    print(f"Started at: {datetime.now()}")
    
    try:
        # Combine embedding chunks
        embeddings, labels = combine_embedding_chunks()
        
        # Train the classifier
        results = train_kmeans_enhanced_classifier(embeddings, labels)
        
        # Cleanup
        cleanup_temp_files()
        
        print(f"\n🎉 SUCCESS! Training completed on {results['training_sequences']} sequences")
        print(f"Final F1-Score: {results['cv_f1_mean']:.3f} ± {results['cv_f1_std']:.3f}")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()

