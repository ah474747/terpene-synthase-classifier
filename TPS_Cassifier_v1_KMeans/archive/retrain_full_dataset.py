#!/usr/bin/env python3
"""
Retrain Full Dataset Model with Proper Approach
==============================================

This script retrains the germacrene classifier on the complete MARTS-DB dataset
using the same successful approach as the 500-sequence model, but with all data.

Key improvements:
1. Use real ESM embeddings (not synthetic)
2. Proper class imbalance handling
3. Same feature engineering as successful model
4. Comprehensive evaluation

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
import joblib

warnings.filterwarnings('ignore')


def load_complete_training_data():
    """Load the complete MARTS-DB dataset"""
    print("=== Loading Complete Training Data ===")
    
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


def generate_real_embeddings(sequences, chunk_size=25):
    """Generate real ESM embeddings for all sequences"""
    print(f"\n=== Generating Real ESM Embeddings ===")
    print(f"Processing {len(sequences)} sequences in chunks of {chunk_size}...")
    
    # Initialize embedding generator
    embedding_generator = RobustEmbeddingGenerator(
        model_name="esm2_t33_650M_UR50D",
        max_length=512
    )
    
    all_embeddings = []
    failed_sequences = []
    
    # Process in chunks
    num_chunks = (len(sequences) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sequences))
        chunk_sequences = sequences[start_idx:end_idx]
        
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_sequences)} sequences)...")
        
        try:
            # Generate embeddings for this chunk
            chunk_embeddings_df = embedding_generator.generate_embeddings(chunk_sequences)
            
            if chunk_embeddings_df is not None and len(chunk_embeddings_df) > 0:
                # Extract embeddings from DataFrame
                chunk_embeddings = chunk_embeddings_df['embedding'].tolist()
                all_embeddings.extend(chunk_embeddings)
                print(f"  ✓ Generated {len(chunk_embeddings)} embeddings")
            else:
                print(f"  ✗ Failed to generate embeddings for chunk {chunk_idx + 1}")
                failed_sequences.extend(range(start_idx, end_idx))
                
        except Exception as e:
            print(f"  ✗ Error processing chunk {chunk_idx + 1}: {e}")
            failed_sequences.extend(range(start_idx, end_idx))
    
    if failed_sequences:
        print(f"⚠️  Failed to process {len(failed_sequences)} sequences")
        print(f"   Failed indices: {failed_sequences[:10]}..." if len(failed_sequences) > 10 else f"   Failed indices: {failed_sequences}")
    
    if not all_embeddings:
        raise ValueError("No embeddings were generated successfully!")
    
    embeddings_array = np.array(all_embeddings)
    print(f"✓ Generated {len(all_embeddings)} embeddings")
    print(f"✓ Embedding shape: {embeddings_array.shape}")
    
    return embeddings_array, failed_sequences


def train_with_cross_validation(X, y, n_splits=5):
    """Train model with stratified k-fold cross-validation"""
    print(f"\n=== Cross-Validation Training ({n_splits} folds) ===")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'auc_pr_scores': [],
        'models': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Calculate class weights for imbalance
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Training samples: {len(y_train)} (pos: {pos_count}, neg: {neg_count})")
        print(f"Test samples: {len(y_test)}")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Initialize XGBoost classifier with optimized parameters
        xgb_model = xgb.XGBClassifier(
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
        
        # Train model
        print("Training model...")
        fold_start_time = time.time()
        xgb_model.fit(X_train, y_train)
        fold_time = time.time() - fold_start_time
        
        # Make predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        print(f"Training time: {fold_time:.1f} seconds")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        
        # Store results
        results['f1_scores'].append(f1)
        results['precision_scores'].append(precision)
        results['recall_scores'].append(recall)
        results['auc_pr_scores'].append(auc_pr)
        results['models'].append(xgb_model)
    
    # Print summary
    print(f"\n=== Cross-Validation Results ===")
    print(f"F1-Score: {np.mean(results['f1_scores']):.4f} ± {np.std(results['f1_scores']):.4f}")
    print(f"Precision: {np.mean(results['precision_scores']):.4f} ± {np.std(results['precision_scores']):.4f}")
    print(f"Recall: {np.mean(results['recall_scores']):.4f} ± {np.std(results['recall_scores']):.4f}")
    print(f"AUC-PR: {np.mean(results['auc_pr_scores']):.4f} ± {np.std(results['auc_pr_scores']):.4f}")
    
    return results


def train_final_model(X, y):
    """Train the final model on the complete dataset"""
    print(f"\n=== Training Final Model ===")
    
    # Calculate class weights
    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Total samples: {len(y)} (pos: {pos_count}, neg: {neg_count})")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Initialize XGBoost classifier
    final_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_estimators=300,  # More trees for final model
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    # Train model
    print("Training final model...")
    start_time = time.time()
    final_model.fit(X, y)
    training_time = time.time() - start_time
    
    print(f"✓ Final model trained in {training_time:.1f} seconds")
    
    return final_model


def create_visualizations(results, save_dir="results"):
    """Create visualization plots"""
    print(f"\n=== Creating Visualizations ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Full Dataset Model Performance', fontsize=16)
    
    # F1-Score distribution
    axes[0, 0].hist(results['f1_scores'], bins=10, alpha=0.7, color='blue')
    axes[0, 0].axvline(np.mean(results['f1_scores']), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(results["f1_scores"]):.3f}')
    axes[0, 0].set_title('F1-Score Distribution')
    axes[0, 0].set_xlabel('F1-Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Precision-Recall scatter
    axes[0, 1].scatter(results['recall_scores'], results['precision_scores'], alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC-PR distribution
    axes[1, 0].hist(results['auc_pr_scores'], bins=10, alpha=0.7, color='orange')
    axes[1, 0].axvline(np.mean(results['auc_pr_scores']), color='red', linestyle='--',
                      label=f'Mean: {np.mean(results["auc_pr_scores"]):.3f}')
    axes[1, 0].set_title('AUC-PR Distribution')
    axes[1, 0].set_xlabel('AUC-PR')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Performance comparison
    metrics = ['F1-Score', 'Precision', 'Recall', 'AUC-PR']
    means = [np.mean(results['f1_scores']), np.mean(results['precision_scores']), 
             np.mean(results['recall_scores']), np.mean(results['auc_pr_scores'])]
    stds = [np.std(results['f1_scores']), np.std(results['precision_scores']),
            np.std(results['recall_scores']), np.std(results['auc_pr_scores'])]
    
    axes[1, 1].bar(metrics, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/full_dataset_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {save_dir}/full_dataset_performance.png")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("RETRAIN FULL DATASET MODEL")
    print("Using Real ESM Embeddings and Proper Approach")
    print("=" * 70)
    
    # Step 1: Load complete training data
    sequences, labels = load_complete_training_data()
    
    # Step 2: Generate real ESM embeddings
    try:
        embeddings, failed_indices = generate_real_embeddings(sequences, chunk_size=25)
        
        # Remove failed sequences from labels
        if failed_indices:
            print(f"Removing {len(failed_indices)} failed sequences from labels...")
            valid_indices = [i for i in range(len(labels)) if i not in failed_indices]
            labels = labels[valid_indices]
            print(f"Final dataset: {len(embeddings)} sequences, {np.sum(labels)} positive samples")
        
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False
    
    # Step 3: Cross-validation training
    try:
        cv_results = train_with_cross_validation(embeddings, labels, n_splits=5)
        
        # Select best model
        best_fold = np.argmax(cv_results['f1_scores'])
        best_model = cv_results['models'][best_fold]
        
        print(f"✓ Best model from fold {best_fold + 1} with F1-score: {cv_results['f1_scores'][best_fold]:.4f}")
        
    except Exception as e:
        print(f"✗ Cross-validation training failed: {e}")
        return False
    
    # Step 4: Train final model
    try:
        final_model = train_final_model(embeddings, labels)
        print("✓ Final model trained successfully")
        
    except Exception as e:
        print(f"✗ Final model training failed: {e}")
        return False
    
    # Step 5: Save model and results
    print(f"\n=== Saving Model and Results ===")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save final model
    model_data = {
        'model': final_model,
        'embedding_dimension': embeddings.shape[1],
        'training_sequences': len(embeddings),
        'positive_samples': int(np.sum(labels)),
        'negative_samples': int(len(labels) - np.sum(labels)),
        'class_balance': float(np.mean(labels))
    }
    
    model_path = "models/germacrene_classifier_full_retrained.pkl"
    joblib.dump(model_data, model_path)
    
    # Save comprehensive results
    results_data = {
        'training_sequences': int(len(embeddings)),
        'embedding_dimension': int(embeddings.shape[1]),
        'positive_samples': int(np.sum(labels)),
        'negative_samples': int(len(labels) - np.sum(labels)),
        'class_balance': float(np.mean(labels)),
        'cv_f1_mean': float(np.mean(cv_results['f1_scores'])),
        'cv_f1_std': float(np.std(cv_results['f1_scores'])),
        'cv_precision_mean': float(np.mean(cv_results['precision_scores'])),
        'cv_precision_std': float(np.std(cv_results['precision_scores'])),
        'cv_recall_mean': float(np.mean(cv_results['recall_scores'])),
        'cv_recall_std': float(np.std(cv_results['recall_scores'])),
        'cv_auc_pr_mean': float(np.mean(cv_results['auc_pr_scores'])),
        'cv_auc_pr_std': float(np.std(cv_results['auc_pr_scores'])),
        'best_fold': int(best_fold + 1),
        'best_f1_score': float(cv_results['f1_scores'][best_fold]),
        'best_precision': float(cv_results['precision_scores'][best_fold]),
        'best_recall': float(cv_results['recall_scores'][best_fold]),
        'best_auc_pr': float(cv_results['auc_pr_scores'][best_fold]),
        'failed_sequences': len(failed_indices) if 'failed_indices' in locals() else 0,
        'training_completed': True,
        'timestamp': datetime.now().isoformat()
    }
    
    with open("results/full_dataset_retrained_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Step 6: Create visualizations
    create_visualizations(cv_results)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Results saved to: results/full_dataset_retrained_results.json")
    
    # Step 7: Performance comparison
    print(f"\n=== Performance Comparison ===")
    print(f"Full Dataset Model (Retrained):")
    print(f"  - F1-Score: {np.mean(cv_results['f1_scores']):.4f} ± {np.std(cv_results['f1_scores']):.4f}")
    print(f"  - AUC-PR: {np.mean(cv_results['auc_pr_scores']):.4f} ± {np.std(cv_results['auc_pr_scores']):.4f}")
    print(f"  - Best F1-Score: {cv_results['f1_scores'][best_fold]:.4f}")
    
    print(f"\nPrevious 500-Sequence Model:")
    print(f"  - F1-Score: 0.593 ± 0.174")
    print(f"  - AUC-PR: 0.723 ± 0.187")
    print(f"  - Best F1-Score: 0.833")
    
    # Check if full dataset model is better
    full_dataset_f1 = np.mean(cv_results['f1_scores'])
    if full_dataset_f1 > 0.593:
        print(f"\n🎉 SUCCESS: Full dataset model ({full_dataset_f1:.4f}) performs better than 500-sequence model (0.593)!")
    else:
        print(f"\n⚠️  Full dataset model ({full_dataset_f1:.4f}) still underperforms 500-sequence model (0.593)")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✓ Training completed successfully!")
    else:
        print(f"\n✗ Training failed!")
        sys.exit(1)
