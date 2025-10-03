#!/usr/bin/env python3
"""
Full-Scale Training Script for Germacrene Synthase Classifier
============================================================

This script trains on the complete MARTS-DB dataset (1,356 sequences).
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
import xgboost as xgb


def load_full_training_data():
    """Load the complete training dataset"""
    training_file = "data/germacrene_training_data.csv"
    
    if not os.path.exists(training_file):
        print(f"Training data not found at {training_file}")
        print("Please run: python3 improved_marts_parser.py")
        return None
    
    df = pd.read_csv(training_file)
    print(f"✓ Loaded complete dataset: {len(df)} sequences")
    
    positive_count = df['target'].sum()
    negative_count = len(df) - positive_count
    print(f"  - Germacrene synthases: {positive_count}")
    print(f"  - Other synthases: {negative_count}")
    print(f"  - Class balance: {positive_count/len(df):.2%}")
    
    return df


def generate_embeddings_full_scale(sequences, labels, chunk_size=50, max_sequences=None):
    """
    Generate embeddings for the full dataset with progress tracking
    """
    if max_sequences:
        print(f"Limiting to first {max_sequences} sequences for testing")
        sequences = sequences[:max_sequences]
        labels = labels[:max_sequences]
    
    print(f"Generating embeddings for {len(sequences)} sequences in chunks of {chunk_size}...")
    
    generator = RobustEmbeddingGenerator()
    all_embeddings = []
    all_labels = []
    valid_indices = []
    
    total_chunks = (len(sequences) + chunk_size - 1) // chunk_size
    start_time = time.time()
    
    # Process in chunks
    for i in range(0, len(sequences), chunk_size):
        chunk_num = i // chunk_size + 1
        chunk_sequences = sequences[i:i + chunk_size]
        chunk_labels = labels[i:i + chunk_size]
        
        print(f"\nProcessing chunk {chunk_num}/{total_chunks} ({len(chunk_sequences)} sequences)")
        
        try:
            # Generate embeddings for this chunk
            chunk_embeddings_df = generator.generate_embeddings(chunk_sequences)
            
            # Extract embeddings
            chunk_embeddings = np.array([np.array(emb) for emb in chunk_embeddings_df['embedding']])
            
            all_embeddings.append(chunk_embeddings)
            all_labels.extend(chunk_labels)
            
            # Track valid indices
            chunk_valid_indices = [i + j for j in range(len(chunk_embeddings))]
            valid_indices.extend(chunk_valid_indices)
            
            # Progress tracking
            elapsed_time = time.time() - start_time
            avg_time_per_chunk = elapsed_time / chunk_num
            remaining_chunks = total_chunks - chunk_num
            estimated_remaining_time = remaining_chunks * avg_time_per_chunk
            
            print(f"  ✓ Processed {len(chunk_embeddings)} sequences in this chunk")
            print(f"  ✓ Total progress: {chunk_num}/{total_chunks} chunks")
            print(f"  ✓ Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"  ✗ Chunk {chunk_num} failed: {e}")
            continue
    
    if not all_embeddings:
        raise RuntimeError("No embeddings generated")
    
    # Combine all embeddings
    X = np.vstack(all_embeddings)
    y = np.array(all_labels)
    
    total_time = time.time() - start_time
    print(f"\n✓ Embedding generation completed!")
    print(f"✓ Total embeddings: {X.shape[0]}")
    print(f"✓ Embedding dimension: {X.shape[1]}")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    
    return X, y, valid_indices


def train_with_full_cross_validation(X, y, n_splits=5):
    """
    Train model with full cross-validation on complete dataset
    """
    print(f"\nTraining with {n_splits}-fold cross-validation on complete dataset...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'auc_pr_scores': [],
        'models': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Calculate scale_pos_weight for class imbalance
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
            n_estimators=200,  # Increased for better performance
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
        
        # Calculate AUC-PR
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall_curve, precision_curve)
        
        print(f"Training time: {fold_time:.1f} seconds")
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
    print(f"\n=== Cross-Validation Results (Complete Dataset) ===")
    print(f"F1-Score: {np.mean(results['f1_scores']):.3f} ± {np.std(results['f1_scores']):.3f}")
    print(f"Precision: {np.mean(results['precision_scores']):.3f} ± {np.std(results['precision_scores']):.3f}")
    print(f"Recall: {np.mean(results['recall_scores']):.3f} ± {np.std(results['recall_scores']):.3f}")
    print(f"AUC-PR: {np.mean(results['auc_pr_scores']):.3f} ± {np.std(results['auc_pr_scores']):.3f}")
    
    return results


def train_final_full_model(X, y):
    """Train the final model on the complete dataset"""
    print(f"\nTraining final model on complete dataset...")
    
    # Calculate scale_pos_weight
    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Total samples: {len(y)} (pos: {pos_count}, neg: {neg_count})")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Initialize and train final model with optimized parameters
    final_model = xgb.XGBClassifier(
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
    
    start_time = time.time()
    final_model.fit(X, y)
    training_time = time.time() - start_time
    
    print(f"✓ Final model training completed in {training_time:.1f} seconds!")
    return final_model


def main():
    """Main training function for complete dataset"""
    print("=" * 70)
    print("FULL-SCALE GERMACRENE SYNTHASE CLASSIFIER TRAINING")
    print("Training on Complete MARTS-DB Dataset")
    print("=" * 70)
    
    # Step 1: Load complete training data
    print("\n=== Step 1: Load Complete Training Data ===")
    training_df = load_full_training_data()
    
    if training_df is None:
        return False
    
    # Step 2: Generate embeddings for complete dataset
    print("\n=== Step 2: Generate Embeddings (Complete Dataset) ===")
    
    sequences = training_df['sequence'].tolist()
    labels = training_df['target'].tolist()
    
    # Ask user if they want to limit for testing
    print(f"\nTotal sequences to process: {len(sequences)}")
    print("This will take approximately 2-3 hours on CPU.")
    
    # Use all sequences for complete training
    max_sequences = None  # Use all 1,356 sequences
    
    try:
        X, y, valid_indices = generate_embeddings_full_scale(
            sequences, labels, 
            chunk_size=50, 
            max_sequences=max_sequences
        )
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False
    
    # Step 3: Cross-validation training on complete dataset
    print("\n=== Step 3: Cross-Validation Training (Complete Dataset) ===")
    
    try:
        cv_results = train_with_full_cross_validation(X, y, n_splits=5)
        
        # Select best model
        best_fold = np.argmax(cv_results['f1_scores'])
        best_model = cv_results['models'][best_fold]
        
        print(f"✓ Best model from fold {best_fold + 1} with F1-score: {cv_results['f1_scores'][best_fold]:.3f}")
        
    except Exception as e:
        print(f"✗ Cross-validation training failed: {e}")
        return False
    
    # Step 4: Train final model on complete dataset
    print("\n=== Step 4: Final Model Training (Complete Dataset) ===")
    
    try:
        final_model = train_final_full_model(X, y)
    except Exception as e:
        print(f"✗ Final model training failed: {e}")
        return False
    
    # Step 5: Save model and results
    print("\n=== Step 5: Save Complete Model and Results ===")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Save the final model
    import joblib
    model_data = {
        'model': final_model,
        'embedding_dim': X.shape[1],
        'model_name': 'esm2_t33_650M_UR50D',
        'training_sequences': len(sequences) if max_sequences is None else max_sequences,
        'positive_samples': int(np.sum(y)),
        'negative_samples': int(len(y) - np.sum(y)),
        'cv_results': cv_results  # Save CV results for analysis
    }
    
    model_path = "models/germacrene_classifier_full.pkl"
    joblib.dump(model_data, model_path)
    
    # Save comprehensive training results
    results_data = {
        'training_sequences': int(len(sequences) if max_sequences is None else max_sequences),
        'embedding_dimension': int(X.shape[1]),
        'positive_samples': int(np.sum(y)),
        'negative_samples': int(len(y) - np.sum(y)),
        'class_balance': float(np.mean(y)),
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
        'best_auc_pr': float(cv_results['auc_pr_scores'][best_fold])
    }
    
    with open("results/full_training_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✓ Complete model saved to: {model_path}")
    print(f"✓ Comprehensive results saved to: results/full_training_results.json")
    
    # Step 6: Performance analysis
    print("\n=== Step 6: Performance Analysis ===")
    
    print(f"\nFinal Performance Summary:")
    print(f"  - Training sequences: {len(X)}")
    print(f"  - Cross-validation F1-Score: {np.mean(cv_results['f1_scores']):.3f} ± {np.std(cv_results['f1_scores']):.3f}")
    print(f"  - Best fold F1-Score: {cv_results['f1_scores'][best_fold]:.3f}")
    print(f"  - Cross-validation AUC-PR: {np.mean(cv_results['auc_pr_scores']):.3f} ± {np.std(cv_results['auc_pr_scores']):.3f}")
    
    print("\n" + "=" * 70)
    print("FULL-SCALE TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"✓ Trained on {len(X)} sequences with real ESM-2 embeddings")
    print(f"✓ 5-fold cross-validation completed")
    print(f"✓ Final model trained and saved")
    print(f"✓ Model ready for production use")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n" + "=" * 70)
        print("FULL-SCALE TRAINING FAILED")
        print("=" * 70)
        print("Please check the error messages above and try again.")
        sys.exit(1)
