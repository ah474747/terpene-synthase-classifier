#!/usr/bin/env python3
"""
Robust Training Script for Germacrene Synthase Classifier
========================================================

This script uses the robust embedding generator to avoid memory issues.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_embedding_generator import RobustEmbeddingGenerator
from terpene_classifier import TerpeneClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
import xgboost as xgb


def load_training_data():
    """Load the training data"""
    training_file = "data/germacrene_training_data.csv"
    
    if not os.path.exists(training_file):
        print(f"Training data not found at {training_file}")
        print("Please run: python3 improved_marts_parser.py")
        return None
    
    df = pd.read_csv(training_file)
    print(f"✓ Loaded {len(df)} training sequences")
    
    positive_count = df['target'].sum()
    negative_count = len(df) - positive_count
    print(f"  - Germacrene synthases: {positive_count}")
    print(f"  - Other synthases: {negative_count}")
    print(f"  - Class balance: {positive_count/len(df):.2%}")
    
    return df


def generate_embeddings_in_chunks(sequences, labels, chunk_size=50):
    """
    Generate embeddings in chunks to manage memory
    """
    print(f"Generating embeddings in chunks of {chunk_size}...")
    
    generator = RobustEmbeddingGenerator()
    all_embeddings = []
    all_labels = []
    valid_indices = []
    
    # Process in chunks
    for i in range(0, len(sequences), chunk_size):
        chunk_sequences = sequences[i:i + chunk_size]
        chunk_labels = labels[i:i + chunk_size]
        
        print(f"Processing chunk {i//chunk_size + 1}/{(len(sequences) + chunk_size - 1)//chunk_size}")
        
        try:
            # Generate embeddings for this chunk
            chunk_embeddings_df = generator.generate_embeddings(chunk_sequences)
            
            # Extract embeddings
            embedding_columns = [col for col in chunk_embeddings_df.columns if col != 'id']
            # Convert embedding lists to numpy arrays
            chunk_embeddings = np.array([np.array(emb) for emb in chunk_embeddings_df['embedding']])
            
            all_embeddings.append(chunk_embeddings)
            all_labels.extend(chunk_labels)
            
            # Track valid indices
            chunk_valid_indices = [i + j for j in range(len(chunk_embeddings))]
            valid_indices.extend(chunk_valid_indices)
            
            print(f"  ✓ Processed {len(chunk_embeddings)} sequences in this chunk")
            
        except Exception as e:
            print(f"  ✗ Chunk {i//chunk_size + 1} failed: {e}")
            continue
    
    if not all_embeddings:
        raise RuntimeError("No embeddings generated")
    
    # Combine all embeddings
    X = np.vstack(all_embeddings)
    y = np.array(all_labels)
    
    print(f"✓ Total embeddings generated: {X.shape[0]}")
    print(f"✓ Embedding dimension: {X.shape[1]}")
    
    return X, y, valid_indices


def train_with_cross_validation(X, y, n_splits=3):
    """
    Train model with cross-validation
    """
    print(f"Training with {n_splits}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'auc_pr_scores': [],
        'models': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Calculate scale_pos_weight for class imbalance
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"  Training samples: {len(y_train)} (pos: {pos_count}, neg: {neg_count})")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        
        # Initialize XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
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
        
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  AUC-PR: {auc_pr:.3f}")
        
        # Store results
        results['f1_scores'].append(f1)
        results['precision_scores'].append(precision)
        results['recall_scores'].append(recall)
        results['auc_pr_scores'].append(auc_pr)
        results['models'].append(xgb_model)
    
    # Print summary
    print(f"\n=== Cross-Validation Results ===")
    print(f"F1-Score: {np.mean(results['f1_scores']):.3f} ± {np.std(results['f1_scores']):.3f}")
    print(f"Precision: {np.mean(results['precision_scores']):.3f} ± {np.std(results['precision_scores']):.3f}")
    print(f"Recall: {np.mean(results['recall_scores']):.3f} ± {np.std(results['recall_scores']):.3f}")
    print(f"AUC-PR: {np.mean(results['auc_pr_scores']):.3f} ± {np.std(results['auc_pr_scores']):.3f}")
    
    return results


def train_final_model(X, y):
    """Train the final model on all data"""
    print("Training final model on complete dataset...")
    
    # Calculate scale_pos_weight
    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Total samples: {len(y)} (pos: {pos_count}, neg: {neg_count})")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Initialize and train final model
    final_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    final_model.fit(X, y)
    
    print("✓ Final model training completed!")
    return final_model


def main():
    """Main training function"""
    print("=" * 60)
    print("ROBUST GERMACRENE SYNTHASE CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Step 1: Load training data
    print("\n=== Step 1: Load Training Data ===")
    training_df = load_training_data()
    
    if training_df is None:
        return False
    
    # Step 2: Generate embeddings
    print("\n=== Step 2: Generate Embeddings ===")
    
    sequences = training_df['sequence'].tolist()
    labels = training_df['target'].tolist()
    
    # Limit sequences for initial training (we can increase this later)
    max_sequences = 200  # Start with 200 sequences
    if len(sequences) > max_sequences:
        print(f"Limiting to first {max_sequences} sequences for initial training")
        sequences = sequences[:max_sequences]
        labels = labels[:max_sequences]
    
    try:
        X, y, valid_indices = generate_embeddings_in_chunks(sequences, labels, chunk_size=25)
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False
    
    # Step 3: Cross-validation training
    print("\n=== Step 3: Cross-Validation Training ===")
    
    try:
        cv_results = train_with_cross_validation(X, y, n_splits=3)
        
        # Select best model
        best_fold = np.argmax(cv_results['f1_scores'])
        best_model = cv_results['models'][best_fold]
        
        print(f"✓ Best model from fold {best_fold + 1} with F1-score: {cv_results['f1_scores'][best_fold]:.3f}")
        
    except Exception as e:
        print(f"✗ Cross-validation training failed: {e}")
        return False
    
    # Step 4: Train final model
    print("\n=== Step 4: Final Model Training ===")
    
    try:
        final_model = train_final_model(X, y)
    except Exception as e:
        print(f"✗ Final model training failed: {e}")
        return False
    
    # Step 5: Save model and results
    print("\n=== Step 5: Save Model and Results ===")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Save the final model
    import joblib
    model_data = {
        'model': final_model,
        'embedding_dim': X.shape[1],
        'model_name': 'esm2_t33_650M_UR50D',
        'training_sequences': len(sequences),
        'positive_samples': np.sum(y),
        'negative_samples': len(y) - np.sum(y)
    }
    
    model_path = "models/germacrene_classifier_robust.pkl"
    joblib.dump(model_data, model_path)
    
    # Save training results
    results_data = {
        'training_sequences': int(len(sequences)),
        'embedding_dimension': int(X.shape[1]),
        'positive_samples': int(np.sum(y)),
        'negative_samples': int(len(y) - np.sum(y)),
        'class_balance': float(np.mean(y)),
        'cv_f1_mean': float(np.mean(cv_results['f1_scores'])),
        'cv_f1_std': float(np.std(cv_results['f1_scores'])),
        'cv_precision_mean': float(np.mean(cv_results['precision_scores'])),
        'cv_recall_mean': float(np.mean(cv_results['recall_scores'])),
        'cv_auc_pr_mean': float(np.mean(cv_results['auc_pr_scores'])),
        'best_fold': int(best_fold + 1),
        'best_f1_score': float(cv_results['f1_scores'][best_fold])
    }
    
    with open("results/robust_training_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Results saved to: results/robust_training_results.json")
    
    # Step 6: Test predictions
    print("\n=== Step 6: Test Predictions ===")
    
    try:
        # Test on a few sequences
        test_indices = [0, 1, 2] if len(X) > 2 else [0, 1] if len(X) > 1 else [0]
        
        print("Testing predictions on sample sequences:")
        for i, idx in enumerate(test_indices):
            if idx < len(sequences):
                confidence = final_model.predict_proba(X[idx:idx+1])[0, 1]
                prediction = "Germacrene" if confidence > 0.5 else "Other"
                actual = "Germacrene" if y[idx] == 1 else "Other"
                
                print(f"  Sequence {i+1}:")
                print(f"    Prediction: {prediction} (confidence: {confidence:.3f})")
                print(f"    Actual: {actual}")
                print(f"    Correct: {'✓' if (confidence > 0.5) == (y[idx] == 1) else '✗'}")
    
    except Exception as e:
        print(f"⚠ Prediction testing failed: {e}")
    
    print("\n" + "=" * 60)
    print("ROBUST TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✓ Generated real protein embeddings")
    print(f"✓ Cross-validation training completed")
    print(f"✓ Final model trained and saved")
    print(f"✓ Model ready for predictions")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n" + "=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)
        print("Please check the error messages above and try again.")
        sys.exit(1)
