#!/usr/bin/env python3
"""
Enhanced ESM2+Engineered Baseline with Adaptive Thresholding
=============================================================

An improved baseline incorporating:
1. ESM2 embeddings + Engineered biochemical features
2. Per-class adaptive threshold optimization

This aims to replicate V3's ESM2+Engineered performance (~8.57% F1)
by using the same adaptive thresholding strategy.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import sys
from tqdm import tqdm

# --- Configuration ---
DATA_FILE = '../TPS_Classifier_v3_Early/TS-GSD_consolidated.csv'
ESM2_EMBEDDINGS_FILE = 'data/esm2_embeddings.npy'
ENGINEERED_FEATURES_FILE = 'data/engineered_features.npy'
N_SPLITS = 5
EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
N_CLASSES = 30

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"🔧 Device: {DEVICE}")
print(f"📊 Configuration: {N_SPLITS}-fold CV, {EPOCHS} epochs, batch size {BATCH_SIZE}")


# --- 1. Enhanced MLP Model (for combined features) ---
class EnhancedMLP(nn.Module):
    """MLP for ESM2 + Engineered features"""
    def __init__(self, input_dim, output_dim):
        super(EnhancedMLP, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.layer_stack(x)


# --- 2. Per-Class Adaptive Threshold Finding ---
def find_optimal_thresholds(y_true, y_pred_proba, n_classes=N_CLASSES):
    """
    Find optimal decision threshold for each class independently.
    
    For each class, iterate through candidate thresholds and select
    the one that maximizes the F1 score for that class.
    
    Args:
        y_true: Ground truth labels (N_samples, N_classes)
        y_pred_proba: Predicted probabilities (N_samples, N_classes)
        n_classes: Number of classes
        
    Returns:
        optimal_thresholds: Array of shape (N_classes,) with best threshold per class
    """
    optimal_thresholds = np.zeros(n_classes)
    
    # Candidate thresholds to test
    threshold_candidates = np.arange(0.05, 0.96, 0.01)
    
    for class_idx in range(n_classes):
        best_f1 = 0.0
        best_threshold = 0.5  # Default
        
        # Get true labels and predictions for this class
        y_true_class = y_true[:, class_idx]
        y_pred_class_proba = y_pred_proba[:, class_idx]
        
        # Skip if no positive samples for this class
        if y_true_class.sum() == 0:
            optimal_thresholds[class_idx] = 0.5
            continue
        
        # Test each threshold
        for threshold in threshold_candidates:
            y_pred_binary = (y_pred_class_proba >= threshold).astype(int)
            
            # Calculate F1 for this class at this threshold
            f1 = f1_score(y_true_class, y_pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[class_idx] = best_threshold
    
    return optimal_thresholds


# --- 3. Data Loading with Engineered Features ---
def load_and_process_data():
    """Load ESM2 embeddings, engineered features, and labels"""
    print("📂 Loading data...")
    
    # Load TS-GSD for labels
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df)} samples")
    
    # Load ESM2 embeddings
    if not Path(ESM2_EMBEDDINGS_FILE).exists():
        print(f"❌ Error: ESM2 embeddings not found at {ESM2_EMBEDDINGS_FILE}")
        print("Please run train_baseline.py first to generate ESM2 embeddings")
        sys.exit(1)
    
    esm2_embeddings = np.load(ESM2_EMBEDDINGS_FILE)
    print(f"✅ Loaded ESM2 embeddings: {esm2_embeddings.shape}")
    
    # Load engineered features
    if not Path(ENGINEERED_FEATURES_FILE).exists():
        print(f"❌ Error: Engineered features not found at {ENGINEERED_FEATURES_FILE}")
        print("Please run generate_engineered_features.py first")
        sys.exit(1)
    
    engineered_features = np.load(ENGINEERED_FEATURES_FILE)
    print(f"✅ Loaded engineered features: {engineered_features.shape}")
    
    # Concatenate ESM2 + Engineered features
    combined_features = np.concatenate([esm2_embeddings, engineered_features], axis=1)
    print(f"✅ Combined features: {combined_features.shape} (ESM2 + Engineered)")
    
    # Extract labels
    target_vectors = []
    for target_str in df['target_vector'].tolist():
        target_vec = eval(target_str)
        target_vectors.append(target_vec)
    
    labels = np.array(target_vectors, dtype=np.float32)
    print(f"✅ Labels: {labels.shape}")
    
    return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# --- 4. Training Function with Threshold Tuning ---
def train_one_fold_with_thresholding(X_train, y_train, X_val, y_val, input_dim, n_classes, epochs):
    """
    Train model for one fold with adaptive threshold tuning.
    
    Steps:
    1. Split training data into sub-train (90%) and threshold-tuning (10%)
    2. Train model on sub-train set
    3. Find optimal thresholds using tuning set
    4. Evaluate on validation set with adaptive thresholds
    """
    
    # Split training data for threshold tuning (random split to avoid stratification issues)
    indices = np.arange(len(X_train))
    
    train_indices, tune_indices = train_test_split(
        indices, 
        test_size=0.1, 
        random_state=RANDOM_SEED
        # No stratification due to sparse/single-sample classes
    )
    
    X_subtrain = X_train[train_indices]
    y_subtrain = y_train[train_indices]
    X_tune = X_train[tune_indices]
    y_tune = y_train[tune_indices]
    
    print(f"      Sub-train: {len(X_subtrain)}, Threshold-tune: {len(X_tune)}, Val: {len(X_val)}")
    
    # Create data loader for sub-training set
    train_dataset = TensorDataset(X_subtrain, y_subtrain)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = EnhancedMLP(input_dim=input_dim, output_dim=n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train model
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Periodic logging
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val.to(DEVICE))
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_preds_fixed = (val_probs > 0.5).astype(int)
                macro_f1_fixed = f1_score(y_val.numpy(), val_preds_fixed, average='macro', zero_division=0)
                print(f"         Epoch {epoch+1}/{epochs} - Val F1 (fixed 0.5): {macro_f1_fixed:.4f}")
    
    # Find optimal thresholds using tuning set
    model.eval()
    with torch.no_grad():
        tune_logits = model(X_tune.to(DEVICE))
        tune_probs = torch.sigmoid(tune_logits).cpu().numpy()
    
    print(f"      🔍 Finding optimal thresholds on tuning set...")
    optimal_thresholds = find_optimal_thresholds(y_tune.numpy(), tune_probs, n_classes)
    
    # Report threshold statistics
    print(f"      📊 Threshold range: [{optimal_thresholds.min():.2f}, {optimal_thresholds.max():.2f}]")
    print(f"      📊 Threshold mean: {optimal_thresholds.mean():.2f}")
    
    # Evaluate on validation set with adaptive thresholds
    with torch.no_grad():
        val_logits = model(X_val.to(DEVICE))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        
        # Apply per-class thresholds
        val_preds_adaptive = (val_probs >= optimal_thresholds[None, :]).astype(int)
        
        # Calculate metrics
        macro_f1 = f1_score(y_val.numpy(), val_preds_adaptive, average='macro', zero_division=0)
        micro_f1 = f1_score(y_val.numpy(), val_preds_adaptive, average='micro', zero_division=0)
        macro_prec = precision_score(y_val.numpy(), val_preds_adaptive, average='macro', zero_division=0)
        macro_rec = recall_score(y_val.numpy(), val_preds_adaptive, average='macro', zero_division=0)
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'precision': macro_prec,
        'recall': macro_rec,
        'optimal_thresholds': optimal_thresholds
    }


# --- 5. Main Cross-Validation Loop ---
def run_cross_validation():
    """Run 5-fold CV with adaptive thresholding"""
    print("\n" + "="*60)
    print("🚀 Enhanced Baseline Training (ESM2 + Engineered + Adaptive Thresholds)")
    print("="*60 + "\n")
    
    # Load data
    X, y = load_and_process_data()
    
    print(f"\n📊 Data summary:")
    print(f"   - Samples: {len(X)}")
    print(f"   - Input dim: {X.shape[1]} (ESM2 {X.shape[1]-64} + Engineered 64)")
    print(f"   - Classes: {N_CLASSES}")
    
    # Stratification variable
    y_stratify = y.argmax(axis=1).numpy()
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    all_thresholds = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_stratify)):
        print(f"\n{'='*60}")
        print(f"📁 Fold {fold+1}/{N_SPLITS}")
        print(f"{'='*60}")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"\n   🏋️  Training with threshold tuning...")
        
        result = train_one_fold_with_thresholding(
            X_train, y_train, X_val, y_val,
            input_dim=X.shape[1],
            n_classes=N_CLASSES,
            epochs=EPOCHS
        )
        
        fold_results.append(result)
        all_thresholds.append(result['optimal_thresholds'])
        
        print(f"\n   ✅ Fold {fold+1} Results (Adaptive Thresholds):")
        print(f"      Macro F1:   {result['macro_f1']:.4f}")
        print(f"      Micro F1:   {result['micro_f1']:.4f}")
        print(f"      Precision:  {result['precision']:.4f}")
        print(f"      Recall:     {result['recall']:.4f}")
    
    # Aggregate results
    print(f"\n" + "="*60)
    print("📊 CROSS-VALIDATION RESULTS (ADAPTIVE THRESHOLDS)")
    print("="*60)
    
    macro_f1_scores = [r['macro_f1'] for r in fold_results]
    micro_f1_scores = [r['micro_f1'] for r in fold_results]
    precision_scores = [r['precision'] for r in fold_results]
    recall_scores = [r['recall'] for r in fold_results]
    
    print(f"\n🎯 Macro F1 Score (Primary Metric):")
    print(f"   Mean:   {np.mean(macro_f1_scores):.4f}")
    print(f"   Std:    {np.std(macro_f1_scores):.4f}")
    print(f"   95% CI: [{np.mean(macro_f1_scores) - 1.96*np.std(macro_f1_scores):.4f}, "
          f"{np.mean(macro_f1_scores) + 1.96*np.std(macro_f1_scores):.4f}]")
    
    print(f"\n📈 Additional Metrics:")
    print(f"   Micro F1:   {np.mean(micro_f1_scores):.4f} (± {np.std(micro_f1_scores):.4f})")
    print(f"   Precision:  {np.mean(precision_scores):.4f} (± {np.std(precision_scores):.4f})")
    print(f"   Recall:     {np.mean(recall_scores):.4f} (± {np.std(recall_scores):.4f})")
    
    # Threshold analysis
    avg_thresholds = np.mean(all_thresholds, axis=0)
    print(f"\n🔍 Threshold Analysis (Averaged Across Folds):")
    print(f"   Range: [{avg_thresholds.min():.2f}, {avg_thresholds.max():.2f}]")
    print(f"   Mean:  {avg_thresholds.mean():.2f}")
    print(f"   Std:   {avg_thresholds.std():.2f}")
    
    # Save results
    results_summary = {
        'configuration': {
            'model': 'Enhanced MLP (ESM2 + Engineered)',
            'features': 'ESM2 (1280D) + Engineered (64D) = 1344D',
            'thresholding': 'Per-class adaptive (F1-optimized)',
            'n_folds': N_SPLITS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'n_classes': N_CLASSES
        },
        'cross_validation_results': {
            'macro_f1_mean': float(np.mean(macro_f1_scores)),
            'macro_f1_std': float(np.std(macro_f1_scores)),
            'micro_f1_mean': float(np.mean(micro_f1_scores)),
            'precision_mean': float(np.mean(precision_scores)),
            'recall_mean': float(np.mean(recall_scores)),
            'fold_results': [
                {
                    'fold': i+1,
                    'macro_f1': float(r['macro_f1']),
                    'micro_f1': float(r['micro_f1']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'optimal_thresholds': r['optimal_thresholds'].tolist()
                }
                for i, r in enumerate(fold_results)
            ]
        },
        'threshold_analysis': {
            'avg_thresholds': avg_thresholds.tolist(),
            'threshold_mean': float(avg_thresholds.mean()),
            'threshold_std': float(avg_thresholds.std()),
            'threshold_min': float(avg_thresholds.min()),
            'threshold_max': float(avg_thresholds.max())
        }
    }
    
    Path('results').mkdir(exist_ok=True)
    results_path = 'results/enhanced_baseline_cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    print(f"\n{'='*60}")
    print(f"✅ Enhanced baseline training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_cross_validation()

