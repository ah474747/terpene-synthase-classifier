#!/usr/bin/env python3
"""
Train XGBoost Classifier using ESM2 Embeddings
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load ESM2 embeddings and labels"""
    print("Loading data...")
    
    # Load embeddings
    embeddings = np.load('esm2_embeddings.npy')
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    
    # Load sequence info
    sequence_info = pd.read_csv('sequence_info.csv')
    print(f"✅ Loaded sequence info: {len(sequence_info)} sequences")
    
    # Load expanded dataset with labels
    expanded_data = pd.read_csv('data/expanded_dataset/expanded_germacrene_dataset.csv')
    print(f"✅ Loaded expanded dataset: {len(expanded_data)} sequences")
    
    # Merge to get labels
    merged = sequence_info.merge(
        expanded_data[['id', 'is_germacrene']], 
        on='id', 
        how='left'
    )
    
    # Check for missing labels
    if merged['is_germacrene'].isna().any():
        print(f"⚠️  Warning: {merged['is_germacrene'].isna().sum()} sequences missing labels")
        merged = merged.dropna(subset=['is_germacrene'])
        embeddings = embeddings[merged.index]
    
    labels = merged['is_germacrene'].values.astype(int)
    
    print(f"\n📊 Dataset summary:")
    print(f"   Total sequences: {len(labels)}")
    print(f"   Germacrene synthases: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"   Non-germacrene: {len(labels) - labels.sum()} ({(1-labels.sum()/len(labels))*100:.1f}%)")
    
    return embeddings, labels, merged

def train_model(X, y):
    """Train XGBoost classifier with repeated stratified k-fold CV"""
    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)
    
    # Calculate class weights
    n_samples = len(y)
    n_positive = y.sum()
    n_negative = n_samples - n_positive
    scale_pos_weight = n_negative / n_positive
    
    print(f"\nClass balance:")
    print(f"   Positive samples: {n_positive}")
    print(f"   Negative samples: {n_negative}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    # XGBoost parameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cpu'
    }
    
    print(f"\nModel parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Repeated Stratified K-Fold Cross-Validation
    n_splits = 5
    n_repeats = 3
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    print(f"\nCross-validation: {n_splits}-fold, {n_repeats} repeats")
    
    # Store results
    cv_results = {
        'fold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'avg_precision': []
    }
    
    fold = 1
    for train_idx, val_idx in rskf.split(X, y):
        print(f"\nFold {fold}/{n_splits * n_repeats}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        avg_prec = average_precision_score(y_val, y_pred_proba)
        
        cv_results['fold'].append(fold)
        cv_results['accuracy'].append(accuracy)
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        cv_results['f1'].append(f1)
        cv_results['roc_auc'].append(roc_auc)
        cv_results['avg_precision'].append(avg_prec)
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        
        fold += 1
    
    # Summary statistics
    cv_df = pd.DataFrame(cv_results)
    
    print("\n" + "="*60)
    print("Cross-Validation Results Summary")
    print("="*60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']:
        mean = cv_df[metric].mean()
        std = cv_df[metric].std()
        print(f"{metric.upper():15s}: {mean:.4f} ± {std:.4f}")
    
    # Train final model on all data
    print("\n" + "="*60)
    print("Training Final Model on Full Dataset")
    print("="*60)
    
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)
    
    print("✅ Final model trained")
    
    return final_model, cv_df

def evaluate_model(model, X, y, sequence_info):
    """Evaluate model and create visualizations"""
    print("\n" + "="*60)
    print("Final Model Evaluation")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Non-germacrene', 'Germacrene']))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-germacrene', 'Germacrene'],
                yticklabels=['Non-germacrene', 'Germacrene'])
    plt.title('Confusion Matrix - ESM2 Embeddings')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/esm2_confusion_matrix.png', dpi=300)
    print("\n✅ Confusion matrix saved to results/esm2_confusion_matrix.png")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ESM2 Embeddings')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/esm2_roc_curve.png', dpi=300)
    print("✅ ROC curve saved to results/esm2_roc_curve.png")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    avg_prec = average_precision_score(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {avg_prec:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - ESM2 Embeddings')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/esm2_pr_curve.png', dpi=300)
    print("✅ Precision-Recall curve saved to results/esm2_pr_curve.png")
    
    plt.close('all')

def save_results(model, cv_results, embeddings, labels, sequence_info):
    """Save model and results"""
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save model
    model.save_model('models/esm2_germacrene_classifier.json')
    print("✅ Model saved to models/esm2_germacrene_classifier.json")
    
    # Save CV results
    cv_results.to_csv('results/esm2_cv_results.csv', index=False)
    print("✅ CV results saved to results/esm2_cv_results.csv")
    
    # Save summary statistics
    summary = {
        'model': 'XGBoost with ESM2 embeddings',
        'embedding_dim': embeddings.shape[1],
        'n_sequences': len(labels),
        'n_positive': int(labels.sum()),
        'n_negative': int(len(labels) - labels.sum()),
        'cv_metrics': {
            'accuracy': f"{cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}",
            'precision': f"{cv_results['precision'].mean():.4f} ± {cv_results['precision'].std():.4f}",
            'recall': f"{cv_results['recall'].mean():.4f} ± {cv_results['recall'].std():.4f}",
            'f1': f"{cv_results['f1'].mean():.4f} ± {cv_results['f1'].std():.4f}",
            'roc_auc': f"{cv_results['roc_auc'].mean():.4f} ± {cv_results['roc_auc'].std():.4f}",
            'avg_precision': f"{cv_results['avg_precision'].mean():.4f} ± {cv_results['avg_precision'].std():.4f}"
        }
    }
    
    with open('results/esm2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✅ Summary saved to results/esm2_summary.json")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("ESM2 Germacrene Synthase Classifier Training")
    print("="*60)
    
    # Load data
    embeddings, labels, sequence_info = load_data()
    
    # Train model
    model, cv_results = train_model(embeddings, labels)
    
    # Evaluate model
    evaluate_model(model, embeddings, labels, sequence_info)
    
    # Save results
    save_results(model, cv_results, embeddings, labels, sequence_info)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review results in results/esm2_summary.json")
    print("2. Check visualizations in results/")
    print("3. Compare with previous model performance")
    print("4. Use model for predictions on new sequences")

if __name__ == "__main__":
    main()

