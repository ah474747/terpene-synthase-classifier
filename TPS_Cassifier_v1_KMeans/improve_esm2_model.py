#!/usr/bin/env python3
"""
Improve ESM2 Model Performance with K-means and Advanced Techniques
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

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
        expanded_data[['id', 'is_germacrene', 'sequence']], 
        on='id', 
        how='left',
        suffixes=('', '_expanded')
    )
    
    # Get sequences for feature engineering (prioritize from expanded_data)
    if 'sequence_expanded' in merged.columns:
        sequences = merged['sequence_expanded'].values
    else:
        sequences = merged['sequence'].values
    
    # Check for missing labels
    if merged['is_germacrene'].isna().any():
        print(f"⚠️  Warning: {merged['is_germacrene'].isna().sum()} sequences missing labels")
        merged = merged.dropna(subset=['is_germacrene'])
        embeddings = embeddings[merged.index]
        sequences = sequences[merged.index]
    
    labels = merged['is_germacrene'].values.astype(int)
    
    print(f"\n📊 Dataset summary:")
    print(f"   Total sequences: {len(labels)}")
    print(f"   Germacrene synthases: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"   Non-germacrene: {len(labels) - labels.sum()} ({(1-labels.sum()/len(labels))*100:.1f}%)")
    
    return embeddings, labels, sequences, merged

def create_kmeans_features(embeddings, n_clusters_range=[10, 20, 30, 40, 50]):
    """Create K-means cluster features"""
    print("\n" + "="*60)
    print("Creating K-means Features")
    print("="*60)
    
    # Normalize embeddings for better clustering
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    kmeans_features = []
    
    for n_clusters in n_clusters_range:
        print(f"\nClustering with {n_clusters} clusters...")
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        
        # Calculate distances to cluster centers
        distances = kmeans.transform(embeddings_scaled)
        
        # Add cluster assignment as one-hot encoding
        for i in range(n_clusters):
            cluster_feature = (cluster_labels == i).astype(float)
            kmeans_features.append(cluster_feature)
        
        # Add minimum distance to any cluster
        min_distances = distances.min(axis=1)
        kmeans_features.append(min_distances)
        
        print(f"   Added {n_clusters + 1} features (clusters + min distance)")
    
    kmeans_features = np.column_stack(kmeans_features)
    print(f"\n✅ Created {kmeans_features.shape[1]} K-means features")
    
    return kmeans_features

def create_sequence_features(sequences):
    """Create additional sequence-based features"""
    print("\n" + "="*60)
    print("Creating Sequence Features")
    print("="*60)
    
    features = []
    
    for seq in sequences:
        seq_features = {}
        
        # Length
        seq_features['length'] = len(seq)
        
        # Amino acid composition
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        for aa in aa_list:
            seq_features[f'aa_{aa}'] = seq.count(aa) / len(seq)
        
        # Hydrophobicity
        hydrophobic = 'AILMFWV'
        seq_features['hydrophobic_ratio'] = sum(seq.count(aa) for aa in hydrophobic) / len(seq)
        
        # Charged residues
        positive = 'KR'
        negative = 'DE'
        seq_features['positive_ratio'] = sum(seq.count(aa) for aa in positive) / len(seq)
        seq_features['negative_ratio'] = sum(seq.count(aa) for aa in negative) / len(seq)
        seq_features['charge_ratio'] = seq_features['positive_ratio'] - seq_features['negative_ratio']
        
        # Aromatic residues
        aromatic = 'FWY'
        seq_features['aromatic_ratio'] = sum(seq.count(aa) for aa in aromatic) / len(seq)
        
        features.append(seq_features)
    
    features_df = pd.DataFrame(features)
    print(f"✅ Created {features_df.shape[1]} sequence features")
    
    return features_df.values

def combine_features(embeddings, kmeans_features, sequence_features):
    """Combine all features"""
    print("\n" + "="*60)
    print("Combining Features")
    print("="*60)
    
    combined = np.hstack([embeddings, kmeans_features, sequence_features])
    
    print(f"ESM2 embeddings: {embeddings.shape[1]} features")
    print(f"K-means features: {kmeans_features.shape[1]} features")
    print(f"Sequence features: {sequence_features.shape[1]} features")
    print(f"Combined: {combined.shape[1]} features")
    
    return combined

def optimize_hyperparameters(X, y):
    """Find best hyperparameters using GridSearchCV"""
    print("\n" + "="*60)
    print("Optimizing Hyperparameters")
    print("="*60)
    
    # Calculate scale_pos_weight
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    scale_pos_weight = n_negative / n_positive
    
    # Parameter grid
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        device='cpu'
    )
    
    # Grid search with F1 scoring
    from sklearn.metrics import make_scorer
    f1_scorer = make_scorer(f1_score)
    
    print("Running GridSearchCV (this may take 10-20 minutes)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\nBest F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

def train_improved_model(X, y, best_params=None):
    """Train improved model with all enhancements"""
    print("\n" + "="*60)
    print("Training Improved Model")
    print("="*60)
    
    # Calculate scale_pos_weight
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    scale_pos_weight = n_negative / n_positive
    
    # Use best params or defaults
    if best_params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cpu'
        }
    else:
        params = {
            **best_params,
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
        
        print(f"   F1-score: {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
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
    
    # Train final model
    print("\n" + "="*60)
    print("Training Final Model on Full Dataset")
    print("="*60)
    
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)
    
    print("✅ Final model trained")
    
    return final_model, cv_df

def compare_models(baseline_results, improved_results):
    """Compare baseline vs improved model"""
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    metrics = ['f1', 'precision', 'recall', 'roc_auc']
    
    comparison = pd.DataFrame({
        'Metric': metrics,
        'Baseline': [baseline_results[m].mean() for m in metrics],
        'Improved': [improved_results[m].mean() for m in metrics],
        'Improvement': [improved_results[m].mean() - baseline_results[m].mean() for m in metrics]
    })
    
    print("\n", comparison.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, [baseline_results[m].mean() for m in metrics], 
           width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, [improved_results[m].mean() for m in metrics], 
           width, label='Improved', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300)
    print("\n✅ Comparison plot saved to results/model_comparison.png")
    
    return comparison

def main():
    """Main improvement pipeline"""
    print("\n" + "="*60)
    print("ESM2 Model Improvement Pipeline")
    print("="*60)
    
    # Load data
    embeddings, labels, sequences, merged = load_data()
    
    # Create K-means features
    kmeans_features = create_kmeans_features(embeddings)
    
    # Create sequence features
    sequence_features = create_sequence_features(sequences)
    
    # Combine all features
    X_improved = combine_features(embeddings, kmeans_features, sequence_features)
    
    # Option 1: Quick training with default enhanced parameters
    print("\n" + "="*60)
    print("Option 1: Quick Training (Recommended)")
    print("="*60)
    print("Training with enhanced features and good default parameters...")
    
    model_improved, cv_improved = train_improved_model(X_improved, labels)
    
    # Load baseline results for comparison
    baseline_cv = pd.read_csv('results/esm2_cv_results.csv')
    
    # Compare models
    comparison = compare_models(baseline_cv, cv_improved)
    
    # Save improved model
    model_improved.save_model('models/esm2_improved_classifier.json')
    cv_improved.to_csv('results/esm2_improved_cv_results.csv', index=False)
    comparison.to_csv('results/model_comparison.csv', index=False)
    
    print("\n" + "="*60)
    print("Improvement Complete!")
    print("="*60)
    print("\nFiles saved:")
    print("  • models/esm2_improved_classifier.json")
    print("  • results/esm2_improved_cv_results.csv")
    print("  • results/model_comparison.csv")
    print("  • results/model_comparison.png")
    
    # Option 2: Hyperparameter tuning (optional, takes longer)
    print("\n" + "="*60)
    print("Option 2: Hyperparameter Tuning (Optional)")
    print("="*60)
    print("Would you like to run hyperparameter tuning? (10-20 minutes)")
    print("If yes, uncomment the lines below in the script and re-run")
    print("="*60)
    
    # Run hyperparameter tuning
    best_params = optimize_hyperparameters(X_improved, labels)
    
    print("\n" + "="*60)
    print("Training with Tuned Hyperparameters")
    print("="*60)
    
    model_tuned, cv_tuned = train_improved_model(X_improved, labels, best_params)
    model_tuned.save_model('models/esm2_tuned_classifier.json')
    cv_tuned.to_csv('results/esm2_tuned_cv_results.csv', index=False)
    
    # Compare tuned vs improved
    print("\n" + "="*60)
    print("Tuned vs Improved Comparison")
    print("="*60)
    
    metrics = ['f1', 'precision', 'recall', 'roc_auc']
    tuned_comparison = pd.DataFrame({
        'Metric': metrics,
        'Improved': [cv_improved[m].mean() for m in metrics],
        'Tuned': [cv_tuned[m].mean() for m in metrics],
        'Improvement': [cv_tuned[m].mean() - cv_improved[m].mean() for m in metrics]
    })
    
    print("\n", tuned_comparison.to_string(index=False))
    tuned_comparison.to_csv('results/tuned_comparison.csv', index=False)
    
    print("\n✅ Tuned model saved to models/esm2_tuned_classifier.json")

if __name__ == "__main__":
    main()

