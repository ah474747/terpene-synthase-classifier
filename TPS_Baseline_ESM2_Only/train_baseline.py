#!/usr/bin/env python3
"""
ESM2-Only Baseline for Terpene Synthase Classification
========================================================

A simple MLP baseline using pre-computed ESM2 embeddings to establish
a robust performance benchmark before adding AlphaFold structures.

This script:
1. Loads MARTS-DB enhanced dataset
2. Generates ESM2 embeddings (esm2_t33_650M_UR50D)
3. Runs 5-fold stratified cross-validation
4. Reports macro F1 scores with confidence intervals
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import sys
from tqdm import tqdm

# --- Configuration ---
DATA_FILE = '../TPS_Classifier_v3_Early/TS-GSD_consolidated.csv'
N_SPLITS = 5    # Number of folds for cross-validation
EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
N_CLASSES = 30  # Fixed number of functional ensemble classes

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"🔧 Device: {DEVICE}")
print(f"📊 Configuration: {N_SPLITS}-fold CV, {EPOCHS} epochs, batch size {BATCH_SIZE}")

# --- 1. Simple MLP Model Definition ---
class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for baseline classification"""
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.layer_stack(x)


# --- 2. ESM2 Embedding Generator ---
def generate_esm2_embeddings(sequences):
    """Generate ESM2 embeddings for sequences"""
    print("📥 Loading ESM2 model (esm2_t33_650M_UR50D)...")
    
    try:
        import esm
    except ImportError:
        print("❌ Error: fair-esm not installed. Install with: pip install fair-esm")
        sys.exit(1)
    
    # Load ESM2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(DEVICE)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    embeddings = []
    batch_size = 8  # Smaller batch for memory
    
    print(f"🧬 Computing embeddings for {len(sequences)} sequences...")
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]
            # Format: [(id, sequence), ...]
            batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
            
            # Convert to tokens
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(DEVICE)
            
            # Truncate to max length
            if batch_tokens.shape[1] > 1024:
                batch_tokens = batch_tokens[:, :1024]
            
            # Get representations
            results = model(batch_tokens, repr_layers=[33])
            
            # Extract and mean pool
            token_representations = results["representations"][33]
            # Mean over sequence length (excluding special tokens)
            sequence_embeddings = token_representations[:, 1:-1, :].mean(dim=1)
            
            embeddings.append(sequence_embeddings.cpu().numpy())
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    print(f"✅ Generated embeddings: shape {embeddings.shape}")
    
    return embeddings


# --- 3. Data Loading and Preprocessing ---
def load_and_process_data():
    """Loads and preprocesses the TS-GSD consolidated data"""
    print("📂 Loading TS-GSD consolidated dataset...")
    df = pd.read_csv(DATA_FILE)
    
    print(f"📊 Dataset: {len(df)} unique enzymes")
    print(f"   Columns: {list(df.columns)}")
    
    # Extract sequences
    sequences = df['aa_sequence'].tolist()
    print(f"✅ Loaded {len(sequences)} protein sequences")
    
    # Extract multi-label target vectors
    # Format: "[0, 0, 1, 0, ...]" as strings, need to parse
    target_vectors = []
    for target_str in df['target_vector'].tolist():
        # Parse string representation of list
        target_vec = eval(target_str)  # Safe here since data is from our pipeline
        target_vectors.append(target_vec)
    
    labels = np.array(target_vectors, dtype=np.float32)
    print(f"📊 Label matrix shape: {labels.shape} (N_samples × {N_CLASSES} classes)")
    
    # Analyze label distribution
    n_positives_per_class = labels.sum(axis=0)
    n_classes_per_sample = labels.sum(axis=1)
    
    print(f"\n📊 Multi-Label Statistics:")
    print(f"   Samples with labels: {(n_classes_per_sample > 0).sum()}/{len(labels)}")
    print(f"   Avg labels per sample: {n_classes_per_sample.mean():.2f}")
    print(f"   Max labels per sample: {int(n_classes_per_sample.max())}")
    print(f"   Classes with data: {(n_positives_per_class > 0).sum()}/{N_CLASSES}")
    print(f"   Samples per class (range): [{int(n_positives_per_class.min())}, {int(n_positives_per_class.max())}]")
    
    # Show top classes by frequency
    top_classes_idx = np.argsort(n_positives_per_class)[::-1][:10]
    print(f"\n🔝 Top 10 Most Frequent Classes:")
    for i, idx in enumerate(top_classes_idx):
        print(f"   {i+1}. Class {idx}: {int(n_positives_per_class[idx])} samples")
    
    # Generate ESM2 embeddings
    embeddings = generate_esm2_embeddings(sequences)
    
    # Save embeddings for future use
    embeddings_path = 'data/esm2_embeddings.npy'
    Path('data').mkdir(exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"💾 Saved embeddings to {embeddings_path}")
    
    # Save label info
    label_info = {
        'n_classes': N_CLASSES,
        'n_samples': len(labels),
        'positives_per_class': n_positives_per_class.tolist(),
        'labels_per_sample_mean': float(n_classes_per_sample.mean()),
        'labels_per_sample_max': int(n_classes_per_sample.max())
    }
    with open('data/label_info.json', 'w') as f:
        json.dump(label_info, f, indent=2)
    
    return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), N_CLASSES


# --- 4. Training Function ---
def train_one_fold(model, train_loader, val_X, val_y, optimizer, criterion, epochs):
    """Train model for one fold"""
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_X.to(DEVICE))
                val_preds = (torch.sigmoid(val_outputs) > 0.5).cpu().numpy()
                macro_f1 = f1_score(val_y.numpy(), val_preds, average='macro', zero_division=0)
                best_f1 = max(best_f1, macro_f1)
                print(f"      Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}, Val F1: {macro_f1:.4f}")
    
    return best_f1


# --- 5. Main Cross-Validation Loop ---
def run_cross_validation():
    """Runs the full stratified k-fold cross-validation"""
    print("\n" + "="*60)
    print("🚀 Starting ESM2-Only Baseline Training")
    print("="*60 + "\n")
    
    # Load data
    X, y, n_classes = load_and_process_data()
    
    print(f"\n📊 Data loaded:")
    print(f"   - Samples: {len(X)}")
    print(f"   - Embedding dim: {X.shape[1]}")
    print(f"   - Classes: {n_classes}")
    
    # Create stratification variable (use argmax for multi-label)
    y_stratify = y.argmax(axis=1).numpy()
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_stratify)):
        print(f"\n{'='*60}")
        print(f"📁 Fold {fold+1}/{N_SPLITS}")
        print(f"{'='*60}")
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize model
        model = SimpleMLP(input_dim=X.shape[1], output_dim=n_classes).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        
        # Train
        print(f"\n   🏋️  Training...")
        best_f1 = train_one_fold(model, train_loader, X_val, y_val, optimizer, criterion, EPOCHS)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(DEVICE))
            val_preds = (torch.sigmoid(val_outputs) > 0.5).cpu().numpy()
            
            # Calculate metrics
            macro_f1 = f1_score(y_val.numpy(), val_preds, average='macro', zero_division=0)
            micro_f1 = f1_score(y_val.numpy(), val_preds, average='micro', zero_division=0)
            macro_prec = precision_score(y_val.numpy(), val_preds, average='macro', zero_division=0)
            macro_rec = recall_score(y_val.numpy(), val_preds, average='macro', zero_division=0)
            
            fold_results.append({
                'fold': fold + 1,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'precision': macro_prec,
                'recall': macro_rec,
                'best_f1': best_f1
            })
            
            print(f"\n   ✅ Fold {fold+1} Results:")
            print(f"      Macro F1:   {macro_f1:.4f}")
            print(f"      Micro F1:   {micro_f1:.4f}")
            print(f"      Precision:  {macro_prec:.4f}")
            print(f"      Recall:     {macro_rec:.4f}")
    
    # Aggregate results
    print(f"\n" + "="*60)
    print("📊 CROSS-VALIDATION RESULTS")
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
    
    # Save results
    results_summary = {
        'configuration': {
            'model': 'Simple MLP (512 hidden)',
            'embeddings': 'ESM2 (esm2_t33_650M_UR50D)',
            'n_folds': N_SPLITS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'n_classes': n_classes
        },
        'cross_validation_results': {
            'macro_f1_mean': float(np.mean(macro_f1_scores)),
            'macro_f1_std': float(np.std(macro_f1_scores)),
            'micro_f1_mean': float(np.mean(micro_f1_scores)),
            'precision_mean': float(np.mean(precision_scores)),
            'recall_mean': float(np.mean(recall_scores)),
            'fold_results': fold_results
        }
    }
    
    Path('results').mkdir(exist_ok=True)
    results_path = 'results/baseline_cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    print(f"\n{'='*60}")
    print(f"✅ Baseline training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_cross_validation()

