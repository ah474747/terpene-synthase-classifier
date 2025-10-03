# Google Colab Training Instructions

Since the Jupyter notebook isn't loading properly in Colab, here's a step-by-step manual guide:

## Step 1: Setup Environment
In a new Colab cell, run:
```python
# Install dependencies
!pip install fair-esm torch scikit-learn pandas numpy seaborn matplotlib

# Import libraries
import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(f'PyTorch: {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
```

## Step 2: Upload Data Files
1. Go to Files tab in Colab (left sidebar)
2. Upload `colab_training_data.tar.gz` 
3. Run: `!tar -xzf colab_training_data.tar.gz`
4. Verify: `!ls data/`

## Step 3: Create ESM Embedder
```python
from typing import List, Optional
import esm

class ESMEmbedder:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or "esm2_t33_650M_UR50D"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
    def _ensure_model(self):
        if self.model is None:
            print(f"Loading ESM model: {self.model_id}")
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
    
    def embed_mean(self, seqs: List[str]):
        self._ensure_model()
        embeddings = []
        batch_size = 8
        
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            
            with torch.no_grad():
                if tokens.size(1) > 1024:
                    tokens = tokens[:, :1024]
                results = self.model(tokens, return_pairs=False)
                token_reprs = results["representations"][33]
                mean_emb = token_reprs.mean(dim=1).cpu().numpy()
                embeddings.append(mean_emb)
        
        return np.vstack(embeddings)
```

## Step 4: Multimodal Classifier
```python
import torch.nn as nn

class FinalMultiModalClassifier(nn.Module):
    def __init__(self, plm_dim: int = 1280, eng_dim: int = 24, struct_dim: int = 32, n_classes: int = 6):
        super().__init__()
        self.plm_enc = nn.Sequential(
            nn.Linear(plm_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128)
        )
        self.eng_enc = nn.Sequential(nn.Linear(eng_dim, 16), nn.ReLU())
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 20), nn.ReLU())
        self.final_head = nn.Linear(128 + 16 + 20, n_classes)
    
    def forward(self, plm_y: 128 + eng_feat + struct_feat, dim=1)
        concat_feat = torch.cat([plm_feat, eng_feat, struct_feat], dim=1)
        return self.final_head(concat_feat)
    def forward(self, plm_x, eng_x, struct_x):
        plm_feat = self.plm_enc(plm_x)
        eng_feat = self.eng_enc(eng_x)
        struct_feat = self.struct_enc(struct_x)
        concat_feat = torch.cat([plm_feat, eng_feat, struct_feat], dim=1)
        return self.final_head(concat_feat)
```

## Step 5: Engineered Features
```python
def _make_engineered_features(seqs: List[str]):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    eng = []
    
    for seq in seqs:
        # Amino acid composition
        aa_comp = [seq.count(aa) / len(seq) for aa in amino_acids]
        
        # Hydrophobicity index
        hydrophobicity = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
        avg_hydro = sum(hydrophobicity.get(aa, 0.0) for aa in seq) / len(seq)
        
        # Charge at pH 7
        pos_aa = sum(1 for aa in seq if aa in ['R', 'K', 'H'])
        neg_aa = sum(1 for aa in seq if aa in ['D', 'E'])
        charge = (pos_aa - neg_aa) / len(seq)
        
        feat_vec = aa_comp + [avg_hydro, charge]
        eng.append(feat_vec)
    
    return np.array(eng, dtype=np.float32)
```

## Step 6: Dataset Class
```python
from torch.utils.data import Dataset

class TPSDataset(Dataset):
    def __init__(self, fasta_path: str, labels_path: str, embedder: ESMEmbedder):
        # Load sequences
        self.seqs = []
        with open(fasta_path) as f:
            for line in f:
                if not line.startswith('>'):
                    self.seqs.append(line.strip())
        
        # Load labels
        self.labels = pd.read_csv(labels_path).values
        
        # Create embeddings
        print(f"Computing embeddings for {len(self.seqs)} sequences...")
        self.embeddings = embedder.embed_mean(self.seqs)
        print(f"✓ Embeddings computed: {self.embeddings.shape}")
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        plm = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        eng = torch.tensor(_make_engineered_features([self.seqs[idx]])[0], dtype=torch.float32)
        struct = torch.zeros(32, dtype=torch.float32)  # Placeholder
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return plm, eng, struct, label
```

## Step 7: Train Model
```python
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedder = ESMEmbedder(device=device)

# Load classes
with open('data/classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]
n_classes = len(classes)

# Create datasets
train_dataset = TPSDataset('data/train.fasta', 'data/train_labels.csv', embedder)
val_dataset = TPSDataset('data/val.fasta', 'data/val_labels_binary.csv', embedder)

# Data loaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
plm_dim = train_dataset.embeddings.shape[1]
model = FinalMultiModalClassifier(plm_dim=plm_dim, n_classes=n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print(f"Model initialized with plm_dim={plm_dim}")
```

## Step 8: Training Loop
```python
import torch.optim as optim
from sklearn.metrics import f1_score

def compute_macro_f1(predictions, targets):
    predictions_binary = (torch.sigmoid(predictions) > 0.35).float()
    f1_scores = []
    for class_idx in range(targets.shape[1]):
        pred_class = predictions_binary[:, class_idx]
        true_class = targets[:, class_idx]
        if true_class.sum() == 0:
            continue
        f1 = f1_score(true_class.cpu().numpy(), pred_class.cpu().numpy()).item()
        f1_scores.append(f1)
    return np.mean(f1_scores)

# Training parameters
epochs = 15
best_val_f1 = 0

print("Starting training...")
print("{'Epoch':<6} | {'Train Loss':<12} | {'Val Macro-F1':<15} | {'Best F1':<10}")
print("-" * 55)

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    
    for batch_idx, (plm_x, eng_x, struct_x, targets) in enumerate(train_loader):
        plm_x, eng_x, struct_x, targets = plm_x.to(device), eng_x.to(device), struct_x.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(plm_x, eng_x, struct_x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for plm_x, eng_x, struct_x, targets in val_loader:
            plm_x, eng_x, struct_x, targets = plm_x.to(device), eng_x.to(device), struct_x.to(device), targets.to(device)
            outputs = model(plm_x, eng_x, struct_x)
            all_preds.append(outputs)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    val_f1 = compute_macro_f1(all_preds, all_targets)
    
    # Track best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'state_dict': model.state_dict(),
            'plm_dim': plm_dim,
            'n_classes': n_classes,
            'classes': classes,
            'esm_model_id': embedder.model_id
        }, 'best_model.pth')
    
    print(f"{epoch+1:6d} | {avg_train_loss:12.4f} | {val_f1:15.4f} | {best_val_f1:10.4f}")

print(f"Training completed! Best validation F1: {best_val_f1:.4f}")
```

## Step 9: Evaluation
```python
# Load best model
best_state = torch.load('best_model.pth')
model.load_state_dict(best_state['state_dict'])

# Final evaluation
model.eval()
all_preds_probs = []
all_targets = []

with torch.no_grad():
    for plm_x, eng_x, struct_x, targets in val_loader:
        plm_x, eng_x, struct_x, targets = plm_x.to(device), eng_x.to(device), struct_x.to(device), targets.to(device)
        outputs = model(plm_x, eng_x, struct_x)
        probs = torch.sigmoid(outputs)
        all_preds_probs.append(probs)
        all_targets.append(targets)

all_probs = torch.cat(all_preds_probs, dim=0).cpu().numpy()
all_targets_np = torch.cat(all_targets, dim=0).cpu().numpy()

# Classification report
threshold = 0.35
predictions_binary = (all_probs >= threshold).astype(int)

report = classification_report(all_targets_np, predictions_binary, target_names=classes, zero_division=0)
print("Classification Report:")
print(report)

# Save results
results = {
    'classes': classes,
    'val_predictions': all_probs.tolist(),
    'val_targets': all_targets_np.tolist(),
    'best_val_f1': best_val_f1,
    'threshold': threshold
}

with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved!")
```

## Step 10: Download Results
```python
from google.colab import files

files.download('best_model.pth')
files.download('training_results.json')
```

## Expected Outcomes
- **Training time**: ~45-90 minutes on GPU
- **Performance**: >70% macro F1 on validation set
- **Output**: Trained model + evaluation metrics

This should give you a complete working terpene classifier trained on the enhanced MARTS dataset!
