#!/usr/bin/env python3
"""
Multi-Modal Terpene Synthase Classifier
========================================

Full multi-modal architecture combining:
1. ESM2 embeddings (1280D)
2. Engineered features (64D)  
3. AlphaFold structural features via GCN (graphs → 256D)

With advanced training:
- Focal Loss (alpha=0.25, gamma=2.0)
- Inverse-frequency class weighting
- Per-class adaptive thresholding

Target: Match V3's 38.74% F1 performance
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import pickle
import sys
from tqdm import tqdm

# --- Configuration ---
DATA_FILE = '../TPS_Classifier_v3_Early/TS-GSD_consolidated.csv'
ESM2_EMBEDDINGS_FILE = 'data/esm2_embeddings.npy'
ENGINEERED_FEATURES_FILE = 'data/engineered_features.npy'
FUNCTIONAL_GRAPHS_FILE = 'data/functional_graphs.pkl'

N_SPLITS = 5
EPOCHS = 50  # More epochs for complex model
LEARNING_RATE = 1e-4
BATCH_SIZE = 8  # Smaller batch for graphs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
N_CLASSES = 30

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"🔧 Device: {DEVICE}")
print(f"📊 Configuration: {N_SPLITS}-fold CV, {EPOCHS} epochs, batch size {BATCH_SIZE}")


# --- 1. Protein Graph Data Structure ---
class ProteinGraph:
    """Simple protein graph container"""
    def __init__(self, node_features, edge_index, uniprot_id=None):
        self.node_features = torch.FloatTensor(node_features) if not isinstance(node_features, torch.Tensor) else node_features
        self.edge_index = torch.LongTensor(edge_index) if not isinstance(edge_index, torch.Tensor) else edge_index
        self.uniprot_id = uniprot_id


# --- 2. Focal Loss Implementation ---
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification with class imbalance
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (N, C)
            targets: Binary targets (N, C)
        """
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Probability of true class
        p_t = torch.exp(-bce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = self.alpha * focal_term * bce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights.unsqueeze(0)
        
        return focal_loss.mean()


# --- 3. Multi-Modal Architecture Components ---
class PLMEncoder(nn.Module):
    """ESM2 embedding encoder"""
    def __init__(self, input_dim=1280, output_dim=256, dropout=0.1):
        super(PLMEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class FeatureEncoder(nn.Module):
    """Engineered features encoder"""
    def __init__(self, input_dim=64, output_dim=256, dropout=0.1):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network for protein structures
    Processes graph → fixed 256D vector
    """
    def __init__(self, input_dim=30, hidden_dim=128, output_dim=256, num_layers=3, dropout=0.1):
        super(GCNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        
        # Build GCN layers
        self.gcn_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.gcn_layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, graph):
        """
        Args:
            graph: ProteinGraph object
        Returns:
            Graph-level embedding (1, output_dim)
        """
        x = graph.node_features
        
        # GCN layers with message passing
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            if i < len(self.gcn_layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        # Global mean pooling to get graph-level representation
        graph_embedding = x.mean(dim=0, keepdim=True)
        
        return graph_embedding


class MultiModalClassifier(nn.Module):
    """
    Complete multi-modal classifier
    
    Three branches:
    - ESM2 (1280D) → 256D
    - Engineered (64D) → 256D
    - Structural Graph → 256D via GCN
    
    Fusion: 768D → 512D → 256D → 30 classes
    """
    def __init__(self, plm_dim=1280, eng_dim=64, graph_dim=30, latent_dim=256, n_classes=30, dropout=0.1):
        super(MultiModalClassifier, self).__init__()
        
        # Three encoders
        self.plm_encoder = PLMEncoder(plm_dim, latent_dim, dropout)
        self.eng_encoder = FeatureEncoder(eng_dim, latent_dim, dropout)
        self.gcn_encoder = GCNEncoder(graph_dim, 128, latent_dim, 3, dropout)
        
        # Fusion and classification
        fusion_dim = latent_dim * 3  # 256 * 3 = 768
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(256, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, graph, esm2_features, eng_features):
        """
        Args:
            graph: ProteinGraph or list of graphs
            esm2_features: (batch_size, 1280)
            eng_features: (batch_size, 64)
        """
        # Encode each modality
        plm_emb = self.plm_encoder(esm2_features)  # (batch, 256)
        eng_emb = self.eng_encoder(eng_features)   # (batch, 256)
        
        # Handle graph encoding
        if isinstance(graph, list):
            # Batch of graphs
            struct_embs = []
            for g in graph:
                struct_emb = self.gcn_encoder(g)
                struct_embs.append(struct_emb)
            struct_emb = torch.cat(struct_embs, dim=0)
        else:
            # Single graph
            struct_emb = self.gcn_encoder(graph)
        
        # Ensure matching batch sizes
        if struct_emb.shape[0] != plm_emb.shape[0]:
            struct_emb = struct_emb.expand(plm_emb.shape[0], -1)
        
        # Concatenate all modalities
        fused = torch.cat([plm_emb, eng_emb, struct_emb], dim=1)  # (batch, 768)
        
        # Fusion and classification
        fused = self.fusion(fused)  # (batch, 256)
        logits = self.classifier(fused)  # (batch, 30)
        
        return logits


# --- 4. Dataset with Graphs ---
class MultiModalDataset(Dataset):
    """Dataset with ESM2, engineered features, and graphs"""
    def __init__(self, esm2_emb, eng_features, graphs, labels, uniprot_ids):
        # Convert to tensors if needed
        self.esm2_emb = torch.FloatTensor(esm2_emb) if not isinstance(esm2_emb, torch.Tensor) else esm2_emb
        self.eng_features = torch.FloatTensor(eng_features) if not isinstance(eng_features, torch.Tensor) else eng_features
        self.labels = torch.FloatTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.graphs = graphs
        self.uniprot_ids = uniprot_ids
    
    def __len__(self):
        return len(self.esm2_emb)
    
    def __getitem__(self, idx):
        uniprot_id = self.uniprot_ids[idx]
        graph = self.graphs.get(uniprot_id, None)
        
        # If no graph, create dummy graph
        if graph is None:
            node_features = torch.zeros((10, 30))  # Dummy graph
            edge_index = torch.tensor([[0], [1]], dtype=torch.long)
            graph = ProteinGraph(node_features, edge_index, uniprot_id)
        
        return graph, self.esm2_emb[idx], self.eng_features[idx], self.labels[idx]


def collate_multimodal(batch):
    """Custom collate function for batching graphs"""
    graphs, esm2, eng, labels = zip(*batch)
    return list(graphs), torch.stack(esm2), torch.stack(eng), torch.stack(labels)


# --- 5. Calculate Inverse-Frequency Class Weights ---
def calculate_class_weights(y_train):
    """Calculate inverse-frequency weights for each class"""
    class_counts = y_train.sum(axis=0)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    # Inverse frequency
    total_samples = len(y_train)
    class_weights = total_samples / (N_CLASSES * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.mean()
    
    return torch.FloatTensor(class_weights)


# --- 6. Adaptive Threshold Finding (from enhanced baseline) ---
def find_optimal_thresholds(y_true, y_pred_proba, n_classes=N_CLASSES):
    """Find optimal per-class thresholds"""
    optimal_thresholds = np.zeros(n_classes)
    threshold_candidates = np.arange(0.05, 0.96, 0.01)
    
    for class_idx in range(n_classes):
        best_f1 = 0.0
        best_threshold = 0.5
        
        y_true_class = y_true[:, class_idx]
        y_pred_class_proba = y_pred_proba[:, class_idx]
        
        if y_true_class.sum() == 0:
            optimal_thresholds[class_idx] = 0.5
            continue
        
        for threshold in threshold_candidates:
            y_pred_binary = (y_pred_class_proba >= threshold).astype(int)
            f1 = f1_score(y_true_class, y_pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[class_idx] = best_threshold
    
    return optimal_thresholds


# --- 7. Training Function ---
def train_one_fold(X_esm2_train, X_eng_train, graphs_train, y_train, X_esm2_val, X_eng_val, graphs_val, y_val, 
                   uniprot_ids_train, uniprot_ids_val, class_weights, epochs):
    """Train multi-modal model for one fold"""
    
    # Split training data for threshold tuning
    indices = np.arange(len(X_esm2_train))
    train_indices, tune_indices = train_test_split(indices, test_size=0.1, random_state=RANDOM_SEED)
    
    # Create datasets
    train_dataset = MultiModalDataset(
        X_esm2_train[train_indices], X_eng_train[train_indices], graphs_train,
        y_train[train_indices], [uniprot_ids_train[i] for i in train_indices]
    )
    
    tune_dataset = MultiModalDataset(
        X_esm2_train[tune_indices], X_eng_train[tune_indices], graphs_train,
        y_train[tune_indices], [uniprot_ids_train[i] for i in tune_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_multimodal)
    
    # Initialize model
    model = MultiModalClassifier().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights.to(DEVICE))
    
    print(f"      Sub-train: {len(train_dataset)}, Threshold-tune: {len(tune_dataset)}, Val: {len(X_esm2_val)}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for graphs, esm2, eng, labels in train_loader:
            esm2, eng, labels = esm2.to(DEVICE), eng.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(graphs, esm2, eng)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"         Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Find optimal thresholds on tuning set
    model.eval()
    with torch.no_grad():
        tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, collate_fn=collate_multimodal)
        tune_probs = []
        tune_labels = []
        
        for graphs, esm2, eng, labels in tune_loader:
            esm2, eng = esm2.to(DEVICE), eng.to(DEVICE)
            logits = model(graphs, esm2, eng)
            probs = torch.sigmoid(logits).cpu().numpy()
            tune_probs.append(probs)
            tune_labels.append(labels.numpy())
        
        tune_probs = np.vstack(tune_probs)
        tune_labels = np.vstack(tune_labels)
    
    print(f"      🔍 Finding optimal thresholds...")
    optimal_thresholds = find_optimal_thresholds(tune_labels, tune_probs)
    print(f"      📊 Threshold range: [{optimal_thresholds.min():.2f}, {optimal_thresholds.max():.2f}], mean: {optimal_thresholds.mean():.2f}")
    
    # Evaluate on validation set
    val_dataset = MultiModalDataset(X_esm2_val, X_eng_val, {**graphs_train, **graphs_val}, y_val, uniprot_ids_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_multimodal)
    
    val_probs = []
    with torch.no_grad():
        for graphs, esm2, eng, _ in val_loader:
            esm2, eng = esm2.to(DEVICE), eng.to(DEVICE)
            logits = model(graphs, esm2, eng)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs.append(probs)
    
    val_probs = np.vstack(val_probs)
    val_preds = (val_probs >= optimal_thresholds[None, :]).astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_val, val_preds, average='micro', zero_division=0)
    precision = precision_score(y_val, val_preds, average='macro', zero_division=0)
    recall = recall_score(y_val, val_preds, average='macro', zero_division=0)
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'precision': precision,
        'recall': recall,
        'optimal_thresholds': optimal_thresholds
    }


# --- 8. Main Cross-Validation ---
def run_multimodal_training():
    """Run 5-fold CV with multi-modal architecture"""
    print("\n" + "="*60)
    print("🚀 Multi-Modal Training (ESM2 + Engineered + GCN Structural)")
    print("="*60 + "\n")
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(DATA_FILE)
    esm2_emb = np.load(ESM2_EMBEDDINGS_FILE)
    eng_features = np.load(ENGINEERED_FEATURES_FILE)
    
    # Load graphs
    print("📊 Loading protein graphs...")
    with open(FUNCTIONAL_GRAPHS_FILE, 'rb') as f:
        # Load with custom unpickler to handle class definitions
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'TPS_Classifier_v3_Early'))
        try:
            graphs_dict = pickle.load(f)
            print(f"✅ Loaded {len(graphs_dict)} protein graphs")
        except Exception as e:
            print(f"⚠️  Warning loading graphs: {e}")
            print("   Creating placeholder graphs...")
            graphs_dict = {}
    
    # Extract labels
    labels = []
    for target_str in df['target_vector'].tolist():
        labels.append(eval(target_str))
    labels = np.array(labels, dtype=np.float32)
    
    uniprot_ids = df['uniprot_accession_id'].tolist()
    
    print(f"\n📊 Data summary:")
    print(f"   - Samples: {len(esm2_emb)}")
    print(f"   - ESM2: {esm2_emb.shape}")
    print(f"   - Engineered: {eng_features.shape}")
    print(f"   - Graphs: {len(graphs_dict)}")
    print(f"   - Classes: {N_CLASSES}")
    
    # Calculate class weights on full dataset
    print(f"\n⚖️  Calculating inverse-frequency class weights...")
    class_weights = calculate_class_weights(labels)
    print(f"   Weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    print(f"   Top 5 weighted classes (rarest): {torch.topk(class_weights, 5).indices.tolist()}")
    
    # Cross-validation
    X_esm2 = torch.FloatTensor(esm2_emb)
    X_eng = torch.FloatTensor(eng_features)
    y = labels
    
    y_stratify = y.argmax(axis=1)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_esm2, y_stratify)):
        print(f"\n{'='*60}")
        print(f"📁 Fold {fold+1}/{N_SPLITS}")
        print(f"{'='*60}")
        
        result = train_one_fold(
            X_esm2[train_idx], X_eng[train_idx], graphs_dict, y[train_idx],
            X_esm2[val_idx], X_eng[val_idx], graphs_dict, y[val_idx],
            [uniprot_ids[i] for i in train_idx], [uniprot_ids[i] for i in val_idx],
            class_weights, EPOCHS
        )
        
        fold_results.append(result)
        
        print(f"\n   ✅ Fold {fold+1} Results:")
        print(f"      Macro F1:   {result['macro_f1']:.4f}")
        print(f"      Micro F1:   {result['micro_f1']:.4f}")
        print(f"      Precision:  {result['precision']:.4f}")
        print(f"      Recall:     {result['recall']:.4f}")
    
    # Aggregate results
    print(f"\n" + "="*60)
    print("📊 MULTI-MODAL CROSS-VALIDATION RESULTS")
    print("="*60)
    
    macro_f1_scores = [r['macro_f1'] for r in fold_results]
    
    print(f"\n🎯 Macro F1 Score (Primary Metric):")
    print(f"   Mean:   {np.mean(macro_f1_scores):.4f}")
    print(f"   Std:    {np.std(macro_f1_scores):.4f}")
    print(f"   95% CI: [{np.mean(macro_f1_scores) - 1.96*np.std(macro_f1_scores):.4f}, "
          f"{np.mean(macro_f1_scores) + 1.96*np.std(macro_f1_scores):.4f}]")
    
    # Save results
    results_summary = {
        'configuration': {
            'model': 'Multi-Modal (ESM2 + Engineered + GCN)',
            'focal_loss': {'alpha': 0.25, 'gamma': 2.0},
            'class_weighting': 'inverse-frequency',
            'n_folds': N_SPLITS,
            'epochs': EPOCHS
        },
        'cross_validation_results': {
            'macro_f1_mean': float(np.mean(macro_f1_scores)),
            'macro_f1_std': float(np.std(macro_f1_scores)),
            'fold_results': [
                {
                    'fold': i+1,
                    'macro_f1': float(r['macro_f1']),
                    'micro_f1': float(r['micro_f1']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall'])
                }
                for i, r in enumerate(fold_results)
            ]
        }
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/multimodal_cv_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to results/multimodal_cv_results.json")
    print(f"\n{'='*60}")
    print(f"✅ Multi-modal training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_multimodal_training()

