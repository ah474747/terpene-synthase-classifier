# Technical Overview: Multi-Modal Terpene Synthase Classifier

## 🏗️ **System Architecture**

### **High-Level Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Modal TPS Classifier                   │
├─────────────────────────────────────────────────────────────────┤
│  Input: UniProt ID + Amino Acid Sequence                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Feature Generation Pipeline                   │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │ │
│  │  │ ESM2 Stream │  │ Engineered  │  │ GCN Stream  │         │ │
│  │  │   (PLM)     │  │   Stream    │  │(Structure)  │         │ │
│  │  │             │  │             │  │             │         │ │
│  │  │ 1280D → 256D│  │  64D → 256D │  │ 30D → 256D  │         │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │ │
│  │         │                │                │                │ │
│  │         └────────────────┼────────────────┘                │ │
│  │                          │                                 │ │
│  │                  Fusion Layer                              │ │
│  │                  768D → 512D → 256D                       │ │
│  │                          │                                 │ │
│  │                  Classifier Head                           │ │
│  │                  256D → 30D (Sigmoid)                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                       │
│                  Adaptive Thresholding                           │
│                          │                                       │
│  Output: 30 Functional Ensemble Predictions                     │
└─────────────────────────────────────────────────────────────────┘
```

### **Component Details**

#### **1. ESM2 Stream (Protein Language Model)**
- **Model**: `facebook/esm2_t33_650M_UR50D`
- **Input**: Raw amino acid sequence
- **Processing**: Tokenization → Transformer layers → Mean pooling
- **Output**: 1,280-dimensional protein embedding
- **Encoder**: Linear(1280) → ReLU → Dropout → Linear(256)

#### **2. Engineered Features Stream**
- **Components**:
  - One-hot encoded terpene type (mono, sesq, di, tri, pt, etc.)
  - One-hot encoded enzyme class (1, 2)
  - Simulated structural placeholders (pocket volume, domain counts)
- **Total Dimensions**: 64
- **Encoder**: Linear(64) → ReLU → Linear(256)

#### **3. GCN Stream (Structural Features)**
- **Structure Source**: AlphaFold predicted structures (EBI AlphaFold DB)
- **Graph Construction**:
  - Nodes: Protein residues (C-alpha atoms)
  - Edges: Spatial contacts (distance < 8.0 Å)
  - Node Features: 30D (20D one-hot AA + 5D physicochemical + 5D functional)
- **Ligand Integration**: 4 additional nodes (3×Mg²⁺, 1×substrate)
- **GCN Architecture**: 3-layer Graph Convolutional Network
- **Output**: 256-dimensional structural embedding

## 🧠 **Model Architecture Details**

### **Complete Multi-Modal Classifier**
```python
class FinalMultiModalClassifier(nn.Module):
    def __init__(self, plm_dim=1280, eng_dim=64, latent_dim=256, n_classes=30):
        # Three encoders
        self.plm_encoder = PLMEncoder(plm_dim, latent_dim)
        self.feature_encoder = FeatureEncoder(eng_dim, latent_dim)
        self.structural_encoder = FinalGCNEncoder(input_dim=30, hidden_dim=128, output_dim=256)
        
        # Fusion architecture
        self.fusion_dim = 768  # latent_dim * 3
        self.fusion_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
            nn.Sigmoid()
        )
```

### **GCN Architecture**
```python
class FinalGCNEncoder(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=128, output_dim=256, num_layers=3):
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.gcn_layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
```

## 🎯 **Training Strategy**

### **Loss Function: Weighted Focal Loss**
```python
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # Inverse-frequency class weights
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
```

### **Training Configuration**
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Batch Size**: 16 (with gradient accumulation of 4 steps)
- **Mixed Precision**: torch.cuda.amp for memory efficiency
- **Early Stopping**: Based on validation Macro F1 score
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Regularization**: Dropout (0.3), weight decay

### **Adaptive Thresholding**
```python
def find_optimal_thresholds(y_true, y_pred_proba, threshold_range):
    """Find optimal threshold for each class to maximize F1 score"""
    optimal_thresholds = np.zeros(y_true.shape[1])
    
    for class_idx in range(y_true.shape[1]):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in threshold_range:
            y_pred_binary = (y_pred_proba[:, class_idx] >= threshold).astype(int)
            f1 = f1_score(y_true[:, class_idx], y_pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[class_idx] = best_threshold
    
    return optimal_thresholds
```

## 📊 **Data Pipeline**

### **Dataset Structure**
```
TS-GSD (Terpene Synthase Gold Standard Dataset)
├── 1,273 unique enzymes
├── 30 functional ensembles
├── Multi-label targets (each enzyme can have multiple active ensembles)
└── Features:
    ├── ESM2 embeddings (1,280D)
    ├── Engineered features (64D)
    └── Structural graphs (variable size, 30D node features)
```

### **Functional Ensemble Mapping**
```python
functional_ensembles = {
    # Monoterpenes (0-9)
    'mono_pinene': 0,           # α-pinene, β-pinene
    'mono_limonene': 1,         # limonene
    'mono_myrcene': 2,          # myrcene
    # ... (30 total ensembles)
    
    # Sesquiterpenes (10-19)
    'sesq_germacrane': 10,      # germacrene A, D, etc.
    'sesq_caryophyllane': 11,   # caryophyllene
    
    # Diterpenes (20-24)
    'di_kaurane': 20,           # kaurene
    
    # Triterpenes (25-27)
    'tri_squalene': 25,         # squalene
    
    # Specialized (28-29)
    'specialized_oxygenated': 28,
    'specialized_cyclic': 29
}
```

## 🔧 **Production Deployment**

### **Inference Pipeline**
```python
def predict_ensemble(uniprot_id, sequence, annotation=None):
    # 1. Generate ESM2 features
    e_plm = generate_esm2_features(sequence)
    
    # 2. Generate engineered features
    e_eng = generate_engineered_features(terpene_type, enzyme_class)
    
    # 3. Generate GCN features (with AlphaFold structure)
    gcn_graph, has_structure = generate_gcn_features(sequence, uniprot_id)
    
    # 4. Run model inference
    with torch.no_grad():
        plm_latent = model.plm_encoder(e_plm)
        eng_latent = model.feature_encoder(e_eng)
        gcn_latent = model.structural_encoder(gcn_graph)
        
        # Fusion
        fused = torch.cat([plm_latent, eng_latent, gcn_latent], dim=1)
        fused = model.fusion_layer(fused)
        
        # Prediction
        predictions = model.classifier(fused)
    
    # 5. Apply adaptive thresholds
    binary_predictions = (predictions >= optimal_thresholds).astype(int)
    
    return {
        'probabilities': predictions,
        'binary_predictions': binary_predictions,
        'top_predictions': get_top_k_predictions(predictions, k=3)
    }
```

### **AlphaFold Integration**
```python
def _download_alphafold_structure(uniprot_id):
    """Download AlphaFold structure from EBI AlphaFold DB"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Save and parse PDB file
    pdb_path = f"alphafold_structures/pdb/AF-{uniprot_id}-F1-model_v4.pdb"
    with open(pdb_path, 'w') as f:
        f.write(response.text)
    
    return parse_alphafold_pdb(pdb_path)
```

## 📈 **Performance Metrics**

### **Evaluation Metrics**
- **Macro F1 Score**: Primary metric for multi-label classification
- **Micro F1 Score**: Overall performance across all classes
- **Precision@K**: How often true labels are in top-K predictions
- **Mean Average Precision (mAP)**: Ranking quality metric
- **Per-Class F1**: Individual class performance analysis

### **Validation Strategy**
1. **5-Fold Cross-Validation**: Train/validation splits with proper stratification
2. **External Generalization**: 30 external UniProt sequences
3. **Statistical Significance**: Confidence intervals and error analysis
4. **Ablation Studies**: Component-wise performance analysis

## 🛠️ **Technical Implementation**

### **Key Dependencies**
```python
# Core ML/AI
torch>=1.12.0
torch-geometric>=2.0.0
transformers>=4.20.0
scikit-learn>=1.1.0

# Bioinformatics
biopython>=1.79
requests>=2.28.0

# Data Processing
numpy>=1.21.0
pandas>=1.4.0
```

### **Hardware Requirements**
- **Training**: GPU with 8GB+ VRAM (recommended: RTX 3080/4080 or better)
- **Inference**: CPU sufficient for single predictions, GPU for batch processing
- **Storage**: 10GB+ for models, datasets, and AlphaFold structures
- **Memory**: 16GB+ RAM for full dataset processing

### **Performance Optimizations**
- **Mixed Precision Training**: torch.cuda.amp for 2x speedup
- **Gradient Accumulation**: Simulate larger batch sizes
- **Structure Caching**: Local storage of downloaded AlphaFold files
- **Batch Processing**: Efficient ESM2 embedding generation
- **Memory Management**: Careful tensor handling and cleanup

## 🔍 **Code Quality Standards**

### **Code Organization**
- **Modular Design**: Separate concerns (data, models, training, deployment)
- **Type Hints**: Full type annotation for better maintainability
- **Documentation**: Comprehensive docstrings and inline comments
- **Error Handling**: Robust exception handling with graceful fallbacks
- **Logging**: Structured logging for debugging and monitoring

### **Testing Strategy**
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Benchmarking and regression testing
- **Validation Tests**: Scientific correctness verification

---

**Technical Overview Version**: 1.0  
**Generated**: $(date)  
**Architecture Status**: Production Ready



