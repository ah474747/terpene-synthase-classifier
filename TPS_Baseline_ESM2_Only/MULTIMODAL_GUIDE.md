# Multi-Modal Architecture Guide

## 🎯 Objective

Upgrade from Enhanced Baseline (19.15% F1) to Full Multi-Modal (Target: ~39% F1) by adding AlphaFold structural features via Graph Convolutional Networks.

## 🏗️ Architecture

### Three Parallel Branches

```
Input Layer:
├── ESM2 Embeddings (1280D)        →  PLMEncoder       →  256D
├── Engineered Features (64D)       →  FeatureEncoder   →  256D
└── Protein Graphs (N×30D nodes)    →  GCNEncoder       →  256D

Fusion Layer:
└── Concatenate [256+256+256]       →  768D
    ↓
    MLP Fusion (768→512→256)
    ↓
    Classifier (256→30)
```

### Key Components

**1. GCN Encoder**
- Input: Protein graph with 30D node features (20D amino acids + 10D functional)
- 3-layer GCN with message passing
- Global mean pooling → single 256D vector per protein
- Handles variable-size graphs

**2. Focal Loss**
- Formula: `FL(p_t) = -α(1-p_t)^γ * log(p_t)`
- Parameters: α=0.25, γ=2.0
- Focuses learning on hard examples
- Down-weights easy examples

**3. Inverse-Frequency Class Weighting**
- Weight per class: `w_i = N_total / (N_classes × N_i)`
- Normalized to mean=1.0
- Applied within Focal Loss
- Handles extreme imbalance (0-83 samples/class)

## 📊 Training Strategy

### 5-Fold Cross-Validation
- Stratified split on primary label
- 90% train, 10% threshold-tuning, validation
- Per-class adaptive thresholds (F1-optimized)
- 50 epochs per fold

### Optimization
- AdamW optimizer (LR=1e-4, weight_decay=1e-5)
- Batch size: 8 (smaller for graphs)
- Xavier initialization for all linear layers

## 🔍 Expected Improvements

### From Enhanced Baseline (19.15% F1)

**Adding Structural Features:**
- V3 showed: 8.57% → 38.74% F1 with AlphaFold
- Expected: **19.15% → ~35-40% F1**
- Structural features provide largest gain

**With Advanced Loss:**
- Focal Loss + Class Weighting
- Better handling of rare classes
- Expected additional: **+2-5% F1**

## 📝 Usage

### Prerequisites
```bash
# Ensure you have:
- data/esm2_embeddings.npy (1273 × 1280)
- data/engineered_features.npy (1273 × 64)
- data/functional_graphs.pkl (protein graphs)
```

### Run Training
```bash
python train_multimodal.py
```

### Note on Graphs
If functional_graphs.pkl loading fails (pickling issues), the script will create placeholder graphs. For full performance, real AlphaFold graphs are needed.

## 📈 Comparison Table

| Model | Features | Loss | Thresholds | F1 Score |
|-------|----------|------|------------|----------|
| Simple Baseline | ESM2 | BCE | Fixed 0.5 | 0.66% |
| Enhanced Baseline | ESM2+Eng | BCE | Adaptive | 19.15% |
| **Multi-Modal** | **ESM2+Eng+GCN** | **Focal** | **Adaptive** | **~35-40%** ⭐ |
| V3 Reference | ESM2+Eng+GCN | Focal | Adaptive | 38.74% |

## 🔬 Model Parameters

- **Total trainable params**: ~2.8M
  - PLM Encoder: ~1.05M
  - Eng Encoder: ~0.04M
  - GCN Encoder: ~0.13M
  - Fusion + Classifier: ~1.58M

## ⚙️ Hyperparameters

```python
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Focal Loss
ALPHA = 0.25
GAMMA = 2.0

# Architecture
LATENT_DIM = 256
GCN_HIDDEN = 128
GCN_LAYERS = 3
```

## 📁 Outputs

- `results/multimodal_cv_results.json` - Full CV results
- Per-fold macro F1, micro F1, precision, recall
- Optimal thresholds per fold
- Class weights and configuration

---

**Target**: Match or exceed V3's 38.74% F1 performance with full multi-modal integration!

