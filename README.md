# Terpene Synthase Classifier

A multi-modal deep learning system for predicting terpene synthase products using protein sequences, AlphaFold structures, and biochemical features.

## 🎯 Project Overview

This project develops production-ready classifiers for predicting terpene synthase (TPS) products from protein sequences. The system integrates multiple data modalities to achieve state-of-the-art performance on sparse multi-label classification tasks.

## 📊 Model Performance (V3 - Current Baseline)

**Test Set Results:**
- **Macro F1**: 38.74%
- **Macro Recall**: 77.52%
- **Macro Precision**: 32.30%
- **Classes**: 25/30 with data coverage

### Performance Evolution

| Stage | Architecture | F1 Score | Improvement |
|-------|-------------|----------|-------------|
| **ESM2 + Engineered** | Sequence only + adaptive thresholds | 8.57% | Baseline |
| **Multi-Modal (20D)** | + AlphaFold structures | 20.08% | +134.3% |
| **Enhanced (25D)** | + Physicochemical properties | 38.74% | +350.0% |

## 🏗️ Architecture

### V3 Model (TPS_Classifier_v3_Early)
- **ESM2 Embeddings**: 1280D protein language model → 256D
- **AlphaFold Structures**: Graph convolutional network on 3D structures → 256D
- **Engineered Features**: Biochemical/mechanistic features → 256D
- **Fusion**: Multi-modal late fusion with attention
- **Training**: Focal loss, adaptive thresholds, class weighting, mixed precision

### V4 Model (TPS_Classifier_v4_Enhanced) - In Development
- Enhanced predictor with stabilized training
- kNN soft voting for improved generalization
- Per-class calibration (Platt scaling)
- Hierarchy-aware masking

## 📁 Project Structure

```
Terpene_Classifier/
├── TPS_Classifier_v3_Early/          # V3 baseline model
│   ├── module8_functional_geometric_integration.py  # Main training
│   ├── complete_multimodal_classifier.py           # Architecture
│   ├── structural_graph_pipeline.py                # AlphaFold GCN
│   ├── adaptive_threshold_fix.py                   # Threshold optimization
│   └── TPS_Predictor.py                           # Feature generation
│
└── TPS_Classifier_v4_Enhanced/       # V4 development
    ├── tps/                          # Core modules
    ├── scripts/                      # Training & evaluation
    └── models/                       # Checkpoints
```

## 🚀 Key Features

### Multi-Modal Integration
- **ESM2**: State-of-the-art protein language model embeddings
- **AlphaFold**: Real predicted 3D structures with GCN processing
- **Engineered**: Domain-specific biochemical features
- **Ligand Binding**: Mg²⁺ cofactor and substrate binding site features

### Advanced Training
- **Focal Loss**: Handles extreme class imbalance
- **Adaptive Thresholds**: Per-class F1-optimized thresholds
- **Class Weighting**: Balanced learning across sparse labels
- **Mixed Precision**: Efficient GPU utilization

### Production Features
- **Identity-Aware Evaluation**: Performance on novel sequences (≤40% identity)
- **Bootstrap CI**: Robust confidence intervals
- **kNN Blending**: Soft voting for improved generalization
- **Hierarchy Masking**: Type-based prediction constraints

## 📈 Datasets

- **MARTS-DB**: Enhanced terpene synthase database (~1,276 sequences)
- **AlphaFold Structures**: 1,222 high-confidence structures (96% coverage)
- **Multiple Products**: Multi-label classification with 30 terpene classes

## 🔬 Next Steps

1. **Larger Dataset Training**: Retrain v3 model on expanded MARTS-DB
2. **Novel Sequence Evaluation**: Focus on ≤40% identity performance
3. **V4 Integration**: Combine v3's structural features with v4's stabilized training
4. **Production Deployment**: API and batch prediction tools

## 📝 Citation

Project developed for terpene synthase product prediction research.

## 🔗 Related Projects

- V3 Model: AlphaFold + GCN + Ligand features
- V4 Model: Stabilized training + kNN + calibration
- V2 Model: ESM2 binary classification (archived)

---

**Status**: Active development | **Latest Model**: V3 (38.74% test F1) | **Next**: Large-scale training
