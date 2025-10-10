# Terpene Synthase Product Classifier: Model Iterations Summary

This document summarizes all iterations of the terpene synthase product classifier model development, including training data, methodologies, and best results.

## 📊 Overview Table

| Version | Dataset | Size | Classes | Architecture | Methodology | Macro F1 | Notes |
|---------|---------|------|---------|--------------|-------------|----------|-------|
| **v0.1 - Simple Baseline** | TS-GSD consolidated | 1,273 seqs | 30 (multi-label) | ESM2 + Simple MLP | ESM2 embeddings (1280D) → 2-layer MLP (512) → 30 classes. Fixed 0.5 threshold, BCE loss | **0.66%** | Baseline established; poor performance due to fixed threshold |
| **v0.2 - Enhanced Baseline** | TS-GSD consolidated | 1,273 seqs | 30 (multi-label) | ESM2 + Engineered + MLP | ESM2 (1280D) + Engineered features (64D) → 3-layer MLP (512→256) → 30 classes. Adaptive per-class thresholds | **19.15%** | 29× improvement; adaptive thresholds crucial |
| **v0.3 - Multi-Modal (Placeholder)** | TS-GSD consolidated | 1,273 seqs | 30 (multi-label) | ESM2 + Engineered + GCN | Three-branch fusion: ESM2 (→256D) + Engineered (→256D) + GCN placeholder graphs (→256D). Focal Loss (α=0.25, γ=2.0), inverse-freq weighting | **32.87%** | 72% improvement over v0.2; focal loss effective |
| **v0.3.1 - Multi-Modal (Real Graphs)** | TS-GSD consolidated | 1,273 seqs | 30 (multi-label) | ESM2 + Engineered + GCN | Same as v0.3 but with real AlphaFold graphs (1,222 structures, 96% coverage). 3-layer GCN with 30D node features | **32.94%** | Only +0.07% vs placeholder; ESM2 likely encodes structure |
| **V3 Phase 1 - ESM2+Engineered** | MARTS-DB | 1,273 seqs | 30 (multi-label) | ESM2 + Engineered | ESM2 (1280D) + Engineered (64D) with adaptive thresholds | **8.57%** | V3 baseline; lower than v0.2 likely due to different data split |
| **V3 Phase 2 - Multi-Modal (20D)** | MARTS-DB | 1,273 seqs | 30 (multi-label) | ESM2 + AlphaFold GCN + Engineered | Three-branch fusion with 20D node features (amino acid one-hot) | **20.08%** | +134% improvement; GCN+AlphaFold introduced together |
| **V3 Phase 3 - Enhanced (25D)** | MARTS-DB | 1,273 seqs | 30 (multi-label) | ESM2 + AlphaFold GCN + Engineered | Enhanced with 25D node features (20D AA + 5D physicochemical) | **38.56%** | +92% improvement; richer node features |
| **V3 Phase 4 - Functional (30D)** | MARTS-DB | 1,273 seqs | 30 (multi-label) | ESM2 + AlphaFold GCN + Engineered | Final version with 30D node features (25D + 5D ligand/cofactor Mg²⁺, FPP/GPP/DMAPP) | **40.19% (test: 40.59%)** | +4% improvement; functional geometry integration |
| **Clean Multilabel** | Cleaned expanded dataset | 1,947 seqs | 246 (multi-label) | ESM2 + Engineered + MLP | ESM2 (1280D) + Engineered (64D) → MLP → 246 classes. Multi-label classification | **1.29%** | Very poor due to extreme class count (246) and sparsity |
| **Binary ent-kaurene** | Cleaned expanded dataset | 1,947 seqs | 2 (binary) | ESM2 + Engineered + MLP | ESM2 (1280D) + Engineered (64D) → Binary classifier for ent-kaurene detection | **95.77%** | Excellent performance on simplified binary task |
| **V4 - Stabilized (development)** | MARTS-DB Enhanced | 1,326 seqs | 6 (multi-label) | ESM2 + kNN + Calibration | ESM2 embeddings + kNN soft voting + Platt scaling calibration | **8.78%** (baseline) | In development; focus on novel sequences (≤40% identity) |

---

## 📁 Detailed Iteration Breakdown

### v0.1 - Simple Baseline
**Date**: October 3, 2024  
**Training Data**: 
- File: `TPS_Baseline_ESM2_Only/data/esm2_embeddings.npy`
- Source: TS-GSD consolidated dataset
- Size: 1,273 sequences, 30 classes
- Label density: 50% of samples have labels (636/1,273)
- Avg labels per sample: 0.75

**Methodology**:
- ESM2 embeddings: `esm2_t33_650M_UR50D` (1280D)
- Architecture: 2-layer MLP (1280 → 512 → 30)
- Loss: BCEWithLogitsLoss
- Threshold: Fixed 0.5
- Training: 30 epochs, AdamW (lr=1e-4)
- Cross-validation: 5-fold stratified

**Results**:
- **Macro F1**: 0.66% (± 0.67%)
- Micro F1: 0.97%
- Precision: 2.00%
- Recall: 0.42%

**Key Finding**: Fixed threshold severely limits performance on imbalanced multi-label tasks.

---

### v0.2 - Enhanced Baseline
**Date**: October 3, 2024  
**Training Data**: 
- Files: 
  - `TPS_Baseline_ESM2_Only/data/esm2_embeddings.npy`
  - `TPS_Baseline_ESM2_Only/data/engineered_features.npy`
- Source: TS-GSD consolidated dataset
- Size: 1,273 sequences, 30 classes
- Engineered features: 64D (terpene type, enzyme class, kingdom, product count)

**Methodology**:
- ESM2 embeddings (1280D) + Engineered features (64D) = 1344D input
- Architecture: 3-layer MLP (1344 → 512 → 256 → 30)
- Dropout: 0.5 (layer 1), 0.3 (layer 2)
- Loss: BCEWithLogitsLoss
- **Adaptive per-class thresholds**: F1-optimized on 10% tuning set
- Training: 30 epochs, AdamW (lr=1e-4)
- Cross-validation: 5-fold stratified

**Results**:
- **Macro F1**: 19.15% (± 1.63%)
- Micro F1: 31.43%
- Precision: 17.49%
- Recall: 27.92%
- Mean threshold: 0.29 (vs 0.5 fixed)

**Key Finding**: Adaptive thresholds and engineered features provide 29× improvement over baseline.

---

### v0.3 - Multi-Modal (Placeholder Graphs)
**Date**: October 3, 2024  
**Training Data**: 
- Files: 
  - `TPS_Baseline_ESM2_Only/data/esm2_embeddings.npy`
  - `TPS_Baseline_ESM2_Only/data/engineered_features.npy`
  - Placeholder graphs (synthetic 10 nodes × 30D)
- Source: TS-GSD consolidated dataset
- Size: 1,273 sequences, 30 classes

**Methodology**:
- **Three-branch multi-modal fusion**:
  - PLMEncoder: ESM2 (1280D) → 512 → 256D
  - FeatureEncoder: Engineered (64D) → 128 → 256D
  - GCNEncoder: Graphs (N×30D nodes) → 128 → 256D via 3-layer GCN
- Fusion: Concatenate (768D) → 512 → 256D
- Classifier: 256D → 30 classes
- **Focal Loss**: α=0.25, γ=2.0
- **Inverse-frequency class weighting**: Range [0.07, 5.62]
- Adaptive per-class thresholds
- Training: 50 epochs, AdamW (lr=1e-4, weight_decay=1e-5)
- Batch size: 8
- Cross-validation: 5-fold

**Results**:
- **Macro F1**: 32.87% (± 2.81%)
- Micro F1: 41.55%
- Precision: 32.56%
- Recall: 41.76%
- Mean threshold: 0.37

**Key Finding**: Multi-modal fusion + Focal Loss provides 72% improvement over v0.2, even with placeholder graphs.

---

### v0.3.1 - Multi-Modal (Real AlphaFold Graphs)
**Date**: October 3, 2024  
**Training Data**: 
- Files: 
  - `TPS_Baseline_ESM2_Only/data/esm2_embeddings.npy`
  - `TPS_Baseline_ESM2_Only/data/engineered_features.npy`
  - `TPS_Classifier_v3_Early/functional_graphs.pkl` (1,222 real AlphaFold structures)
- Source: TS-GSD consolidated dataset
- Size: 1,273 sequences, 30 classes
- Graph coverage: 96% (1,222/1,273)

**Methodology**:
- Same architecture as v0.3
- **Real AlphaFold protein graphs**:
  - 30D node features: 20D AA one-hot + 5D physicochemical + 5D ligand/cofactor
  - Variable nodes (100-600 per protein)
  - Edge construction: 8Å spatial contact threshold
  - GCN: 3 layers with global mean pooling
- Fallback to dummy graphs for 51 proteins without structures

**Results**:
- **Macro F1**: 32.94% (± 2.01%)
- Micro F1: 40.32%

**Key Finding**: Real graphs provided only +0.07% improvement vs placeholders, suggesting ESM2 already encodes structural information. Performance gap to V3 (~6%) likely due to GCN architecture sophistication (attention, edge features), not graph quality.

---

### V3 Phase 1 - ESM2+Engineered
**Date**: 2024 (V3 Reference)  
**Training Data**: 
- Source: MARTS-DB
- File: Various in `TPS_Classifier_v3_Early/`
- Size: 1,273 sequences, 30 classes

**Methodology**:
- ESM2 embeddings (1280D) + Engineered features (64D)
- Adaptive per-class thresholds
- Train/validation/test split (not cross-validation)

**Results**:
- **Macro F1**: 8.57%

**Key Finding**: V3 baseline performance; lower than v0.2 (19.15%) likely due to different evaluation protocol (single split vs 5-fold CV).

---

### V3 Phase 2 - Multi-Modal (20D nodes)
**Date**: 2024 (V3 Reference)  
**Training Data**: 
- Source: MARTS-DB with AlphaFold structures
- File: Various in `TPS_Classifier_v3_Early/`
- Size: 1,273 sequences, 30 classes
- Structures: AlphaFold predictions from UniProt

**Methodology**:
- **Three-branch multi-modal architecture**:
  - ESM2 (1280D) → 256D
  - AlphaFold GCN with 20D node features (amino acid one-hot) → 256D
  - Engineered (64D) → 256D
- Fusion layer combines all three modalities
- Adaptive thresholds
- Advanced GCN architecture (likely with attention mechanisms)

**Results**:
- **Macro F1**: 20.08%

**Key Finding**: +134% improvement over ESM2-only. GCN and AlphaFold introduced simultaneously, so contributions cannot be separated.

---

### V3 Phase 3 - Enhanced Multi-Modal (25D nodes)
**Date**: 2024 (V3 Reference)  
**Training Data**: 
- Source: MARTS-DB with AlphaFold structures
- File: `TPS_Classifier_v3_Early/functional_graphs.pkl`
- Size: 1,273 sequences, 30 classes
- Structures: 1,222 high-confidence AlphaFold structures (96% coverage)

**Methodology**:
- Same three-branch architecture as Phase 2
- **Enhanced 25D node features**:
  - 20D amino acid one-hot encoding
  - 5D physicochemical properties (hydrophobicity, charge, volume, pI)
- Training: 50 epochs
- Mixed precision training
- Gradient accumulation (4 steps)
- Early stopping (patience=10)

**Results**:
- **Validation Macro F1**: 38.56%
- **Test Macro F1**: 38.74%

**Key Finding**: +92% improvement over Phase 2. Richer node features substantially improve GCN performance.

---

### V3 Phase 4 - Functional Multi-Modal (30D nodes)
**Date**: 2024 (V3 Reference)  
**Training Data**: 
- Source: MARTS-DB with AlphaFold structures + ligand information
- File: `TPS_Classifier_v3_Early/functional_graphs.pkl`
- Size: 1,273 sequences, 30 classes

**Methodology**:
- Same three-branch architecture
- **Final 30D node features**:
  - 20D amino acid one-hot encoding
  - 5D physicochemical properties
  - **5D ligand/cofactor features**: Mg²⁺ ions, FPP/GPP/DMAPP substrates
- **Functional geometric integration**: Active site modeling, protein-ligand contacts
- Training: Mixed precision, gradient accumulation

**Results**:
- **Validation Macro F1**: 40.19%
- **Test Macro F1**: 40.59%
- Macro Recall: 83.82%
- Macro Precision: 32.65%

**Key Finding**: +4% improvement from functional binding site features. Final V3 model achieves 368% improvement over broken baseline.

---

### Clean Multilabel Classifier
**Date**: October 2024  
**Training Data**: 
- Files: 
  - `data/cleaned_TPS_training_set_final_v2_multilabel_clean.csv`
  - `colab_package/esm2_embeddings_clean.npy`
  - `colab_package/engineered_features_clean.npy`
- Source: Cleaned and expanded dataset from multiple databases
- Size: 1,947 sequences, 246 classes
- Avg labels per sequence: 1.33

**Methodology**:
- ESM2 embeddings (1280D) + Engineered features (64D)
- Architecture: MLP with dropout
- Multi-label classification across 246 terpene product classes
- Training: 50 epochs
- Cross-validation: 5-fold stratified

**Results**:
- **Macro F1**: 1.29% (± 0.09%)
- Micro F1: 43.87%
- Hamming Loss: 0.40%

**Key Finding**: Very poor macro F1 due to extreme class count (246) and sparse labels. High micro F1 indicates model captures frequent classes but fails on rare ones.

---

### Binary ent-kaurene Classifier
**Date**: October 2024  
**Training Data**: 
- Files: 
  - `data/cleaned_TPS_training_set_final_v2_multilabel_clean.csv`
  - `colab_package/esm2_embeddings_clean.npy`
  - `colab_package/engineered_features_clean.npy`
- Source: Same cleaned dataset, binary labels created
- Size: 1,947 sequences
- Class distribution: 570 positive (29.3%), 1,377 negative (70.7%)

**Methodology**:
- ESM2 embeddings (1280D) + Engineered features (64D)
- Binary classifier for ent-kaurene detection
- Architecture: MLP with dropout
- Training: 50 epochs, BCEWithLogitsLoss
- Cross-validation: 5-fold stratified

**Results**:
- **F1**: 95.77% (± 1.64%)
- Precision: 92.83%
- Recall: 98.95%
- Accuracy: 97.43%

**Key Finding**: Excellent performance on simplified binary classification task. Demonstrates model can learn well when task complexity is reduced.

---

### V4 - Stabilized Pipeline (In Development)
**Date**: October 2024  
**Training Data**: 
- Files: 
  - `TPS_Classifier_v4_Enhanced/data/train.fasta`
  - `TPS_Classifier_v4_Enhanced/data/train_labels.csv`
  - `TPS_Classifier_v4_Enhanced/data/val.fasta`
- Source: MARTS-DB Enhanced dataset
- Size: 1,326 sequences, 6 broader product classes
- Focus: Identity-aware evaluation (≤40% identity sequences)

**Methodology**:
- **ESM2 embeddings** (`esm2_t33_650M_UR50D`)
- **kNN soft voting**: Alpha blending with training data neighbors
- **Calibrated thresholds**: Per-class Platt scaling
- **Hierarchy-aware masking**: Type-based prediction constraints
- Deterministic structural features (no random placeholders)
- Comprehensive testing: Leakage guards, calibration validation

**Results** (Current):
- **Validation Baseline Macro F1**: 8.78%
- With calibration: 0.00% (calibration issues)

**Status**: In active development. Target: >5% improvement vs baseline on novel sequences.

---

## 🎯 Key Insights Across Iterations

### 1. **Adaptive Thresholds are Critical**
- Fixed 0.5 threshold: 0.66% F1 (v0.1)
- Adaptive thresholds: 19.15% F1 (v0.2)
- **29× improvement** from threshold optimization alone

### 2. **Multi-Modal Fusion Provides Substantial Gains**
- ESM2 + Engineered: 19.15% F1
- + GCN (placeholder): 32.87% F1 (72% improvement)
- + GCN (real AlphaFold): 32.94% F1 (minimal improvement)

### 3. **ESM2 Already Encodes Structural Information**
- Real vs placeholder graphs: Only +0.07% difference
- Suggests protein language models learn structural features from sequence
- Performance gap to V3 likely due to GCN architecture sophistication, not graph quality

### 4. **Node Feature Richness Matters**
- 20D nodes: 20.08% F1
- 25D nodes (+ physicochemical): 38.56% F1 (+92%)
- 30D nodes (+ ligands): 40.19% F1 (+4%)

### 5. **Focal Loss Effective for Imbalanced Data**
- Standard BCE with fixed threshold: 0.66% F1
- Focal Loss + adaptive thresholds: 32.87% F1
- Focal Loss focuses learning on hard examples

### 6. **Task Complexity Impacts Performance**
- 246 classes (sparse): 1.29% macro F1
- 30 classes: 32-40% macro F1
- Binary classification: 95.77% F1

### 7. **GCN Architecture Quality Drives Performance**
- Simple 3-layer GCN: 32.94% F1
- Advanced GCN (V3, likely with attention): 40.19% F1
- **~7% gap** attributed to architecture sophistication (attention, edge features, advanced pooling)

---

## 📈 Performance Evolution Summary

| Stage | Best F1 | Key Innovation |
|-------|---------|----------------|
| v0.1 Baseline | 0.66% | ESM2 embeddings |
| v0.2 Enhanced | 19.15% | + Adaptive thresholds + Engineered features |
| v0.3 Multi-Modal | 32.87% | + Multi-modal fusion + Focal Loss |
| v0.3.1 Real Graphs | 32.94% | + Real AlphaFold structures |
| V3 Final | **40.59%** | + Advanced GCN + Ligand features |

**Total Improvement**: 0.66% → 40.59% = **6,053% improvement** over initial baseline

---

## 🔬 Training Data Files Reference

### Primary Datasets

1. **TS-GSD Consolidated** (1,273 sequences, 30 classes)
   - Location: `TPS_Baseline_ESM2_Only/data/`
   - ESM2 embeddings: `esm2_embeddings.npy` (1273 × 1280)
   - Engineered features: `engineered_features.npy` (1273 × 64)
   - Label info: `label_info.json`

2. **Cleaned Expanded Dataset** (1,947 sequences, 246 classes)
   - Location: `colab_package/`
   - CSV: `cleaned_TPS_training_set_final_v2_multilabel_clean.csv`
   - ESM2 embeddings: `esm2_embeddings_clean.npy` (1947 × 1280)
   - Engineered features: `engineered_features_clean.npy` (1947 × 64)

3. **MARTS-DB Enhanced** (1,326 sequences, 6 classes)
   - Location: `TPS_Classifier_v4_Enhanced/data/`
   - Training: `train.fasta`, `train_labels.csv`
   - Validation: `val.fasta`, `val_labels_binary.csv`

### Structural Data

4. **AlphaFold Protein Graphs** (1,222 structures)
   - Location: `TPS_Classifier_v3_Early/functional_graphs.pkl`
   - Coverage: 96% of TS-GSD dataset
   - Node features: 30D (20D AA + 5D physicochemical + 5D ligand)
   - Edge construction: 8Å contact threshold

---

## 🚀 Future Directions

### Immediate Next Steps
1. **Enhance GCN Architecture** (Target: Close 6% gap to V3)
   - Add attention mechanisms (GATConv)
   - Incorporate edge features (distances, angles)
   - Implement advanced pooling (attention-based)
   - Add residual connections

2. **Fix V4 Calibration Issues**
   - Debug Platt scaling implementation
   - Improve kNN blending performance
   - Achieve >5% improvement on novel sequences

### Long-term Goals
1. **Scale to Larger Datasets**
   - Retrain on expanded MARTS-DB (1,326+ sequences)
   - Transfer learning from large protein databases

2. **Production Deployment**
   - API endpoints for predictions
   - Batch prediction tools
   - Confidence calibration

3. **Novel Sequence Focus**
   - Evaluate on sequences ≤40% identity
   - Improve generalization to distant homologs

---

**Last Updated**: October 10, 2025  
**Status**: Active Development  
**Best Model**: V3 Phase 4 (40.59% test F1)  
**Repository**: https://github.com/ah474747/terpene-synthase-classifier

