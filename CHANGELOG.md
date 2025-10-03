# Changelog

All notable changes to the Terpene Synthase Classifier project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Enhance GCN architecture with attention mechanisms
- Incorporate edge features (distances, angles) in graph processing
- Implement V3's AdaptiveWeightedFocalLoss
- Hyperparameter tuning (learning rate, dropout, focal loss params)
- Data augmentation (sequence/graph perturbations)
- Target: 38.74% F1 (V3 multi-modal parity)

## [0.3.1-real-graphs] - 2024-10-03

### Added
- **Real AlphaFold protein graphs** successfully integrated (1,222 structures, 96% coverage)
- Fixed graph unpickling by defining ProteinGraph class hierarchy
- Module namespace registration for pickle compatibility
- Fallback handling for proteins without graphs

### Performance
- **Macro F1**: 32.94% (± 2.01%) - Stable performance with real graphs
- **Micro F1**: 40.32%
- Real graphs vs placeholders: +0.07% F1 (minimal difference)

### Key Findings
- **Surprising result**: Real AlphaFold graphs provided virtually identical performance to placeholder graphs
- **Hypothesis**: ESM2 embeddings already encode sufficient structural information
- **Implication**: Performance gap to V3 (5.8% F1) likely due to GCN architecture sophistication, not graph quality

### Technical Details
- Graph structure: Variable nodes (100-600), 30D features per node
- Node features: 20D amino acid + 5D physicochemical + 5D ligand/cofactor
- Edge construction: 8Å spatial contact threshold from AlphaFold predictions
- Successfully loaded: 1,222/1,273 proteins (96% coverage)

### Fixed
- `Can't get attribute 'FunctionalProteinGraph'` unpickling error
- Graph class hierarchy: ProteinGraph → EnhancedProteinGraph → FunctionalProteinGraph
- Module registration for structural_graph_pipeline, module6_feature_enhancement, module8_functional_geometric_integration

### Changed
- Updated train_multimodal.py with proper graph class definitions
- Improved fallback graph creation for missing structures

## [0.3.0-multimodal] - 2024-10-03

### Added
- **Full multi-modal architecture** with three-branch fusion
- Graph Convolutional Network (GCN) encoder for protein structures
- Focal Loss implementation (α=0.25, γ=2.0)
- Inverse-frequency class weighting for extreme imbalance
- 50-epoch training with Adam optimization

### Performance
- **Macro F1**: 32.87% (± 2.81%) - **72% improvement over v0.2**
- **Micro F1**: 41.55% (± 3.97%)
- 50× improvement over v0.1 baseline
- 85% of V3's 38.74% target achieved

### Architecture
- Three parallel branches:
  - PLM Encoder: ESM2 (1280D) → 256D
  - Feature Encoder: Engineered (64D) → 256D
  - GCN Encoder: Graphs (N×30D) → 256D
- Fusion layer: 768D → 512D → 256D
- Classifier: 256D → 30 classes

### Features
- Focal Loss for class imbalance (down-weights easy examples)
- Inverse-frequency class weights (range: 0.07-5.62)
- Xavier weight initialization
- 3-layer GCN with global mean pooling
- Adaptive per-class thresholds (mean: 0.37)

### Known Issues
- Using placeholder graphs (real AlphaFold unpickling issue)
- Expected improvement with real structures

## [0.2.0-enhanced] - 2024-10-03

### Added
- **Enhanced baseline model** with ESM2 + Engineered features + Adaptive thresholds
- 64D engineered feature generation (`generate_engineered_features.py`)
- Per-class adaptive threshold optimization (F1-optimized)
- 3-layer MLP architecture with improved dropout
- Threshold tuning set (10% of training data)
- Comprehensive results documentation (`ENHANCED_BASELINE_RESULTS.md`)

### Performance
- **Macro F1**: 19.15% (± 1.63%) - **29× improvement over v0.1**
- **Micro F1**: 31.43% (± 1.70%)
- **Exceeds V3 target** of 8.57% ESM2+Engineered performance
- Consistent across all 5 CV folds

### Features
- Engineered features (64D):
  - Terpene type (6D one-hot)
  - Enzyme class (2D one-hot)
  - Kingdom (11D one-hot)
  - Product count (1D normalized)
  - Placeholder features (44D)
- Adaptive thresholds per class (range: 0.07-0.50, mean: 0.29)
- Threshold-tuning pipeline for robust optimization

### Changed
- Updated model architecture from 2-layer to 3-layer MLP
- Improved dropout strategy (0.5 → 0.3 progressive)
- Random split for threshold tuning (avoid stratification issues)

## [0.1.0-baseline] - 2024-10-03

### Added
- **ESM2-only baseline model** using simple 2-layer MLP
- 5-fold stratified cross-validation framework
- Automated ESM2 embedding generation (esm2_t33_650M_UR50D)
- Baseline results documentation (`BASELINE_RESULTS.md`)
- Project status tracking (`PROJECT_STATUS.md`)
- Reusable embeddings for 1,273 TS-GSD enzymes
- Label statistics and analysis

### Performance
- **Macro F1**: 0.66% (± 0.67%)
- **Micro F1**: 0.97% (± 1.04%)
- Dataset: TS-GSD consolidated (1,273 enzymes, 30 classes)

### Known Issues
- Performance significantly below V3 ESM2-only (8.57% F1)
- Fixed 0.5 threshold instead of adaptive thresholds
- No class weighting or focal loss
- 50% of samples lack labels (data quality concern)
- Extreme class imbalance (0-83 samples per class)

## [0.0.0] - 2024-10-03

### Added
- Initial repository setup with V3 baseline code
- Multi-modal architecture (ESM2 + AlphaFold + Engineered features)
- V3 verified performance: 38.74% test F1
- TS-GSD consolidated dataset (1,273 enzymes)
- V4 development framework (stabilized training, kNN, calibration)
- Comprehensive documentation and training guides
- `.gitignore` configured for ML projects

### Reference Models
- **V3 Multi-Modal**: 38.74% test F1 (proven performance)
- **V3 ESM2-only**: 8.57% F1 (target for baseline)

---

## Version Tags

- `v0.1-baseline` - ESM2-only simple MLP baseline (October 3, 2024)

## Links

- [Repository](https://github.com/ah474747/terpene-synthase-classifier)
- [Latest Release](https://github.com/ah474747/terpene-synthase-classifier/releases)
- [Issues](https://github.com/ah474747/terpene-synthase-classifier/issues)

