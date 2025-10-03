# Changelog

All notable changes to the Terpene Synthase Classifier project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Implement adaptive threshold optimization (F1β strategy)
- Add focal loss for class imbalance
- Add engineered biochemical features
- Verify and fix label quality issues

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

