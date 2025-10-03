# Project Status

## 🎯 Current Phase: Baseline Establishment & V3 Replication

### Latest Updates

**October 3, 2024** - Enhanced baseline **exceeds V3 ESM2-only performance!**
- ✅ **19.15% macro F1** with ESM2 + Engineered + Adaptive Thresholds
- ✅ **29× improvement** over simple baseline (0.66% → 19.15%)
- ✅ **Exceeds V3 target** of 8.57% ESM2+Engineered performance
- ✅ Adaptive per-class thresholding successfully implemented
- ✅ 64D engineered features add critical signal
- 📋 Next: Add AlphaFold structures to reach 38.74% F1

## 📊 Model Progression

| Model | Architecture | F1 Score | Status |
|-------|------------|----------|--------|
| **Baseline (v0.1)** | ESM2 + Simple MLP | 0.66% | ✅ Complete |
| **Enhanced Baseline (v0.2)** | ESM2 + Engineered + Adaptive Thresholds | **19.15%** | ✅ **Complete** |
| **V3 ESM2-only** | ESM2 + Engineered + Adaptive Thresholds | 8.57% | ✅ **Exceeded** |
| **V3 Multi-Modal** | + AlphaFold + GCN | 38.74% | 🎯 Goal |
| **V4 Enhanced** | + kNN + Calibration | TBD | 🔮 Future |

## 🔬 Active Work

### ✅ Sprint 1 Complete: ESM2+Engineered Baseline

**Achievement**: 19.15% F1 - **Exceeded 8.57% target!**

**What Worked:**
1. ✅ Adaptive per-class thresholds (F1-optimized)
2. ✅ 64D engineered features (terpene type, enzyme class, kingdom, etc.)
3. ✅ Deeper 3-layer MLP architecture
4. ✅ Robust 5-fold cross-validation

### Current Sprint: Add AlphaFold Structures

**Objective**: Reach V3's multi-modal performance (38.74% F1)

**Next Steps:**
1. Extract AlphaFold predicted structures for TS-GSD dataset
2. Implement Graph Convolutional Network (GCN) for structural encoding
3. Add ligand binding site features (Mg²⁺, substrate)
4. Integrate multi-modal fusion (ESM2 + Engineered + Structural)
5. Target: 38.74% F1 (V3 multi-modal parity)

## 📁 Repository Structure

```
terpene-synthase-classifier/
├── TPS_Baseline_ESM2_Only/      # v0.1 - Simple baseline
├── TPS_Classifier_v3_Early/     # V3 - Proven 38.74% F1
├── TPS_Classifier_v4_Enhanced/  # V4 - Development
├── BASELINE_RESULTS.md          # Baseline analysis
├── PROJECT_STATUS.md            # This file
└── README.md                    # Main overview
```

## 🐛 Known Issues

1. **Label Sparsity**: Only 636/1,273 samples (50%) have labels
   - May indicate data quality issues
   - Need to verify TS-GSD consolidation pipeline

2. **Extreme Class Imbalance**: 0-83 samples per class
   - Requires focal loss and class weighting
   - 4 classes have zero samples

3. **Performance Gap**: 0.66% vs 8.57% (V3 ESM2-only)
   - Primary cause: Fixed threshold vs adaptive
   - Secondary: No class balancing

## 📈 Success Metrics

### Phase 1: ESM2 Parity ✅
- [x] Establish baseline (< 1% F1)
- [ ] Reach V3 ESM2-only performance (8.57% F1)
- [ ] Validate on same dataset

### Phase 2: Multi-Modal Replication
- [ ] Add AlphaFold structural features
- [ ] Implement GCN encoding
- [ ] Reach V3 multi-modal performance (38.74% F1)

### Phase 3: Enhancement
- [ ] Scale to larger dataset
- [ ] Add kNN soft voting
- [ ] Implement per-class calibration
- [ ] Target: > 40% F1 on novel sequences

## 🔗 References

- **V3 Benchmark**: `VERIFY_V3_BENCHMARK.md`
- **Baseline Results**: `BASELINE_RESULTS.md`
- **Data Pipeline**: `TPS_Classifier_v3_Early/marts_consolidation_pipeline.py`

## 🏷️ Tags & Releases

- `v0.1-baseline` - ESM2-only simple MLP (0.66% F1)
- `v3-reference` - (Planned) Tag for proven V3 code

---

**Last Updated**: October 3, 2024  
**Branch**: `main`  
**Repository**: https://github.com/ah474747/terpene-synthase-classifier

