# Project Status

## 🎯 Current Phase: Baseline Establishment & V3 Replication

### Latest Updates

**October 3, 2024** - Baseline ESM2-only model completed
- ✅ Established ESM2-only baseline: **0.66% macro F1** (5-fold CV)
- ✅ Identified performance gap vs V3 (8.57% F1)
- ✅ Generated reusable ESM2 embeddings for 1,273 enzymes
- ⚠️ Revealed critical issues: class imbalance, sparse labels, fixed thresholds

## 📊 Model Progression

| Model | Architecture | F1 Score | Status |
|-------|------------|----------|--------|
| **Baseline (v0.1)** | ESM2 + Simple MLP | 0.66% | ✅ Complete |
| **V3 ESM2-only** | ESM2 + Engineered + Adaptive Thresholds | 8.57% | 📋 Target |
| **V3 Multi-Modal** | + AlphaFold + GCN | 38.74% | 🎯 Goal |
| **V4 Enhanced** | + kNN + Calibration | TBD | 🔮 Future |

## 🔬 Active Work

### Current Sprint: Close the 8% Gap

**Objective**: Replicate V3's ESM2-only performance (8.57% F1)

**Known Differences:**
1. ❌ Fixed threshold (0.5) → Need adaptive per-class thresholds
2. ❌ Standard BCE loss → Need focal loss for imbalance
3. ❌ No class weighting → Need balanced training
4. ❌ Simple MLP → May need deeper architecture

**Next Steps:**
1. Implement adaptive threshold optimization (F1β strategy)
2. Add focal loss and class weighting
3. Verify label quality (50% unlabeled concerning)
4. Add engineered features from V3
5. Re-run 5-fold CV with improvements

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

