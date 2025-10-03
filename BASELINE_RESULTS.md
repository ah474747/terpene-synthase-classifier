# Baseline Results Summary

## ESM2-Only Baseline (October 2024)

### 📊 Performance Metrics

**5-Fold Cross-Validation Results:**
- **Macro F1 Score**: 0.66% (± 0.67%)
- **Micro F1 Score**: 0.97% (± 1.04%)
- **Precision**: 2.00% (± 1.63%)
- **Recall**: 0.42% (± 0.46%)

### 🔍 Key Findings

**Critical Issue Identified**: The simple MLP baseline is performing very poorly (< 1% F1), which is **significantly worse** than V3's reported 8.57% ESM2-only performance.

### 📈 Per-Fold Breakdown

| Fold | Macro F1 | Micro F1 | Precision | Recall |
|------|----------|----------|-----------|--------|
| 1    | 1.82%    | 2.86%    | 3.33%     | 1.25%  |
| 2    | 0.74%    | 1.04%    | 3.33%     | 0.42%  |
| 3    | 0.00%    | 0.00%    | 0.00%     | 0.00%  |
| 4    | 0.74%    | 0.98%    | 3.33%     | 0.42%  |
| 5    | 0.00%    | 0.00%    | 0.00%     | 0.00%  |

### 🔧 Model Configuration

- **Architecture**: Simple 2-layer MLP (1280 → 512 → 30)
- **Embeddings**: ESM2 (esm2_t33_650M_UR50D)
- **Dataset**: TS-GSD consolidated (1,273 unique enzymes, 30 classes)
- **Training**: 30 epochs, AdamW optimizer, LR=1e-4
- **Loss**: BCEWithLogitsLoss

### 🧬 Dataset Characteristics

- **Total Samples**: 1,273 unique enzymes
- **Samples with Labels**: 636/1,273 (50%)
- **Avg Labels per Sample**: 0.75 (multi-label)
- **Max Labels per Sample**: 8
- **Classes with Data**: 26/30
- **Class Imbalance**: Range [0, 83] samples per class

**Top Classes:**
1. Class 10: 83 samples
2. Class 11: 73 samples  
3. Class 1: 66 samples
4. Class 0: 64 samples
5. Class 18: 56 samples

### ⚠️ Analysis & Next Steps

**Why is performance so low?**

1. **Class Imbalance**: Extreme imbalance (0-83 samples per class)
2. **Sparse Labels**: Only 50% of samples have labels, avg 0.75 labels/sample
3. **Simple Architecture**: 2-layer MLP may be too simple
4. **No Class Weighting**: Using standard BCE loss without balancing
5. **Fixed Threshold**: Using 0.5 threshold, but V3 used adaptive thresholds

**Comparison with V3:**
- V3 ESM2-only: **8.57% F1** (with adaptive thresholds)
- Current baseline: **0.66% F1** (with fixed 0.5 threshold)
- Gap: ~8% F1 points

**Required Improvements:**
1. ✅ Implement **adaptive per-class thresholds** (like V3)
2. ✅ Add **class weighting** to handle imbalance
3. ✅ Use **focal loss** instead of BCE
4. ✅ Optimize thresholds on validation set
5. ⚠️ Verify label quality (50% unlabeled is concerning)

### 📝 Reproduction

To reproduce these results:
```bash
cd TPS_Baseline_ESM2_Only
./run_baseline.sh
```

Results saved in:
- `results/baseline_cv_results.json` - Full CV results
- `data/esm2_embeddings.npy` - Pre-computed embeddings (1273 × 1280)
- `data/label_info.json` - Label statistics

---

**Conclusion**: The baseline establishes that a simple MLP with fixed thresholds achieves < 1% F1. To reach V3's 8.57% ESM2-only performance, we need adaptive thresholds, class weighting, and better loss functions. The dramatic improvement to 38.74% F1 in V3 came from adding AlphaFold structures.

