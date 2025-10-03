# ESM2-Only Baseline Model

## Overview

This is a **simple baseline model** for terpene synthase classification using only ESM2 embeddings and a 2-layer MLP. The purpose is to establish a robust performance benchmark before adding AlphaFold structural features.

## Model Architecture

```
Input: ESM2 Embeddings (1280D from esm2_t33_650M_UR50D)
    ↓
Linear(1280 → 512)
    ↓
ReLU + Dropout(0.5)
    ↓
Linear(512 → N_classes)
    ↓
Output: Multi-label predictions
```

## Training Strategy

- **5-fold stratified cross-validation** for robust evaluation
- **BCEWithLogitsLoss** for multi-label classification
- **AdamW optimizer** with learning rate 1e-4
- **30 epochs** per fold
- **Batch size**: 16

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Baseline Training

```bash
python train_baseline.py
```

This will:
- Load TS-GSD consolidated dataset (1,273 unique enzymes with multi-label targets)
- Generate ESM2 embeddings (saved to `data/esm2_embeddings.npy`)
- Run 5-fold cross-validation
- Report macro F1 scores with confidence intervals
- Save results to `results/baseline_cv_results.json`

## Expected Output

The script will print:
- Per-fold training progress
- Final metrics for each fold (F1, precision, recall)
- Average metrics across all folds with standard deviation
- 95% confidence interval for macro F1

## Dataset

**TS-GSD (Terpene Synthase Gold Standard Dataset)**:
- **1,273 unique enzymes** (consolidated from MARTS-DB)
- **30 functional ensemble classes** (multi-label classification)
- **Multi-product enzymes**: Each enzyme can produce multiple terpene products
- **Rich metadata**: terpene_type, enzyme_class, product_names, etc.

## Comparison with V3

This baseline will be compared against the V3 model's ESM2-only performance (8.57% F1) to validate improvements from:
1. Same consolidated dataset used in V3
2. Robust 5-fold cross-validation
3. Proper multi-label evaluation

Then we'll add AlphaFold structures to see if we can replicate V3's dramatic improvement (8.57% → 38.74% F1).

## Files Generated

- `data/esm2_embeddings.npy` - Pre-computed embeddings (reusable)
- `data/label_info.json` - Class information
- `results/baseline_cv_results.json` - Complete CV results

## Next Steps

After establishing this baseline:
1. Add AlphaFold structural features (GCN encoding)
2. Add engineered biochemical features
3. Compare with V3 multi-modal performance
4. Scale to larger datasets

