# Identity-aware TPS evaluation (OOD focus)

This document describes the evaluation pipeline for the stabilized TPS classifier, focusing on out-of-distribution performance on novel sequences with ≤40% identity to training data.

## Prerequisites

1. **Install dependencies**: `pip install -r requirements.txt`
   - If ESM installation fails: `pip install fair-esm torch --extra-index-url https://download.pytorch.org/whl/cpu`
2. **Ensure training data is available**:
   - `data/train.fasta` - Training sequences
   - `data/train_labels.csv` - Training labels (CSV: id,class)
3. **Ensure validation data is available**:
   - `data/val.fasta` - Validation sequences  
   - `data/val_labels_binary.csv` - Validation labels (binary format)
4. **Ensure external holdout data is available**:
   - `data/external_30.fasta` - External holdout sequences
   - `data/external_30_labels_binary.csv` - External holdout labels
5. **Optional**: Place trained model checkpoint at `models/checkpoints/complete_multimodal_best.pth`

## Evaluation Pipeline

### 1) Build kNN index (train only)

Build the kNN index using only training data to avoid leakage:

```bash
python scripts/build_index.py \
  --train_fasta "data/train.fasta" \
  --labels "data/train_labels.csv" \
  --class_list "data/classes.txt" \
  --out_index "models/knn/index.npy" \
  --out_meta "models/knn/index_meta.json"
```

### 2) Baseline predictions on val (emit logits for Platt calibration)

Generate baseline predictions WITH logits to enable proper Platt scaling:

```bash
python scripts/predict.py \
  --input "data/val.fasta" \
  --class_list "data/classes.txt" \
  --out "preds/val_base.jsonl" \
  --emit_logits
```

### 3) Calibrate thresholds on val (Platt scaling + F1β optimization)

Calibrate per-class thresholds using Platt scaling on logits, then optimize thresholds:

```bash
python scripts/calibrate_thresholds.py \
  --preds "preds/val_base.jsonl" \
  --labels "data/val_labels_binary.csv" \
  --class_list "data/classes.txt" \
  --use_logits \
  --f1beta --beta 0.7 \
  --out_dir "models/calibration/"
```

### 4) Final predictions (kNN + hierarchy + calibration + thresholds)

Generate final predictions using all components:

```bash
python scripts/predict.py \
  --input "data/val.fasta" \
  --class_list "data/classes.txt" \
  --out "preds/val_final.jsonl" \
  --use_knn --alpha 0.7 \
  --use_hierarchy \
  --calibration_dir "models/calibration/"
```

### 5) Leakage guard check

Verify no training sequences leaked into evaluation:

```bash
python scripts/assert_no_leakage.py preds/val_final.jsonl
```

### 6) Evaluate with bootstrap CI

Compare baseline and final predictions with bootstrap confidence intervals:

```bash
python scripts/evaluate.py \
  --preds "preds/val_base.jsonl" "preds/val_final.jsonl" \
  --labels "data/val_labels_binary.csv" \
  --bootstrap 1000 \
  --report "reports/val_compare.json"
```

## External holdout evaluation

Evaluate on the external 30-sequence holdout set:

### External holdout predictions

```bash
python scripts/predict.py \
  --input "data/external_30.fasta" \
  --class_list "data/classes.txt" \
  --out "preds/ext30_final.jsonl" \
  --use_knn --alpha 0.7 \
  --use_hierarchy \
  --calibration_dir "models/calibration/"
```

### External holdout evaluation

```bash
python scripts/evaluate.py \
  --preds "preds/ext30_final.jsonl" \
  --labels "data/external_30_labels_binary.csv" \
  --bootstrap 1000 \
  --report "reports/ext30.json"
```

## Expected Results

### Success Criteria

1. **Identity-aware validation (≤40% identity)**: 
   - Macro-F1 improvement ≥5 points vs baseline
   - 95% bootstrap CI excludes 0 improvement
   - Precision increases materially
   - Recall drop ≤5% absolute

2. **External holdout**:
   - Macro-F1 >> prior ~0.0569
   - Top-3 accuracy improved
   - ECE decreased ≥30%

3. **Determinism**: Identical outputs with same seed

4. **No leakage**: kNN index built from train only

## Ablation Studies

Run ablation studies to understand component contributions:

```bash
bash scripts/ablate.sh
```

This will test:
- Base (no kNN, no calibration)
- + calibration only
- + kNN with different alpha values (0.5-0.9)
- + kNN + calibration combinations

Results will be saved to `reports/ablation.json`.

## Quick Start

Run the complete pipeline:

```bash
bash run.sh
```

## Troubleshooting

- **ESM not found**: Install with `pip install fair-esm`
- **CUDA errors**: The code will fallback to CPU if CUDA is not available
- **Missing data files**: Ensure all required data files exist in the `data/` directory
- **Memory issues**: The ESM model can be memory-intensive; consider using smaller models or batch processing
