# Stabilized TPS Classifier Pipeline

A production-ready pipeline for terpene synthase (TPS) classification with enhanced out-of-distribution performance on novel sequences.

## Key Features

- **ESM2 Integration**: Real protein language model embeddings (esm2_t33_650M_UR50D)
- **Idenity-Aware Evaluation**: Focus on ≤40% identity sequences for OOD performance
- **kNN Augmentation**: Soft voting with configurable alpha blending
- **Calibrated Thresholds**: Per-class Platt scaling with F1-β optimization
- **No Random Features**: Deterministic structural fallbacks with zeros
- **Comprehensive Testing**: Leakage guards, determinism checks, and calibration validation

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data** (place in `data/` directory):
   - `train.fasta` - Training sequences
   - `train_labels.csv` - Training labels  
   - `val.fasta` - Validation sequences
   - `val_labels_binary.csv` - Validation labels
   - `external_30.fasta` - External holdout sequences
   - `external_30_labels_binary.csv` - External holdout labels

3. **Run complete pipeline**:
   ```bash
   ./run.sh
   ```

## Pipeline Overview

The evaluation pipeline follows this workflow:

1. **Build kNN Index**: Create searchable index from training data only
2. **Baseline Predictions**: Generate predictions without calibration
3. **Calibrate Thresholds**: Optimize per-class decision thresholds
4. **Final Predictions**: Combine kNN + calibration for improved performance
5. **Bootstrap Evaluation**: Statistical comparison with confidence intervals

## Key Files

### Core Components
- `TPS_Predictor_Stabilized.py` - Main predictor with ESM integration
- `tps/esm_embed.py` - ESM2 embedding module
- `tps/config.py` - Configuration settings
- `tps/models/multimodal.py` - Multi-modal classifier architecture

### Scripts
- `scripts/build_index.py` - kNN index construction
- `scripts/predict.py` - Single prediction script
- `scripts/calibrate_thresholds.py` - Threshold optimization
- `scripts/evaluate.py` - Bootstrap evaluation
- `scripts/ablate.sh` - Ablation studies

### Testing
- `tests/test_no_random_features.py` - Determinism verification
- `tests/test_label_order_lock.py` - Label consistency checks
- `tests/test_knn_leakage_guard.py` - Leakage prevention
- `tests/test_calibration_roundtrip.py` - Reproducibility validation

## Success Criteria

### Validation Performance
- **Macro-F1 improvement**: ≥5 points vs baseline
- **Identity constraint**: Focus on ≤40% identity sequences  
- **Precision gains**: Material improvement
- **Recall preservation**: ≤5% absolute drop

### External Holdout
- **Macro-F1**: >> prior ~0.0569
- **Top-3 accuracy**: Improved performance
- **Calibration error**: ≥30% reduction (ECE)

### System Properties
- **Deterministic outputs**: Identical results with same seed
- **No data leakage**: kNN built from train-only data

## Configuration

Key settings in `tps/config.py`:

```python
ESM_MODEL_ID = "esm2_t33_650M_UR50D"  # ESM2 model variant
ALPHA_KNN = 0.7                      # kNN blending weight
F1_BETA = 0.7                         # F1-β optimization target
IDENTITY_THRESHOLD = 0.4              # Novel sequence threshold
RANDOM_SEED = 42                      # For reproducibility
```

## Usage Examples

### Individual Component Testing
```bash
# Build index
make knn

# Generate predictions  
make predict

# Calibrate thresholds
make calibrate

# Full evaluation
make eval
```

### Ablation Studies
```bash
./scripts/ablate.sh
```

### Custom Prediction
```bash
python scripts/predict.py \
  --input "data/new_sequences.fasta" \
  --class_list "data/classes.txt" \
  --out "predictions.jsonl" \
  --use_knn --alpha 0.7 \
  --calibration_dir "models/calibration/"
```

## Model Architecture

The system uses a multi-modal architecture:

1. **ESM Embeddings**: Mean-pooled protein language model representations
2. **Engineered Features**: Computed protein sequence features  
3. **Structural Features**: Graph neural network features (fallback to zeros)
4. **Modality Gates**: Layer normalization + gating mechanism
5. **Classification Head**: Final multilayer perceptron

## Label Order Consistency

Critical for reproducibility - the label order is locked in:
- `models/checkpoints/label_order.json`
- `data/classes.txt`

Both files must match exactly to ensure consistent predictions.

## Troubleshooting

- **ESM errors**: Install with `pip install fair-esm`
- **CUDA unavailable**: Automatically falls back to CPU
- **Memory issues**: Consider smaller ESM models or batch processing
- **Missing data**: Ensure all required files exist in `data/`
- **Import errors**: Verify Python path includes the package directory

## Results

Pipeline generates comprehensive reports:

- `reports/val_compare.json` - Baseline vs final comparison
- `reports/ext30.json` - External holdout evaluation  
- `reports/ablation.json` - Component contribution analysis

Each report includes bootstrap confidence intervals for robust statistical evaluation.


