#!/usr/bin/env bash
set -euo pipefail

echo "Running TPS evaluation pipeline..."
python -V

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Build kNN index (train only)
echo "Building kNN index..."
bash -lc 'python scripts/build_index.py --train_fasta "data/train.fasta" --labels "data/train_labels.csv" --class_list "data/classes.txt" --out_index "models/knn/index.npy" --out_meta "models/knn/index_meta.json"'

# Baseline predictions on val (emit logits for Platt calibration)
echo "Generating baseline predictions with logits..."
bash -lc 'python scripts/predict.py --input "data/val.fasta" --class_list "data/classes.txt" --out "preds/val_base.jsonl" --emit_logits'

# Calibrate thresholds on val (Platt scaling + F1β optimization)
echo "Calibrating thresholds with Platt scaling..."
bash -lc 'python scripts/calibrate_thresholds.py --preds "preds/val_base.jsonl" --labels "data/val_labels_binary.csv" --class_list "data/classes.txt" --use_logits --f1beta --beta 0.7 --out_dir "models/calibration/"'

# Final predictions (kNN + hierarchy + calibration + thresholds)
echo "Generating final predictions..."
bash -lc 'python scripts/predict.py --input "data/val.fasta" --class_list "data/classes.txt" --out "preds/val_final.jsonl" --use_knn --alpha 0.7 --use_hierarchy --calibration_dir "models/calibration/"'

# Leakage guard check
echo "Checking for data leakage..."
bash -lc 'python scripts/assert_no_leakage.py preds/val_final.jsonl'

# Evaluate with bootstrap CI
echo "Running evaluation..."
bash -lc 'python scripts/evaluate.py --preds "preds/val_base.jsonl" "preds/val_final.jsonl" --labels "data/val_labels_binary.csv" --bootstrap 1000 --report "reports/val_compare.json"'

echo "Done. Check reports/val_compare.json for results."
