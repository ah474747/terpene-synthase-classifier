#!/usr/bin/env bash
set -euo pipefail

echo "== Global threshold sweep on baseline logits =="
# Ensure we have baseline with logits
test -f preds/val_base.jsonl || { echo "Missing preds/val_base.jsonl"; exit 1; }

# Temperature scaling
python3 scripts/calibrate_temperature.py --preds preds/val_base.jsonl --labels data/val_labels_binary.csv --out_dir models/calibration_temp

# Evaluate baseline at thresholds 0.10..0.50
for TH in 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
  python3 scripts/evaluate.py \
    --preds preds/val_base.jsonl \
    --labels data/val_labels_binary.csv \
    --bootstrap 1000 --thr ${TH} \
    --report reports/val_base_thr_${TH}.json
done

echo "== Apply temperature + evaluate at 0.35 and 0.25 =="
python3 scripts/predict_with_temperature.py --input preds/val_base.jsonl --calibration_dir models/calibration_temp --thr 0.35 --out preds/val_temp_thr035.jsonl
python3 scripts/predict_with_temperature.py --input preds/val_base.jsonl --calibration_dir models/calibration_temp --thr 0.25 --out preds/val_temp_thr025.jsonl

python3 scripts/evaluate.py --preds preds/val_temp_thr035.jsonl --labels data/val_labels_binary.csv --bootstrap 1000 --report reports/val_temp_thr035.json
python3 scripts/evaluate.py --preds preds/val_temp_thr025.jsonl --labels data/val_labels_binary.csv --bootstrap 1000 --report reports/val_temp_thr025.json

echo "== Top-1 macro-F1 (no thresholds) =="
python3 scripts/predict_with_temperature.py --input preds/val_base.jsonl --calibration_dir models/calibration_temp --top1 --out preds/val_temp_top1.jsonl
python3 scripts/evaluate.py --preds preds/val_temp_top1.jsonl --labels data/val_labels_binary.csv --bootstrap 1000 --report reports/val_temp_top1.json

echo "== kNN α sweep with precision-floor calibrators (if present) =="
CAL=models/calibration_pfloor
if [ -f ${CAL}/thresholds.json ] && [ -f models/knn/index.npy ]; then
  for A in 0.3 0.5 0.7 0.9; do
    python3 scripts/predict.py \
      --input data/val.fasta \
      --class_list data/classes.txt \
      --out preds/val_knn_cal_A${A}.jsonl \
      --use_knn --alpha ${A} \
      --use_hierarchy \
      --calibration_dir ${CAL} \
      --knn_index models/knn/index.npy \
      --knn_meta models/knn/index_meta.json
    python3 scripts/evaluate.py \
      --preds preds/val_knn_cal_A${A}.jsonl \
      --labels data/val_labels_binary.csv \
      --bootstrap 1000 \
      --report reports/val_knn_cal_A${A}.json
  done
else
  echo "Skipping α sweep: calibration_pfloor or kNN index not found."
fi

echo "== DONE =="
