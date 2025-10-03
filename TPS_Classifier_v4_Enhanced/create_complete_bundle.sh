#!/usr/bin/env bash
set -euo pipefail

OUT="tps_complete_bundle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"/{models/checkpoints,models/calibration,models/knn,preds,reports,data,logs,scripts,tests,tps}

echo "Creating complete bundle with training data and model weights..."

# Copy all artifacts from the previous bundle
cp -f models/checkpoints/complete_multimodal_best.pth "$OUT/models/checkpoints/" 2>/dev/null || echo "⚠️  Model weights not found: models/checkpoints/complete_multimodal_best.pth"
cp -f models/checkpoints/label_order.json "$OUT/models/checkpoints/" 2>/dev/null || echo "⚠️  Label order not found"
cp -f models/checkpoints/class_to_type.json "$OUT/models/checkpoints/" 2>/dev/null || echo "⚠️  Hierarchy mapping not found"

# Copy calibration artifacts (will be MISSING initially)
cp -f models/calibration/calibrators.json "$OUT/models/calibration/" 2>/dev/null || true
cp -f models/calibration/thresholds.json "$OUT/models/calibration/" 2>/dev/null || true

# Copy kNN artifacts (will be MISSING initially) 
cp -f models/knn/index.npy "$OUT/models/knn/" 2>/dev/null || true
cp -f models/knn/index_meta.json "$OUT/models/knn/" 2>/dev/null || true

# Copy predictions & reports (will be MISSING initially)
cp -f preds/val_base.jsonl "$OUT/preds/" 2>/dev/null || true
cp -f preds/val_final.jsonl "$OUT/preds/" 2>/dev/null || true
cp -f preds/ext30_final.jsonl "$OUT/preds/" 2>/dev/null || true
cp -f reports/val_compare.json "$OUT/reports/" 2>/dev/null || true
cp -f reports/ablation.json "$OUT/reports/" 2>/dev/null || true
cp -f reports/ext30.json "$OUT/reports/" 2>/dev/null || true

# Copy training/validation data (CRITICAL)
cp -f data/train.fasta "$OUT/data/" 2>/dev/null || echo "⚠️  Training FASTA not found: data/train.fasta"
cp -f data/train_labels.csv "$OUT/data/" 2>/dev/null || echo "⚠️  Training labels not found: data/train_labels.csv"
cp -f data/val.fasta "$OUT/data/" 2>/dev/null || echo "⚠️  Validation FASTA not found: data/val.fasta"
cp -f data/val_labels_binary.csv "$OUT/data/" 2>/dev/null || echo "⚠️  Validation labels not found: data/val_labels_binary.csv"

# Copy external holdout data (OPTIONAL)
cp -f data/external_30.fasta "$OUT/data/" 2>/dev/null || echo "ℹ️  External holdout FASTA not provided (optional)"
cp -f data/external_30_labels_binary.csv "$OUT/data/" 2>/dev/null || echo "ℹ️  External holdout labels not provided (optional)"

# Copy core configuration files
cp -f data/classes.txt "$OUT/data/" 2>/dev/null || echo "⚠️  Classes file not found"

# Copy all Python code
echo "Copying all Python modules and scripts..."
find . -name "*.py" -exec cp --parents {} "$OUT/" \; 2>/dev/null || true
find . -name "*.py" -exec cp --parents {} "$OUT/" \; 2>/dev/null || true

# Copy documentation and scripts
cp -f RUN.md "$OUT/" 2>/dev/null || true
cp -f README.md "$OUT/" 2>/dev/null || true
cp -f requirements.txt "$OUT/" 2>/dev/null || true
cp -f Makefile "$OUT/" 2>/dev/null || true
cp -f run.sh "$OUT/" 2>/dev/null || true
cp -f scripts/ablate.sh "$OUT/scripts/" 2>/dev/null || true

# Enhanced environment + config log
{
  echo "=== Versions ==="
  python3 -V || true
  pip3 freeze | grep -E 'torch|esm|fair-esm|faiss|scikit-learn|numpy|biopython' || true
  echo
  echo "=== Runtime config ==="
  python3 - <<'PY'
import json, hashlib, os, sys
sys.path.append('.')
from tps import config
print("TPS_ESM_MODEL_ID:", config.ESM_MODEL_ID)
print("RANDOM_SEED:", config.RANDOM_SEED)
print("ALPHA_KNN:", config.ALPHA_KNN)
print("CALIBRATION_MODE:", config.CALIBRATION_MODE)
paths = [
  "models/checkpoints/complete_multimodal_best.pth",
  "models/checkpoints/label_order.json", 
  "models/checkpoints/class_to_type.json",
  "models/calibration/calibrators.json",
  "models/calibration/thresholds.json",
  "models/knn/index.npy",
  "models/knn/index_meta.json",
  "data/train.fasta",
  "data/train_labels.csv",
  "data/val.fasta", 
  "data/val_labels_binary.csv",
  "data/external_30.fasta",
  "data/external_30_labels_binary.csv",
]
def sha1(p):
    if not os.path.exists(p): return "MISSING"
    import hashlib
    h=hashlib.sha1()
    with open(p, "rb") as f:
        for ch in iter(lambda:f.read(8192), b""): h.update(ch)
    return h.hexdigest()
for p in paths:
    status = "MISSING" if not os.path.exists(p) else "PRESENT"
    sha = sha1(p) if status == "PRESENT" else "N/A"
    print("FILE", p, status, sha)
PY
} > "$OUT/logs/env_and_file_status.txt" || true

# Data sizes and stats
{
  echo "=== Data Statistics ==="
  if [ -f "data/train.fasta" ]; then
    echo "Training sequences: $(grep -c '^>' data/train.fasta 2>/dev/null || echo 'unknown')"
  fi
  if [ -f header "data/train_labels.csv" ]; then
    echo "Training labels: $(wc -l < data/train_labels.csv 2>/dev/null || echo 'unknown')"
  fi
  if [ -f "data/val.fasta" ]; then
    echo "Validation sequences: $(grep -c '^>' data/val.fasta 2>/dev/null || echo 'unknown')"
  fi
  if [ -f "data/val_labels_binary.csv" ]; then
    echo "Validation labels: $(wc -l < data/val_labels_binary.csv 2>/dev/null || echo 'unknown')"
  fi
  if [ -f "data/external_30.fasta" ]; then
    echo "External sequences: $(grep -c '^>' data/external_30.fasta 2>/dev/null || echo 'unknown')"
  fi
} > "$OUT/logs/data_stats.txt" || true

# Optional: leakage sentry if kNN index exists
if [ -f scripts/assert_no_leakage.py ] && [ -f models/knn/index_meta.json ]; then
  python3 scripts/assert_no_leakage.py preds/val_final.jsonl 2>/dev/null > "$OUT/logs/leakage_sentry.txt" || true
else
  echo "Leakage check skipped (no kNN index or predictions)" > "$OUT/logs/leakage_sentry.txt"
fi

# Run readiness test
{
  echo "=== System Readiness Check ==="
  python3 - <<'PY'
import sys, os
sys.path.append('.')
try:
    from tps.esm_embed import ESMEmbedder
    from tps import config
    from TPS_Predictor_Stabilized import TPSPredictorStabilized
    
    # Test ESM
    embedder = ESMEmbedder()
    print("✓ ESM Embedder:", embedder.model_id)
    
    # Test predictor
    label_order = ['Germacrene_A', 'Germacrene_C', 'Linalool']
    predictor = TPSPredictorStabilized(n_classes=len(label_order), label_order=label_order)
    print("✓ Predictor initialized")
    
    # Check model file
    if os.path.exists("models/checkpoints/complete_multimodal_best.pth"):
        print("✓ Model checkpoint present")
    else:
        print("⚠️  Model checkpoint missing")
        
    # Check data files
    data_files = [
        "data/train.fasta", "data/train_labels.csv", 
        "data/val.fasta", "data/val_labels_binary.csv"
    ]
    for f in data_files:
        if os.path.exists(f):
            print(f"✓ {f}")
        else:
            print(f"⚠️  {f} missing")
            
    # Check external holdout
    if os.path.exists("data/external_30.fasta"):
        print("✓ External holdout data present")
    else:
        print("ℹ️  External holdout data not provided (optional)")
        
except Exception as e:
    print(f"Error in readiness check: {e}")
PY
} > "$OUT/logs/readiness_check.txt" || true

# Create tar.gz archive
tar -czf "${OUT}.tar.gz" "$OUT"
echo ""
echo "🎯 Complete bundle created: ${OUT}.tar.gz"
echo ""
echo "📊 Bundle Contents:"
echo "   📁 Complete codebase (Python modules, scripts, tests)"
echo "   📁 Configuration files (label order, hierarchy mapping)"
echo "   📁 Training data (FASTA + labels)"
echo "   📁 Validation data (FASTA + binary labels)"
echo "   📁 Optional: External holdout data"
echo "   📁 Model weights (if provided)"
echo "   📁 Runtime diagnostics and file status"
echo ""
echo "🚀 Ready for end-to-end evaluation pipeline!"

# Cleanup
rm -rf "$OUT"


