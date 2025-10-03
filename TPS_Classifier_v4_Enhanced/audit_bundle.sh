#!/usr/bin/env bash
set -euo pipefail

OUT="tps_audit_bundle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"/{models/checkpoints,models/calibration,models/knn,preds,reports,data,logs}

# Copy artifacts
cp -f models/checkpoints/complete_multimodal_best.pth "$OUT/models/checkpoints/" 2>/dev/null || true
cp -f models/checkpoints/label_order.json "$OUT/models/checkpoints/" 2>/dev/null || true
cp -f models/checkpoints/class_to_type.json "$OUT/models/checkpoints/" 2>/dev/null || true
cp -f models/calibration/calibrators.json "$OUT/models/calibration/" 2>/dev/null || true
cp -f models/calibration/thresholds.json "$OUT/models/calibration/" 2>/dev/null || true
cp -f models/knn/index.npy "$OUT/models/knn/" 2>/dev/null || true
cp -f models/knn/index_meta.json "$OUT/models/knn/" 2>/dev/null || true

# Copy predictions & reports
cp -f preds/val_base.jsonl "$OUT/preds/" 2>/dev/null || true
cp -f preds/val_final.jsonl "$OUT/preds/" 2>/dev/null || true
cp -f preds/ext30_final.jsonl "$OUT/preds/" 2>/dev/null || true
cp -f reports/val_compare.json "$OUT/reports/" 2>/dev/null || true
cp -f reports/ablation.json "$OUT/reports/" 2>/dev/null || true
cp -f reports/ext30.json "$OUT/reports/" 2>/dev/null || true

# Copy labels & class list (IDs optional)
cp -f data/classes.txt "$OUT/data/" 2>/dev/null || true
cp -f data/val_labels_binary.csv "$OUT/data/" 2>/dev/null || true
cp -f data/external_30_labels_binary.csv "$OUT/data/" 2>/dev/null || true

# Env + config log
{
  echo "=== Versions ==="
  python3 -V || true
  pip3 freeze | grep -E 'torch|esm|fair-esm|faiss|scikit-learn' || true
  echo
  echo "=== Runtime config ==="
  python3 - <<'PY'
import json, hashlib, os, sys
from tps import config
print("TPS_ESM_MODEL_ID:", config.ESM_MODEL_ID)
print("RANDOM_SEED:", config.RANDOM_SEED)
paths = [
  "models/checkpoints/complete_multimodal_best.pth",
  "models/checkpoints/label_order.json",
  "models/calibration/calibrators.json",
  "models/calibration/thresholds.json",
  "models/knn/index.npy",
  "models/knn/index_meta.json",
]
def sha1(p):
    if not os.path.exists(p): return "MISSING"
    import hashlib
    h=hashlib.sha1()
    with open(p, "rb") as f:
        for ch in iter(lambda:f.read(8192), b""): h.update(ch)
    return h.hexdigest()
for p in paths:
    print("SHA1", p, sha1(p))
PY
} > "$OUT/logs/env_and_hashes.txt" || true

# Optional: leakage sentry if present
if [ -f scripts/assert_no_leakage.py ]; then
  python3 scripts/assert_no_leakage.py preds/val_final.jsonl > "$OUT/logs/leakage_sentry.txt" || true
fi

# Tar/zip
tar -czf "${OUT}.tar.gz" "$OUT"
echo "Wrote ${OUT}.tar.gz"


