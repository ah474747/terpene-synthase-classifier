#!/usr/bin/env python3
import os, json, subprocess
import sys
from pathlib import Path

# Add parent directory to path to import from workspace root
sys.path.append(str(Path(__file__).parent.parent))

from tps import config
esm = os.getenv("TPS_ESM_MODEL_ID", config.ESM_MODEL_ID)
print(f"[REPAIR] Retraining with current ESM: {esm}")
# retrain quick head to produce a fresh checkpoint that matches current ESM dim
cmd = [
  "python3","scripts/train_linear_head.py",
  "--train_fa","data/train.fasta",
  "--train_csv","data/train_labels.csv",
  "--val_fa","data/val.fasta",
  "--val_bin","data/val_labels_binary.csv",
  "--classes","data/classes.txt",
  "--epochs","6",
  "--out","models/checkpoints/complete_multimodal_best.pth"
]
print("[REPAIR] Running:", " ".join(cmd))
subprocess.check_call(cmd)
print("[REPAIR] Done. New checkpoint written.")
