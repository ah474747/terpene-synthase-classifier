#!/usr/bin/env python3
import json, sys, numpy as np
P = []
for line in open(sys.argv[1]):
    obj = json.loads(line)
    P.append(obj.get("probs", obj.get("logits")))
A = np.array(P, dtype=float)
if A.size == 0:
    print("No records in", sys.argv[1]); raise SystemExit(1)
if A.max() <= 1.0:
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    Y = (A >= thr).astype(int)
    print("Probs shape:", A.shape, "thr:", thr)
    print("Positives per class:", Y.sum(0).tolist())
else:
    print("Logits shape:", A.shape, "min/max:", float(A.min()), float(A.max()))


