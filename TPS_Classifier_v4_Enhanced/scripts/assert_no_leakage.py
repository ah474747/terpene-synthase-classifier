#!/usr/bin/env python3
import json, sys
meta = json.load(open("models/knn/index_meta.json"))
train_ids = set(meta.get("train_ids", []))
eval_ids = [json.loads(l)["id"] for l in open(sys.argv[1])]
bad = [i for i in eval_ids if i in train_ids]
assert not bad, f"Leakage: {len(bad)} eval IDs appear in kNN index (first few: {bad[:5]})"
print("✓ No leakage: eval IDs are disjoint from kNN train IDs.")