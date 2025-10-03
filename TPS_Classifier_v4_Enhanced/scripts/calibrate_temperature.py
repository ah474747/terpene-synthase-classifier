#!/usr/bin/env python3
import argparse, json, numpy as np

def load_logits_and_labels(preds_path, labels_path, n_classes=None):
    logits, y = [], []
    with open(preds_path) as f:
        for line in f:
            obj = json.loads(line)
            logits.append(obj["logits"])
            if n_classes is None: n_classes = len(obj["logits"])
    with open(labels_path) as f:
        for line in f:
            rr = [int(x) for x in line.strip().split(",")[:n_classes]]
            y.append(rr)
    return np.array(logits, dtype=np.float32), np.array(y, dtype=np.int32)

def softmax(x, T=1.0):
    z = x / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def nll(logits, y, T):
    p = softmax(logits, T=T)
    # assume single-label; take argmax target
    tgt = y.argmax(axis=1)
    rows = np.arange(len(tgt))
    return -np.log(p[rows, tgt] + 1e-12).mean()

def main(args):
    logits, y = load_logits_and_labels(args.preds, args.labels)
    # simple 1D search over T
    Ts = np.linspace(0.5, 3.0, 26)
    bestT, bestL = None, 1e9
    for T in Ts:
        L = nll(logits, y, T)
        if L < bestL: bestL, bestT = L, T
    print(f"Best temperature T={bestT:.3f} (NLL={bestL:.4f})")
    out = {"temperature": float(bestT)}
    import os, json
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(f"{args.out_dir}/temperature.json","w"), indent=2)
    print("Saved:", f"{args.out_dir}/temperature.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds/val_base.jsonl")
    ap.add_argument("--labels", default="data/val_labels_binary.csv")
    ap.add_argument("--out_dir", default="models/calibration_temp")
    main(ap.parse_args())

