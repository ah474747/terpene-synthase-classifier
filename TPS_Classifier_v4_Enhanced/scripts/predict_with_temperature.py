#!/usr/bin/env python3
import argparse, json, numpy as np, os

def softmax(x, T):
    z = x / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def main(a):
    temp = json.load(open(os.path.join(a.calibration_dir,"temperature.json")))["temperature"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as w:
        with open(a.input) as f:
            for line in f:
                obj = json.loads(line)
                logits = np.array(obj["logits"], dtype=float)[None, :]
                probs = softmax(logits, temp)[0]
                rec = {"id": obj["id"], "label_order": obj["label_order"]}
                if a.top1:
                    onehot = [0]*len(probs); onehot[int(np.argmax(probs))] = 1
                    rec["probs"] = probs.tolist()
                    rec["decisions"] = onehot
                else:
                    thr = a.thr
                    rec["probs"] = probs.tolist()
                    rec["decisions"] = (probs >= thr).astype(int).tolist()
                w.write(json.dumps(rec) + "\n")
    print("Wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="preds/val_base.jsonl")
    p.add_argument("--calibration_dir", default="models/calibration_temp")
    p.add_argument("--out", default="preds/val_temp_thr.jsonl")
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--top1", action="store_true", help="emit top-1 decisions instead of thresholding")
    main(p.parse_args())