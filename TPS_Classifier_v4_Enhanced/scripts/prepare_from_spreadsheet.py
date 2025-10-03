#!/usr/bin/env python3
import csv, os, json, argparse, random
from collections import defaultdict

def stratified_split(rows, val_frac=0.2, seed=42):
    random.seed(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    train, val = [], []
    for lbl, lst in by_label.items():
        random.shuffle(lst)
        k = max(1, int(len(lst) * val_frac))
        val += lst[:k]
        train += lst[k:]
    return train, val

def write_fasta(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(f">{r['id']}\n{r['sequence']}\n")

def write_train_labels(rows, class_list, path_csv):
    with open(path_csv, "w") as f:
        for r in rows:
            f.write(f"{r['id']},{r['label']}\n")

def write_val_binary(rows, class_list, out_csv):
    idx = {c:i for i,c in enumerate(class_list)}
    with open(out_csv, "w") as f:
        for r in rows:
            onehot = ["0"] * len(class_list)
            onehot[idx[r["label"]]] = "1"
            f.write(",".join(onehot) + "\n")

def main(args):
    rows = []
    with open(args.input) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            row = {k: v.strip() for k,v in row.items()}
            if not row.get("id") or not row.get("sequence") or not row.get("label"):
                continue
            rows.append(row)
    labels = sorted(list({r["label"] for r in rows}))
    os.makedirs("data", exist_ok=True)
    with open("data/classes.txt", "w") as g: g.write("\n".join(labels) + "\n")
    os.makedirs("models/checkpoints", exist_ok=True)
    with open("models/checkpoints/label_order.json", "w") as g: json.dump(labels, g, indent=2)
    train, val = stratified_split(rows, val_frac=args.val_frac, seed=args.seed)
    write_fasta(train, "data/train.fasta")
    write_train_labels(train, labels, "data/train_labels.csv")
    write_fasta(val, "data/val.fasta")
    write_val_binary(val, labels, "data/val_labels_binary.csv")
    print(f"Prepared: train={len(train)}  val={len(val)}  classes={len(labels)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw_sequences.csv")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())


