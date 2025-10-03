import argparse, json, os, numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import from workspace root
sys.path.append(str(Path(__file__).parent.parent))

from tps.esm_embed import ESMEmbedder

def read_fasta(path):
    seqs, ids = [], []
    with open(path) as f:
        sid, s = None, []
        for line in f:
            if line.startswith('>'):
                if sid is not None:
                    seqs.append(''.join(s)); s = []
                sid = line.strip()[1:]; ids.append(sid)
            else:
                s.append(line.strip())
        if sid is not None:
            seqs.append(''.join(s))
    return ids, seqs

def read_labels(path, id_to_idx, classes):
    y = np.zeros((len(id_to_idx), len(classes)), dtype=np.int32)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2: continue
            sid, cname = parts[0], parts[1]
            if sid in id_to_idx and cname in classes:
                y[id_to_idx[sid], classes.index(cname)] = 1
    return y

def main(args):
    ids, seqs = read_fasta(args.train_fasta)
    id_to_idx = {i:k for k,i in enumerate(ids)}
    classes = [c.strip() for c in open(args.class_list).read().splitlines() if c.strip()]

    emb = ESMEmbedder().embed_mean(seqs).astype(np.float32)
    y = read_labels(args.labels, id_to_idx, classes)
    y_single = y.argmax(axis=1)

    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    np.save(args.out_index, emb)
    meta = {"labels": y_single.tolist(), "classes": classes, "k": args.k, "train_ids": ids}
    with open(args.out_meta, "w") as f: json.dump(meta, f)
    print("Saved kNN index:", args.out_index, "meta:", args.out_meta)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_fasta", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--class_list", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out_index", default="models/knn/index.npy")
    p.add_argument("--out_meta", default="models/knn/index_meta.json")
    main(p.parse_args())