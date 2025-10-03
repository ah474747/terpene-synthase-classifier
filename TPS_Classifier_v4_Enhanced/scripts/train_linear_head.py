#!/usr/bin/env python3
import os, json, math, argparse, numpy as np, torch
import sys
from pathlib import Path

# Add parent directory to path to import from workspace root
sys.path.append(str(Path(__file__).parent.parent))

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tps import config
from tps.features.engineered import generate_engineered_features
from tps.models.multimodal import FinalMultiModalClassifier
from tps.esm_embed import ESMEmbedder

def read_fasta_ids(path):
    ids, seqs = [], []
    with open(path) as f:
        sid, buf = None, []
        for line in f:
            if line.startswith(">"):
                if sid is not None:
                    seqs.append("".join(buf)); buf=[]
                sid = line.strip()[1:]; ids.append(sid)
            else:
                buf.append(line.strip())
        if sid is not None:
            seqs.append("".join(buf))
    return ids, seqs

def read_train_labels(path_csv, classes):
    idx = {c:i for i,c in enumerate(classes)}
    y, id_order = [], []
    with open(path_csv) as f:
        for line in f:
            sid, cname = line.strip().split(",")[:2]
            id_order.append(sid)
            y.append(idx[cname])
    return id_order, np.array(y, dtype=np.int64)

class SeqDataset(Dataset):
    def __init__(self, ids, seqs, y_idx, embeddings):
        self.ids, self.seqs, self.y = ids, seqs, y_idx
        self.emb = embeddings
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        plm = self.emb[i]
        eng = generate_engineered_features(self.seqs[i])
        struct = np.zeros((32,), dtype=np.float32)
        return torch.from_numpy(plm), torch.from_numpy(eng), torch.from_numpy(struct), torch.tensor(self.y[i])

def cache_embeddings(seqs, cache_path, model_id):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        return np.load(cache_path)
    emb = ESMEmbedder(model_id=model_id).embed_mean(seqs).astype(np.float32)
    np.save(cache_path, emb)
    return emb

def train(train_ds, val_ds, plm_dim, n_classes, epochs=6, lr=3e-4, seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = FinalMultiModalClassifier(plm_dim=plm_dim, eng_dim=24, struct_dim=32, n_classes=n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    best_val = math.inf; best_state = None
    for ep in range(1, epochs+1):
        model.train(); tot=0.0
        for plm, eng, struct, y in train_loader:
            opt.zero_grad()
            logits = model(plm.float(), eng.float(), struct.float(), None, torch.zeros(plm.size(0)))
            loss = ce(logits, y); loss.backward(); opt.step()
            tot += float(loss)
        model.eval(); vtot=0.0
        with torch.no_grad():
            for plm, eng, struct, y in val_loader:
                logits = model(plm.float(), eng.float(), struct.float(), None, torch.zeros(plm.size(0)))
                vtot += float(ce(logits, y))
        print(f"epoch {ep}: train_loss={tot/len(train_loader):.4f}  val_loss={vtot/len(val_loader):.4f}")
        if vtot < best_val:
            best_val = vtot
            best_state = {"state_dict": model.state_dict()}
    
    # Add metadata before returning
    meta = {
      "esm_model_id": os.getenv("TPS_ESM_MODEL_ID", config.ESM_MODEL_ID),
      "plm_dim": int(plm_dim),
      "n_classes": int(n_classes)
    }
    best_state = {"state_dict": best_state["state_dict"], "meta": meta}
    return best_state

def main(args):
    classes = [c.strip() for c in open(args.classes).read().splitlines() if c.strip()]
    tr_ids, tr_seqs = read_fasta_ids(args.train_fa)
    tr_id_order, y_train = read_train_labels(args.train_csv, classes)
    assert tr_ids == tr_id_order, "train.fasta order must match train_labels.csv order"
    va_ids, va_seqs = read_fasta_ids(args.val_fa)
    y_val = []
    with open(args.val_bin) as f:
        for line in f:
            onehot = [int(x) for x in line.strip().split(",")[:len(classes)]]
            y_val.append(int(np.argmax(onehot)))
    y_val = np.array(y_val, dtype=np.int64)
    plm_model = os.getenv("TPS_ESM_MODEL_ID", config.ESM_MODEL_ID)
    tr_emb = cache_embeddings(tr_seqs, "data/cache_train_esm.npy", plm_model)
    va_emb = cache_embeddings(va_seqs, "data/cache_val_esm.npy", plm_model)
    plm_dim = tr_emb.shape[1]
    train_ds = SeqDataset(tr_ids, tr_seqs, y_train, tr_emb)
    val_ds = SeqDataset(va_ids, va_seqs, y_val, va_emb)
    state = train(train_ds, val_ds, plm_dim, len(classes), epochs=args.epochs, lr=args.lr, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(state, args.out)
    print("Saved checkpoint to", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_fa", default="data/train.fasta")
    ap.add_argument("--train_csv", default="data/train_labels.csv")
    ap.add_argument("--val_fa", default="data/val.fasta")
    ap.add_argument("--val_bin", default="data/val_labels_binary.csv")
    ap.add_argument("--classes", default="data/classes.txt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/checkpoints/complete_multimodal_best.pth")
    main(ap.parse_args())
