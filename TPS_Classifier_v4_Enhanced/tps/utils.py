
import random, hashlib, json, logging
from typing import Dict, Any
import numpy as np

def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Dict[str, Any]):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a:i for i,a in enumerate(AA)}

def amino_acid_composition(seq: str) -> np.ndarray:
    counts = np.zeros(len(AA), dtype=np.float32)
    for ch in seq:
        if ch in AA_TO_IDX:
            counts[AA_TO_IDX[ch]] += 1.0
    total = counts.sum() or 1.0
    return counts / total

def simple_embed(seq: str) -> np.ndarray:
    comp = amino_acid_composition(seq)
    length = len(seq)
    length_feats = np.array([length, length**0.5, np.log1p(length)], dtype=np.float32)
    return np.concatenate([comp, length_feats])
