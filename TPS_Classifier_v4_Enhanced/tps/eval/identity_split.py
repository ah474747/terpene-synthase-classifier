
from typing import List

def approx_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0: return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n

def map_val_to_train_identity(train_seqs: List[str], val_seqs: List[str]) -> List[float]:
    out = []
    for v in val_seqs:
        best = 0.0
        for t in train_seqs:
            best = max(best, approx_identity(v, t))
        out.append(best)
    return out

def filter_val_by_identity(train_seqs: List[str], val_seqs: List[str], threshold: float) -> List[int]:
    idents = map_val_to_train_identity(train_seqs, val_seqs)
    return [i for i, idv in enumerate(idents) if idv <= threshold]
