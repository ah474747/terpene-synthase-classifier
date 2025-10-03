
from typing import Dict
import numpy as np

def apply_type_mask(class_probs: np.ndarray, type_probs: np.ndarray, class_to_type: Dict[int, int]) -> np.ndarray:
    N, C = class_probs.shape
    out = class_probs.copy()
    for i in range(N):
        for c in range(C):
            t = class_to_type.get(c, None)
            if t is not None:
                out[i, c] *= type_probs[i, t]
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out
