
from typing import Dict
import numpy as np

class PlattScaler:
    def __init__(self):
        self.A = 1.0
        self.B = 0.0
    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.1, steps: int = 500):
        A, B = 1.0, 0.0
        for _ in range(steps):
            z = A * logits + B
            p = 1.0 / (1.0 + np.exp(-z))
            gradA = np.mean((p - labels) * logits)
            gradB = np.mean(p - labels)
            A -= lr * gradA
            B -= lr * gradB
        self.A, self.B = float(A), float(B)
        return self
    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = self.A * logits + self.B
        return 1.0 / (1.0 + np.exp(-z))

def fit_platt_per_class(logits: np.ndarray, y_true: np.ndarray) -> Dict[int, PlattScaler]:
    N, C = logits.shape
    scalers = {}
    for c in range(C):
        scaler = PlattScaler().fit(logits[:, c], y_true[:, c])
        scalers[c] = scaler
    return scalers

def apply_scalers(logits: np.ndarray, scalers: Dict[int, PlattScaler]) -> np.ndarray:
    C = logits.shape[1]
    out = np.zeros_like(logits, dtype=np.float32)
    for c in range(C):
        out[:, c] = scalers[c].transform(logits[:, c])
    return out

def optimize_thresholds(probs: np.ndarray, y_true: np.ndarray, mode: str = "f1beta", beta: float = 0.7, precision_min: float = 0.6):
    C = probs.shape[1]
    thresholds = np.zeros(C, dtype=np.float32)
    for c in range(C):
        p = probs[:, c]
        y = y_true[:, c].astype(bool)
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            yhat = p >= t
            tp = (yhat & y).sum()
            fp = (yhat & ~y).sum()
            fn = (~yhat & y).sum()
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            if mode == "precision_floor" and prec < precision_min:
                continue
            if mode == "f1beta":
                b2 = beta**2
                score = (1 + b2) * prec * rec / max(1e-8, b2 * prec + rec)
            else:
                score = 2 * prec * rec / max(1e-8, prec + rec)
            if score > best_score:
                best_score, best_t = score, t
        thresholds[c] = best_t
    return thresholds
