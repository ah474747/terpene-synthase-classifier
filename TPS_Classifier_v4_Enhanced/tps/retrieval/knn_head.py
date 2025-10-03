
import os, json
import numpy as np

class KNNSoftClassifier:
    def __init__(self, k: int = 10):
        self.k = k
        self._emb = None
        self._labels = None
        self._classes = None

    def build(self, embeddings: np.ndarray, labels: np.ndarray, class_names):
        assert embeddings.shape[0] == labels.shape[0]
        self._emb = embeddings.astype(np.float32)
        self._labels = labels.astype(np.int32)
        self._classes = list(class_names)

    def save(self, index_path: str, meta_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        np.save(index_path, self._emb)
        meta = {"labels": self._labels.tolist(), "classes": self._classes, "k": self.k}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, index_path: str, meta_path: str):
        self._emb = np.load(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._labels = np.array(meta["labels"], dtype=np.int32)
        self._classes = meta["classes"]
        self.k = meta.get("k", self.k)

    def predict_proba(self, queries: np.ndarray) -> np.ndarray:
        def l2norm(x):
            n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            return x / n
        E = l2norm(self._emb)
        Q = l2norm(queries.astype(np.float32))
        sims = Q @ E.T
        idx = np.argpartition(-sims, self.k, axis=1)[:, :self.k]
        probs = np.zeros((Q.shape[0], len(self._classes)), dtype=np.float32)
        for i in range(Q.shape[0]):
            neigh_labels = self._labels[idx[i]]
            for lab in neigh_labels:
                probs[i, lab] += 1.0
        probs /= (self.k + 1e-8)
        return probs
