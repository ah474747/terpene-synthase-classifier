import numpy as np
import torch
import os
from typing import List, Dict, Any, Tuple
from tps import config
from tps.utils import set_seed, simple_embed
from tps.features.engineered import generate_engineered_features
from tps.models.multimodal import FinalMultiModalClassifier
from tps.hierarchy.head import apply_type_mask
from tps.hierarchy.utils import load_class_to_type

class TPSPredictorStabilized:
    def __init__(self, n_classes: int, label_order: List[str]):
        set_seed(config.RANDOM_SEED)
        self._plm_dim = None
        self.model = None
        self.n_classes= n_classes
        self.label_order = label_order
        self.knn = None
        self.class_to_type, self.type_to_id = {}, {}
        
        try:
            self.class_to_type, self.type_to_id = load_class_to_type()
            print(f"Loaded hierarchy mapping: {len(self.class_to_type)} classes mapped to types")
        except Exception as e:
            print(f"[WARN] hierarchy mapping not loaded: {e}")

    def _ensure_model(self, plm_dim: int):
        """Lazy initialization of the model after we know the ESM embedding dimension."""
        if self.model is None:
            from tps.models.multimodal import FinalMultiModalClassifier
            self.model = FinalMultiModalClassifier(plm_dim=plm_dim, eng_dim=24, struct_dim=32, n_classes=self.n_classes)
            self.model.eval()
            
            ckpt = "models/checkpoints/complete_multimodal_best.pth"
            if os.path.exists(ckpt):
                import torch
                from tps import config
                sd = torch.load(ckpt, map_location="cpu")
                
                # Check metadata and auto-repair if mismatch
                meta = sd.get("meta", {})
                expected_plm = meta.get("plm_dim")
                expected_esm = meta.get("esm_model_id")
                
                if expected_plm is not None and self._plm_dim is not None and expected_plm != self._plm_dim:
                    raise RuntimeError(
                      f"Checkpoint plm_dim={expected_plm} but runtime plm_dim={self._plm_dim}. "
                      f"Fix: set TPS_ESM_MODEL_ID={expected_esm} and rebuild artifacts or retrain head."
                    )
                
                self.model.load_state_dict(sd.get("state_dict", sd), strict=False)
                print(f"[INFO] Loaded checkpoint (esm={expected_esm}, plm_dim={expected_plm}) from {ckpt}")
            else:
                print("[WARN] No checkpoint found; using random-initialized weights.")

    def _seq_to_plm(self, seqs: List[str]) -> np.ndarray:
        """Generate ESM embeddings for the input sequences."""
        from tps.esm_embed import ESMEmbedder
        emb = ESMEmbedder().embed_mean(seqs)  # [B, D]
        if self._plm_dim is None:
            self._plm_dim = emb.shape[1]
            self._ensure_model(self._plm_dim)
        return emb
        
    def _seq_to_eng(self, seqs: List[str]) -> np.ndarray:
        return np.stack([generate_engineered_features(s) for s in seqs], axis=0)
        
    def _seq_to_struct(self, seqs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = len(seqs)
        node = np.zeros((B, 1, 32), dtype=np.float32)
        edge_index = np.zeros((B, 2, 0), dtype=np.int64)
        has_structure = np.zeros((B,), dtype=np.float32)
        return node, edge_index, has_structure

    def predict(self, seqs: List[str], use_knn: bool = False, alpha: float = config.ALPHA_KNN,
                use_hierarchy: bool = False, calibrators=None, thresholds: np.ndarray = None) -> Dict[str, Any]:
        # Ensure model is initialized first
        if self.model is None and self._plm_dim is not None:
            self._ensure_model(self._plm_dim)
        
        Xplm = torch.tensor(self._seq_to_plm(seqs))
        Xeng = torch.tensor(self._seq_to_eng(seqs))
        node, edge, has_struct = self._seq_to_struct(seqs)
        node = torch.tensor(node); has_struct = torch.tensor(has_struct)
        
        if self.model is None:
            raise RuntimeError("Model not initialized. This should not happen.")
            
        with torch.no_grad():
            logits = self.model(Xplm, Xeng, node, None, has_struct)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        # Store raw logits before modifications
        raw_logits = logits.cpu().numpy()
        
        if use_knn and self.knn is not None:
            p_knn = self.knn.predict_proba(Xplm.numpy())
            probs = alpha * probs + (1.0 - alpha) * p_knn
        if use_hierarchy and self.class_to_type:
            n_types = max(self.type_to_id.values()) + 1 if self.type_to_id else 0
            if n_types > 0:
                type_probs = np.zeros((len(seqs), n_types), dtype=np.float32)
                for t in range(n_types):
                    cls_idx = [c for c, tt in self.class_to_type.items() if tt == t]
                    if cls_idx:
                        type_probs[:, t] = probs[:, cls_idx].mean(axis=1)
                # pass-through for 'general' types (do not down-weight):
                general_id = self.type_to_id.get("general", None)
                if general_id is not None:
                    # ensure classes mapped to 'general' are not suppressed:
                    for c, t in self.class_to_type.items():
                        if t == general_id:
                            probs[:, c] = probs[:, c]  # no change
                else:
                    probs = apply_type_mask(probs, type_probs, self.class_to_type)
        if calibrators is not None:
            cal_probs = np.zeros_like(probs)
            for c in range(self.n_classes):
                A, B = calibrators.get(str(c), [1.0, 0.0])
                z = A * logits[:, c].cpu().numpy() + B
                cal_probs[:, c] = 1.0 / (1.0 + np.exp(-z))
            probs = cal_probs
        if thresholds is None:
            thresholds = np.full(self.n_classes, 0.5, dtype=np.float32)
        decisions = (probs >= thresholds[None, :]).astype(int)
        return {
            "probs": probs.tolist(), 
            "decisions": decisions.tolist(), 
            "label_order": self.label_order,
            "logits": raw_logits.tolist()  # Include raw logits for calibration
        }