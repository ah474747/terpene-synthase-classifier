# tps/esm_embed.py
from typing import List, Optional
import torch
import numpy as np

class ESMEmbedder:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        from tps import config
        self.model_id = model_id or config.ESM_MODEL_ID  # Use config default
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.alphabet = None
        self.batch_converter = None

    def _lazy_load(self):
        if self.model is not None:
            return
        # Try to import ESM - fail loudly if not available
        try:
            import esm
        except ImportError as e:
            raise RuntimeError("ESM not installed. `pip install fair-esm` or adjust to your env.") from e
        
        try:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_id)
            self.model.eval().to(self.device)
            self.batch_converter = self.alphabet.get_batch_converter()
        except Exception as e:
            raise RuntimeError(f"Failed to load ESM model {self.model_id}: {e}") from e

    @torch.no_grad()
    def embed_mean(self, seqs: List[str]) -> np.ndarray:
        """Generate mean-pooled embeddings for a list of sequences."""
        self._lazy_load()
        
        if not seqs:
            return np.array([])
        
        # Convert sequences to batch format
        data = [(f"seq{i}", s) for i, s in enumerate(seqs)]
        _, _, toks = self.batch_converter(data)
        toks = toks.to(self.device)
        
        # Get representations from the last layer
        out = self.model(toks, repr_layers=[self.model.num_layers], return_contacts=False)
        rep = out["representations"][self.model.num_layers]  # [B, L, D]
        
        # Mean pool over tokens excluding BOS/EOS (positions 0 and L-1)
        # Simple approach: average across positions 1 to L-2
        emb = rep[:, 1:-1, :].mean(dim=1).cpu().numpy()
        
        return emb.astype(np.float32)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension after loading the model."""
        self._lazy_load()
        # Return the hidden dimension of the model
        return self.model.cfg.hidden_size