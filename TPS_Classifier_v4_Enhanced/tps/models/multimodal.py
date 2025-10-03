
from typing import Optional
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

class FinalMultiModalClassifier(nn.Module):
    def __init__(self, plm_dim: int, eng_dim: int, struct_dim: int, n_classes: int, latent: int = 256):
        super().__init__()
        self.plm_enc = MLP(plm_dim, latent)
        self.eng_enc = MLP(eng_dim, latent)
        self.struct_enc = MLP(struct_dim, latent)
        self.norm_plm = nn.LayerNorm(latent)
        self.norm_eng = nn.LayerNorm(latent)
        self.norm_struct = nn.LayerNorm(latent)
        self.modality_gate = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, latent))
        self.head = nn.Sequential(
            nn.Linear(latent * 3, latent),
            nn.ReLU(),
            nn.Linear(latent, n_classes)
        )

    def forward(self, plm_x: torch.Tensor, eng_x: torch.Tensor,
                struct_node_x: Optional[torch.Tensor] = None,
                struct_edge_index: Optional[torch.Tensor] = None,
                has_structure: Optional[torch.Tensor] = None) -> torch.Tensor:
        plm = self.norm_plm(self.plm_enc(plm_x))
        eng = self.norm_eng(self.eng_enc(eng_x))
        if has_structure is None:
            has_structure = torch.zeros(plm.shape[0], device=plm.device)
        has_structure = has_structure.float().view(-1, 1)
        if struct_node_x is None or struct_node_x.numel() == 0 or has_structure.sum() == 0:
            struct = torch.zeros_like(plm)
        else:
            pooled = struct_node_x.mean(dim=1)
            struct = self.struct_enc(pooled)
        struct = self.norm_struct(struct)
        gate = self.modality_gate(has_structure)
        struct = struct * has_structure + gate * (1.0 - has_structure)
        x = torch.cat([plm, eng, struct], dim=-1)
        logits = self.head(x)
        return logits
