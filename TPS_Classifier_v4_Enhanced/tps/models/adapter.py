import torch.nn as nn
class DimAdapter(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
    def forward(self, x):  # x: [B,Din]
        return self.proj(x)

