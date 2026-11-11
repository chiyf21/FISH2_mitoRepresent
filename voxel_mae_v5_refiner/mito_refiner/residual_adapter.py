"""Small residual adapter for the final 256D embedding."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualEmbeddingAdapter(nn.Module):
    def __init__(self, dim=256, hidden_dim=256, alpha_init=0.1):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_init = float(alpha_init)
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)
        self.alpha = nn.Parameter(torch.tensor(self.alpha_init))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        residual = self.fc2(F.gelu(self.fc1(z)))
        return F.normalize(z + self.alpha * residual, dim=-1, eps=1e-8)

    def config(self):
        return {"dim": self.dim, "hidden_dim": self.hidden_dim,
                "alpha_init": self.alpha_init}
