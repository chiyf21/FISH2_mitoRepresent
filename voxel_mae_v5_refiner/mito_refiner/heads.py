from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class IDHead(nn.Module):
    """MIM-Refiner-style 3-layer projector + 2-layer predictor."""

    def __init__(self, input_dim=256, output_dim=256, hidden_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        projected = self.projector(x)
        predicted = self.predictor(projected)
        # Official NnclrHead uses detached projector outputs as queue/swap keys
        # when no momentum projector is configured. The predictor still sees
        # the live projector tensor, so gradients reach the projector/encoder.
        preswap_key = F.normalize(projected, dim=-1).detach()
        predicted = F.normalize(predicted, dim=-1)
        return preswap_key, predicted
