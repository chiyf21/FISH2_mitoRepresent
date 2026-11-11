from __future__ import annotations

from torch import nn

from .heads import IDHead


class MultiLayerID(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hidden_dim=512,
                 layers=("block4", "block5", "final"), input_dims=None):
        super().__init__()
        input_dims = input_dims or {name: input_dim for name in layers}
        missing = set(layers) - set(input_dims)
        if missing:
            raise ValueError(f"Missing input dimensions for heads: {sorted(missing)}")
        self.heads = nn.ModuleDict({
            name: IDHead(input_dims[name], output_dim, hidden_dim) for name in layers
        })

    def forward(self, features):
        return {name: self.heads[name](features[name]) for name in self.heads}

