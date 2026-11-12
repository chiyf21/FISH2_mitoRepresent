"""Symmetric in-batch cross-view InfoNCE for frozen-MAE CLS features."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class InBatchContrastiveLoss(nn.Module):
    """Each instance's opposite view is the sole positive; other IDs are negatives."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    @staticmethod
    def _gather_with_grad(x: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return x
        from torch.distributed.nn.functional import all_gather
        return torch.cat(all_gather(x), dim=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[1] != 2:
            raise ValueError("expected projected features with shape [B, 2 views, D]")
        z = F.normalize(z, dim=-1)
        z_all = F.normalize(self._gather_with_grad(z), dim=-1)
        batch = z.shape[0]
        offset = dist.get_rank() * batch if dist.is_available() and dist.is_initialized() else 0
        labels = torch.arange(batch, device=z.device) + offset

        # View 0 predicts view 1 and vice versa. The denominator contains only
        # the opposite view of every instance, so each row has one positive and
        # all other instances in the global batch as negatives.
        logits_01 = z[:, 0] @ z_all[:, 1].T / self.temperature
        logits_10 = z[:, 1] @ z_all[:, 0].T / self.temperature
        return 0.5 * (F.cross_entropy(logits_01, labels) +
                      F.cross_entropy(logits_10, labels))


class InBatchContrastiveHead(nn.Module):
    """Shared-view projector; intentionally no predictor and no feature queue."""

    def __init__(self, input_dim: int = 384, output_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048, affine=False), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048, affine=False), nn.GELU(),
            nn.Linear(2048, output_dim), nn.BatchNorm1d(output_dim, affine=False),
        )
        self.projector.apply(self._init_module)

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 2:
            raise ValueError("features must have shape [B, 2 views, D]")
        batch, views, dim = features.shape
        # Keep both views in the same local projector/BatchNorm call.
        flat = features.transpose(0, 1).reshape(views * batch, dim)
        projected = self.projector(flat)
        return projected.reshape(views, batch, -1).transpose(0, 1)


class InBatchContrastiveStack(nn.Module):
    """Independent projector and symmetric InfoNCE loss for each selected layer."""

    def __init__(self, layers=(9, 10, 11, 12), temperature: float = 0.2):
        super().__init__()
        self.layers = tuple(layers)
        self.heads = nn.ModuleDict({
            str(layer): InBatchContrastiveHead() for layer in self.layers
        })
        self.objective = InBatchContrastiveLoss(temperature=temperature)

    def forward(self, features, instance_ids=None):
        if set(self.layers) - set(features):
            raise ValueError("missing one or more layer features")
        if instance_ids is not None and len(instance_ids) != next(iter(features.values())).shape[0]:
            raise ValueError("instance ID count does not match local batch")
        parts = {
            str(layer): self.objective(self.heads[str(layer)](features[layer]))
            for layer in self.layers
        }
        return sum(parts.values()), parts


def selfcheck():
    loss = InBatchContrastiveLoss(temperature=0.2)
    z = torch.tensor([
        [[1.0, 0.0], [0.99, 0.01]],
        [[0.0, 1.0], [0.01, 0.99]],
    ])
    value = loss(z)
    assert torch.isfinite(value) and value.item() < 0.05, value


if __name__ == "__main__":
    selfcheck()
    print("in-batch contrastive selfcheck: PASS")
