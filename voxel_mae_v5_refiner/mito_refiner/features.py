from __future__ import annotations

from collections.abc import Mapping

import torch


def extract_features(adapter, model, batch):
    """Validate the stable adapter contract: named, batch-first global features [B,D]."""
    features = adapter.extract_features(model, batch)
    if not isinstance(features, Mapping) or not features:
        raise TypeError("Backbone adapter must return a non-empty mapping of feature names to tensors")
    batch_sizes = set()
    normalized = {}
    for name, value in features.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Feature names must be non-empty strings")
        if not torch.is_tensor(value) or value.ndim != 2:
            raise ValueError(f"Feature '{name}' must be a Tensor shaped [B,D]")
        if value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError(f"Feature '{name}' has an empty batch or embedding dimension")
        batch_sizes.add(value.shape[0])
        normalized[name] = value
    if len(batch_sizes) != 1:
        raise ValueError("All adapted features must have the same batch dimension")
    return normalized
