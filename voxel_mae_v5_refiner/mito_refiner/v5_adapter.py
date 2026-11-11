from __future__ import annotations

import torch

from ..model import VoxelMAE


class V5VoxelMAEAdapter:
    """Adapter from the generic Refiner contract to the copied MAE-v5 model."""

    identifier = "mito_voxel_mae_v5_refiner_r0"

    def load_model(self, checkpoint_path, device):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = state.get("model_config", state.get("config", {}).get("model"))
        if config is None:
            raise KeyError("The v5 checkpoint has no model_config")
        model = VoxelMAE(**config).to(device)
        weights = state["model"]
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError:
            if not all(str(key).startswith("module.") for key in weights):
                raise
            model.load_state_dict({str(key)[7:]: value for key, value in weights.items()}, strict=True)
        return model, state, config

    def extract_features(self, model, batch):
        if not hasattr(model, "encode_features"):
            raise TypeError("The copied v5 model must expose encode_features(batch)")
        features = model.encode_features(batch)
        expected = ("block4", "block5", "block6_mean", "final")
        missing = [name for name in expected if name not in features]
        if missing:
            raise KeyError(f"v5 feature taps missing: {missing}")
        normalized = {}
        for name in expected:
            value = features[name]
            if value.ndim == 3:
                value = value.mean(dim=1)
            if value.ndim != 2:
                raise ValueError(f"v5 feature '{name}' must become [B,D], got {tuple(value.shape)}")
            normalized[name] = value
        return normalized
