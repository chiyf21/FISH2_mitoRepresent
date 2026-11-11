from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

import torch
from torch import nn


class BackboneAdapter(Protocol):
    """Model-specific loading and feature extraction; the refiner only sees [B,D] tensors."""

    identifier: str

    def load_model(self, checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict, Any]: ...

    def extract_features(self, model: nn.Module, batch: Any) -> dict[str, torch.Tensor]: ...


def load_adapter(spec: str) -> BackboneAdapter:
    module_name, separator, class_name = spec.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("Adapter must be specified as 'importable.module:ClassName'")
    adapter_class = getattr(import_module(module_name), class_name)
    adapter = adapter_class()
    missing = [name for name in ("identifier", "load_model", "extract_features")
               if not hasattr(adapter, name)]
    if missing:
        raise TypeError(f"Backbone adapter {spec} is missing: {', '.join(missing)}")
    if not isinstance(adapter.identifier, str) or not adapter.identifier:
        raise TypeError("Backbone adapter identifier must be a non-empty string")
    return adapter
