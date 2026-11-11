"""Minimal Point-MAE style organelle representation learning components.

The cache stores coordinates in ZYX order after physical-spacing conversion.
The order is intentional: it follows the input volume convention and keeps
anisotropic spacing explicit.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


FEATURE_NAMES = ("intensity", "gradient", "distance", "surface")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def feature_indices(names: Sequence[str]) -> list[int]:
    unknown = sorted(set(names) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"Unknown point features: {unknown}")
    return [FEATURE_NAMES.index(name) for name in names]


def load_instance_map(path: str | None) -> dict[str, int | str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("neuron map must be a JSON object: instance_id -> neuron_id")
    out = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            value = value.get("neuron_id")
        if value is None:
            raise ValueError(f"Missing neuron_id for instance {key}")
        out[str(key)] = value
    return out


def stable_int(value: int | str) -> int:
    if isinstance(value, int):
        return value
    # Stable across Python processes; unlike hash(), this is reproducible.
    import hashlib

    return int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=8).digest(), "little") & ((1 << 63) - 1)


def _fps_indices(points_nm: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points_nm) == 0:
        raise ValueError("Cannot sample from an empty point set")
    if len(points_nm) <= n:
        return rng.choice(len(points_nm), size=n, replace=True)
    selected = np.empty(n, dtype=np.int64)
    selected[0] = int(rng.integers(len(points_nm)))
    distances = np.full(len(points_nm), np.inf, dtype=np.float64)
    for i in range(1, n):
        last = points_nm[selected[i - 1]]
        distances = np.minimum(distances, np.sum((points_nm - last) ** 2, axis=1))
        selected[i] = int(np.argmax(distances))
    return selected


def _sample_candidate(
    coords_zyx: np.ndarray,
    n: int,
    spacing_zyx: np.ndarray,
    rng: np.random.Generator,
    max_candidates: int,
) -> np.ndarray:
    if len(coords_zyx) == 0:
        raise ValueError("Empty sampling candidate set")
    if len(coords_zyx) > max_candidates:
        coords_zyx = coords_zyx[rng.choice(len(coords_zyx), max_candidates, replace=False)]
    points_nm = coords_zyx.astype(np.float64) * spacing_zyx
    return coords_zyx[_fps_indices(points_nm, n, rng)]


def sample_instance(
    labels: np.ndarray,
    raw: np.ndarray,
    instance_id: int,
    spacing_zyx: Sequence[float],
    coord_scale_nm: float,
    num_points: int = 2048,
    surface_points: int = 1024,
    seed: int = 0,
    max_candidates: int = 100_000,
    bbox: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``xyz_model[N,3]`` and full attributes ``[N,4]``.

    Coordinates are centered physical ZYX coordinates divided by one global
    ``coord_scale_nm``. Attributes are robust intensity, physical gradient
    magnitude, distance-to-boundary divided by the same global scale, and a
    surface flag.
    """
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if surface_points < 0 or surface_points > num_points:
        raise ValueError("surface_points must be in [0, num_points]")
    if bbox is None:
        where = np.argwhere(labels == instance_id)
        if len(where) == 0:
            raise ValueError(f"Instance {instance_id} is absent")
        lo = where.min(axis=0)
        hi = where.max(axis=0) + 1
        bbox = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    else:
        lo = np.asarray([s.start or 0 for s in bbox], dtype=np.int64)
    crop_labels = np.asarray(labels[bbox])
    crop_raw = np.asarray(raw[bbox], dtype=np.float32)
    if not np.any(crop_labels == instance_id):
        raise ValueError(f"Instance {instance_id} is absent from bbox")
    mask = crop_labels == instance_id
    local_coords = np.argwhere(mask)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface = mask & ~eroded
    surface_coords = np.argwhere(surface)
    interior_coords = np.argwhere(mask & eroded)
    if len(interior_coords) == 0:
        interior_coords = local_coords

    rng = np.random.default_rng(seed)
    n_surface = surface_points
    n_interior = num_points - n_surface
    surface_selected = _sample_candidate(surface_coords, n_surface, spacing, rng, max_candidates) if n_surface else np.empty((0, 3), dtype=np.int64)
    interior_selected = _sample_candidate(interior_coords, n_interior, spacing, rng, max_candidates) if n_interior else np.empty((0, 3), dtype=np.int64)
    selected = np.concatenate([surface_selected, interior_selected], axis=0)
    surface_flag = np.concatenate([
        np.ones(len(surface_selected), dtype=np.float32),
        np.zeros(len(interior_selected), dtype=np.float32),
    ])
    order = rng.permutation(len(selected))
    selected = selected[order]
    surface_flag = surface_flag[order]

    global_coords = selected.astype(np.float64) + lo[None, :]
    physical = global_coords * spacing[None, :]
    centroid = ((local_coords.astype(np.float64) + lo[None, :]) * spacing[None, :]).mean(axis=0)
    xyz_model = (physical - centroid[None, :]) / float(coord_scale_nm)

    values = crop_raw.reshape(-1)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    intensity = (crop_raw[tuple(selected.T)] - median) / (1.4826 * mad + 1e-6)
    intensity = np.clip(intensity, -5.0, 5.0).astype(np.float32)

    gradient_sq = np.zeros_like(crop_raw, dtype=np.float32)
    for axis, step in enumerate(spacing):
        if crop_raw.shape[axis] >= 2:
            gradient = np.gradient(crop_raw, float(step), axis=axis, edge_order=1)
            gradient_sq += gradient.astype(np.float32) ** 2
    gradient_mag = np.sqrt(gradient_sq)[tuple(selected.T)].astype(np.float32)
    distance_nm = ndimage.distance_transform_edt(mask, sampling=spacing)[tuple(selected.T)].astype(np.float32)
    distance_scaled = distance_nm / float(coord_scale_nm)
    attrs = np.column_stack([intensity, gradient_mag, distance_scaled, surface_flag]).astype(np.float32)
    return xyz_model.astype(np.float32), attrs


class PointCloudCache(Dataset):
    """Lazy dataset for one Zarr cache; no per-instance files are created."""

    def __init__(self, path: str, indices: Sequence[int] | None = None, feature_names: Sequence[str] = FEATURE_NAMES):
        import zarr

        self.group = zarr.open_group(path, mode="r")
        self.xyz = self.group["xyz"]
        self.attrs = self.group["attrs"]
        self.instance_id = self.group["instance_id"]
        self.neuron_id = self.group["neuron_id"]
        self.indices = np.arange(self.xyz.shape[0], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.attr_indices = feature_indices(feature_names)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        i = int(self.indices[item])
        return {
            "xyz": torch.from_numpy(np.asarray(self.xyz[i], dtype=np.float32)),
            "attrs": torch.from_numpy(np.asarray(self.attrs[i][:, self.attr_indices], dtype=np.float32)),
            "instance_id": int(self.instance_id[i]),
            "neuron_id": int(self.neuron_id[i]),
        }


def split_by_neuron(neuron_ids: np.ndarray, seed: int = 0, val_fraction: float = 0.1, test_fraction: float = 0.1) -> dict[str, np.ndarray]:
    unique = np.unique(neuron_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    if len(unique) < 3:
        # Useful for smoke tests only; real training should provide neuron IDs.
        n = len(neuron_ids)
        order = rng.permutation(n)
        nv = max(1, int(n * val_fraction)) if n > 1 else 0
        nt = max(1, int(n * test_fraction)) if n > 2 else 0
        return {"train": order[: n - nv - nt], "val": order[n - nv - nt : n - nt], "test": order[n - nt :]}
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = max(1, round(len(unique) * val_fraction))
    test_ids = set(unique[:n_test].tolist())
    val_ids = set(unique[n_test : n_test + n_val].tolist())
    return {
        "train": np.flatnonzero(~np.isin(neuron_ids, list(test_ids | val_ids))),
        "val": np.flatnonzero(np.isin(neuron_ids, list(val_ids))),
        "test": np.flatnonzero(np.isin(neuron_ids, list(test_ids))),
    }


def split_stratified_by_group(group_ids: np.ndarray, seed: int = 0, val_fraction: float = 0.1, test_fraction: float = 0.1) -> dict[str, np.ndarray]:
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1 or not len(group_ids):
        raise ValueError("group_ids must be a non-empty 1D array")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be non-negative and sum to less than one")
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in ("train", "val", "test")}
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * val_fraction)) if val_fraction else 0
        n_test = max(1, round(len(indices) * test_fraction)) if test_fraction else 0
        if n_val + n_test >= len(indices):
            raise ValueError(f"group {group_id!r} is too small for the requested split")
        parts["train"].append(indices[: len(indices) - n_val - n_test])
        parts["val"].append(indices[len(indices) - n_val - n_test : len(indices) - n_test])
        parts["test"].append(indices[len(indices) - n_test :])
    return {name: np.sort(np.concatenate(indices)).astype(np.int64) for name, indices in parts.items()}


def _gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    b, groups, k = indices.shape
    channels = points.shape[-1]
    expanded = points[:, None].expand(b, groups, points.shape[1], channels)
    return torch.gather(expanded, 2, indices[..., None].expand(b, groups, k, channels))


def fps_torch(points: torch.Tensor, n: int) -> torch.Tensor:
    b, total, _ = points.shape
    if n > total:
        raise ValueError("num_groups cannot exceed num_points")
    selected = torch.zeros((b, n), dtype=torch.long, device=points.device)
    distances = torch.full((b, total), float("inf"), device=points.device)
    farthest = torch.zeros(b, dtype=torch.long, device=points.device)
    batch = torch.arange(b, device=points.device)
    for i in range(n):
        selected[:, i] = farthest
        centroid = points[batch, farthest].unsqueeze(1)
        d = torch.sum((points - centroid) ** 2, dim=-1)
        distances = torch.minimum(distances, d)
        farthest = torch.argmax(distances, dim=-1)
    return selected


def augmented_point_view(
    xyz: torch.Tensor,
    attrs: torch.Tensor,
    attr_names: Sequence[str],
    keep_ratio: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample each cloud and rotate in the isotropic YX plane."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    surface_idx = attr_names.index("surface") if "surface" in attr_names else None
    xyz_views, attr_views = [], []
    for points, features in zip(xyz, attrs):
        pools = (
            [torch.where(features[:, surface_idx] > 0.5)[0], torch.where(features[:, surface_idx] <= 0.5)[0]]
            if surface_idx is not None
            else [torch.arange(len(points), device=points.device)]
        )
        selected = []
        for pool in pools:
            if not len(pool):
                continue
            n_keep = max(1, round(len(pool) * keep_ratio))
            chosen = pool[torch.randperm(len(pool), device=pool.device)[:n_keep]]
            if n_keep < len(pool):
                fill = chosen[torch.randint(n_keep, (len(pool) - n_keep,), device=pool.device)]
                chosen = torch.cat((chosen, fill))
            selected.append(chosen)
        indices = torch.cat(selected)
        indices = indices[torch.randperm(len(indices), device=indices.device)]
        view_xyz = points[indices].clone()
        theta = torch.rand((), device=points.device, dtype=points.dtype) * (2 * math.pi)
        c, s = torch.cos(theta), torch.sin(theta)
        y, x = view_xyz[:, 1].clone(), view_xyz[:, 2].clone()
        view_xyz[:, 1] = c * y - s * x
        view_xyz[:, 2] = s * y + c * x
        xyz_views.append(view_xyz)
        attr_views.append(features[indices])
    return torch.stack(xyz_views), torch.stack(attr_views)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric in-batch contrastive loss for two views of each instance."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if z1.shape != z2.shape:
        raise ValueError("the two embedding batches must have the same shape")
    batch = z1.shape[0]
    if batch < 2:
        return (z1.sum() + z2.sum()) * 0
    z = F.normalize(torch.cat((z1.float(), z2.float()), dim=0), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat((torch.arange(batch, 2 * batch), torch.arange(batch))).to(z.device)
    return F.cross_entropy(logits, targets)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointMAE(nn.Module):
    def __init__(self, cfg: Mapping, attr_names: Sequence[str]):
        super().__init__()
        self.num_groups = int(cfg["num_groups"])
        self.group_size = int(cfg["group_size"])
        self.mask_ratio = float(cfg["mask_ratio"])
        self.embedding_dim = int(cfg["embedding_dim"])
        self.attr_names = list(attr_names)
        self.attr_dim = len(self.attr_names)
        self.loss_weights = dict(cfg.get("loss_weights", {}))
        dim = int(cfg["encoder_dim"])
        decoder_dim = int(cfg["decoder_dim"])
        heads = int(cfg["encoder_heads"])
        decoder_heads = int(cfg["decoder_heads"])
        if dim % heads or decoder_dim % decoder_heads:
            raise ValueError("Transformer dimensions must be divisible by head counts")
        self.tokenizer = MLP(3 + self.attr_dim, dim, dim)
        self.position = MLP(3, dim, dim)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, int(dim * float(cfg["mlp_ratio"])), batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, int(cfg["encoder_depth"]))
        self.to_decoder = nn.Linear(dim, decoder_dim)
        self.decoder_position = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        dec_layer = nn.TransformerEncoderLayer(decoder_dim, decoder_heads, int(decoder_dim * float(cfg["mlp_ratio"])), batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(dec_layer, int(cfg["decoder_depth"]))
        self.reconstruction = nn.Linear(decoder_dim, self.group_size * (3 + self.attr_dim))
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.embedding_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def group(self, xyz: torch.Tensor, attrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center_idx = fps_torch(xyz, self.num_groups)
        centers = torch.gather(xyz, 1, center_idx[..., None].expand(-1, -1, 3))
        distances = torch.cdist(centers, xyz)
        knn_idx = distances.topk(self.group_size, dim=-1, largest=False).indices
        grouped_xyz = _gather_points(xyz, knn_idx) - centers[:, :, None, :]
        grouped_attrs = _gather_points(attrs, knn_idx)
        return centers, grouped_xyz, grouped_attrs

    def _tokens(self, centers: torch.Tensor, grouped_xyz: torch.Tensor, grouped_attrs: torch.Tensor) -> torch.Tensor:
        local = torch.cat([grouped_xyz, grouped_attrs], dim=-1)
        return self.tokenizer(local).max(dim=2).values + self.position(centers)

    def encode(self, xyz: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        centers, grouped_xyz, grouped_attrs = self.group(xyz, attrs)
        tokens = self.encoder(self._tokens(centers, grouped_xyz, grouped_attrs))
        return self.projection(tokens.mean(dim=1))

    def decode_tokens(self, tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        position = self.decoder_position(self.position(centers))
        decoded = tokens
        for layer in self.decoder.layers:
            decoded = layer(decoded + position)
        if self.decoder.norm is not None:
            decoded = self.decoder.norm(decoded)
        return decoded

    def forward(self, xyz: torch.Tensor, attrs: torch.Tensor, return_loss: bool = True) -> dict[str, torch.Tensor]:
        centers, grouped_xyz, grouped_attrs = self.group(xyz, attrs)
        tokens = self._tokens(centers, grouped_xyz, grouped_attrs)
        b, groups, _ = tokens.shape
        if not return_loss:
            return {"embedding": self.projection(self.encoder(tokens).mean(dim=1))}
        n_mask = max(1, min(groups - 1, round(groups * self.mask_ratio)))
        order = torch.argsort(torch.rand(b, groups, device=xyz.device), dim=1)
        visible_idx, masked_idx = order[:, n_mask:], order[:, :n_mask]
        visible = torch.gather(tokens, 1, visible_idx[..., None].expand(-1, -1, tokens.shape[-1]))
        visible = self.encoder(visible)
        decoder_visible = self.to_decoder(visible)
        full = self.mask_token.expand(b, groups, -1).clone().to(decoder_visible.dtype)
        full.scatter_(1, visible_idx[..., None].expand(-1, -1, full.shape[-1]), decoder_visible)
        decoded = self.decode_tokens(full, centers)
        pred = self.reconstruction(decoded).reshape(b, groups, self.group_size, 3 + self.attr_dim)
        target = torch.cat([grouped_xyz, grouped_attrs], dim=-1)
        mask = torch.zeros((b, groups), dtype=torch.bool, device=xyz.device)
        mask.scatter_(1, masked_idx, True)
        loss = masked_reconstruction_loss(pred, target, mask, self.attr_names, self.loss_weights)
        return loss


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, attr_names: Sequence[str], weights: Mapping[str, float] | None = None) -> dict[str, torch.Tensor]:
    pred_xyz = pred[..., :3][mask]
    target_xyz = target[..., :3][mask]
    pred_attr = pred[..., 3:][mask]
    target_attr = target[..., 3:][mask]
    if pred_xyz.shape[0] == 0:
        raise RuntimeError("No masked groups")
    distances = torch.cdist(pred_xyz, target_xyz)
    chamfer = distances.min(dim=-1).values.mean() + distances.min(dim=-2).values.mean()
    nearest = distances.argmin(dim=-1)
    matched = torch.gather(target_attr, 1, nearest[..., None].expand(-1, -1, target_attr.shape[-1]))
    terms: dict[str, torch.Tensor] = {"geo": chamfer}
    for i, name in enumerate(attr_names):
        if name == "surface":
            terms[name] = nn.functional.binary_cross_entropy_with_logits(pred_attr[..., i], matched[..., i])
        else:
            terms[name] = nn.functional.smooth_l1_loss(pred_attr[..., i], matched[..., i])
    weights = weights or {"geo": 1.0, "intensity": 1.0, "gradient": 1.0, "distance": 1.0, "surface": 1.0}
    total = sum(float(weights.get(k, 1.0)) * v for k, v in terms.items())
    terms["loss"] = total
    return terms


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def require_trained_projection(model: nn.Module, checkpoint: Mapping) -> None:
    optimizer = checkpoint.get("optimizer", {})
    parameter_ids = [pid for group in optimizer.get("param_groups", []) for pid in group.get("params", [])]
    named_parameters = list(model.named_parameters())
    state_ids = {str(key) for key in optimizer.get("state", {})}
    projection_ids = [
        str(pid)
        for (name, _), pid in zip(named_parameters, parameter_ids)
        if name.startswith("projection.")
    ]
    if len(parameter_ids) != len(named_parameters) or not projection_ids or not all(pid in state_ids for pid in projection_ids):
        raise RuntimeError(
            "This checkpoint has no optimizer state for every projection parameter; "
            "its 128D embeddings were not trained. Use a checkpoint trained with "
            "the two-view contrastive objective."
        )


def model_config_from_yaml(cfg: Mapping) -> dict:
    model = dict(cfg["model"])
    presets = {
        "small": dict(encoder_dim=128, encoder_depth=4, encoder_heads=4, decoder_dim=64, decoder_depth=2, decoder_heads=4),
        "medium": dict(encoder_dim=256, encoder_depth=6, encoder_heads=8, decoder_dim=128, decoder_depth=3, decoder_heads=4),
        "large": dict(encoder_dim=384, encoder_depth=10, encoder_heads=8, decoder_dim=192, decoder_depth=4, decoder_heads=8),
    }
    preset = model.get("preset", "medium")
    if preset not in presets:
        raise ValueError(f"Unknown model preset: {preset}")
    values = {**presets[preset], **model}
    values.setdefault("num_groups", 64)
    values.setdefault("group_size", 32)
    values.setdefault("mlp_ratio", 4)
    values.setdefault("mask_ratio", 0.60)
    values.setdefault("embedding_dim", 128)
    return values
