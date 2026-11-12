import numpy as np
import torch
SEED=20261027
def point_xyz(record, points, view, device, rotation=None, jitter=0.0, seed=0):
    raw = np.asarray(points[record["point_row"], view], dtype=np.float32).copy()
    raw *= record["normalization_radius_voxels"] * np.asarray(record["spacing_xyz_nm"], dtype=np.float32)
    raw -= raw.mean(axis=0, keepdims=True)
    radius = max(float(np.linalg.norm(raw, axis=-1).max()), 1e-8)
    xyz = raw / radius
    if rotation is not None:
        xyz = xyz @ rotation
    rng = np.random.default_rng(seed)
    if jitter:
        xyz += rng.normal(0.0, jitter, size=xyz.shape).astype(np.float32)
    xyz -= xyz.mean(axis=0, keepdims=True)
    xyz /= max(float(np.linalg.norm(xyz, axis=-1).max()), 1e-8)
    return torch.as_tensor(xyz, dtype=torch.float32, device=device).unsqueeze(0)

def view_transform(record, points, view, epoch, training, device):
    seed = SEED + int(record["cohort_index"]) * 1009 + view * 9176 + (epoch + 1) * 100003
    rng = np.random.default_rng(seed)
    if training:
        k = int(rng.integers(0, 4))
        flip_x, flip_y = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
        noise, jitter = 0.01, 0.01
    else:
        # Deterministic validation views: independent surface samples and mild noise only.
        k, flip_x, flip_y = 0, False, False
        noise, jitter = 0.003, 0.002

    em = np.asarray(np.load(record["em_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32).copy()
    mask = np.asarray(np.load(record["mask_path"], mmap_mode="r", allow_pickle=False), dtype=np.uint8).copy()
    if em.ndim != 3 or mask.shape != em.shape or not np.isfinite(em).all():
        raise ValueError(f"Invalid EM/mask arrays for {record['key']}")
    if not np.all((mask == 0) | (mask == 1)) or not mask.any():
        raise ValueError(f"Mask is not nonempty binary for {record['key']}")

    quarter = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)
    for _ in range(k):
        rotation = rotation @ quarter
    if flip_x:
        rotation = rotation @ np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    if flip_y:
        rotation = rotation @ np.diag([1.0, -1.0, 1.0]).astype(np.float32)

    if k:
        em, mask = np.rot90(em, k, axes=(-2, -1)).copy(), np.rot90(mask, k, axes=(-2, -1)).copy()
    if flip_x:
        em, mask = em[..., ::-1].copy(), mask[..., ::-1].copy()
    if flip_y:
        em, mask = em[..., ::-1, :].copy(), mask[..., ::-1, :].copy()
    em -= em.min()
    em /= max(float(em.max()), 1e-8)
    em = np.clip(em + rng.normal(0.0, noise, size=em.shape).astype(np.float32), 0.0, 1.0)
    xyz = point_xyz(record, points, view, device, rotation, jitter, seed + 31)
    return (torch.as_tensor(em, dtype=torch.float32, device=device),
            torch.as_tensor(mask, dtype=torch.uint8, device=device), xyz, seed)

def estimate_point_cls_stats(point, grouping, records, points, device, limit=64):
    total = torch.zeros(384, dtype=torch.float64, device=device)
    total_sq = torch.zeros_like(total)
    count = 0
    for record in records[:min(limit, len(records))]:
        xyz = point_xyz(record, points, 0, device)
        groups, centers = grouping(xyz)
        before_block1 = point.encoder(groups) + point.pos_embed(centers)
        values = before_block1[0].double()
        total += values.sum(dim=0)
        total_sq += values.square().sum(dim=0)
        count += values.shape[0]
    if not count:
        raise ValueError("No train-split Point tokens available for CLS initialization")
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(), std.float(), count
