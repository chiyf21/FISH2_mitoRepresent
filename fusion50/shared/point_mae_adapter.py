"""Isolated DDP adapter for the cloned Point-MAE and mito surface point clouds.

This reuses the upstream Point_MAE network, but replaces its CUDA-only FPS/KNN
and Chamfer extensions with small PyTorch equivalents. The upstream checkout is
never modified. Coordinates in the cache are restored to physical XYZ scale
from the manifest and divided by one fixed global scale (not per-instance).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler


def install_optional_extension_stubs() -> None:
    """Allow importing upstream model classes without compiling CUDA ops."""
    # The model definition needs only EasyDict's attribute-access behavior;
    # the actual project config loader is not used by this adapter.
    ed_mod = types.ModuleType("easydict")

    class EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    ed_mod.EasyDict = EasyDict
    sys.modules.setdefault("easydict", ed_mod)

    termcolor_mod = types.ModuleType("termcolor")
    termcolor_mod.colored = lambda value, *args, **kwargs: value
    sys.modules.setdefault("termcolor", termcolor_mod)

    knn_mod = types.ModuleType("knn_cuda")

    class UnusedKNN:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def __call__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("Upstream KNN is replaced by TorchGroup in this adapter")

    knn_mod.KNN = UnusedKNN
    sys.modules.setdefault("knn_cuda", knn_mod)

    ext_pkg = types.ModuleType("extensions")
    ext_pkg.__path__ = []
    chamfer_mod = types.ModuleType("extensions.chamfer_dist")

    class UnusedChamfer(torch.nn.Module):
        def cuda(self, *args: Any, **kwargs: Any):
            # Upstream constructs its compiled loss with .cuda() in __init__;
            # this inert placeholder is replaced immediately after construction.
            return self

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise RuntimeError("Upstream Chamfer is replaced by TorchChamferL2")

    chamfer_mod.ChamferDistanceL1 = UnusedChamfer
    chamfer_mod.ChamferDistanceL2 = UnusedChamfer
    sys.modules.setdefault("extensions", ext_pkg)
    sys.modules.setdefault("extensions.chamfer_dist", chamfer_mod)

    p2_mod = types.ModuleType("pointnet2_ops")
    p2_mod.__path__ = []
    p2_utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
    p2_utils.furthest_point_sample = lambda *a, **k: None
    p2_utils.gather_operation = lambda *a, **k: None
    p2_mod.pointnet2_utils = p2_utils
    sys.modules.setdefault("pointnet2_ops", p2_mod)
    sys.modules.setdefault("pointnet2_ops.pointnet2_utils", p2_utils)


class TorchGroup(torch.nn.Module):
    """FPS centers and KNN groups, matching Point-MAE's B,G,S,3 interface."""

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = xyz.shape
        if n < max(self.num_group, self.group_size):
            raise ValueError(f"need >= {max(self.num_group, self.group_size)} points, got {n}")
        batch = torch.arange(b, device=xyz.device)
        xyz32 = xyz.float()
        farthest = torch.randint(n, (b,), device=xyz.device)
        min_dist = torch.full((b, n), float("inf"), device=xyz.device)
        center_idx = torch.empty((b, self.num_group), dtype=torch.long, device=xyz.device)
        for i in range(self.num_group):
            center_idx[:, i] = farthest
            c = xyz32[batch, farthest].unsqueeze(1)
            d = ((xyz32 - c) ** 2).sum(-1)
            min_dist = torch.minimum(min_dist, d)
            farthest = min_dist.argmax(-1)
        center = xyz[batch[:, None], center_idx]
        # Use physical-space Euclidean distance for neighborhoods.
        dmat = torch.cdist(center.float(), xyz32)
        nn_idx = dmat.topk(self.group_size, dim=-1, largest=False, sorted=False).indices
        neighborhood = xyz[batch[:, None, None], nn_idx]
        return neighborhood - center.unsqueeze(2), center


class TorchChamferL2(torch.nn.Module):
    """Symmetric squared Chamfer distance, averaged over points and batch."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(pred.float(), target.float()).square()
        return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    with (data_dir / "manifest.tsv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return metadata, rows


def validate_dataset(data_dir: Path, full_scan: bool = True) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata, rows = load_metadata(data_dir)
    if "surface" not in metadata.get("point_sampling", "").lower():
        raise ValueError("Refusing to train: dataset metadata does not identify surface-only sampling")
    arr_path = data_dir / metadata.get("array_file", "pointclouds.npy")
    arr = np.load(arr_path, mmap_mode="r")
    expected = tuple(metadata.get("array_shape", ()))
    if arr.ndim != 4 or arr.shape[1:] != (2, 2048, 3):
        raise ValueError(f"expected [instances,2,2048,3], found {arr.shape}")
    if expected and tuple(arr.shape) != expected:
        raise ValueError(f"manifest shape {expected} does not match array {arr.shape}")
    if arr.shape[0] != len(rows) or arr.shape[0] != int(metadata["instances"]):
        raise ValueError(f"instance count mismatch: array={arr.shape[0]}, rows={len(rows)}")
    required = {"instance_index", "body_id", "cell_type", "normalization_radius_voxels", "surface_voxel_count"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    body_ids = [r["body_id"] for r in rows]
    if len(set(body_ids)) < 3:
        raise ValueError("need at least three distinct body_id groups for train/val/test")
    radii = np.asarray([float(r["normalization_radius_voxels"]) for r in rows])
    surface_counts = np.asarray([int(r["surface_voxel_count"]) for r in rows])
    if not np.isfinite(radii).all() or np.any(radii <= 0) or np.any(surface_counts <= 0):
        raise ValueError("invalid radius or surface voxel count in manifest")
    if full_scan:
        for start in range(0, arr.shape[0], 256):
            if not np.isfinite(arr[start:start + 256]).all():
                raise ValueError(f"non-finite point coordinates in instances {start}:"
                                 f"{min(start + 256, arr.shape[0])}")
    # Lightweight normalization sanity check; unit-sphere coordinates should lie in [0,1].
    probe = np.asarray(arr[::max(1, arr.shape[0] // 128), 0], dtype=np.float32)
    max_norm = np.linalg.norm(probe, axis=-1).max()
    if not np.isfinite(max_norm) or max_norm > 1.05:
        raise ValueError(f"unexpected normalized coordinate radius (max={max_norm:.4g})")
    return metadata, rows


def make_group_split(rows: list[dict[str, str]], seed: int, val_fraction: float, test_fraction: float) -> dict[str, list[int]]:
    """Deterministic 80/10/10-style split with no body_id crossing splits."""
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["body_id"], []).append(i)
    body_ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(body_ids)
    n_test = max(1, round(len(body_ids) * test_fraction))
    n_val = max(1, round(len(body_ids) * val_fraction))
    if n_test + n_val >= len(body_ids):
        raise ValueError("too few body_id groups for requested split")
    test_bodies = set(body_ids[:n_test].tolist())
    val_bodies = set(body_ids[n_test:n_test + n_val].tolist())
    train_bodies = set(body_ids[n_test + n_val:].tolist())
    split = {
        "train": sorted(i for b in train_bodies for i in groups[str(b)]),
        "val": sorted(i for b in val_bodies for i in groups[str(b)]),
        "test": sorted(i for b in test_bodies for i in groups[str(b)]),
        "body_ids": {"train": sorted(train_bodies), "val": sorted(val_bodies), "test": sorted(test_bodies)},
    }
    assert not (train_bodies & val_bodies or train_bodies & test_bodies or val_bodies & test_bodies)
    return split


class SurfacePointDataset(Dataset):
    def __init__(self, data_dir: Path, rows: list[dict[str, str]], indices: list[int],
                 spacing_xyz_nm: tuple[float, float, float], common_scale_nm: float,
                 training: bool, unit_sphere: bool = False,
                 num_points: int = 2048, seed: int = 42):
        self.data_dir = data_dir
        self.array_path = data_dir / "pointclouds.npy"
        self.points: np.ndarray | None = None
        self.rows = rows
        self.indices = indices
        self.spacing = np.asarray(spacing_xyz_nm, dtype=np.float32)
        self.common_scale = float(common_scale_nm)
        self.training = training
        self.unit_sphere = bool(unit_sphere)
        self.num_points = int(num_points)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> torch.Tensor:
        if self.points is None:
            self.points = np.load(self.array_path, mmap_mode="r")
        ix = self.indices[item]
        row = self.rows[ix]
        view = random.randrange(2) if self.training else 0
        # The cache stores (voxel_xyz - instance_center_xyz) / radius_voxels.
        # First restore anisotropic physical XYZ dimensions. The standard
        # point-cloud ablation then recenters each sampled view and scales it
        # to a unit sphere, removing instance-specific absolute size.
        xyz_unit = np.asarray(self.points[ix, view], dtype=np.float32)
        radius_vox = float(row["normalization_radius_voxels"])
        xyz = xyz_unit * radius_vox * self.spacing
        if self.unit_sphere:
            xyz = xyz - xyz.mean(axis=0, keepdims=True)
            max_radius = float(np.linalg.norm(xyz, axis=-1).max())
            xyz = xyz / max(max_radius, 1e-8)
        else:
            xyz = xyz / self.common_scale
        if self.training:
            # XY spacing is isotropic (8 nm), so in-plane rotation is physically valid.
            angle = random.random() * (2.0 * math.pi)
            c, s = math.cos(angle), math.sin(angle)
            x = xyz[:, 0].copy()
            y = xyz[:, 1].copy()
            xyz[:, 0] = c * x - s * y
            xyz[:, 1] = s * x + c * y
            if random.random() < 0.5:
                xyz[:, 0] *= -1
            if random.random() < 0.5:
                xyz[:, 1] *= -1
        # Keep the original full-view physical/unit-sphere normalization above,
        # then take a reproducible, uniform subset to isolate input point count.
        if self.num_points > xyz.shape[0]:
            raise ValueError(f"requested {self.num_points} points from a cloud of {xyz.shape[0]}")
        if self.num_points < xyz.shape[0]:
            rng = np.random.default_rng(self.seed + 2 * ix + view)
            selected = rng.choice(xyz.shape[0], self.num_points, replace=False)
            xyz = xyz[selected]
        return torch.from_numpy(np.ascontiguousarray(xyz))


class StridedSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int):
        self.indices = list(range(rank, length, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def lr_for_epoch(epoch: int, epochs: int, warmup_epochs: int, base_lr: float, min_lr: float) -> float:
    if epoch < warmup_epochs:
        return 1e-6 + (base_lr - 1e-6) * ((epoch + 1) / max(1, warmup_epochs))
    span = max(1, epochs - warmup_epochs)
    progress = min(1.0, (epoch - warmup_epochs) / span)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def optimizer_for(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "token" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}],
        lr=lr, betas=(0.9, 0.999), eps=1e-8,
    )


def build_model(repo: Path, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    install_optional_extension_stubs()
    sys.path.insert(0, str(repo))
    from models.Point_MAE import Point_MAE  # noqa: E402

    cfg = types.SimpleNamespace(
        group_size=args.group_size,
        num_group=args.num_group,
        loss="cdl2",
        transformer_config=types.SimpleNamespace(
            mask_ratio=args.mask_ratio,
            mask_type="rand",
            trans_dim=args.trans_dim,
            encoder_dims=args.encoder_dims,
            depth=args.depth,
            drop_path_rate=args.drop_path,
            num_heads=args.heads,
            decoder_depth=args.decoder_depth,
            decoder_num_heads=args.decoder_heads,
        ),
    )
    model = Point_MAE(cfg)
    model.group_divider = TorchGroup(args.num_group, args.group_size)
    model.loss_func = TorchChamferL2()
    return model.to(device)


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path, help="clean cloned Point-MAE checkout")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--smoke-ddp-steps", type=int, default=0,
                   help="run this many real distributed optimizer steps, then exit without checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-per-gpu", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--spacing-xyz-nm", type=float, nargs=3, default=(8.0, 8.0, 30.0))
    p.add_argument("--common-scale-nm", type=float, default=10000.0)
    p.add_argument("--unit-sphere", action="store_true",
                   help="after physical XYZ correction, center each view and normalize its max radius to 1")
    p.add_argument("--num-group", type=int, default=64)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--num-points", type=int, default=2048,
                   help="number of cached surface points passed to Point-MAE")
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--trans-dim", type=int, default=384)
    p.add_argument("--encoder-dims", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--decoder-depth", type=int, default=4)
    p.add_argument("--decoder-heads", type=int, default=6)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    metadata, rows = validate_dataset(args.data, full_scan=(world_size == 1))
    if args.validate_only:
        print(json.dumps({"status": "PASS", "shape": metadata["array_shape"],
                          "instances": len(rows), "bodies": len(set(r["body_id"] for r in rows)),
                          "point_sampling": metadata["point_sampling"]}, ensure_ascii=False), flush=True)
        return
    if not args.repo.is_dir() or not (args.repo / "models/Point_MAE.py").is_file():
        raise FileNotFoundError(f"not a Point-MAE checkout: {args.repo}")
    if args.epochs < 1 or args.batch_per_gpu < 1 or args.num_points < 1:
        raise ValueError("epochs, batch-per-gpu, and num-points must be positive")
    if args.num_points > int(metadata["array_shape"][2]):
        raise ValueError(f"num-points={args.num_points} exceeds cached point count {metadata['array_shape'][2]}")

    if args.smoke_only:
        torch.set_num_threads(2)
        model = build_model(args.repo.resolve(), args, torch.device("cpu"))
        model.train()
        sample = torch.randn(2, 2048, 3) * 0.2
        loss = model(sample)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite smoke-test loss")
        print(json.dumps({"status": "PASS", "smoke_loss": float(loss.detach()),
                          "gradients": "backward succeeded", "device": "cpu"}), flush=True)
        return

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    split = make_group_split(rows, args.seed, args.val_fraction, args.test_fraction)
    train_ds = SurfacePointDataset(args.data, rows, split["train"], tuple(args.spacing_xyz_nm),
                                   args.common_scale_nm, training=True,
                                   unit_sphere=args.unit_sphere,
                                   num_points=args.num_points, seed=args.seed)
    val_ds = SurfacePointDataset(args.data, rows, split["val"], tuple(args.spacing_xyz_nm),
                                 args.common_scale_nm, training=False,
                                 unit_sphere=args.unit_sphere,
                                 num_points=args.num_points, seed=args.seed)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        val_sampler = StridedSampler(len(val_ds), rank, world_size)
    else:
        train_sampler = None
        val_sampler = None
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader_kwargs = dict(batch_size=args.batch_per_gpu, num_workers=args.workers,
                         pin_memory=True, worker_init_fn=worker_seed, generator=generator,
                         persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=train_sampler is None,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, drop_last=False,
                            **loader_kwargs)

    # Same initialization on all ranks before DDP synchronizes parameters.
    seed_all(args.seed)
    model = build_model(args.repo.resolve(), args, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=True, find_unused_parameters=False)
    bare_model = model.module if distributed else model
    optimizer = optimizer_for(bare_model, args.lr, args.weight_decay)
    scaler = None  # A100 BF16 autocast does not need GradScaler.

    if args.smoke_ddp_steps > 0:
        model.train()
        last_loss = 0.0
        for step, points in enumerate(train_loader):
            points = points.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(points)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
            if step + 1 >= args.smoke_ddp_steps:
                break
        if distributed:
            dist.barrier()
        if rank == 0:
            print(json.dumps({"status": "PASS", "ddp_steps": args.smoke_ddp_steps,
                              "world_size": world_size, "batch_per_gpu": args.batch_per_gpu,
                              "last_loss_rank0": last_loss, "device": str(device)}), flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    out = args.output
    if rank == 0:
        unexpected = [x.name for x in out.iterdir()
                      if x.name != "code" and not x.name.endswith(".log")] if out.exists() else []
        if unexpected and args.resume is None:
            raise FileExistsError(f"output directory is non-empty: {out}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
        config = vars(args).copy()
        config["repo"] = str(args.repo.resolve())
        config["data"] = str(args.data.resolve())
        config["output"] = str(out.resolve())
        config["dataset_manifest"] = metadata
        config["world_size"] = world_size
        config["effective_batch"] = args.batch_per_gpu * world_size
        config["optimizer_updates_per_epoch"] = len(train_loader)
        (out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    if distributed:
        dist.barrier()

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        bare_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("best_val", best_val))

    log_path = out / "metrics.jsonl"
    if rank == 0 and not log_path.exists():
        log_path.touch()
    if distributed:
        dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        lr = lr_for_epoch(epoch, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        train_sum = torch.zeros(2, device=device, dtype=torch.float64)
        for points in train_loader:
            points = points.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(points)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_sum[0] += loss.detach().double()
            train_sum[1] += 1
        model.eval()
        val_sum = torch.zeros(2, device=device, dtype=torch.float64)
        with torch.no_grad():
            for points in val_loader:
                points = points.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Bare model avoids DDP buffer broadcasts if ranks receive
                    # different numbers of validation batches.
                    loss = bare_model(points)
                val_sum[0] += loss.detach().double() * points.shape[0]
                val_sum[1] += points.shape[0]
        if distributed:
            dist.all_reduce(train_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_sum, op=dist.ReduceOp.SUM)
        train_loss = (train_sum[0] / train_sum[1].clamp_min(1)).item()
        val_loss = (val_sum[0] / val_sum[1].clamp_min(1)).item()
        elapsed = time.time() - t0
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        row = {"epoch": epoch + 1, "global_step": global_step, "optimizer_updates_this_epoch": len(train_loader),
               "effective_batch": args.batch_per_gpu * world_size, "lr": lr,
               "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val,
               "seconds": elapsed}
        if rank == 0:
            print(json.dumps(row), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            state = {"epoch": epoch, "global_step": global_step, "model": bare_model.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_val": best_val,
                     "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "split_body_ids": split["body_ids"]}
            save_state(out / f"epoch_{epoch + 1:03d}.pt", state)
            save_state(out / "last.pt", state)
            if improved:
                save_state(out / "best.pt", state)
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
