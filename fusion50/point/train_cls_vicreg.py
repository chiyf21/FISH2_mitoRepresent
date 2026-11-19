"""Train a shared Point CLS with VICReg and optional encoder fine-tuning."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from data import coordinates, write_json
from rank_checkpoints import covariance_stats
from train import Trainer, append, save, seed

LAYERS = (9, 10, 11, 12)
COEFFICIENTS = (25., 25., 1.)


def backbone_digest(point, *, trainable=False):
    affected = {name for name, p in point.named_parameters() if p.requires_grad}
    digest = hashlib.sha256()
    for name, value in point.state_dict().items():
        if (name in affected) != trainable:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PointCls(nn.Module):
    def __init__(self, point, initial):
        super().__init__()
        self.point = point.requires_grad_(False).eval()
        self.point_cls = nn.Parameter(initial.reshape(1, 1, -1).clone())
        width = initial.numel()
        self.heads = nn.SyncBatchNorm.convert_sync_batchnorm(nn.ModuleDict({
            str(layer): nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
                                      nn.Linear(width, 384, bias=False))
            for layer in LAYERS
        }))

    def train(self, mode=True):
        super().train(mode)
        self.point.eval()
        return self

    def forward(self, tokens, positions):
        h = torch.cat([self.point_cls.expand(len(tokens), -1, -1), tokens], 1)
        positions = torch.cat([positions.new_zeros(len(tokens), 1, positions.shape[-1]), positions], 1)
        features, projections = {}, {}
        for layer, block in enumerate(self.point.blocks.blocks, 1):
            u = h + positions
            h = checkpoint(block, u, use_reentrant=False) if self.training else block(u)
            if layer in LAYERS:
                cls = h[:, 0].float()
                cls = self.point.norm(cls) if layer == 12 else F.layer_norm(cls, (cls.shape[-1],))
                features[layer] = cls.reshape(-1, 2, cls.shape[-1])
                projections[layer] = self.heads[str(layer)](cls).reshape(-1, 2, 384)
        return features, projections


def vicreg(z):
    with torch.autocast("cuda", enabled=False):
        a, b = z[:, 0].float(), z[:, 1].float()
        invariance = F.mse_loss(a, b)
        stats = []
        for local in (a, b):
            x = torch.cat(all_gather(local), dim=0)
            x = x - x.mean(0)
            variance = F.relu(1 - (x.var(0, unbiased=True) + 1e-4).sqrt()).mean()
            c = (x.T @ x) / (len(x) - 1)
            covariance = (c - torch.diag_embed(c.diagonal())).square().sum() / x.shape[1]
            stats.append((variance, covariance))
        return torch.stack([invariance, (stats[0][0] + stats[1][0]) / 2,
                            stats[0][1] + stats[1][1]])


def score(projections):
    per_layer = {layer: vicreg(x) for layer, x in projections.items()}
    terms = torch.stack(list(per_layer.values())).mean(0)
    return (terms * terms.new_tensor(COEFFICIENTS)).sum(), per_layer


def reduce_terms(per_layer):
    parts = torch.stack([per_layer[l].detach() for l in LAYERS])
    dist.all_reduce(parts, op=dist.ReduceOp.AVG)
    return parts


def format_terms(parts):
    names = ("invariance", "variance", "covariance")
    mean = parts.mean(0).tolist()
    return {**dict(zip(names, mean)), "loss": sum(a * b for a, b in zip(mean, COEFFICIENTS)),
            "layers": {l: dict(zip(names, values)) for l, values in zip(LAYERS, parts.tolist())}}


@torch.no_grad()
def group_inputs(trainer, model, grouping, xyz):
    with trainer.autocast():
        groups, centers = grouping(xyz.flatten(0, 1))
        return model.point.encoder(groups), model.point.pos_embed(centers)


@torch.no_grad()
def evaluate(trainer, model, data, rank):
    model.eval()
    with trainer.autocast():
        raw, projected = model(*data)
    _, terms = score(projected)
    result = format_terms(reduce_terms(terms).cpu())
    metrics = {}
    for name, features in (("cls", raw), ("projection", projected)):
        metrics[name] = {}
        for layer, z in features.items():
            x = torch.cat(all_gather(z.float().mean(1)), dim=0)
            if rank == 0:
                metrics[name][layer] = covariance_stats(x.cpu().numpy())
    return result, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = args.config.parent.resolve()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if not 384 < cfg["global_batch"] or cfg["global_batch"] % world:
        raise ValueError("Use a rank-divisible global batch above projection width")
    point_run = Path(cfg["point_run"])
    point_cfg = json.loads((point_run / "config.resolved.json").read_text())
    point_cfg["device"] = f"cuda:{local_rank}"
    trainer = Trainer(point_cfg)
    ck = torch.load(cfg["point_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    if (ck["stage"], ck["epoch"]) != ("mae", 300):
        raise ValueError("Expected the canonical E300 reconstruction weights")
    point = trainer.mae.MAE_encoder
    prefix = "MAE_encoder."
    point.load_state_dict({k.removeprefix(prefix): v for k, v in ck["model"].items()
                           if k.startswith(prefix)}, strict=True)
    grouping = trainer.mae.group_divider
    del ck, trainer.mae
    seed(cfg["seed"])
    init = torch.tensor(np.load(cfg["point_cls_seed"], allow_pickle=False),
                        device=trainer.device, dtype=torch.float32)
    if init.shape != (768,) or not torch.isfinite(init).all():
        raise ValueError("Expected a finite native 768D Point CLS seed")
    model = PointCls(point, init).to(trainer.device)
    unfreeze_last = cfg.get("unfreeze_last", 0)
    if not 0 <= unfreeze_last <= len(point.blocks.blocks):
        raise ValueError("Invalid number of trainable encoder blocks")
    blocks = list(point.blocks.blocks[-unfreeze_last:]) if unfreeze_last else []
    heads = [model.point_cls, *model.heads.parameters()]
    groups = [{"params": heads, "lr": cfg["lr"]}]
    if blocks:
        for block in blocks:
            block.requires_grad_(True)
        groups.append({"params": [p for block in blocks for p in block.parameters()],
                       "lr": cfg["encoder_lr"]})
    peak_lrs = [g["lr"] for g in groups]
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=1e-4)
    val_ids = sorted(np.random.default_rng(cfg["seed"]).choice(trainer.split["val"],
                          size=cfg["global_batch"], replace=False).tolist())
    if rank == 0:
        write_json(root / "validation_ids.json", val_ids)
    with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
        torch.cuda.manual_seed(cfg["seed"] + rank)
        val_points = np.stack([[coordinates(trainer.points[i, v], trainer.rows[i],
                          point_cfg["spacing_xyz_nm"], point_cfg["normalization"]) for v in range(2)]
                          for i in val_ids[rank::world]])
        val_inputs = group_inputs(trainer, model, grouping,
                                   torch.as_tensor(val_points, device=trainer.device))
    frozen_hash = backbone_digest(point) if rank == 0 else None
    encoder_initial_hash = backbone_digest(point, trainable=True) if rank == 0 else None
    updates = 0
    for epoch in range(cfg["epochs"] + 1):
        started = time.monotonic()
        train_parts = []
        if epoch:
            ddp.train()
            batches = trainer.batch_ids(epoch, cfg["global_batch"])
            for i, ids in enumerate(batches, 1):
                elapsed = epoch - 1 + (i - 1) / len(batches)
                factor = min(1., elapsed + 1 / len(batches)) * (.5 + .5 * math.cos(math.pi * elapsed / cfg["epochs"]))
                for param_group, peak in zip(optimizer.param_groups, peak_lrs):
                    param_group["lr"] = peak * factor
                xyz = trainer.xyz(ids[rank::world], epoch, augment=True, jitter=cfg["jitter"])
                xyz = xyz - xyz.mean(2, keepdim=True)
                xyz = xyz / xyz.norm(dim=-1).amax(dim=2, keepdim=True).clamp_min(1e-8).unsqueeze(-1)
                inp = group_inputs(trainer, model, grouping, xyz)
                optimizer.zero_grad(set_to_none=True)
                with trainer.autocast():
                    _, projected = ddp(*inp)
                loss, parts = score(projected)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite VICReg loss")
                loss.backward()
                gradients = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(gradients, 5., error_if_nonfinite=True)
                if updates == 0:
                    cls_grad = float(model.point_cls.grad.norm())
                    head_grad = float(model.heads["12"][-1].weight.grad.norm())
                    encoder_grad = sum(float(p.grad.norm()) for p in point.parameters()
                                       if p.requires_grad and p.grad is not None)
                    frozen_grad = any(p.grad is not None for p in point.parameters() if not p.requires_grad)
                    if cls_grad <= 0 or head_grad <= 0 or frozen_grad or (blocks and encoder_grad <= 0):
                        raise RuntimeError("Invalid first-step gradient routing")
                optimizer.step()
                updates += 1
                summed = reduce_terms(parts).cpu()
                train_parts.append(summed)
                if rank == 0 and (updates == 1 or updates % 10 == 0):
                    row = dict(epoch=epoch, batch=i, step=updates,
                               lrs=[g["lr"] for g in optimizer.param_groups],
                               grad_norm=float(grad_norm), time=time.time(), **format_terms(summed))
                    if updates == 1:
                        row.update(cls_grad=cls_grad, projector_grad=head_grad, encoder_grad=encoder_grad,
                                   cls_delta=float((model.point_cls.detach().reshape(-1)-init).norm()),
                                   frozen_unchanged=backbone_digest(point)==frozen_hash,
                                   encoder_changed=backbone_digest(point, trainable=True)!=encoder_initial_hash)
                        if not row["frozen_unchanged"] or row["cls_delta"] <= 0 or (blocks and not row["encoder_changed"]):
                            raise RuntimeError("Point-MAE freeze or expected parameter updates check failed")
                    append(root / "steps.jsonl", row)
                    print(json.dumps(row), flush=True)
        val, ranks = evaluate(trainer, model, val_inputs, rank)
        if rank == 0:
            row = dict(epoch=epoch, step=updates, validation=val, ranks=ranks, seconds=time.monotonic()-started)
            if train_parts:
                row["train"] = format_terms(torch.stack(train_parts).mean(0))
            if epoch == cfg["epochs"] and backbone_digest(point) != frozen_hash:
                raise RuntimeError("Frozen Point-MAE weights or buffers changed")
            state = dict(stage="point_cls_vicreg", epoch=epoch, step=updates, config=cfg,
                         point_cls=model.point_cls.detach().cpu(), projector=model.heads.state_dict(),
                         optimizer=optimizer.state_dict(), backbone_sha256=frozen_hash,
                         point_last_blocks=[block.state_dict() for block in blocks])
            save(root / "checkpoints" / f"point_cls_epoch{epoch:03d}.pt", state)
            append(root / "epochs.jsonl", row)
            print(json.dumps(row), flush=True)
        dist.barrier()
    if rank == 0:
        write_json(root / "complete.json", dict(epoch=epoch, step=updates,
                   unfreeze_last=unfreeze_last, frozen_backbone_sha256=frozen_hash,
                   trained_encoder_changed=backbone_digest(point, trainable=True)!=encoder_initial_hash))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
