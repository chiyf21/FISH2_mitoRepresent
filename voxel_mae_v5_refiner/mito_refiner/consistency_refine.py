"""Positive-only two-view consistency pilot for the v5 encoder.

This is deliberately separate from the NNA trainer: no projector, negatives,
queue, VICReg terms, or MAE reconstruction loss are used in this refinement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

def cosine_consistency(z):
    """Positive-only loss for instance-major [B, 2, D] embeddings."""
    if z.ndim != 3 or z.shape[1] != 2:
        raise ValueError(f"Expected [B,2,D] paired embeddings, got {tuple(z.shape)}")
    return (1.0 - F.cosine_similarity(z[:, 0].float(), z[:, 1].float(), dim=-1)).mean()


def self_test():
    z = torch.randn(4, 2, 32, requires_grad=True)
    loss = cosine_consistency(z)
    loss.backward()
    assert torch.isfinite(loss) and z.grad is not None
    assert z.grad[:, 0].abs().sum() > 0 and z.grad[:, 1].abs().sum() > 0
    same = torch.randn(4, 1, 32).expand(-1, 2, -1)
    assert cosine_consistency(same).abs() < 1e-6
    print("PASS cosine consistency shape, identity, finiteness, and two-view gradients")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--cache")
    p.add_argument("--checkpoint")
    p.add_argument("--out")
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-updates", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--slice-microbatch", type=int, default=8)
    p.add_argument("--tile-batch", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    from .backbone import load_adapter
    from .data_utils import groups_of, load_paired_items, make_dataset
    from .features import extract_features
    from ..data import collate_batch, to_device
    from .training_step import configure_encoder, precision_context
    if not a.cache or not a.checkpoint or not a.out:
        p.error("--cache, --checkpoint, and --out are required")
    if min(a.limit, a.batch_size, a.max_updates) < 1 or a.batch_size < 2 or a.lr <= 0:
        p.error("limit, updates, batch size >=2, and learning rate must be positive")
    device = torch.device(a.device)
    if device.type == "cpu" and a.precision != "fp32":
        p.error("Use --precision fp32 on CPU")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)

    adapter = load_adapter("mito_voxel_mae_v5_refiner_r0.mito_refiner.v5_adapter:V5VoxelMAEAdapter")
    model, _, model_config = adapter.load_model(a.checkpoint, device)
    model.slice_microbatch, model.tile_batch = a.slice_microbatch, a.tile_batch
    model.activation_checkpointing = True
    encoder_parameters = configure_encoder(model, ("final",), activation_checkpointing=True)
    optimizer = torch.optim.AdamW(encoder_parameters, lr=a.lr, weight_decay=0.05)
    ds = make_dataset(a.cache, "train", a.limit, a.seed, train_views=True)
    ids = [str(r["record_id"]) for r in ds.records]
    bodies = [str(r["body_id"]) for r in ds.records]
    if len(ids) < a.batch_size or len(set(ids)) != len(ids):
        raise ValueError("Not enough unique training instances for the requested batch")

    config = vars(a).copy()
    config.update({"objective": "mean(1-cos(z_view0,z_view2))",
                   "representation": "final 256D bottleneck",
                   "other_losses": [], "train_instances": len(ids),
                   "train_bodies": len(set(bodies)),
                   "views": "view 0 and mild XY-rotated view 2; existing v5 augmentations"})
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_path = out / "training_metrics.jsonl"
    updates = 0
    processed_steps = 0
    start = time.perf_counter()
    model.eval()
    while updates < a.max_updates:
        ds.epoch = processed_steps // max(1, (len(ids) + a.batch_size - 1) // a.batch_size)
        order = np.random.default_rng(a.seed + ds.epoch).permutation(len(ids)).tolist()
        batches = groups_of(order, a.batch_size)
        for group in batches:
            if updates >= a.max_updates:
                break
            items, kept, skipped = load_paired_items(ds, group)
            processed_steps += 1
            if len(kept) < 2:
                continue
            batch = to_device(collate_batch(items), device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with precision_context(device, a.precision):
                features = extract_features(adapter, model, batch)
                z = features["final"].float().reshape(len(kept), 2, -1)
                loss = cosine_consistency(z)
            loss.backward()
            grad_sq = [p.grad.detach().float().square().sum() for p in encoder_parameters if p.grad is not None]
            grad_norm = float(torch.sqrt(torch.stack(grad_sq).sum())) if grad_sq else 0.0
            if not np.isfinite(float(loss.detach())) or not np.isfinite(grad_norm) or grad_norm <= 0:
                raise FloatingPointError("Consistency loss or encoder gradient is non-finite/zero")
            torch.nn.utils.clip_grad_norm_(encoder_parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            row = {"update": updates, "processed_steps": processed_steps,
                   "loss_consistency": float(loss.detach()), "encoder_grad_norm_preclip": grad_norm,
                   "batch_instances": len(kept), "batch_views": len(items),
                   "skipped_pairs": skipped, "runtime_s": time.perf_counter() - start,
                   "gpu_peak_allocated_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                                               if device.type == "cuda" else 0.0)}
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            if updates == 1 or updates % 10 == 0 or updates == a.max_updates:
                print(json.dumps(row), flush=True)

    state = {
        "phase": "positive_only_consistency",
        "model": model.state_dict(), "model_config": model_config,
        "backbone_adapter": adapter.identifier, "source_checkpoint": str(a.checkpoint),
        "source_checkpoint_sha256": sha256(a.checkpoint), "config": config,
        "head_layers": [], "bottleneck_refined": True,
        "embedding_adapter": None, "ema_model": None,
        "record_ids": ids, "body_ids": bodies, "optimizer": optimizer.state_dict(),
        "optimizer_updates": updates, "consistency_objective": config["objective"],
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
    }
    torch.save(state, out / "consistency_refined.pt")
    print(json.dumps({"status": "complete", "updates": updates,
                      "runtime_s": time.perf_counter() - start,
                      "checkpoint": str(out / "consistency_refined.pt")}))


if __name__ == "__main__":
    main()
