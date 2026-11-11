"""Run one in-memory optimizer step on real cached point clouds."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pointmae import PointCloudCache, PointMAE, model_config_from_yaml, require_trained_projection
from train import run_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = cfg["data"]["feature_names"]
    dataset = PointCloudCache(args.cache, np.arange(args.samples), features)
    loader = DataLoader(dataset, batch_size=args.samples, shuffle=False)
    model = PointMAE(model_config_from_yaml(cfg), features).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = [parameter.detach().clone() for parameter in model.projection.parameters()]
    train_cfg = cfg["train"]
    stats = run_epoch(
        model,
        loader,
        optimizer,
        scaler,
        device,
        amp=False,
        train=True,
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        contrastive_weight=float(train_cfg.get("contrastive_weight", 0.1)),
        contrastive_temperature=float(train_cfg.get("contrastive_temperature", 0.1)),
        view_keep_ratio=float(train_cfg.get("view_keep_ratio", 0.9)),
    )
    changed = [
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.projection.parameters())
    ]
    assert all(changed), "At least one projection parameter did not update"
    require_trained_projection(model, {"optimizer": optimizer.state_dict()})
    print(json.dumps({"device": str(device), "stats": stats, "projection_updated": changed}, indent=2))


if __name__ == "__main__":
    main()
