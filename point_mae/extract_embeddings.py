"""Extract unmasked fixed-dimensional mitochondrial embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pointmae import PointCloudCache, PointMAE, model_config_from_yaml, require_trained_projection


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="NPZ output")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    features = cfg["data"]["feature_names"]
    dataset = PointCloudCache(args.cache, feature_names=features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = PointMAE(model_config_from_yaml(cfg), features).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    require_trained_projection(model, state)
    model.eval()
    embeddings, instance_ids, neuron_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model.encode(batch["xyz"].to(device), batch["attrs"].to(device))
            embeddings.append(out.cpu().numpy())
            instance_ids.append(batch["instance_id"].numpy())
            neuron_ids.append(batch["neuron_id"].numpy())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, embedding=np.concatenate(embeddings), instance_id=np.concatenate(instance_ids), neuron_id=np.concatenate(neuron_ids), organelle_type="mitochondria")
    print(f"wrote {len(np.concatenate(embeddings))} embeddings to {args.out}")


if __name__ == "__main__":
    main()
