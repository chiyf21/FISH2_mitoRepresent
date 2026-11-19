"""Evaluate Point-MAE checkpoint ranks as they appear during a training run."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from data import coordinates
from train import Trainer, append


def covariance_stats(features):
    x = np.asarray(features, dtype=np.float64)
    x = x - x.mean(0)
    sigma = np.linalg.svd(x, compute_uv=False)
    eigenvalues = sigma ** 2
    total = eigenvalues.sum()
    if total == 0:
        return {"effective_rank": 0., "pc1": 0., "total_variance": 0.}
    p = eigenvalues[eigenvalues > 0] / total
    return {"effective_rank": float(np.exp(-np.sum(p * np.log(p)))),
            "pc1": float(p[0]), "total_variance": float(total / (len(x) - 1))}


@torch.inference_mode()
def prepare_groups(trainer, ids):
    chunks = []
    for start in range(0, len(ids), 16):
        groups, centers = [], []
        for i in ids[start:start + 16]:
            for view in range(2):
                coord = coordinates(trainer.points[i, view], trainer.rows[i],
                                    trainer.cfg["spacing_xyz_nm"], trainer.cfg["normalization"])
                xyz = torch.as_tensor(coord, device=trainer.device)[None]
                with torch.random.fork_rng(devices=[trainer.device.index]):
                    torch.cuda.manual_seed(20261029 + i * 1009 + view * 9176 + 17)
                    grouped, center = trainer.mae.group_divider(xyz)
                groups.append(grouped)
                centers.append(center)
        chunks.append((torch.cat(groups), torch.cat(centers)))
    return chunks


@torch.inference_mode()
def evaluate(trainer, ckpt, groups, n):
    state = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    point = trainer.mae.MAE_encoder
    prefix = "MAE_encoder."
    point.load_state_dict({k.removeprefix(prefix): v for k, v in state["model"].items()
                           if k.startswith(prefix)}, strict=True)
    layers = {layer: [] for layer in (9, 10, 11, 12)}
    with trainer.autocast():
        for g, c in groups:
            h = point.encoder(g)
            pos = point.pos_embed(c)
            for layer, block in enumerate(point.blocks.blocks, 1):
                h = block(h + pos)
                if layer in layers:
                    tokens = point.norm(h.float()) if layer == 12 else h.float()
                    layers[layer].append(tokens.mean(1).cpu().numpy())
    return {
        "epoch": state["epoch"], "updates": state["step"], "instances": n,
        "layers": {layer: covariance_stats(np.concatenate(chunks).reshape(n, 2, -1).mean(1))
                   for layer, chunks in layers.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    root = args.run.resolve()
    config = json.loads((root / "config.resolved.json").read_text())
    ids = json.loads((root / "monitor_ids.json").read_text())
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Expected distinct fixed monitor IDs")
    trainer = Trainer(config)
    trainer.mae.eval()
    groups = prepare_groups(trainer, ids)
    out = root / "mae" / "rank_every5.jsonl"
    completed = {json.loads(row)["epoch"] for row in out.read_text().splitlines()} if out.exists() else set()
    final = config["mae"]["epochs"]
    epochs = sorted({*range(5, final + 1, 5), final})
    launcher_pid = json.loads((root / "production_launcher.json").read_text())["pid"]
    for epoch in epochs:
        if epoch in completed:
            continue
        checkpoint = root / "mae" / f"epoch_{epoch:03d}.pt"
        while not checkpoint.exists():
            if not Path(f"/proc/{launcher_pid}/cmdline").exists():
                raise RuntimeError(f"Training exited before checkpoint {checkpoint.name}")
            time.sleep(5)
        row = evaluate(trainer, checkpoint, groups, len(ids))
        append(out, row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
