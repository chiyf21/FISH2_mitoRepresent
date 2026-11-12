"""Export 384-dimensional L12 Fusion50 vectors in manifest order."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train import core, load_models, load_points, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = json.loads(args.config.read_text())
    rows = records(config)
    points = load_points(config, rows)
    device = torch.device("cuda:0")
    point, grouping, image = load_models(config, device)
    model = core.System(point, image).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["stage"] != "refine":
        raise ValueError("Use a fusion-stage checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.phase = "refine"
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output / "fused_12.partial.npy"
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32",
                                      shape=(len(rows), 384))
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            inputs = core.prepare(batch, points, point, grouping, image, device, 0, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, fused, _ = model(inputs, views=1, loss=False)
            vectors = fused[12][:, 0].float().cpu().numpy()
            if not np.isfinite(vectors).all():
                raise FloatingPointError(f"Nonfinite embeddings at row {start}")
            result[start:start + len(batch)] = vectors
    result.flush()
    del result
    with (args.output / "instance_map.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["embedding_row", "source_run_id", "body_id", "record_id", "split"])
        writer.writerows((i, *r["key"], r["split"]) for i, r in enumerate(rows))
    temporary.rename(args.output / "fused_12.npy")


if __name__ == "__main__":
    main()
