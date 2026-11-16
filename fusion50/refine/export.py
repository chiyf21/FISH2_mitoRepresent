"""Export L12 Point, Image, Concat1536 and Project384 in manifest order."""
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["stage"] != "refine":
        raise ValueError("Use a fusion-stage checkpoint")
    saved = checkpoint['config']
    for key in ('point_run', 'point_checkpoint', 'image_checkpoint'):
        if config[key] != saved[key]:
            raise ValueError(f'Export config and checkpoint disagree on {key}')
    point, grouping, image = load_models(config, device)
    model = core.System(point, image).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.phase = "refine"
    args.output.mkdir(parents=True, exist_ok=False)
    dimensions = {'point': 768, 'image': 768, 'concat': 1536, 'fused': 384}
    results = {name: np.lib.format.open_memmap(
        args.output / f'{name}_12.partial.npy', mode='w+', dtype='float32',
        shape=(len(rows), dim)) for name, dim in dimensions.items()}
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            inputs = core.prepare(batch, points, point, grouping, image, device, 0, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                point_cls, image_cls, fused, concat = model(inputs, views=1, loss=False)
            for name, by_layer in zip(dimensions, (point_cls, image_cls, concat, fused)):
                vectors = by_layer[12][:, 0].float().cpu().numpy()
                if not np.isfinite(vectors).all():
                    raise FloatingPointError(f'Nonfinite {name} embeddings at row {start}')
                results[name][start:start + len(batch)] = vectors
    for result in results.values():
        result.flush()
    results.clear()
    del result
    with (args.output / "instance_map.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["embedding_row", "source_run_id", "body_id", "record_id", "split"])
        writer.writerows((i, *r["key"], r["split"]) for i, r in enumerate(rows))
    for name in dimensions:
        (args.output / f'{name}_12.partial.npy').rename(args.output / f'{name}_12.npy')


if __name__ == "__main__":
    main()
