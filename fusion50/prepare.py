"""Prepare a Fusion50 surface cache and optional image blocks from an instance manifest."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

BLOCK = (32, 128, 128)


def windows(length, width):
    stop = max(0, length - width)
    return sorted(set(range(0, stop + 1, width)) | {stop})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Base for relative instance directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    records = source["records"] if isinstance(source, dict) else source
    if not records:
        raise ValueError("Empty instance manifest")
    root = (args.source_root or args.manifest.parent).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ("body_id", "record_id", "instance_id", "em_path", "mask_path",
              "normalization_radius_voxels", "split")
    rng = np.random.default_rng(args.seed)
    points = np.lib.format.open_memmap(args.output / "pointclouds.npy", mode="w+",
                                       dtype="float32", shape=(len(records), 2, 2048, 3))
    image_dir = args.output / "image_blocks"
    blocks = []
    if args.image_blocks:
        image_dir.mkdir()
    with (args.output / "manifest.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, row in enumerate(records):
            directory = root / row.get("relative_dir", "")
            em_path = (root / row["em_path"] if "em_path" in row else directory / "em.npy").resolve()
            mask_path = (root / row["mask_path"] if "mask_path" in row else directory / "mask.npy").resolve()
            em = np.load(em_path, mmap_mode="r", allow_pickle=False)
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False) > 0
            if mask.ndim != 3 or em.shape != mask.shape:
                raise ValueError(f"Expected aligned ZYX arrays: {mask_path}")
            surface = mask & ~binary_erosion(mask, structure=generate_binary_structure(3, 1))
            xyz = np.argwhere(surface)[:, ::-1].astype(np.float32)
            if len(xyz) < 2:
                raise ValueError(f"Degenerate surface: {mask_path}")
            xyz -= xyz.mean(axis=0)
            radius = float(np.linalg.norm(xyz, axis=1).max())
            for view in range(2):
                sample = rng.choice(len(xyz), size=2048, replace=len(xyz) < 2048)
                points[i, view] = xyz[sample] / radius
            identity = row.get("record_id", row.get("global_instance", i))
            writer.writerow(dict(body_id=row["body_id"], record_id=identity, instance_id=identity,
                                 em_path=em_path, mask_path=mask_path,
                                 normalization_radius_voxels=radius, split=row.get("split", "")))
            if args.image_blocks and row["split"] == "train":
                for start in itertools.product(*(windows(n, b) for n, b in zip(mask.shape, BLOCK))):
                    sl = tuple(slice(j, j + b) for j, b in zip(start, BLOCK))
                    if mask[sl].any():
                        name = f"{i:06d}_{len(blocks):06d}.npy"
                        np.save(image_dir / name, em[sl], allow_pickle=False)
                        blocks.append(name)
    points.flush()
    if args.image_blocks:
        if not blocks:
            raise ValueError("No training image blocks")
        (image_dir / "manifest.tsv").write_text("dataset_em\n" + "\n".join(blocks) + "\n")
    (args.output / "dataset_manifest.json").write_text(json.dumps(
        dict(instances=len(records), point_sampling="6-neighbor surface, two independent views",
             image_blocks=len(blocks), image_axes="zyx"), indent=2) + "\n")
    print(f"{len(records)} surface point pairs, {len(blocks)} training image blocks")


if __name__ == "__main__":
    main()
