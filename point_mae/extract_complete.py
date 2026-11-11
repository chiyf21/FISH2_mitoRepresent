"""Keep only instances that do not touch any of the six volume boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr


def open_volume(path: str):
    if path.endswith(".zarr"):
        return zarr.open(path, mode="r")
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract complete mitochondria from an instance volume")
    p.add_argument("--instance", required=True)
    p.add_argument("--out", required=True, help="Filtered instance Zarr")
    p.add_argument("--summary", required=True, help="JSON summary")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--chunk-z", type=int, default=8)
    p.add_argument("--min-voxels", type=int, default=0)
    args = p.parse_args()

    labels = open_volume(args.instance)
    shape = tuple(labels.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected ZYX 3D labels, got {shape}")
    labels_np = np.asarray(labels)
    ids, counts = np.unique(labels_np, return_counts=True)
    foreground = ids > 0
    ids = ids[foreground].astype(np.int64)
    counts = counts[foreground].astype(np.int64)
    if len(ids) == 0:
        raise ValueError("No foreground instances found")

    boundary_ids: set[int] = set()
    boundary_views = (
        labels_np[0], labels_np[-1],
        labels_np[:, 0, :], labels_np[:, -1, :],
        labels_np[:, :, 0], labels_np[:, :, -1],
    )
    for view in boundary_views:
        boundary_ids.update(int(x) for x in np.unique(view) if int(x) > 0)
    count_map = {int(i): int(c) for i, c in zip(ids, counts)}
    all_ids = set(count_map)
    size_ok_ids = {i for i, c in count_map.items() if c >= args.min_voxels}
    complete_ids = sorted((all_ids - boundary_ids) & size_ok_ids)
    keep = np.zeros(int(ids.max()) + 1, dtype=bool)
    keep[np.asarray(complete_ids, dtype=np.int64)] = True

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open(str(out_path), mode="w", shape=shape, chunks=(args.chunk_z, min(512, shape[1]), min(512, shape[2])), dtype=labels_np.dtype, compressor=None)
    for z0 in range(0, shape[0], args.chunk_z):
        z1 = min(shape[0], z0 + args.chunk_z)
        block = labels_np[z0:z1]
        out[z0:z1] = np.where(keep[block], block, 0).astype(labels_np.dtype, copy=False)
    out.attrs.update({
        "axes": ["z", "y", "x"],
        "source": args.instance,
        "dataset_name": args.dataset_name,
        "completeness": "instance does not touch any of six volume boundary faces",
        "min_voxels": args.min_voxels,
        "complete_instance_count": len(complete_ids),
        "min_voxels": args.min_voxels,
        "incomplete_instance_count": len(all_ids - set(complete_ids)),
        "boundary_instance_count": len(all_ids & boundary_ids),
        "small_instance_count": len(all_ids - size_ok_ids),
    })

    summary = {
        "dataset_name": args.dataset_name,
        "source": args.instance,
        "output": str(out_path),
        "shape_zyx": list(shape),
        "total_instance_count": len(count_map),
        "complete_instance_count": len(complete_ids),
        "incomplete_instance_count": len(count_map) - len(complete_ids),
        "min_voxels": args.min_voxels,
        "boundary_instance_count": len(all_ids & boundary_ids),
        "small_instance_count": len(all_ids - size_ok_ids),
        "total_foreground_voxels": int(sum(count_map.values())),
        "complete_foreground_voxels": int(sum(count_map[i] for i in complete_ids)),
        "boundary_rule": "exclude labels touching z=0,z=Z-1,y=0,y=Y-1,x=0,x=X-1",
        "complete_instance_ids": complete_ids,
        "complete_instance_voxels": {str(i): count_map[i] for i in complete_ids},
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
