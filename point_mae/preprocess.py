"""Build one compact Zarr point-cloud cache for mitochondria instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage

from pointmae import FEATURE_NAMES, load_instance_map, sample_instance, stable_int


def open_volume(path: str):
    if path.endswith(".zarr"):
        return zarr.open(path, mode="r")
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)


def append(array, values: np.ndarray) -> None:
    old = array.shape[0]
    new = old + values.shape[0]
    array.resize((new,) + array.shape[1:])
    array[old:new] = values


def main() -> None:
    p = argparse.ArgumentParser(description="Cache mitochondria as attributed physical point clouds")
    p.add_argument("--instance", required=True, help="ZYX instance-label TIFF or Zarr")
    p.add_argument("--raw", required=True, help="ZYX raw EM TIFF")
    p.add_argument("--out", required=True, help="Output cache directory, one Zarr group")
    p.add_argument("--spacing-zyx", nargs=3, type=float, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--coord-scale-nm", type=float, default=1000.0)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--surface-points", type=int, default=1024)
    p.add_argument("--max-candidates", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--neuron-map", help="JSON object: instance_id -> neuron_id")
    p.add_argument("--default-neuron-id", default="unknown", help="Fallback group ID when no per-instance map is available")
    p.add_argument("--max-instances", type=int)
    args = p.parse_args()

    labels = np.asarray(open_volume(args.instance))
    raw = open_volume(args.raw)
    if labels.ndim != 3 or tuple(labels.shape) != tuple(raw.shape):
        raise ValueError(f"Expected matching 3D ZYX arrays, got labels={labels.shape}, raw={raw.shape}")
    if args.surface_points > args.num_points:
        raise ValueError("surface-points cannot exceed num-points")

    ids = np.unique(labels)
    ids = ids[ids > 0].astype(np.int64)
    if args.max_instances:
        ids = ids[: args.max_instances]
    boxes = ndimage.find_objects(labels)
    neuron_map = load_instance_map(args.neuron_map)
    if not neuron_map and args.default_neuron_id == "unknown":
        print("WARNING: no neuron map supplied; all instances will use instance-specific placeholder IDs.")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cache: {out}")
    out.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(out), mode="w")
    group.attrs.update({
        "organelle_type": "mitochondria",
        "axes": "ZYX",
        "spacing_zyx": list(args.spacing_zyx),
        "coord_scale_nm": args.coord_scale_nm,
        "num_points": args.num_points,
        "surface_points": args.surface_points,
        "feature_names": list(FEATURE_NAMES),
    })
    chunk = min(args.batch_size, 64)
    xyz = group.create_dataset("xyz", shape=(0, args.num_points, 3), maxshape=(None, args.num_points, 3), chunks=(chunk, args.num_points, 3), dtype="f4")
    attrs = group.create_dataset("attrs", shape=(0, args.num_points, len(FEATURE_NAMES)), maxshape=(None, args.num_points, len(FEATURE_NAMES)), chunks=(chunk, args.num_points, len(FEATURE_NAMES)), dtype="f4")
    instance_ids = group.create_dataset("instance_id", shape=(0,), maxshape=(None,), chunks=(chunk,), dtype="i8")
    neuron_ids = group.create_dataset("neuron_id", shape=(0,), maxshape=(None,), chunks=(chunk,), dtype="i8")
    records_path = out / "records.jsonl"
    errors_path = out / "errors.jsonl"
    xyz_batch, attrs_batch, instance_batch, neuron_batch, records = [], [], [], [], []
    good = 0

    with records_path.open("w", encoding="utf-8") as records_file, errors_path.open("w", encoding="utf-8") as errors_file:
        for position, instance_id in enumerate(ids):
            bbox = boxes[int(instance_id) - 1] if int(instance_id) - 1 < len(boxes) else None
            if bbox is None:
                continue
            try:
                point_xyz, point_attrs = sample_instance(
                    labels,
                    raw,
                    int(instance_id),
                    args.spacing_zyx,
                    args.coord_scale_nm,
                    args.num_points,
                    args.surface_points,
                    args.seed + position,
                    args.max_candidates,
                    bbox=bbox,
                )
                raw_neuron = neuron_map.get(str(int(instance_id)), args.default_neuron_id if not neuron_map else f"unknown_{int(instance_id)}")
                neuron_id = stable_int(raw_neuron)
                xyz_batch.append(point_xyz)
                attrs_batch.append(point_attrs)
                instance_batch.append(int(instance_id))
                neuron_batch.append(neuron_id)
                record = {"cache_index": good, "instance_id": int(instance_id), "neuron_id": str(raw_neuron), "organelle_type": "mitochondria"}
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
                good += 1
                if len(xyz_batch) >= args.batch_size:
                    append(xyz, np.stack(xyz_batch))
                    append(attrs, np.stack(attrs_batch))
                    append(instance_ids, np.asarray(instance_batch, dtype=np.int64))
                    append(neuron_ids, np.asarray(neuron_batch, dtype=np.int64))
                    xyz_batch.clear(); attrs_batch.clear(); instance_batch.clear(); neuron_batch.clear()
            except Exception as exc:
                errors_file.write(json.dumps({"instance_id": int(instance_id), "error": repr(exc)}) + "\n")
        if xyz_batch:
            append(xyz, np.stack(xyz_batch))
            append(attrs, np.stack(attrs_batch))
            append(instance_ids, np.asarray(instance_batch, dtype=np.int64))
            append(neuron_ids, np.asarray(neuron_batch, dtype=np.int64))

    print(json.dumps({"cache": str(out), "instances": good, "errors": len(ids) - good}, indent=2))


if __name__ == "__main__":
    main()
