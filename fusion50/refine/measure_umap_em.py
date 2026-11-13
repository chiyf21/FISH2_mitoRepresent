"""Measure EM intensity and texture on the matched FlyEM cohort."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, find_objects

from em_texture import METRICS as TEXTURE_METRICS, texture


METRICS = (
    "em_mean", "em_std", "em_shell_mean", "em_shell_minus_inside",
    "em_neighbor_abs_diff", *TEXTURE_METRICS,
)


def measure(row):
    em = np.load(row["em_path"], mmap_mode="r", allow_pickle=False)
    mask = np.load(row["mask_path"], mmap_mode="r", allow_pickle=False)
    if em.ndim != 3 or em.shape != mask.shape:
        raise ValueError(f"EM/mask shape mismatch for {row['record_id']}")
    box, = find_objects(mask, max_label=1)
    crop = tuple(slice(max(0, s.start - 2), min(n, s.stop + 2))
                 for s, n in zip(box, em.shape))
    foreground = mask[crop].astype(bool)
    image = np.asarray(em[crop], dtype=np.float32)
    inside = image[foreground]
    # Two face-connected voxel steps, clipped at the downloaded crop boundary.
    shell = image[binary_dilation(foreground, iterations=2) & ~foreground]
    total = 0.0
    pairs = 0
    for axis in range(3):
        m = np.moveaxis(foreground, axis, 0)
        v = np.moveaxis(image, axis, 0)
        valid = m[:-1] & m[1:]
        total += np.abs(v[:-1][valid] - v[1:][valid]).sum(dtype=np.float64)
        pairs += valid.sum()
    values = (inside.mean(), inside.std(), shell.mean(),
              shell.mean() - inside.mean(), total / pairs)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite EM feature for {row['record_id']}")
    return [str(row["body_id"]), str(row["record_id"]), *values, *texture(image, foreground)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", type=Path)
    parser.add_argument("author", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with args.author.open(newline="") as stream:
        keys = [(r["body_id"], r["record_id"]) for r in csv.DictReader(stream)]
    instances = json.loads(args.instances.read_text())
    by_key = {(str(row["body_id"]), str(row["record_id"])): row for row in instances}
    if len(by_key) != len(instances) or len(set(keys)) != len(keys):
        raise ValueError("Duplicate instance keys")
    selected = [by_key[key] for key in keys]
    temporary = args.output.with_suffix(".partial")
    with temporary.open("w", newline="") as stream, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.writer(stream)
        writer.writerow(["body_id", "record_id", *METRICS])
        for index, row in enumerate(pool.map(measure, selected), 1):
            writer.writerow(row)
            if index % 400 == 0 or index == len(selected):
                stream.flush()
                print(f"{index}/{len(selected)}", flush=True)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
