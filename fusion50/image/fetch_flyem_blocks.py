"""Fetch grayscale FlyEM blocks from a global XYZ candidate CSV."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import itertools
import json
from pathlib import Path
import threading
import time

from cloudvolume import CloudVolume
import numpy as np

EM_URL = ("precomputed://https://storage.googleapis.com/"
          "neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg")
SIZE = 128
_tls = threading.local()


def volume():
    if not hasattr(_tls, "em"):
        _tls.em = CloudVolume(EM_URL, mip=0, progress=False, fill_missing=False,
                              cache=False, parallel=1)
    return _tls.em


def heldout_blocks(manifest):
    heldout = set()
    origins = set()
    if manifest is None:
        return heldout, origins
    data = json.loads(manifest.read_text())
    for row in data["records"] if isinstance(data, dict) else data:
        if row["split"] == "train":
            continue
        heldout.add(str(row["body_id"]))
        box = row["bbox_xyz"]
        axes = [range(lo // SIZE * SIZE, hi, SIZE) for lo, hi in zip(box[:3], box[3:])]
        origins.update(itertools.product(*axes))
    return heldout, origins


def fetch(origin, output):
    name = "_".join(map(str, origin)) + ".npy"
    path = output / name
    if path.exists():
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        if cached.shape == (SIZE,) * 3 and cached.dtype == np.uint8:
            return name, None
    for attempt in range(3):
        try:
            selection = tuple(slice(x, x + SIZE) for x in origin)
            data = np.asarray(volume()[selection])
            if data.shape == (SIZE, SIZE, SIZE, 1):
                data = data[..., 0]
            if data.shape != (SIZE,) * 3 or data.dtype != np.uint8:
                raise ValueError(f"Unexpected shape/dtype: {data.shape} {data.dtype}")
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.save(stream, data, allow_pickle=False)
            temporary.replace(path)
            return name, None
        except Exception as exc:
            if attempt == 2:
                return name, f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    blocked_bodies, blocked_origins = heldout_blocks(args.holdout_manifest)
    source = volume()
    if tuple(source.resolution) != (8, 8, 8):
        raise ValueError("Expected 8 nm isotropic source")
    minimum = tuple(source.bounds.minpt)
    maximum = tuple(source.bounds.maxpt)
    eligible, seen = [], set()
    skipped = {"heldout": 0, "outside_volume": 0}
    with args.candidates.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            lo = tuple(int(row[f"block_{a}0"]) for a in "xyz")
            hi = tuple(int(row[f"block_{a}1"]) for a in "xyz")
            if lo in seen or any(a % SIZE or b-a != SIZE for a, b in zip(lo, hi)):
                raise ValueError(f"Duplicate or invalid {SIZE}-voxel block: {lo}")
            seen.add(lo)
            if blocked_bodies.intersection(row["body_ids"].split(";")) or lo in blocked_origins:
                skipped["heldout"] += 1
            elif any(a < left or b > right for a,b,left,right in zip(lo,hi,minimum,maximum)):
                skipped["outside_volume"] += 1
            else:
                eligible.append(lo)
    if not eligible:
        raise ValueError("No eligible blocks")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(eligible)} eligible blocks; excluded {skipped}", flush=True)
    filenames, failures = [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, error) in enumerate(
            pool.map(lambda origin: fetch(origin, args.output), eligible), 1
        ):
            if error is None:
                filenames.append(name)
            else:
                failures[name] = error
            if index % 100 == 0 or index == len(eligible):
                print(f"{index}/{len(eligible)} processed; {len(failures)} failures", flush=True)
    summary = dict(candidates=len(seen), eligible=len(eligible), downloaded=len(filenames),
                   excluded=skipped, failures=failures)
    (args.output / "fetch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise RuntimeError("Some blocks could not be read; rerun to retry missing blocks")
    temporary = args.output / "manifest.partial"
    temporary.write_text("dataset_em\n" + "\n".join(filenames) + "\n")
    temporary.replace(args.output / "manifest.tsv")


if __name__ == "__main__":
    main()
