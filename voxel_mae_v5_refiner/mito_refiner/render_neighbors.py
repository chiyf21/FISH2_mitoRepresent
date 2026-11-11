from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from .data_utils import make_dataset
from .evaluate import plot_nearest_neighbor_shapes
from .neighbor_consensus import cosine_knn


def main():
    parser = argparse.ArgumentParser(description="Render physical-space nearest-neighbor masks from saved embeddings")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = np.load(args.embeddings)
    ids, bodies = data["record_id"], data["body_id"]
    methods = [name for name in ("pretrained", "refined_online", "refined_ema")
               if f"{name}_final" in data]
    neighbors = {name: cosine_knn(data[f"{name}_final"], 15) for name in methods}
    ds = make_dataset(args.cache, "test", 0, args.seed)
    if [str(row["record_id"]) for row in ds.records] != list(map(str, ids)):
        raise ValueError("Saved embedding order does not match the current held-out test manifest")
    if not plot_nearest_neighbor_shapes(args.out, ds, ids, bodies, neighbors, methods):
        raise RuntimeError("Matplotlib is unavailable; nearest-neighbor figure was not created")
    print(f"Wrote readable physical-space nearest-neighbor figure to {args.out}")


if __name__ == "__main__":
    main()
