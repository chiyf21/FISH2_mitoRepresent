"""Held-out cross-layer neighbor-overlap diagnostic; does not train a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import TARGET_NAMES
from .neighbor_consensus import consensus_summary, shared_neighbor_target_differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="E0 embeddings_test_heldout.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = np.load(args.npz)
    layers = {name: data[f"pretrained_{name}"]
              for name in ("block4", "block5", "block6_mean")}
    bodies, targets = data["body_id"], data["targets"]
    result = {
        "n_heldout": int(len(bodies)),
        "layers": list(layers),
        "neighbor_consensus": consensus_summary(layers, bodies, ks=(5, 20)),
        "shared_neighbor_proxy_differences_top20": shared_neighbor_target_differences(
            layers, targets, TARGET_NAMES, k=20),
        "interpretation": "diagnostic only; no consensus loss or model training was run",
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
