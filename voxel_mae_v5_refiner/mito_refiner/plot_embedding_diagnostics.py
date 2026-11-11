"""Plot held-out final-embedding PCA, optionally UMAP, colored by metadata proxies."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGETS = ((0, "log10 volume"), (1, "elongation"),
           (3, "sphericity"), (4, "EM intensity mean"),
           (5, "EM intensity std"), (7, "gradient magnitude"))


def plot(npz_path: Path, out_dir: Path, reduction: str) -> Path:
    data = np.load(npz_path)
    targets = data["targets"]
    methods = [name for name in ("pretrained", "refined_online", "refined_ema")
               if f"{name}_final" in data]
    if not methods:
        raise ValueError(f"No final embeddings found in {npz_path}")
    transform = None
    if reduction == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("umap-learn is not installed") from exc
        transform = umap.UMAP(n_components=2, n_neighbors=15, min_dist=.25,
                              metric="cosine", random_state=42)
    fig, axes = plt.subplots(len(methods), len(TARGETS), figsize=(4.0 * len(TARGETS),
                            3.4 * len(methods)), squeeze=False, constrained_layout=True)
    for row, method in enumerate(methods):
        values = data[f"{method}_final"].astype(np.float32)
        if reduction == "pca":
            centered = values - values.mean(axis=0, keepdims=True)
            _, _, directions = np.linalg.svd(centered, full_matrices=False)
            xy = centered @ directions[:2].T
        else:
            xy = transform.fit_transform(values)
        for col, (idx, label) in enumerate(TARGETS):
            ax = axes[row, col]
            value = targets[:, idx]
            vmin, vmax = np.nanquantile(value, [.02, .98])
            points = ax.scatter(xy[:, 0], xy[:, 1], c=value, s=5, alpha=.85,
                                cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
            ax.set_title(f"{method}: {label}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(points, ax=ax, fraction=.045, pad=.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{reduction}_final_embedding_metadata.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reduction", choices=("pca", "umap"), default="pca")
    args = parser.parse_args()
    print(plot(Path(args.npz), Path(args.out_dir), args.reduction))


if __name__ == "__main__":
    main()
