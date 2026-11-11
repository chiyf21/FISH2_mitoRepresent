"""Plot PCA and UMAP views of Scale-aware Voxel MAE embeddings."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA


def l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--pca-components", type=int, default=30)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    x = np.asarray(data["embedding"], dtype=np.float32)
    record_ids = np.asarray(data["record_id"]).astype(str)
    body_ids = np.asarray(data["body_id"]).astype(str)
    manifest = json.loads(Path(args.manifest).read_text())
    metadata = {r["record_id"]: r for r in manifest["records"]}
    missing = [rid for rid in record_ids if rid not in metadata]
    if missing:
        raise ValueError(f"{len(missing)} embedding record IDs are missing from manifest")
    cell_types = np.asarray([metadata[rid].get("cell_type") or "NA" for rid in record_ids])
    splits = np.asarray([metadata[rid]["split"] for rid in record_ids])

    x_norm = l2_normalize(x)
    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(x)
    pca = PCA(n_components=min(args.pca_components, x.shape[0] - 1, x.shape[1]), random_state=args.seed)
    x_pc = l2_normalize(pca.fit_transform(x_norm).astype(np.float32))
    reducer_args = dict(n_neighbors=args.neighbors, min_dist=args.min_dist, metric="cosine",
                        random_state=args.seed, n_components=2, n_jobs=1)
    umap_direct = umap.UMAP(**reducer_args).fit_transform(x_norm)
    umap_pc = umap.UMAP(**reducer_args).fit_transform(x_pc)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    categories = sorted(np.unique(cell_types).tolist())
    colors = plt.get_cmap("tab20", max(1, len(categories)))
    panels = [(pca2, "PCA of raw 256-D latent"),
              (umap_direct, "Direct UMAP (cosine on L2-normalized latent)"),
              (umap_pc, f"PCA-{x_pc.shape[1]} → UMAP (cosine)")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
    for ax, (xy, title) in zip(axes, panels):
        for i, label in enumerate(categories):
            take = cell_types == label
            ax.scatter(xy[take, 0], xy[take, 1], s=7, alpha=0.72, color=colors(i),
                       label=f"{label} (n={int(take.sum())})", linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.16, linewidth=0.5)
    if len(categories) <= 20:
        axes[-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Scale-aware Voxel MAE: unmasked encoder embeddings (n={len(x)}, dim={x.shape[1]})")
    fig.savefig(out / "pca_umap_by_cell_type.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (out / "embedding_pca_umap_coordinates.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "body_id", "split", "cell_type", "pca1", "pca2",
                         "umap_direct1", "umap_direct2", "umap_pc1", "umap_pc2"])
        writer.writerows(zip(record_ids, body_ids, splits, cell_types, pca2[:, 0], pca2[:, 1],
                             umap_direct[:, 0], umap_direct[:, 1], umap_pc[:, 0], umap_pc[:, 1]))
    summary = {
        "embedding_file": args.embeddings,
        "n_instances": int(len(x)),
        "embedding_dim": int(x.shape[1]),
        "embedding_norm_mean": float(np.linalg.norm(x, axis=1).mean()),
        "pca_input": "raw latent; mean-centered by sklearn PCA",
        "pca_components_for_umap": int(x_pc.shape[1]),
        "pca_explained_variance_ratio_2d": float(pca.explained_variance_ratio_[:2].sum()),
        "pca_explained_variance_ratio_used": float(pca.explained_variance_ratio_.sum()),
        "umap": {"metric": "cosine", "neighbors": args.neighbors, "min_dist": args.min_dist,
                 "seed": args.seed, "input_l2_normalized": True},
        "cell_type_counts": {label: int(np.sum(cell_types == label)) for label in categories},
        "split_counts": {label: int(np.sum(splits == label)) for label in sorted(np.unique(splits))},
        "note": "UMAP is a visualization of the unmasked encoder latent, not evidence by itself of cluster validity."
    }
    (out / "pca_umap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
