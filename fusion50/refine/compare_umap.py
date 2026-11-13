"""Compare author morphometrics with four Fusion50 layer-12 representations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


KEY = ("body_id", "record_id")
SPACES = {
    "author22": "Author morphometrics (22D)",
    "image_cls": "Image CLS (768D)",
    "point_cls": "Point-cloud CLS (384D)",
    "balanced_merge": "Variance-balanced merge (1,152D)",
    "fused": "Fusion MLP output (384D)",
}


def cohort(path):
    with path.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    keys = [tuple(row[k] for k in KEY) for row in table]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Expected unique author instance keys")
    return table, keys


def extract(args, keys):
    import torch
    from train import core, load_models, load_points, records

    config = json.loads(args.config.read_text())
    rows = records(config)
    points = load_points(config, rows)
    by_key = {tuple(row["key"][1:]): row for row in rows}
    selected = [by_key[key] for key in keys[args.shard::args.shards]]
    device = torch.device("cuda:0")
    point, grouping, image = load_models(config, device)
    model = core.System(point, image).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["stage"] != "refine":
        raise ValueError("Expected a fusion-stage checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.phase = "refine"
    vectors = {name: [] for name in tuple(SPACES)[1:]}
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start:start + args.batch_size]
            inputs = core.prepare(batch, points, point, grouping, image, device, 0, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p, i, fused, _ = model(inputs, views=1, loss=False)
                balanced, _ = model.balance(p[12], i[12], 12, update=False)
            for name, values in zip(vectors, (i[12], p[12], balanced, fused[12])):
                vector = values[:, 0].float().cpu().numpy()
                if not np.isfinite(vector).all():
                    raise FloatingPointError(f"{name}: nonfinite embedding at row {start}")
                vectors[name].append(vector)
            if start % (args.batch_size * 50) == 0:
                print(f"Shard {args.shard}: {start + len(batch)}/{len(selected)}", flush=True)
    path = args.output / f"vectors_{args.shard}.npz"
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, keys=np.asarray(keys[args.shard::args.shards]),
             **{name: np.concatenate(parts) for name, parts in vectors.items()})
    temporary.replace(path)


def plot(args, table, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from umap import UMAP

    feature_names = list(table[0])
    feature_names = feature_names[feature_names.index("size"):feature_names.index("SBI") + 1]
    author = np.asarray([[float(row[k]) for k in feature_names] for row in table])
    if author.shape != (len(table), 22) or np.any(author.std(axis=0) == 0):
        raise ValueError("Expected 22 nonconstant author features")
    vectors = {"author22": ((author - author.mean(axis=0)) / author.std(axis=0)).astype("float32")}
    for name, width in zip(tuple(SPACES)[1:], (768, 384, 1152, 384)):
        vectors[name] = np.empty((len(table), width), dtype="float32")
    for index in range(args.shards):
        with np.load(args.output / f"vectors_{index}.npz") as data:
            if data["keys"].tolist() != [list(key) for key in keys[index::args.shards]]:
                raise ValueError(f"Instance mismatch in shard {index}")
            for name in tuple(SPACES)[1:]:
                vectors[name][index::args.shards] = data[name]

    coords = {}
    for name, values in vectors.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {name}")
        print(f"Fitting UMAP: {name} {values.shape}", flush=True)
        coords[name] = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                            random_state=13, n_jobs=1).fit_transform(values)
        if not np.isfinite(coords[name]).all():
            raise ValueError(f"Nonfinite UMAP coordinates in {name}")

    columns = ("body_id", "record_id", "source_mito_label", "cell_type", "compartment")
    with (args.output / "umap_coordinates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*columns, *(f"{name}_{axis}" for name in coords for axis in ("x", "y"))])
        for index, row in enumerate(table):
            writer.writerow([*(row[k] for k in columns),
                             *(value for xy in coords.values() for value in xy[index])])

    fields = ("compartment", "cell_type")
    palettes = (
        {"axon": "#0072B2", "dendrite": "#D55E00", "connecting cable": "#009E73"},
        dict(zip(("LC4", "LC6", "LC10", "LC12", "LC17", "LC22"),
                 ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#535C74"))),
    )
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    order = np.random.default_rng(13).permutation(len(table))
    for row_index, (field, palette) in enumerate(zip(fields, palettes)):
        labels = np.asarray([row[field] for row in table])
        if set(labels) != set(palette):
            raise ValueError(f"Unexpected labels for {field}")
        handles = [Line2D([], [], linestyle="", marker="o", color=color,
                          label=f"{label} ({np.count_nonzero(labels == label):,})")
                   for label, color in palette.items()]
        for col_index, (name, xy) in enumerate(coords.items()):
            ax = axes[row_index, col_index]
            ax.scatter(*xy[order].T, c=[palette[label] for label in labels[order]],
                       s=2.6, alpha=0.68, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#D7DEE8")
            if row_index == 0:
                ax.set_title(SPACES[name], fontsize=10, pad=9)
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, (0.505, 0.037)[row_index]),
                   ncol=len(handles), frameon=False, fontsize=10, markerscale=0.85)
        fig.text(0.022, (0.70, 0.27)[row_index], ("Compartment", "Cell type")[row_index],
                 va="center", rotation=90, fontsize=11, weight="bold")
    fig.suptitle(f"FlyEM mitochondrial representations | {len(table):,} matched instances",
                 fontsize=17, weight="bold", y=0.973)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.09, top=0.88,
                        hspace=0.23, wspace=0.09)
    fig.savefig(args.output / "flyem_umap_comparison.png", dpi=200, facecolor="white")
    fig.savefig(args.output / "flyem_umap_comparison.pdf", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "plot"))
    parser.add_argument("--author", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.batch_size < 1:
        parser.error("Invalid shard or batch size")
    if args.mode == "extract" and not (args.config and args.checkpoint):
        parser.error("extract requires --config and --checkpoint")
    table, keys = cohort(args.author)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "extract":
        extract(args, keys)
    else:
        plot(args, table, keys)


if __name__ == "__main__":
    main()
