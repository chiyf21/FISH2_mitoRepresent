"""Export eight individual panels from saved balanced RF metrics; no fitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        for slug, dataset in [("flyem", "FlyEM"), ("fish2", "Fish2")]:
            directory = args.results / slug
            metrics = json.loads((directory / "metrics.json").read_text())
            provenance = json.loads((directory / "provenance.json").read_text())
            classes = provenance["classes"]
            ticks = list(range(len(classes)))
            labels = [c.replace("connecting cable", "connecting\ncable") for c in classes]
            for task, short in [("per-mitochondrion", "per_mito"), ("per-compartment", "per_compartment")]:
                count = provenance["instances" if short == "per_mito" else "groups"]
                for feature in ["handcrafted", "balanced"]:
                    result = metrics[task + "/" + feature]
                    cm = np.asarray(result["confusion"])
                    norm = np.asarray(result["confusion_normalized"])
                    assert cm.shape == norm.shape == (len(classes), len(classes))
                    assert cm.sum() == count
                    np.testing.assert_allclose(norm, cm / cm.sum(1, keepdims=True))
                    fig, ax = plt.subplots(figsize=(6.2, 6.2))
                    fig.subplots_adjust(left=.23, right=.96, bottom=.25, top=.78)
                    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
                    for i in ticks:
                        for j in ticks:
                            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}",
                                    ha="center", va="center", fontsize=14,
                                    color="white" if norm[i, j] > .55 else "black")
                    ax.set_xticks(ticks, labels)
                    ax.set_yticks(ticks, labels)
                    ax.set_xlabel("Predicted compartment", labelpad=8)
                    ax.set_ylabel("True compartment", labelpad=8)
                    name = "Handcrafted 22D" if feature == "handcrafted" else "Balanced 1152D"
                    ax.set_title(f"{dataset} | {task}\n{name}\n"
                                 f"BA={result['balanced_accuracy']:.3f} | "
                                 f"Macro-F1={result['macro_f1']:.3f}", fontsize=12, pad=14)
                    fig.text(.5, .965, f"n={count:,} | bodies={provenance['bodies']} | 5-fold body OOF",
                             ha="center", fontsize=11)
                    footer = "RF by neuron type | Count + true-class normalized %"
                    if short == "per_compartment":
                        footer += "\nOracle grouping: body + true compartment"
                    fig.text(.5, .035, footer, ha="center", va="bottom", fontsize=10)
                    task_name = "per-mito" if short == "per_mito" else "per-compart"
                    path = args.output / f"{slug}_{feature}_{task_name}.png"
                    fig.savefig(path, dpi=220, facecolor="white")
                    plt.close(fig)
                    print(path.name, flush=True)
    assert len(list(args.output.glob("*.png"))) == 8


if __name__ == "__main__":
    main()
