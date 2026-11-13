"""Render only the retained holdout confusion matrices from saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ["axon", "dendrite"]
PANELS = [("handcrafted", "logistic"), ("embedding", "logistic"),
          ("handcrafted", "random_forest"), ("embedding", "random_forest")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    record = dict(evaluation="holdout", rows="true", columns="predicted",
                  classes=CLASSES, normalization="within true class", datasets={})
    style = {"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
             "axes.labelsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11}
    for dataset, expected_n, expected_bodies in [("fish2", 1770, 23), ("flyem", 588, 11)]:
        directory = args.results / dataset
        manifest = pd.read_csv(directory / "instances.csv", dtype={"record_id": str, "body_id": str})
        splits = json.loads((directory / "splits.json").read_text())
        expected = manifest.iloc[splits["holdout"]["test"]].sort_values("record_id").reset_index(drop=True)
        assert expected.partition.eq("holdout").all()
        data = dict(n=len(expected), bodies=expected.body_id.nunique(), panels=[])
        assert (data["n"], data["bodies"]) == (expected_n, expected_bodies)
        with plt.rc_context(style):
            figure, axes = plt.subplots(2, 2, figsize=(9, 8))
            figure.subplots_adjust(left=0.13, right=0.97, bottom=0.09, top=0.88, wspace=0.43, hspace=0.43)
            for axis, (feature, classifier) in zip(axes.flat, PANELS):
                path = directory / feature / classifier / "predictions.csv"
                original = pd.read_csv(path, dtype={"record_id": str, "body_id": str})
                assert original.record_id.is_unique
                frame = original.loc[original.evaluation.eq("holdout")].sort_values("record_id").reset_index(drop=True)
                assert frame.partition.eq("holdout").all() and frame.outer_fold.eq(-1).all()
                for column in ["record_id", "body_id", "label", "compartment"]:
                    assert frame[column].tolist() == expected[column].tolist(), (path, column)
                assert frame.compartment.map({"axon": 0, "dendrite": 1}).tolist() == frame.label.tolist()
                counts = np.zeros((2, 2), dtype=np.int64)
                np.add.at(counts, (frame.label.to_numpy(dtype=int), frame.prediction.to_numpy(dtype=int)), 1)
                result_path = path.parent / "holdout/complete.json"
                saved = json.loads(result_path.read_text())
                assert counts.tolist() == saved["metrics"]["confusion"]
                assert int(counts.sum()) == expected_n
                normalized = counts / counts.sum(axis=1, keepdims=True)
                np.testing.assert_allclose(normalized.sum(1), [1, 1])
                axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
                for i in range(2):
                    for j in range(2):
                        axis.text(j, i, f"{counts[i,j]}\n{normalized[i,j]:.1%}",
                                  ha="center", va="center", fontsize=15,
                                  color="white" if normalized[i,j] > 0.55 else "black")
                axis.set_xticks([0, 1], CLASSES)
                axis.set_yticks([0, 1], CLASSES)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("True")
                axis.set_title(feature + " / " + classifier.replace("_", " "))
                data["panels"].append(dict(feature=feature, classifier=classifier,
                    predictions=str(path.resolve()), predictions_sha256=sha(path),
                    results=str(result_path.resolve()), counts=counts.tolist(),
                    true_class_percent=(normalized * 100).tolist()))
            figure.suptitle(f"{dataset.upper()} / holdout / n={expected_n}, bodies={expected_bodies}", fontsize=16, y=0.975)
            path = args.destination / f"{dataset}_holdout_confusion.png"
            figure.savefig(path, dpi=240, facecolor="white")
            plt.close(figure)
        data["figure"] = str(path.resolve())
        data["figure_sha256"] = sha(path)
        record["datasets"][dataset] = data
        print(dataset, "holdout", data["n"], data["bodies"], path, flush=True)
    (args.destination / "holdout_inputs.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
