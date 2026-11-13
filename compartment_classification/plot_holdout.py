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
    parser.add_argument("--datasets", nargs="+", default=["fish2", "flyem"])
    parser.add_argument("--evaluation", choices=["holdout", "development_oof"], default="holdout")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    record = dict(evaluation=args.evaluation, rows="true", columns="predicted",
                  normalization="within true class", datasets={})
    style = {"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
             "axes.labelsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11}
    for dataset in args.datasets:
        directory = args.results / dataset
        classes = json.loads((directory / "data_provenance.json").read_text())["classes"]
        display_classes = [name.replace("connecting cable", "connecting\ncable") for name in classes]
        manifest = pd.read_csv(directory / "instances.csv", dtype={"record_id": str, "body_id": str})
        splits = json.loads((directory / "splits.json").read_text())
        part = "holdout" if args.evaluation == "holdout" else "development"
        expected = manifest.iloc[splits["holdout"]["test" if part == "holdout" else "train"]].sort_values("record_id").reset_index(drop=True)
        assert expected.partition.eq(part).all()
        expected_n, expected_bodies = len(expected), expected.body_id.nunique()
        data = dict(n=len(expected), bodies=expected.body_id.nunique(), classes=classes, panels=[])
        assert (data["n"], data["bodies"]) == (expected_n, expected_bodies)
        with plt.rc_context(style):
            figure, axes = plt.subplots(2, 2, figsize=(9, 8))
            figure.subplots_adjust(left=0.13, right=0.97, bottom=0.09, top=0.88, wspace=0.43, hspace=0.43)
            for axis, (feature, classifier) in zip(axes.flat, PANELS):
                path = directory / feature / classifier / "predictions.csv"
                original = pd.read_csv(path, dtype={"record_id": str, "body_id": str})
                assert original.record_id.is_unique
                frame = original.loc[original.evaluation.eq(args.evaluation)].sort_values("record_id").reset_index(drop=True)
                assert frame.partition.eq(part).all()
                assert frame.outer_fold.eq(-1).all() if part == "holdout" else frame.outer_fold.ge(0).all()
                for column in ["record_id", "body_id", "label", "compartment"]:
                    assert frame[column].tolist() == expected[column].tolist(), (path, column)
                assert frame.compartment.map(dict(zip(classes, range(len(classes))))).tolist() == frame.label.tolist()
                counts = np.zeros((len(classes), len(classes)), dtype=np.int64)
                np.add.at(counts, (frame.label.to_numpy(dtype=int), frame.prediction.to_numpy(dtype=int)), 1)
                result_path = path.parent / "holdout/complete.json" if part == "holdout" else directory / feature / "complete.json"
                saved = json.loads(result_path.read_text())
                saved_metrics = saved["metrics"] if part == "holdout" else saved[classifier]["development_oof"]
                assert counts.tolist() == saved_metrics["confusion"]
                assert int(counts.sum()) == expected_n
                normalized = counts / counts.sum(axis=1, keepdims=True)
                np.testing.assert_allclose(normalized.sum(1), np.ones(len(classes)))
                axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        axis.text(j, i, f"{counts[i,j]}\n{normalized[i,j]:.1%}",
                                  ha="center", va="center", fontsize=13 if len(classes) == 3 else 15,
                                  color="white" if normalized[i,j] > 0.55 else "black")
                axis.set_xticks(range(len(classes)), display_classes)
                axis.set_yticks(range(len(classes)), display_classes)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("True")
                axis.set_title(feature + " / " + classifier.replace("_", " "))
                data["panels"].append(dict(feature=feature, classifier=classifier,
                    predictions=str(path.resolve()), predictions_sha256=sha(path),
                    results=str(result_path.resolve()), counts=counts.tolist(),
                    true_class_percent=(normalized * 100).tolist()))
            figure.suptitle(f"{dataset.upper()} / {args.evaluation} / n={expected_n}, bodies={expected_bodies}", fontsize=16, y=0.975)
            path = args.destination / f"{dataset}_{args.evaluation}_confusion.png"
            figure.savefig(path, dpi=240, facecolor="white")
            plt.close(figure)
        data["figure"] = str(path.resolve())
        data["figure_sha256"] = sha(path)
        record["datasets"][dataset] = data
        print(dataset, args.evaluation, data["n"], data["bodies"], path, flush=True)
    (args.destination / f"{args.evaluation}_inputs.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
