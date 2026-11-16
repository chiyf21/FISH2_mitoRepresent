"""Regression against saved historical metrics, not a new scientific comparison."""
import argparse
from pathlib import Path
import numpy as np

from .common import BASE, read


def check(actual, expected):
    for key in ["balanced_accuracy", "macro_f1"]:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual["confusion"], expected["confusion"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rf-run", type=Path, required=True)
    p.add_argument("--lr-run", type=Path, required=True)
    args = p.parse_args()
    for directory in [args.rf_run, args.lr_run]:
        if not (directory / "_SUCCESS.json").is_file():
            raise ValueError("Reference verification requires a completed new run")
    rf = read(args.rf_run / "metrics.json")
    lr = read(args.lr_run / "metrics.json")
    compartment = BASE / "compartment_classification"
    gold_rf = read(compartment / "balanced_confusion_20261113_v1/fish2/metrics.json")
    gold_lr = read(compartment / "balanced_lr_20261113_v1/fish2/metrics.json")
    gold_cell = read(BASE / "point-MAE/runs/cell_type_fig2c_rf_20261113/results.json")["reviewed_81"]
    for feature, previous in [("handcrafted", "handcrafted"), ("embedding", "balanced")]:
        for level, old_level in [("per_mito", "per-mitochondrion"), ("per_group", "per-compartment")]:
            check(rf["compartment"][feature][level], gold_rf[old_level + "/" + previous])
            check(lr["compartment"][feature][level], gold_lr[old_level + "/" + previous])
        name = "handcrafted_22" if feature == "handcrafted" else "balanced_pair_12"
        old = next(row for row in gold_cell["results"] if row["representation"] == name)
        check(rf["cell_type"][feature]["per_group"], old)
    print("PASS: both features reproduce RF compartment, RF cell-type body aggregation, and LR compartment")
    print("BA/Macro-F1 tolerance=1e-12; confusion counts exact")


if __name__ == "__main__":
    main()
